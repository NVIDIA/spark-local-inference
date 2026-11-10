/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Placement: which nodes and GPUs of the pool run a core's ranks. See docs/orbit.md,
//! "Placement".
//!
//! [`plan`] is a pure function of what the core needs ([`Needs`]), what every node's STATUS
//! reports, and the operator's [`Constraints`]. It applies these rules:
//!
//! 1. A GPU is a candidate only when its hardware revision is the core's.
//! 2. A GPU is free for a rank when no running group holds it and its free memory covers the
//!    rank's device bytes plus the core's reserve.
//! 3. A multi-host core takes one node per rank and one free GPU on each. A single-host
//!    core takes one free GPU per rank on one node, preferring contiguous NVML indices, or
//!    exactly the constrained indices.
//! 4. A node list narrows the candidates and, for a multi-host core, fixes rank order.
//! 5. Without a node list, a multi-host rank prefers a node holding its files (the complete
//!    core for rank zero, the rank's own directory for the others), then the node whose GPU
//!    has more free memory, then the node name; a single-host core prefers a node holding
//!    the complete core, then the node with more free memory.
//! 6. A rank taking one GPU on a node takes the free candidate with the lowest index.
//!
//! A node holds a cache directory when its STATUS lists the directory as verified; the node a
//! `NODE:PATH` launch names holds the complete core.

use std::cmp::Ordering;

use super::Failure;
use super::controller::{GpuStatus, NodeStatus};

const GIB: f64 = (1u64 << 30) as f64;

/// What a core needs from the pool.
#[derive(Clone, Debug)]
pub struct Needs {
    /// The hardware revision every GPU of the group must have: `gb10`, `gb202`, `gb300`, or
    /// `sm{major}{minor}`.
    pub revision: String,
    /// True for a multi-host core: one node per rank, one GPU each. False for a single-host
    /// tensor-parallel core: one node with one GPU per rank, run as one process.
    pub multi_host: bool,
    /// Per rank, in rank order: the device bytes the rank allocates before its KV cache.
    pub rank_bytes: Vec<u64>,
    /// Device bytes the core keeps free beyond its allocations (its serving reserve), added
    /// to each rank's need.
    pub reserve_bytes: u64,
    /// The cache directory holding the complete core, `{export}-{length}`.
    pub complete_directory: String,
    /// The cache directory holding each rank's files, `{export}-{length}-rank{r}`, indexed by
    /// rank. A rank without an entry has no rank directory, so an empty list suits a core
    /// that is cached only complete.
    pub rank_directories: Vec<String>,
    /// The node holding the core outside the cache, the source of a `NODE:PATH` launch. It
    /// counts as holding the complete core.
    pub source_node: Option<String>,
}

/// Operator constraints on a placement.
#[derive(Clone, Debug, Default)]
pub struct Constraints {
    /// Candidate nodes, from `--nodes`, a group's `nodes`, or a cluster. Empty admits every
    /// node of the pool. For a multi-host core the list also fixes rank order: each rank
    /// takes the earliest listed node after the previous rank's node that can serve it, so a
    /// list of exactly one node per rank places rank `r` on the `r`-th node.
    pub nodes: Vec<String>,
    /// NVML indices every member uses on its node, in the order its ranks use them. Empty lets
    /// the planner choose. A multi-host member uses one GPU, so the list then holds one index.
    pub gpus: Vec<u32>,
}

/// One member of a plan: one process on one node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub rank: usize,
    /// Index into the `nodes` slice given to [`plan`].
    pub node: usize,
    /// GPU UUIDs in the order the member's ranks use them.
    pub gpus: Vec<String>,
    /// The node holds this rank's directory or the complete core verified, or is the
    /// core's source node.
    pub cached: bool,
}

/// Place `needs` on the pool `nodes`, whose STATUS reports `statuses` holds in the same order,
/// within `constraints`.
///
/// A multi-host core yields one member per rank, in rank order, on distinct nodes, each with
/// one GPU. A single-host core yields one member, rank zero, holding one GPU per
/// tensor-parallel rank in rank order; a GPU chosen by the planner serves ranks in ascending
/// index order. The error names the nodes and GPUs that cannot serve and why.
pub fn plan(
    needs: &Needs,
    nodes: &[String],
    statuses: &[NodeStatus],
    constraints: &Constraints,
) -> Result<Vec<Placement>, Failure> {
    if needs.rank_bytes.is_empty() {
        return Err("the core has no ranks".into());
    }
    if nodes.len() != statuses.len() {
        return Err(format!(
            "{} node names were given with {} STATUS reports",
            nodes.len(),
            statuses.len()
        )
        .into());
    }
    for (position, index) in constraints.gpus.iter().enumerate() {
        if constraints.gpus[..position].contains(index) {
            return Err(format!("the GPU list names gpu {index} twice").into());
        }
    }
    let candidates = candidates(nodes, constraints)?;
    let pool = Pool {
        needs,
        nodes,
        statuses,
    };
    if !needs.multi_host {
        return pool.single_host(&candidates, &constraints.gpus);
    }
    if constraints.gpus.len() > 1 {
        return Err(format!(
            "a multi-host core's member uses one GPU and the GPU list names {}",
            constraints.gpus.len()
        )
        .into());
    }
    let index = constraints.gpus.first().copied();
    if constraints.nodes.is_empty() {
        pool.match_ranks(&candidates, index)
    } else {
        pool.in_order(&candidates, index)
    }
}

/// The plan as the controller prints it, one line per member without a trailing newline, such
/// as `rank 0 on maxwell: gpu 0 NVIDIA GB10 (cached)`.
pub fn describe(plan: &[Placement], nodes: &[String], statuses: &[NodeStatus]) -> String {
    plan.iter()
        .map(|member| {
            let gpus: Vec<String> = member
                .gpus
                .iter()
                .map(|uuid| {
                    statuses
                        .get(member.node)
                        .and_then(|status| status.gpus.iter().find(|g| &g.uuid == uuid))
                        .map_or_else(|| uuid.clone(), |g| format!("gpu {} {}", g.index, g.name))
                })
                .collect();
            format!(
                "rank {} on {}: {} ({})",
                member.rank,
                nodes.get(member.node).map_or("?", String::as_str),
                gpus.join(", "),
                if member.cached {
                    "cached"
                } else {
                    "not cached"
                }
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Pool indices of the candidate nodes: every node in pool order, or the listed nodes in list
/// order.
fn candidates(nodes: &[String], constraints: &Constraints) -> Result<Vec<usize>, Failure> {
    if constraints.nodes.is_empty() {
        return Ok((0..nodes.len()).collect());
    }
    let mut listed: Vec<usize> = Vec::with_capacity(constraints.nodes.len());
    for name in &constraints.nodes {
        let Some(index) = nodes.iter().position(|n| n == name) else {
            return Err(if nodes.is_empty() {
                format!("node {name} is not in the pool, which is empty")
            } else {
                format!(
                    "node {name} is not in the pool, which holds {}",
                    nodes.join(", ")
                )
            }
            .into());
        };
        if listed.contains(&index) {
            return Err(format!("the node list names {name} twice").into());
        }
        listed.push(index);
    }
    Ok(listed)
}

/// A core's needs with the pool it is placed on.
struct Pool<'a> {
    needs: &'a Needs,
    nodes: &'a [String],
    statuses: &'a [NodeStatus],
}

impl<'a> Pool<'a> {
    /// Free device bytes `rank` needs on its GPU.
    fn need(&self, rank: usize) -> u64 {
        self.needs.rank_bytes[rank].saturating_add(self.needs.reserve_bytes)
    }

    fn holds(&self, node: usize, directory: &str) -> bool {
        self.statuses[node]
            .cores
            .iter()
            .any(|p| p.verified && p.name == directory)
    }

    fn holds_complete(&self, node: usize) -> bool {
        self.needs.source_node.as_deref() == Some(self.nodes[node].as_str())
            || self.holds(node, &self.needs.complete_directory)
    }

    fn holds_rank(&self, node: usize, rank: usize) -> bool {
        self.needs
            .rank_directories
            .get(rank)
            .is_some_and(|directory| self.holds(node, directory))
    }

    fn cached(&self, node: usize, rank: usize) -> bool {
        self.holds_rank(node, rank) || self.holds_complete(node)
    }

    /// How strongly `rank` prefers `node` for its files, lower first: rank zero prefers the
    /// complete core, from which the other ranks fetch, then its own directory; every other
    /// rank prefers its own directory, then the complete core.
    fn tier(&self, node: usize, rank: usize) -> u8 {
        let (first, second) = if rank == 0 {
            (self.holds_complete(node), self.holds_rank(node, rank))
        } else {
            (self.holds_rank(node, rank), self.holds_complete(node))
        };
        if first {
            0
        } else if second {
            1
        } else {
            2
        }
    }

    /// Orders two serving nodes for `rank`, preferred first: by [`Pool::tier`], then by the free
    /// memory of the GPU the rank would take, more first, then by node name.
    fn prefer(&self, rank: usize, a: (usize, &GpuStatus), b: (usize, &GpuStatus)) -> Ordering {
        self.tier(a.0, rank)
            .cmp(&self.tier(b.0, rank))
            .then(b.1.memory_free.cmp(&a.1.memory_free))
            .then_with(|| self.nodes[a.0].cmp(&self.nodes[b.0]))
    }

    /// The GPU `rank` takes on `node`: the GPU at `index` when the operator names one, otherwise
    /// the free candidate with the lowest index. The error says why every GPU fails.
    fn gpu(&self, node: usize, rank: usize, index: Option<u32>) -> Result<&'a GpuStatus, String> {
        let statuses = self.statuses;
        let status = &statuses[node];
        let need = self.need(rank);
        if let Some(index) = index {
            let gpu = status
                .gpus
                .iter()
                .find(|g| g.index == index)
                .ok_or_else(|| missing(status, index))?;
            return match unfit(gpu, &self.needs.revision, need) {
                Some(reason) => Err(reason),
                None => Ok(gpu),
            };
        }
        if status.gpus.is_empty() {
            return Err(no_gpus(status));
        }
        let mut gpus: Vec<&'a GpuStatus> = status.gpus.iter().collect();
        gpus.sort_by_key(|g| g.index);
        let mut reasons = Vec::new();
        for gpu in gpus {
            match unfit(gpu, &self.needs.revision, need) {
                None => return Ok(gpu),
                Some(reason) => reasons.push(reason),
            }
        }
        Err(reasons.join(", "))
    }

    /// Multi-host placement without a node list: a maximum matching of ranks to distinct
    /// nodes. Each rank in order takes its most preferred unclaimed node; a rank finding every
    /// serving node claimed moves earlier ranks to other nodes of theirs along an augmenting
    /// path, so the placement fails only when no assignment of ranks to nodes exists.
    fn match_ranks(
        &self,
        candidates: &[usize],
        index: Option<u32>,
    ) -> Result<Vec<Placement>, Failure> {
        let ranks = self.needs.rank_bytes.len();
        if candidates.len() < ranks {
            return Err(format!(
                "the core's {ranks} ranks need {ranks} nodes and the pool has {}",
                candidates.len()
            )
            .into());
        }
        let preferences: Vec<Vec<(usize, &'a GpuStatus)>> = (0..ranks)
            .map(|rank| {
                let mut serving: Vec<(usize, &'a GpuStatus)> = candidates
                    .iter()
                    .filter_map(|&node| self.gpu(node, rank, index).ok().map(|gpu| (node, gpu)))
                    .collect();
                serving.sort_by(|a, b| self.prefer(rank, *a, *b));
                serving
            })
            .collect();
        let mut owner: Vec<Option<usize>> = vec![None; self.nodes.len()];
        let mut taken: Vec<Option<usize>> = vec![None; ranks];
        for rank in 0..ranks {
            let direct = preferences[rank]
                .iter()
                .position(|&(node, _)| owner[node].is_none());
            if let Some(position) = direct {
                owner[preferences[rank][position].0] = Some(rank);
                taken[rank] = Some(position);
                continue;
            }
            let mut visited = vec![false; self.nodes.len()];
            if !augment(&preferences, &mut owner, &mut taken, rank, &mut visited) {
                return Err(self.unmatched(candidates, &preferences, rank, index).into());
            }
        }
        Ok((0..ranks)
            .map(|rank| {
                let position = taken[rank].expect("every rank holds a node once matching ends");
                let (node, gpu) = preferences[rank][position];
                Placement {
                    rank,
                    node,
                    gpus: vec![gpu.uuid.clone()],
                    cached: self.cached(node, rank),
                }
            })
            .collect())
    }

    /// Why matching found no node for `rank`.
    fn unmatched(
        &self,
        candidates: &[usize],
        preferences: &[Vec<(usize, &GpuStatus)>],
        rank: usize,
        index: Option<u32>,
    ) -> String {
        let ranks = preferences.len();
        let revision = &self.needs.revision;
        if preferences[rank].is_empty() {
            let reasons: Vec<String> = candidates
                .iter()
                .filter_map(|&node| {
                    self.gpu(node, rank, index)
                        .err()
                        .map(|reason| format!("{}: {reason}", self.nodes[node]))
                })
                .collect();
            return format!(
                "rank {rank} needs a free {revision} GPU with {} free and no node has one: {}",
                gib(self.need(rank)),
                reasons.join("; ")
            );
        }
        let serving: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|node| {
                preferences
                    .iter()
                    .any(|serving| serving.iter().any(|(n, _)| n == node))
            })
            .collect();
        if serving.len() < ranks {
            // A node that serves no rank fails even the smallest need; report that failure.
            let smallest = (0..ranks).min_by_key(|&r| self.need(r)).unwrap_or(0);
            let mut text = format!(
                "the core's {ranks} ranks need {ranks} nodes with a free {revision} GPU, and {} \
                 qualify ({})",
                serving.len(),
                self.names(serving.iter().copied())
            );
            for &node in candidates {
                if serving.contains(&node) {
                    continue;
                }
                if let Err(reason) = self.gpu(node, smallest, index) {
                    text += &format!("; {}: {reason}", self.nodes[node]);
                }
            }
            return text;
        }
        format!(
            "rank {rank} can run only on {}, which the other ranks need",
            self.names(preferences[rank].iter().map(|&(node, _)| node))
        )
    }

    /// Multi-host placement on a node list: rank `r` takes the earliest listed node after rank
    /// `r - 1`'s node that can serve it, leaving enough listed nodes for the ranks after it.
    fn in_order(&self, listed: &[usize], index: Option<u32>) -> Result<Vec<Placement>, Failure> {
        let ranks = self.needs.rank_bytes.len();
        if listed.len() < ranks {
            return Err(format!(
                "the core's {ranks} ranks need {ranks} nodes and the node list names {}",
                listed.len()
            )
            .into());
        }
        let mut placements = Vec::with_capacity(ranks);
        let mut next = 0;
        for rank in 0..ranks {
            let last = listed.len() - (ranks - rank - 1);
            let mut reasons = Vec::new();
            let mut found = None;
            for (position, &node) in listed.iter().enumerate().take(last).skip(next) {
                match self.gpu(node, rank, index) {
                    Ok(gpu) => {
                        found = Some((position, node, gpu));
                        break;
                    }
                    Err(reason) => reasons.push((node, reason)),
                }
            }
            let Some((position, node, gpu)) = found else {
                return Err(match reasons.as_slice() {
                    [(node, reason)] => {
                        format!("rank {rank} cannot run on {}: {reason}", self.nodes[*node])
                    }
                    _ => format!(
                        "rank {rank} cannot run on any of {}: {}",
                        self.names(reasons.iter().map(|(node, _)| *node)),
                        reasons
                            .iter()
                            .map(|(node, reason)| format!("{}: {reason}", self.nodes[*node]))
                            .collect::<Vec<_>>()
                            .join("; ")
                    ),
                }
                .into());
            };
            placements.push(Placement {
                rank,
                node,
                gpus: vec![gpu.uuid.clone()],
                cached: self.cached(node, rank),
            });
            next = position + 1;
        }
        Ok(placements)
    }

    /// Single-host placement: the preferred node among those with a GPU set for every rank.
    fn single_host(
        &self,
        candidates: &[usize],
        indices: &[u32],
    ) -> Result<Vec<Placement>, Failure> {
        let ranks = self.needs.rank_bytes.len();
        if !indices.is_empty() && indices.len() != ranks {
            return Err(format!(
                "the core's {ranks} ranks need {ranks} GPU indices and the GPU list names {}",
                indices.len()
            )
            .into());
        }
        let mut best: Option<(usize, Vec<&'a GpuStatus>)> = None;
        let mut reasons = Vec::new();
        for &node in candidates {
            match self.node_gpus(node, indices) {
                Ok(gpus) => {
                    let better = match &best {
                        None => true,
                        Some((held, held_gpus)) => {
                            self.prefer_node(node, &gpus, *held, held_gpus) == Ordering::Less
                        }
                    };
                    if better {
                        best = Some((node, gpus));
                    }
                }
                Err(reason) => reasons.push((node, reason)),
            }
        }
        let Some((node, gpus)) = best else {
            return Err(match reasons.as_slice() {
                [] => "the pool has no nodes".to_string(),
                [(node, reason)] => format!(
                    "the core's {ranks} ranks cannot run on {}: {reason}",
                    self.nodes[*node]
                ),
                _ => format!(
                    "no node has {ranks} free {} GPUs for the core: {}",
                    self.needs.revision,
                    reasons
                        .iter()
                        .map(|(node, reason)| format!("{}: {reason}", self.nodes[*node]))
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
            }
            .into());
        };
        Ok(vec![Placement {
            rank: 0,
            node,
            gpus: gpus.iter().map(|g| g.uuid.clone()).collect(),
            cached: self.holds_complete(node),
        }])
    }

    /// Orders two single-host candidates, preferred first: the node holding the complete
    /// core, then more total free memory on the chosen GPUs, then node name.
    fn prefer_node(
        &self,
        a: usize,
        a_gpus: &[&GpuStatus],
        b: usize,
        b_gpus: &[&GpuStatus],
    ) -> Ordering {
        let free = |gpus: &[&GpuStatus]| gpus.iter().map(|g| g.memory_free).sum::<u64>();
        self.holds_complete(b)
            .cmp(&self.holds_complete(a))
            .then(free(b_gpus).cmp(&free(a_gpus)))
            .then_with(|| self.nodes[a].cmp(&self.nodes[b]))
    }

    /// The GPUs a single-host core uses on `node`, in rank order: exactly the GPUs at
    /// `indices` when the operator names them, otherwise the free candidates with the most
    /// index-adjacent pairs (a contiguous run when one exists), lowest indices first among
    /// equals, each covering the need of the rank it serves.
    fn node_gpus(&self, node: usize, indices: &[u32]) -> Result<Vec<&'a GpuStatus>, String> {
        let statuses = self.statuses;
        let status = &statuses[node];
        let revision = &self.needs.revision;
        let ranks = self.needs.rank_bytes.len();
        if !indices.is_empty() {
            let mut gpus = Vec::with_capacity(ranks);
            let mut reasons = Vec::new();
            for (rank, &index) in indices.iter().enumerate() {
                match status.gpus.iter().find(|g| g.index == index) {
                    None => reasons.push(missing(status, index)),
                    Some(gpu) => match unfit(gpu, revision, self.need(rank)) {
                        Some(reason) => reasons.push(reason),
                        None => gpus.push(gpu),
                    },
                }
            }
            return if reasons.is_empty() {
                Ok(gpus)
            } else {
                Err(reasons.join(", "))
            };
        }
        if status.gpus.is_empty() {
            return Err(no_gpus(status));
        }
        let mut free: Vec<&'a GpuStatus> = Vec::new();
        let mut reasons = Vec::new();
        for gpu in &status.gpus {
            // A need of zero bytes leaves only the revision and holder checks.
            match unfit(gpu, revision, 0) {
                Some(reason) => reasons.push(reason),
                None => free.push(gpu),
            }
        }
        free.sort_by_key(|g| g.index);
        if free.len() < ranks {
            let mut text = format!(
                "{} of its {} GPUs are free {revision} GPUs, and the core needs {ranks}",
                free.len(),
                status.gpus.len()
            );
            if !reasons.is_empty() {
                text += &format!(" ({})", reasons.join(", "));
            }
            return Err(text);
        }
        let needs: Vec<u64> = (0..ranks).map(|rank| self.need(rank)).collect();
        best_set(&free, &needs).ok_or_else(|| {
            let wanted = if needs.iter().all(|&n| n == needs[0]) {
                format!("{} on each GPU", gib(needs[0]))
            } else {
                format!(
                    "{} on their GPUs in rank order",
                    needs.iter().map(|&n| gib(n)).collect::<Vec<_>>().join(", ")
                )
            };
            let have = free
                .iter()
                .map(|g| format!("gpu {} {} has {} free", g.index, g.name, gib(g.memory_free)))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "the ranks need {wanted}, and too few of its free {revision} GPUs have it: {have}"
            )
        })
    }

    fn names(&self, nodes: impl Iterator<Item = usize>) -> String {
        nodes
            .map(|node| self.nodes[node].as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Kuhn's augmenting path: give `rank` a node from its preferences, moving the rank that holds
/// a node to another node of its own preferences where one can be freed. `visited` marks the
/// nodes this search has tried.
fn augment(
    preferences: &[Vec<(usize, &GpuStatus)>],
    owner: &mut [Option<usize>],
    taken: &mut [Option<usize>],
    rank: usize,
    visited: &mut [bool],
) -> bool {
    for (position, &(node, _)) in preferences[rank].iter().enumerate() {
        if visited[node] {
            continue;
        }
        visited[node] = true;
        let available = match owner[node] {
            None => true,
            Some(holder) => augment(preferences, owner, taken, holder, visited),
        };
        if available {
            owner[node] = Some(rank);
            taken[rank] = Some(position);
            return true;
        }
    }
    false
}

/// The `needs.len()` GPUs of `free` (sorted by index) that cover `needs` position by position
/// with the most index-adjacent pairs, the lexicographically lowest positions among equals.
fn best_set<'g>(free: &[&'g GpuStatus], needs: &[u64]) -> Option<Vec<&'g GpuStatus>> {
    let count = needs.len();
    let mut best: Option<(usize, Vec<usize>)> = None;
    for_each_combination(free.len(), count, |chosen| {
        let fits = chosen
            .iter()
            .zip(needs)
            .all(|(&position, &need)| free[position].memory_free >= need);
        if !fits {
            return false;
        }
        let adjacent = chosen
            .windows(2)
            .filter(|pair| free[pair[0]].index.checked_add(1) == Some(free[pair[1]].index))
            .count();
        if best.as_ref().is_none_or(|(score, _)| adjacent > *score) {
            best = Some((adjacent, chosen.to_vec()));
        }
        // A fully contiguous set cannot be improved on, and later sets have higher indices.
        adjacent + 1 == count
    });
    best.map(|(_, chosen)| chosen.into_iter().map(|position| free[position]).collect())
}

/// Call `visit` with every `count`-element subset of `0..items`, each sorted ascending, in
/// lexicographic order, until `visit` returns true.
fn for_each_combination(items: usize, count: usize, mut visit: impl FnMut(&[usize]) -> bool) {
    if count > items {
        return;
    }
    let mut chosen: Vec<usize> = (0..count).collect();
    loop {
        if visit(&chosen) {
            return;
        }
        let Some(position) = (0..count).rev().find(|&p| chosen[p] < items - count + p) else {
            return;
        };
        chosen[position] += 1;
        for p in position + 1..count {
            chosen[p] = chosen[p - 1] + 1;
        }
    }
}

/// Why `gpu` cannot serve a rank needing `need` free bytes, or None when it can.
fn unfit(gpu: &GpuStatus, revision: &str, need: u64) -> Option<String> {
    let name = format!("gpu {} {}", gpu.index, gpu.name);
    if gpu.revision != revision {
        Some(format!("{name} is {}, not {revision}", gpu.revision))
    } else if let Some(group) = &gpu.group {
        Some(format!("{name} is held by group {group}"))
    } else if gpu.memory_free < need {
        Some(format!(
            "{name} has {} free and needs {}",
            gib(gpu.memory_free),
            gib(need)
        ))
    } else {
        None
    }
}

fn missing(status: &NodeStatus, index: u32) -> String {
    if status.gpus.is_empty() {
        no_gpus(status)
    } else {
        format!("gpu {index} does not exist")
    }
}

fn no_gpus(status: &NodeStatus) -> String {
    match &status.gpus_error {
        Some(error) => format!("reports no GPUs ({error})"),
        None => "reports no GPUs".into(),
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB)
}

#[cfg(test)]
mod tests {
    use super::super::controller::CoreStatus;
    use super::*;

    const GIBIBYTE: u64 = 1 << 30;

    /// A GPU of `revision` with `free_gib` GiB free, held by group `holder` when given. Its
    /// UUID is set by [`node`].
    fn gpu(index: u32, revision: &str, free_gib: u64, holder: Option<&str>) -> GpuStatus {
        GpuStatus {
            index,
            uuid: String::new(),
            name: format!("NVIDIA {}", revision.to_uppercase()),
            revision: revision.into(),
            memory_total: 256 * GIBIBYTE,
            memory_free: free_gib * GIBIBYTE,
            group: holder.map(Into::into),
        }
    }

    /// A node's name and STATUS with `gpus`, whose UUIDs become `{name}-{index}`, and cache
    /// directories given as (name, verified).
    fn node(name: &str, gpus: Vec<GpuStatus>, cores: &[(&str, bool)]) -> (String, NodeStatus) {
        let gpus = gpus
            .into_iter()
            .map(|mut gpu| {
                gpu.uuid = format!("{name}-{}", gpu.index);
                gpu
            })
            .collect();
        let cores = cores
            .iter()
            .map(|&(directory, verified)| CoreStatus {
                name: directory.into(),
                files: Vec::new(),
                verified,
                bytes: 0,
            })
            .collect();
        let status = NodeStatus {
            fingerprint: String::new(),
            free_bytes: 0,
            controllers_revision: 1,
            group: None,
            groups: Vec::new(),
            gpus,
            gpus_error: None,
            cores,
            address_toward: None,
        };
        (name.into(), status)
    }

    fn pool(members: Vec<(String, NodeStatus)>) -> (Vec<String>, Vec<NodeStatus>) {
        members.into_iter().unzip()
    }

    /// A core exported as `e-8` with `ranks` ranks of 20 GiB each and a 2 GiB reserve, so
    /// each rank needs 22 GiB free.
    fn needs(revision: &str, multi_host: bool, ranks: usize) -> Needs {
        Needs {
            revision: revision.into(),
            multi_host,
            rank_bytes: vec![20 * GIBIBYTE; ranks],
            reserve_bytes: 2 * GIBIBYTE,
            complete_directory: "e-8".into(),
            rank_directories: (0..ranks).map(|rank| format!("e-8-rank{rank}")).collect(),
            source_node: None,
        }
    }

    fn member(rank: usize, node: usize, gpus: &[&str], cached: bool) -> Placement {
        Placement {
            rank,
            node,
            gpus: gpus.iter().map(|gpu| gpu.to_string()).collect(),
            cached,
        }
    }

    fn on(names: &[&str]) -> Constraints {
        Constraints {
            nodes: names.iter().map(|name| name.to_string()).collect(),
            gpus: Vec::new(),
        }
    }

    fn indices(gpus: &[u32]) -> Constraints {
        Constraints {
            nodes: Vec::new(),
            gpus: gpus.to_vec(),
        }
    }

    fn error(result: Result<Vec<Placement>, Failure>) -> String {
        result.expect_err("placement should fail").to_string()
    }

    #[test]
    fn multi_host_prefers_cached_ranks_and_the_source_for_rank_zero() {
        let (nodes, statuses) = pool(vec![
            node("ampere", vec![gpu(0, "gb10", 100, None)], &[]),
            node(
                "faraday",
                vec![gpu(0, "gb10", 120, None)],
                &[("e-8-rank1", false)],
            ),
            node(
                "hertz",
                vec![gpu(0, "gb10", 100, None)],
                &[("e-8-rank1", true)],
            ),
            node("maxwell", vec![gpu(0, "gb10", 50, None)], &[]),
        ]);
        let mut core = needs("gb10", true, 3);
        core.source_node = Some("maxwell".into());
        // Rank 2 holds no files anywhere, so it takes the GPU with the most free memory; an
        // unverified rank directory does not count.
        assert_eq!(
            plan(&core, &nodes, &statuses, &Constraints::default()).unwrap(),
            vec![
                member(0, 3, &["maxwell-0"], true),
                member(1, 2, &["hertz-0"], true),
                member(2, 1, &["faraday-0"], false),
            ]
        );

        let (nodes, statuses) = pool(vec![
            node("ampere", vec![gpu(0, "gb10", 60, None)], &[("e-8", true)]),
            node(
                "hertz",
                vec![gpu(0, "gb10", 100, None)],
                &[("e-8-rank0", true)],
            ),
            node("maxwell", vec![gpu(0, "gb10", 110, None)], &[]),
        ]);
        assert_eq!(
            plan(
                &needs("gb10", true, 2),
                &nodes,
                &statuses,
                &Constraints::default()
            )
            .unwrap(),
            vec![
                member(0, 0, &["ampere-0"], true),
                member(1, 2, &["maxwell-0"], false),
            ]
        );
    }

    #[test]
    fn only_gpus_of_the_core_revision_are_candidates() {
        let (nodes, statuses) = pool(vec![
            node("gracie", vec![gpu(0, "gb300", 250, None)], &[]),
            node("hertz", vec![gpu(0, "gb10", 100, None)], &[]),
            node("maxwell", vec![gpu(0, "gb10", 100, None)], &[]),
        ]);
        assert_eq!(
            plan(
                &needs("gb10", true, 2),
                &nodes,
                &statuses,
                &Constraints::default()
            )
            .unwrap(),
            vec![
                member(0, 1, &["hertz-0"], false),
                member(1, 2, &["maxwell-0"], false),
            ]
        );
        assert_eq!(
            error(plan(
                &needs("gb10", true, 3),
                &nodes,
                &statuses,
                &Constraints::default()
            )),
            "the core's 3 ranks need 3 nodes with a free gb10 GPU, and 2 qualify \
             (hertz, maxwell); gracie: gpu 0 NVIDIA GB300 is gb300, not gb10"
        );
        assert_eq!(
            error(plan(
                &needs("gb10", true, 4),
                &nodes,
                &statuses,
                &Constraints::default()
            )),
            "the core's 4 ranks need 4 nodes and the pool has 3"
        );
    }

    #[test]
    fn a_gpu_without_room_for_the_rank_is_skipped() {
        let (nodes, statuses) = pool(vec![node(
            "gracie",
            vec![gpu(0, "gb300", 21, None), gpu(1, "gb300", 60, None)],
            &[],
        )]);
        assert_eq!(
            plan(
                &needs("gb300", true, 1),
                &nodes,
                &statuses,
                &Constraints::default()
            )
            .unwrap(),
            vec![member(0, 0, &["gracie-1"], false)]
        );

        let (nodes, statuses) = pool(vec![node("gracie", vec![gpu(0, "gb300", 21, None)], &[])]);
        assert_eq!(
            error(plan(
                &needs("gb300", true, 1),
                &nodes,
                &statuses,
                &Constraints::default()
            )),
            "rank 0 needs a free gb300 GPU with 22.0 GiB free and no node has one: gracie: \
             gpu 0 NVIDIA GB300 has 21.0 GiB free and needs 22.0 GiB"
        );
    }

    #[test]
    fn a_held_gpu_is_skipped() {
        let (nodes, statuses) = pool(vec![node(
            "gracie",
            vec![
                gpu(0, "gb300", 100, Some("qwen")),
                gpu(1, "gb300", 100, None),
            ],
            &[],
        )]);
        assert_eq!(
            plan(
                &needs("gb300", true, 1),
                &nodes,
                &statuses,
                &Constraints::default()
            )
            .unwrap(),
            vec![member(0, 0, &["gracie-1"], false)]
        );

        let (nodes, statuses) = pool(vec![node(
            "gracie",
            vec![gpu(0, "gb300", 100, Some("qwen"))],
            &[],
        )]);
        assert_eq!(
            error(plan(
                &needs("gb300", true, 1),
                &nodes,
                &statuses,
                &Constraints::default()
            )),
            "rank 0 needs a free gb300 GPU with 22.0 GiB free and no node has one: gracie: \
             gpu 0 NVIDIA GB300 is held by group qwen"
        );
    }

    #[test]
    fn a_node_list_fixes_rank_order_and_names_the_node_that_cannot_serve() {
        let (nodes, statuses) = pool(vec![
            node("ampere", vec![gpu(0, "gb10", 100, Some("glm"))], &[]),
            node("hertz", vec![gpu(0, "gb10", 100, None)], &[]),
            node("maxwell", vec![gpu(0, "gb10", 100, None)], &[("e-8", true)]),
        ]);
        let core = needs("gb10", true, 2);
        // Without the list, rank 0 would take maxwell, which holds the complete core.
        assert_eq!(
            plan(&core, &nodes, &statuses, &on(&["hertz", "maxwell"])).unwrap(),
            vec![
                member(0, 1, &["hertz-0"], false),
                member(1, 2, &["maxwell-0"], true),
            ]
        );
        assert_eq!(
            plan(
                &core,
                &nodes,
                &statuses,
                &on(&["ampere", "hertz", "maxwell"])
            )
            .unwrap(),
            vec![
                member(0, 1, &["hertz-0"], false),
                member(1, 2, &["maxwell-0"], true),
            ]
        );
        assert_eq!(
            error(plan(&core, &nodes, &statuses, &on(&["hertz", "ampere"]))),
            "rank 1 cannot run on ampere: gpu 0 NVIDIA GB10 is held by group glm"
        );
        assert_eq!(
            plan(
                &core,
                &nodes,
                &statuses,
                &on(&["ampere", "maxwell", "hertz"])
            )
            .unwrap(),
            vec![
                member(0, 2, &["maxwell-0"], true),
                member(1, 1, &["hertz-0"], false),
            ]
        );
        let mut large = needs("gb10", true, 1);
        large.rank_bytes = vec![120 * GIBIBYTE];
        assert_eq!(
            error(plan(&large, &nodes, &statuses, &on(&["ampere", "hertz"]))),
            "rank 0 cannot run on any of ampere, hertz: ampere: gpu 0 NVIDIA GB10 is held by \
             group glm; hertz: gpu 0 NVIDIA GB10 has 100.0 GiB free and needs 122.0 GiB"
        );
        assert_eq!(
            error(plan(&core, &nodes, &statuses, &on(&["faraday"]))),
            "node faraday is not in the pool, which holds ampere, hertz, maxwell"
        );
        assert_eq!(
            error(plan(&core, &nodes, &statuses, &on(&["hertz"]))),
            "the core's 2 ranks need 2 nodes and the node list names 1"
        );
    }

    #[test]
    fn single_host_prefers_contiguous_gpus() {
        let four = |held: Option<u32>| {
            node(
                "gracie",
                (0..4)
                    .map(|index| gpu(index, "gb300", 100, (Some(index) == held).then_some("qwen")))
                    .collect(),
                &[],
            )
        };
        let (nodes, statuses) = pool(vec![four(None)]);
        assert_eq!(
            plan(
                &needs("gb300", false, 2),
                &nodes,
                &statuses,
                &Constraints::default()
            )
            .unwrap(),
            vec![member(0, 0, &["gracie-0", "gracie-1"], false)]
        );

        let (nodes, statuses) = pool(vec![four(Some(1))]);
        assert_eq!(
            plan(
                &needs("gb300", false, 2),
                &nodes,
                &statuses,
                &Constraints::default()
            )
            .unwrap(),
            vec![member(0, 0, &["gracie-2", "gracie-3"], false)]
        );
        assert_eq!(
            plan(
                &needs("gb300", false, 3),
                &nodes,
                &statuses,
                &Constraints::default()
            )
            .unwrap(),
            vec![member(0, 0, &["gracie-0", "gracie-2", "gracie-3"], false)]
        );
        assert_eq!(
            error(plan(
                &needs("gb300", false, 4),
                &nodes,
                &statuses,
                &Constraints::default()
            )),
            "the core's 4 ranks cannot run on gracie: 3 of its 4 GPUs are free gb300 GPUs, \
             and the core needs 4 (gpu 1 NVIDIA GB300 is held by group qwen)"
        );
        let mut large = needs("gb300", false, 2);
        large.rank_bytes = vec![120 * GIBIBYTE; 2];
        assert_eq!(
            error(plan(&large, &nodes, &statuses, &Constraints::default())),
            "the core's 2 ranks cannot run on gracie: the ranks need 122.0 GiB on each GPU, \
             and too few of its free gb300 GPUs have it: gpu 0 NVIDIA GB300 has 100.0 GiB free, \
             gpu 2 NVIDIA GB300 has 100.0 GiB free, gpu 3 NVIDIA GB300 has 100.0 GiB free"
        );
    }

    #[test]
    fn single_host_prefers_the_node_holding_the_core_then_free_memory() {
        let (nodes, statuses) = pool(vec![
            node(
                "gracie",
                vec![gpu(0, "gb300", 200, None), gpu(1, "gb300", 200, None)],
                &[],
            ),
            node(
                "newton",
                vec![gpu(0, "gb300", 100, None), gpu(1, "gb300", 100, None)],
                &[("e-8", true)],
            ),
        ]);
        assert_eq!(
            plan(
                &needs("gb300", false, 2),
                &nodes,
                &statuses,
                &Constraints::default()
            )
            .unwrap(),
            vec![member(0, 1, &["newton-0", "newton-1"], true)]
        );
        let mut elsewhere = needs("gb300", false, 2);
        elsewhere.complete_directory = "f-8".into();
        assert_eq!(
            plan(&elsewhere, &nodes, &statuses, &Constraints::default()).unwrap(),
            vec![member(0, 0, &["gracie-0", "gracie-1"], false)]
        );
    }

    #[test]
    fn explicit_gpu_indices_are_used_exactly() {
        let (nodes, statuses) = pool(vec![node(
            "gracie",
            vec![
                gpu(0, "gb300", 100, None),
                gpu(1, "gb300", 100, Some("qwen")),
                gpu(2, "gb300", 100, None),
                gpu(3, "gb300", 100, None),
            ],
            &[],
        )]);
        let pair = needs("gb300", false, 2);
        assert_eq!(
            plan(&pair, &nodes, &statuses, &indices(&[3, 0])).unwrap(),
            vec![member(0, 0, &["gracie-3", "gracie-0"], false)]
        );
        assert_eq!(
            error(plan(&pair, &nodes, &statuses, &indices(&[0, 1]))),
            "the core's 2 ranks cannot run on gracie: gpu 1 NVIDIA GB300 is held by group qwen"
        );
        assert_eq!(
            error(plan(&pair, &nodes, &statuses, &indices(&[0, 7]))),
            "the core's 2 ranks cannot run on gracie: gpu 7 does not exist"
        );
        assert_eq!(
            error(plan(&pair, &nodes, &statuses, &indices(&[0]))),
            "the core's 2 ranks need 2 GPU indices and the GPU list names 1"
        );
        assert_eq!(
            error(plan(&pair, &nodes, &statuses, &indices(&[2, 2]))),
            "the GPU list names gpu 2 twice"
        );

        let single = needs("gb300", true, 1);
        assert_eq!(
            plan(&single, &nodes, &statuses, &indices(&[2])).unwrap(),
            vec![member(0, 0, &["gracie-2"], false)]
        );
        assert_eq!(
            error(plan(&single, &nodes, &statuses, &indices(&[0, 2]))),
            "a multi-host core's member uses one GPU and the GPU list names 2"
        );
    }

    #[test]
    fn earlier_ranks_move_to_make_room_for_a_larger_rank() {
        let (nodes, statuses) = pool(vec![
            node("hertz", vec![gpu(0, "gb10", 30, None)], &[]),
            node("maxwell", vec![gpu(0, "gb10", 100, None)], &[]),
        ]);
        let mut core = needs("gb10", true, 2);
        core.rank_bytes = vec![20 * GIBIBYTE, 60 * GIBIBYTE];
        core.source_node = Some("maxwell".into());
        // Rank 0 prefers maxwell, the source, but rank 1 fits only there.
        assert_eq!(
            plan(&core, &nodes, &statuses, &Constraints::default()).unwrap(),
            vec![
                member(0, 0, &["hertz-0"], false),
                member(1, 1, &["maxwell-0"], true),
            ]
        );

        let (nodes, statuses) = pool(vec![
            node("ampere", vec![gpu(0, "gb10", 30, None)], &[]),
            node("hertz", vec![gpu(0, "gb10", 30, None)], &[]),
            node("maxwell", vec![gpu(0, "gb10", 100, None)], &[]),
        ]);
        core.rank_bytes = vec![20 * GIBIBYTE, 60 * GIBIBYTE, 60 * GIBIBYTE];
        core.rank_directories.push("e-8-rank2".into());
        assert_eq!(
            error(plan(&core, &nodes, &statuses, &Constraints::default())),
            "rank 2 can run only on maxwell, which the other ranks need"
        );
    }

    #[test]
    fn describe_prints_one_line_per_member() {
        let (nodes, statuses) = pool(vec![
            node("hertz", vec![gpu(0, "gb10", 100, None)], &[]),
            node("maxwell", vec![gpu(0, "gb10", 100, None)], &[("e-8", true)]),
        ]);
        let placement = vec![
            member(0, 1, &["maxwell-0"], true),
            member(1, 0, &["hertz-0"], false),
        ];
        assert_eq!(
            describe(&placement, &nodes, &statuses),
            "rank 0 on maxwell: gpu 0 NVIDIA GB10 (cached)\n\
             rank 1 on hertz: gpu 0 NVIDIA GB10 (not cached)"
        );
    }
}
