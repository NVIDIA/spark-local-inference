/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use std::collections::{BTreeSet, VecDeque};

const NODES: usize = 3;
const MAX_DEPTH: usize = 10;
type Action = [u64; 3];

struct Fixture {
    cache: PrefixCheckpointCache,
    ids: [Option<CheckpointId>; NODES],
    histories: [PrefixHistory; NODES],
    original_parent: [usize; NODES],
    retired: [Option<RetiredResources>; NODES],
    last_use: [u64; NODES],
    ready_after: [u64; NODES],
    required: [u64; NODES],
    created: usize,
}

impl Fixture {
    fn new() -> Self {
        let history = |tokens: Vec<u32>| {
            let mut history = PrefixHistory::new([7; 32]);
            history.append(&tokens, &[]).unwrap();
            history
        };
        Self {
            cache: PrefixCheckpointCache::new([7; 32], 8, 8, 64).unwrap(),
            ids: [None; NODES],
            histories: [
                history((0..64).collect()),
                history((0..128).collect()),
                history((0..192).collect()),
            ],
            original_parent: [0; NODES],
            retired: [None, None, None],
            last_use: [0; NODES],
            ready_after: [0; NODES],
            required: [0; NODES],
            created: 0,
        }
    }

    fn fork(&self) -> Self {
        let cache = &self.cache;
        Self {
            cache: PrefixCheckpointCache {
                identity: cache.identity,
                page_tokens: cache.page_tokens,
                page_capacity: cache.page_capacity,
                snapshot_capacity: cache.snapshot_capacity,
                nodes: cache
                    .nodes
                    .iter()
                    .map(|(id, node)| {
                        (
                            *id,
                            Checkpoint {
                                id: node.id,
                                parent: node.parent,
                                prefix: node.prefix,
                                bucket: node.bucket,
                                tail_pages: node.tail_pages.clone(),
                                snapshot: node.snapshot,
                                ready_after: node.ready_after,
                                ready: node.ready,
                                last_use: node.last_use,
                                child_count: node.child_count,
                                direct_pins: node.direct_pins,
                                pins: node.pins,
                                accessed: node.accessed,
                                role: node.role,
                            },
                        )
                    })
                    .collect(),
                prefixes: cache.prefixes.clone(),
                owned_pages: cache.owned_pages.clone(),
                owned_snapshots: cache.owned_snapshots.clone(),
                retirement: cache.retirement.clone(),
                completed: cache.completed,
                next_id: cache.next_id,
                clock: cache.clock,
            },
            ids: self.ids,
            histories: self.histories.clone(),
            original_parent: self.original_parent,
            retired: self.retired.clone(),
            last_use: self.last_use,
            ready_after: self.ready_after,
            required: self.required,
            created: self.created,
        }
    }

    fn semantic(&self, id: CheckpointId) -> usize {
        self.ids
            .iter()
            .position(|candidate| *candidate == Some(id))
            .unwrap()
            + 1
    }

    fn pages(&self, node: usize) -> Vec<PageId> {
        self.cache.page_chain(self.ids[node - 1].unwrap()).unwrap()
    }

    fn record_reads(&mut self, pages: &[PageId], fence: u64) {
        for page in pages {
            self.required[page.0 as usize] = self.required[page.0 as usize].max(fence);
        }
    }

    fn proposal(&self, node: usize, parent: usize, fence: u64, duplicate: bool) -> Publication {
        let endpoint = self.histories[node - 1].token_count();
        Publication {
            parent: if parent == 0 {
                None
            } else {
                self.ids[parent - 1]
            },
            consumed_tokens: endpoint,
            tail_pages: vec![PageId(if duplicate { 3 } else { node as u32 - 1 })],
            snapshot: StateSnapshot {
                id: SnapshotId(if duplicate { 3 } else { node as u32 - 1 }),
                consumed_tokens: endpoint,
            },
            ready_after: Fence(fence),
        }
    }

    fn apply(&mut self, action: Action) -> Action {
        let [kind, n, value] = action;
        let node = n as usize;
        match kind {
            1 => {
                let parent = if node == 1 {
                    0
                } else if node == 2 {
                    1
                } else {
                    value as usize / 3 + 1
                };
                let fence = value % 3;
                if node == 3 && parent == 1 {
                    let tokens: Vec<_> = (0..64).chain(1000..1064).collect();
                    self.histories[2] = PrefixHistory::new([7; 32]);
                    self.histories[2].append(&tokens, &[]).unwrap();
                }
                let parent_pages = if parent == 0 {
                    vec![]
                } else {
                    self.pages(parent)
                };
                let proposal = self.proposal(node, parent, fence, false);
                let result = self
                    .cache
                    .publish(&self.histories[node - 1], &proposal)
                    .unwrap();
                assert!(matches!(result, PublishResult::Created { .. }));
                self.ids[node - 1] = Some(result.id());
                self.original_parent[node - 1] = parent;
                self.ready_after[node - 1] = fence;
                self.required[node - 1] = fence;
                self.record_reads(&parent_pages, fence);
                self.created += 1;
            }
            2 => {
                self.cache.pin(self.ids[node - 1].unwrap()).unwrap();
            }
            3 => {
                self.cache.unpin(self.ids[node - 1].unwrap()).unwrap();
            }
            4 => {
                let pages = self.pages(node);
                self.cache
                    .mark_used(self.ids[node - 1].unwrap(), Fence(value))
                    .unwrap();
                self.record_reads(&pages, value);
            }
            5 => {
                self.last_use[node - 1] = self
                    .cache
                    .checkpoint(self.ids[node - 1].unwrap())
                    .unwrap()
                    .last_use
                    .0;
                let retired = self
                    .cache
                    .cancel_publication(self.ids[node - 1].unwrap())
                    .unwrap();
                self.retired[node - 1] = Some(retired);
            }
            6 => {
                for i in 0..NODES {
                    if let Some(id) = self.ids[i] {
                        if let Ok(checkpoint) = self.cache.checkpoint(id) {
                            self.last_use[i] = checkpoint.last_use.0;
                        }
                    }
                }
                let report = self.cache.evict(EvictionTarget {
                    pages: value as usize,
                    snapshots: 1,
                });
                assert!(report.retired.len() <= 1);
                if let Some(retired) = report.retired.into_iter().next() {
                    let index = self.semantic(retired.checkpoint) - 1;
                    self.retired[index] = Some(retired);
                    return [6, index as u64 + 1, value];
                }
                return [9, 0, 0];
            }
            7 => {
                let pending = self.cache.pending_retirement().to_vec();
                let update = self.cache.advance_completed(Fence(value)).unwrap();
                let expected_pages: BTreeSet<_> = pending
                    .iter()
                    .filter(|r| r.retire_after.0 <= value)
                    .flat_map(|r| r.pages.iter().copied())
                    .collect();
                let expected_snapshots: BTreeSet<_> = pending
                    .iter()
                    .filter(|r| r.retire_after.0 <= value)
                    .map(|r| r.snapshot)
                    .collect();
                assert_eq!(
                    update
                        .reusable
                        .pages
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>(),
                    expected_pages
                );
                assert_eq!(
                    update
                        .reusable
                        .snapshots
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>(),
                    expected_snapshots
                );
                assert_eq!(
                    update.reusable.pages.len(),
                    expected_pages.len(),
                    "page reclaimed more than once"
                );
                assert_eq!(
                    update.reusable.snapshots.len(),
                    expected_snapshots.len(),
                    "snapshot reclaimed more than once"
                );
                for page in &update.reusable.pages {
                    assert!(
                        value >= self.required[page.0 as usize],
                        "page reclaimed before inherited last-use fence"
                    );
                }
            }
            8 => {
                let id = self.ids[node - 1].unwrap();
                let parent = self
                    .cache
                    .checkpoint(id)
                    .unwrap()
                    .parent
                    .map_or(0, |id| self.semantic(id));
                // Only unmodified one-page tails admit this fixed duplicate proposal.
                let pages = if parent == 0 {
                    vec![]
                } else {
                    self.pages(parent)
                };
                let proposal = self.proposal(node, parent, value, true);
                let before_pages = self.cache.owned_pages.clone();
                let before_snapshots = self.cache.owned_snapshots.clone();
                let result = self
                    .cache
                    .publish(&self.histories[node - 1], &proposal)
                    .unwrap();
                assert!(
                    matches!(result, PublishResult::Existing { id: existing, .. } if existing == id)
                );
                assert_eq!(
                    self.cache.owned_pages, before_pages,
                    "duplicate publication transferred caller pages"
                );
                assert_eq!(
                    self.cache.owned_snapshots, before_snapshots,
                    "duplicate publication transferred caller snapshot"
                );
                self.record_reads(&pages, value);
            }
            _ => unreachable!(),
        }
        action
    }

    fn projection(&self) -> Vec<u64> {
        let mut out = vec![self.cache.completed.0, self.created as u64];
        out.extend(self.required);
        for i in 0..NODES {
            let original = self.original_parent[i] as u64;
            if let Some(checkpoint) = self.ids[i].and_then(|id| self.cache.nodes.get(&id)) {
                out.extend([
                    if checkpoint.ready { 2 } else { 1 },
                    checkpoint.parent.map_or(0, |id| self.semantic(id) as u64),
                    original,
                    mask(&checkpoint.tail_pages),
                    checkpoint.ready_after.0,
                    checkpoint.last_use.0,
                    checkpoint.direct_pins,
                    0,
                ]);
            } else if let Some(retired) = &self.retired[i] {
                let pending = self
                    .cache
                    .retirement
                    .iter()
                    .any(|r| r.checkpoint == retired.checkpoint);
                out.extend([
                    if pending { 3 } else { 4 },
                    0,
                    original,
                    mask(&retired.pages),
                    self.ready_after[i],
                    self.last_use[i],
                    0,
                    retired.retire_after.0,
                ]);
            } else {
                out.extend([0; 8]);
            }
        }
        out
    }

    fn assert_ownership(&self) {
        let mut pages = HashSet::new();
        let mut snapshots = HashSet::new();
        for checkpoint in self.cache.nodes.values() {
            assert!(snapshots.insert(checkpoint.snapshot));
            for page in &checkpoint.tail_pages {
                assert!(pages.insert(*page), "live page owners overlap");
            }
            let mut expected = vec![];
            let mut next = self.semantic(checkpoint.id);
            while next != 0 {
                expected.push(PageId(next as u32 - 1));
                next = self.original_parent[next - 1];
            }
            expected.reverse();
            assert_eq!(
                self.cache.page_chain(checkpoint.id).unwrap(),
                expected,
                "compaction changed descendant page chain"
            );
            for page in &checkpoint.tail_pages {
                assert!(
                    checkpoint.last_use.0 >= self.required[page.0 as usize],
                    "compaction lost inherited last-use fence"
                );
            }
            let direct_sum: u64 = self
                .cache
                .nodes
                .values()
                .filter(|descendant| {
                    self.cache
                        .chain(descendant.id)
                        .unwrap()
                        .contains(&checkpoint.id)
                })
                .map(|descendant| descendant.direct_pins)
                .sum();
            assert_eq!(checkpoint.pins, direct_sum);
            assert_eq!(
                checkpoint.child_count,
                self.cache
                    .nodes
                    .values()
                    .filter(|child| child.parent == Some(checkpoint.id))
                    .count()
            );
        }
        for retired in &self.cache.retirement {
            assert!(
                snapshots.insert(retired.snapshot),
                "retired snapshot still has live owner"
            );
            for page in &retired.pages {
                assert!(pages.insert(*page), "retired page still has live owner");
                assert!(
                    retired.retire_after.0 >= self.required[page.0 as usize],
                    "retirement lost inherited last-use fence"
                );
            }
        }
        assert_eq!(pages, self.cache.owned_pages);
        assert_eq!(snapshots, self.cache.owned_snapshots);
    }

    fn assert_rejections_transactional(&self) {
        let mut fixture = self.fork();
        let before = fixture.key();
        let clock = fixture.cache.clock;
        let next_id = fixture.cache.next_id;
        let prefixes = fixture.cache.prefixes.clone();
        let pages = fixture.cache.owned_pages.clone();
        let snapshots = fixture.cache.owned_snapshots.clone();
        let retirement = fixture.cache.retirement.clone();
        let references: Vec<_> = fixture
            .cache
            .nodes
            .values()
            .map(|node| (node.id, node.pins, node.child_count))
            .collect();
        assert!(fixture.cache.pin(CheckpointId::new(99)).is_err());
        assert!(
            fixture
                .cache
                .mark_used(CheckpointId::new(99), Fence(2))
                .is_err()
        );
        if fixture.cache.completed.0 > 0 {
            assert!(
                fixture
                    .cache
                    .advance_completed(Fence(fixture.cache.completed.0 - 1))
                    .is_err()
            );
        }
        for id in fixture.ids.into_iter().flatten() {
            if let Ok(node) = fixture.cache.checkpoint(id) {
                let ready = node.ready;
                let direct = node.direct_pins;
                if !ready {
                    assert!(fixture.cache.pin(id).is_err());
                    assert!(fixture.cache.mark_used(id, Fence(2)).is_err());
                } else {
                    assert!(fixture.cache.cancel_publication(id).is_err());
                }
                if direct == 0 {
                    assert!(fixture.cache.unpin(id).is_err());
                }
            } else {
                assert!(fixture.cache.unpin(id).is_err());
                assert!(fixture.cache.cancel_publication(id).is_err());
            }
        }
        let mut invalid = fixture.proposal(1, 0, 0, false);
        invalid.tail_pages = vec![PageId(8)];
        invalid.snapshot.id = SnapshotId(7);
        assert!(
            fixture
                .cache
                .publish(&fixture.histories[0], &invalid)
                .is_err()
        );
        assert_eq!(
            fixture.key(),
            before,
            "rejected cache operation changed lifecycle state"
        );
        assert_eq!(fixture.cache.clock, clock);
        assert_eq!(fixture.cache.next_id, next_id);
        assert_eq!(fixture.cache.prefixes, prefixes);
        assert_eq!(fixture.cache.owned_pages, pages);
        assert_eq!(fixture.cache.owned_snapshots, snapshots);
        assert_eq!(fixture.cache.retirement, retirement);
        assert_eq!(
            fixture
                .cache
                .nodes
                .values()
                .map(|node| (node.id, node.pins, node.child_count))
                .collect::<Vec<_>>(),
            references
        );
    }

    fn key(&self) -> Vec<u64> {
        let mut key = self.projection();
        // Relative retention order preserves eviction choices without an unbounded clock.
        let mut ordered: Vec<_> = self.cache.nodes.values().collect();
        ordered.sort_by_key(|node| (node.retention_rank(), node.id));
        key.extend(ordered.iter().map(|node| self.semantic(node.id) as u64));
        key
    }

    fn actions(&self) -> Vec<Action> {
        let mut actions = vec![[6, 0, 0], [6, 0, 1]];
        for fence in self.cache.completed.0..=2 {
            actions.push([7, 0, fence]);
        }
        if self.created < NODES {
            let node = self.created + 1;
            let parents = if node == 1 {
                vec![0]
            } else if node == 2 {
                vec![1]
            } else {
                vec![1, 2]
            };
            for parent in parents {
                if parent == 0
                    || self.ids[parent - 1]
                        .and_then(|id| self.cache.nodes.get(&id))
                        .is_some_and(|node| node.ready)
                {
                    for fence in 0..=2 {
                        actions.push([
                            1,
                            node as u64,
                            fence
                                + if node == 3 {
                                    (parent as u64 - 1) * 3
                                } else {
                                    0
                                },
                        ]);
                    }
                }
            }
        }
        for checkpoint in self.cache.nodes.values() {
            let node = self.semantic(checkpoint.id) as u64;
            if checkpoint.ready {
                if checkpoint.direct_pins == 0 {
                    actions.push([2, node, 0]);
                } else {
                    actions.push([3, node, 0]);
                }
                for fence in checkpoint.last_use.0 + 1..=2 {
                    actions.push([4, node, fence]);
                }
            } else {
                actions.push([5, node, 0]);
            }
            let parent_len = checkpoint
                .parent
                .map_or(0, |id| self.cache.nodes[&id].prefix.consumed_tokens);
            if checkpoint.prefix.consumed_tokens - parent_len == 64 {
                for fence in 0..=2 {
                    actions.push([8, node, fence]);
                }
            }
        }
        actions
    }
}

fn mask(pages: &[PageId]) -> u64 {
    pages.iter().map(|page| 1 << page.0).sum()
}
fn replay(path: &[Action]) -> Fixture {
    let mut fixture = Fixture::new();
    for action in path {
        fixture.apply(*action);
    }
    fixture
}

#[test]
fn bounded_cache_ownership_matches_tla() {
    let mut queue = VecDeque::from([vec![]]);
    let mut visited = BTreeSet::from([Fixture::new().key()]);
    let mut edges = BTreeSet::new();
    let mut compactions = 0;
    let mut pending_children = 0;
    let mut delayed_retirements = 0;
    let mut duplicate_publications = 0;
    while let Some(path) = queue.pop_front() {
        let fixture = replay(&path);
        fixture.assert_ownership();
        fixture.assert_rejections_transactional();
        edges.insert((fixture.projection(), [9, 0, 1], fixture.projection()));
        if path.len() == MAX_DEPTH {
            continue;
        }
        for action in fixture.actions() {
            let mut next = fixture.fork();
            let before = next.projection();
            let observed = next.apply(action);
            next.assert_ownership();
            for retired in next.cache.pending_retirement() {
                if retired.retire_after > next.cache.completed {
                    delayed_retirements += 1;
                }
                if let Some(heir) = retired.heir {
                    compactions += 1;
                    if next.cache.nodes.get(&heir).is_some_and(|node| !node.ready) {
                        pending_children += 1;
                    }
                }
            }
            if action[0] == 8 {
                duplicate_publications += 1;
            }
            edges.insert((before, observed, next.projection()));
            if visited.insert(next.key()) {
                let mut extended = path.clone();
                extended.push(action);
                queue.push_back(extended);
            }
        }
    }
    assert!(
        compactions > 0
            && pending_children > 0
            && delayed_retirements > 0
            && duplicate_publications > 0
    );
    for (before, action, after) in &edges {
        println!("CACHE_OWNERSHIP_EDGE [{before:?},{action:?},{after:?}]");
    }
    println!(
        "CACHE_OWNERSHIP_COVERAGE states={} edges={} depth={MAX_DEPTH} compactions={compactions} pending_children={pending_children} delayed_retirements={delayed_retirements} duplicate_publications={duplicate_publications}",
        visited.len(),
        edges.len()
    );
}

#[test]
fn reclaimable_projection_matches_eviction_with_pins() {
    let mut cases = 0;
    for parent in 1..=2 {
        for pending in 0..=1 {
            for initial_pin in 0..=3 {
                let mut fixture = Fixture::new();
                fixture.apply([1, 1, 0]);
                fixture.apply([1, 2, 0]);
                fixture.apply([1, 3, (parent - 1) * 3 + pending]);
                if initial_pin != 0 {
                    let id = fixture.ids[initial_pin as usize - 1].unwrap();
                    if !fixture.cache.checkpoint(id).unwrap().ready {
                        continue;
                    }
                    fixture.cache.pin(id).unwrap();
                }
                for id in fixture.ids.into_iter().flatten() {
                    if !fixture.cache.checkpoint(id).unwrap().ready {
                        continue;
                    }
                    let before = fixture.key();
                    let predicted = fixture.cache.reclaimable_if_pinned(id).unwrap();
                    assert_eq!(
                        fixture.key(),
                        before,
                        "reclaimability projection mutated cache"
                    );
                    let mut evicted = fixture.fork();
                    evicted.cache.pin(id).unwrap();
                    let report = evicted.cache.evict(EvictionTarget {
                        pages: usize::MAX,
                        snapshots: usize::MAX,
                    });
                    let pages: BTreeSet<_> = report
                        .retired
                        .iter()
                        .flat_map(|record| record.pages.iter().copied())
                        .collect();
                    let snapshots: BTreeSet<_> = report
                        .retired
                        .iter()
                        .map(|record| record.snapshot)
                        .collect();
                    assert_eq!(
                        predicted.pages.iter().copied().collect::<BTreeSet<_>>(),
                        pages
                    );
                    assert_eq!(
                        predicted.snapshots.iter().copied().collect::<BTreeSet<_>>(),
                        snapshots
                    );
                    assert!(
                        evicted.cache.checkpoint(id).is_ok(),
                        "pinned endpoint evicted"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert!(cases >= 30);
    println!("CACHE_RECLAIMABLE_CASES {cases}");
}
