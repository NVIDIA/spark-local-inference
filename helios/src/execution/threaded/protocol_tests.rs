/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{protocol::*, *};
use crate::execution::{CompletionRecord, ExecutionTiming, GraphKind};
use std::collections::{BTreeSet, HashSet};

fn manifest() -> ExecutionManifest {
    let mut manifest = super::super::tests::manifest();
    manifest.graphs.push(manifest.graphs[0].clone());
    manifest
}

fn completion(fence: u64, graph: usize) -> FrameCompletion {
    FrameCompletion {
        fence: Fence(fence),
        graph_index: graph,
        timing: ExecutionTiming::default(),
        records: [CompletionRecord::default(); 0].into(),
        logprobs: None,
        prompt_logprobs: None,
        proposal_confidence: None,
        speculative_survival: None,
    }
}

fn frame(fence: u64, graph: usize, grammar: bool) -> Pending {
    Pending {
        step: Arc::new(Step {
            fence: Fence(fence),
            graph,
            grammar,
            uploads: Vec::new(),
            cleared_pages: Vec::new(),
        }),
        dispatched: Instant::now(),
        committed: false,
        queued: fence > 1,
        ready: vec![false; 2],
        completions: vec![None, None, None],
    }
}

struct State {
    submitted: usize,
    completed: usize,
    pending: VecDeque<Pending>,
}

impl State {
    fn fork(&self) -> Self {
        Self {
            submitted: self.submitted,
            completed: self.completed,
            pending: self
                .pending
                .iter()
                .map(|pending| Pending {
                    step: pending.step.clone(),
                    dispatched: pending.dispatched,
                    committed: pending.committed,
                    queued: pending.queued,
                    ready: pending.ready.clone(),
                    completions: pending
                        .completions
                        .iter()
                        .map(|value| {
                            value
                                .as_ref()
                                .map(|value| completion(value.fence.0, value.graph_index))
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    fn projection(&self) -> [usize; 15] {
        let mut result = [0; 15];
        result[..3].copy_from_slice(&[self.submitted, self.completed, self.pending.len()]);
        for (index, pending) in self.pending.iter().enumerate() {
            let start = 3 + index * 6;
            let ready = pending
                .ready
                .iter()
                .enumerate()
                .fold(0, |mask, (rank, value)| {
                    mask | (usize::from(*value) << rank)
                });
            let done = pending
                .completions
                .iter()
                .enumerate()
                .fold(0, |mask, (rank, value)| {
                    mask | (usize::from(value.is_some()) << rank)
                });
            result[start..start + 6].copy_from_slice(&[
                pending.step.fence.0 as usize,
                pending.step.graph,
                usize::from(pending.step.grammar),
                ready,
                usize::from(pending.committed),
                done,
            ]);
        }
        result
    }
}

#[test]
fn rank_coordinator_matches_model_edges() {
    let manifest = manifest();
    let mut work = vec![State {
        submitted: 0,
        completed: 0,
        pending: VecDeque::new(),
    }];
    let mut states = HashSet::new();
    let mut edges = BTreeSet::new();
    while let Some(state) = work.pop() {
        let before = state.projection();
        if !states.insert(before) {
            continue;
        }
        let mut edge = |after: State, action| {
            edges.insert((before, action, after.projection()));
            work.push(after);
        };
        for graph in 0..2 {
            for barrier in [false, true] {
                if state.submitted < 2 && can_submit_pending(&state.pending, barrier, &manifest) {
                    let mut after = state.fork();
                    after.submitted += 1;
                    after
                        .pending
                        .push_back(frame(after.submitted as u64, graph, barrier));
                    edge(after, [1, graph, usize::from(barrier)]);
                }
            }
        }
        for (index, pending) in state.pending.iter().enumerate() {
            let fence = pending.step.fence;
            for peer in 0..2 {
                let mut after = state.fork();
                match record_ready(&mut after.pending, peer, fence, pending.step.graph) {
                    Ok(()) => edge(after, [2, fence.0 as usize, peer]),
                    Err(_) => {
                        assert_eq!(after.projection(), before);
                        edge(after, [6, 0, 0]);
                    }
                }
            }
            if !pending.committed {
                let all_ready = pending.ready.iter().all(|ready| *ready);
                assert_eq!(
                    ready_to_commit(pending),
                    all_ready,
                    "commit must wait for every peer readiness"
                );
                if ready_to_commit(pending) {
                    let mut after = state.fork();
                    // The marker follows successful peer dispatch and local submission in poll_ranks.
                    after.pending[index].committed = true;
                    edge(after, [3, fence.0 as usize, 0]);
                }
            }
            if pending.committed {
                for rank in 0..3 {
                    let mut after = state.fork();
                    let value = completion(fence.0, pending.step.graph);
                    let result = if rank == 0 {
                        record_local_completion(&mut after.pending, value)
                    } else {
                        record_peer_completion(&mut after.pending, rank - 1, value)
                    };
                    match result {
                        Ok(()) => edge(after, [4, fence.0 as usize, rank]),
                        Err(_) => {
                            assert_eq!(after.projection(), before);
                            edge(after, [6, 0, 0]);
                        }
                    }
                }
            }
            let mut rejected = state.fork();
            assert!(record_ready(&mut rejected.pending, 0, fence, pending.step.graph ^ 1).is_err());
            assert_eq!(rejected.projection(), before);
            edge(rejected, [6, 0, 0]);
        }
        let mut rejected = state.fork();
        assert!(record_ready(&mut rejected.pending, 0, Fence(3), 0).is_err());
        assert_eq!(rejected.projection(), before);
        edge(rejected, [6, 0, 0]);
        let mut after = state.fork();
        let complete = state
            .pending
            .front()
            .is_some_and(|pending| pending.completions.iter().all(Option::is_some));
        let popped = pop_completed_front(&mut after.pending);
        assert_eq!(
            popped.is_some(),
            complete,
            "publish must wait for every rank completion at the queue front"
        );
        if popped.is_some() {
            after.completed += 1;
            edge(after, [5, 0, 0]);
        } else {
            assert_eq!(after.projection(), before);
            edge(after, [6, 0, 0]);
        }
    }
    assert!(states.iter().any(|state| state[0] == 2 && state[1] == 2));
    println!("RANK_COUNTS states={} edges={}", states.len(), edges.len());
    for (before, action, after) in edges {
        println!("RANK_EDGE [{before:?}, {action:?}, {after:?}]");
    }
}

fn error(result: Result<()>, expected: &str) {
    assert_eq!(result.unwrap_err().to_string(), expected);
}

#[test]
fn rank_coordinator_matches_model_completion_identity() {
    let mut pending = VecDeque::from([frame(1, 0, false), frame(2, 1, false)]);
    for value in &mut pending {
        value.committed = true;
    }
    error(
        record_ready(&mut pending, 0, Fence(3), 0),
        "unexpected rank readiness fence",
    );
    error(
        record_ready(&mut pending, 0, Fence(1), 1),
        "rank readiness identity or state differs",
    );
    error(
        record_local_completion(&mut pending, completion(3, 0)),
        "unexpected rank-zero completion fence",
    );
    error(
        record_peer_completion(&mut pending, 0, completion(2, 1)),
        "rank completion order or identity differs",
    );
    error(
        record_peer_completion(&mut pending, 0, completion(1, 1)),
        "rank completion order or identity differs",
    );
    record_local_completion(&mut pending, completion(2, 1)).unwrap();
    assert!(pending[0].completions[0].is_none());
    assert!(
        pop_completed_front(&mut pending).is_none(),
        "later completion cannot publish ahead of the queue front"
    );
    record_local_completion(&mut pending, completion(1, 0)).unwrap();
    error(
        record_local_completion(&mut pending, completion(1, 1)),
        "duplicate rank-zero completion",
    );
    assert_eq!(pending[0].completions[0].as_ref().unwrap().graph_index, 1);
    record_peer_completion(&mut pending, 0, completion(1, 0)).unwrap();
    error(
        record_peer_completion(&mut pending, 0, completion(1, 0)),
        "rank completion order or identity differs",
    );
    record_peer_completion(&mut pending, 0, completion(2, 1)).unwrap();
    error(
        record_peer_completion(&mut pending, 0, completion(2, 1)),
        "unexpected rank completion",
    );
    record_peer_completion(&mut pending, 1, completion(1, 0)).unwrap();
    let removed = pop_completed_front(&mut pending).unwrap();
    assert_eq!(removed.step.fence, Fence(1));
    assert!(
        super::super::parallel::agree(
            removed.completions[0].as_ref().unwrap(),
            removed.completions[1].as_ref().unwrap()
        )
        .is_err()
    );
    assert_eq!(
        pending.len(),
        1,
        "agreement errors occur after removing the completed frame"
    );
}

#[test]
fn rank_coordinator_matches_model_pending_window() {
    let mut manifest = manifest();
    let mut pending = VecDeque::new();
    assert!(can_submit_pending(&pending, true, &manifest));
    pending.push_back(frame(1, 0, false));
    assert!(!can_submit_pending(&pending, false, &manifest));
    pending[0].ready = vec![true, false];
    assert!(
        !ready_to_commit(&pending[0]),
        "commit must wait for every peer readiness"
    );
    record_ready(&mut pending, 1, Fence(1), 0).unwrap();
    error(
        record_ready(&mut pending, 1, Fence(1), 0),
        "rank readiness identity or state differs",
    );
    assert!(ready_to_commit(&pending[0]));
    error(
        record_peer_completion(&mut pending, 0, completion(1, 0)),
        "rank completion order or identity differs",
    );
    pending[0].committed = true;
    assert!(!ready_to_commit(&pending[0]));
    assert!(can_submit_pending(&pending, false, &manifest));
    assert!(!can_submit_pending(&pending, true, &manifest));
    manifest.graphs[0].kind = GraphKind::Prefill;
    assert!(!can_submit_pending(&pending, false, &manifest));
    manifest.graphs[0].kind = GraphKind::Speculative;
    assert!(can_submit_pending(&pending, false, &manifest));
    pending[0].step = Arc::new(Step {
        fence: Fence(1),
        graph: 0,
        grammar: true,
        uploads: Vec::new(),
        cleared_pages: Vec::new(),
    });
    assert!(!can_submit_pending(&pending, false, &manifest));
    pending[0].step = Arc::new(Step {
        fence: Fence(1),
        graph: 0,
        grammar: false,
        uploads: Vec::new(),
        cleared_pages: Vec::new(),
    });
    pending.push_back(frame(2, 1, false));
    pending[1].committed = true;
    for rank in 0..3 {
        pending[0].completions[rank] = Some(completion(1, 0));
    }
    assert!(
        !can_submit_pending(&pending, false, &manifest),
        "completed frames retain their window slots until popped"
    );
    pending[0].completions[2] = None;
    assert!(
        pop_completed_front(&mut pending).is_none(),
        "publish must wait for every rank completion at the queue front"
    );
    pending[0].completions[2] = Some(completion(1, 0));
    pop_completed_front(&mut pending).unwrap();
    assert!(can_submit_pending(&pending, false, &manifest));
}

#[test]
fn rank_coordinator_matches_model_partial_dispatch() {
    for failed_peer in 0..=3 {
        let mut peers = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..3 {
            let (sender, commands) = mpsc::channel();
            let (_, receiver) = mpsc::channel();
            peers.push(Peer {
                sender,
                receiver,
                thread: None,
            });
            receivers.push(Some(commands));
        }
        if failed_peer < 3 {
            drop(receivers[failed_peer].take());
        }
        let result = dispatch_commit(&peers, Fence(7), 1);
        if failed_peer < 3 {
            assert!(
                result.is_err(),
                "partial commit dispatch must report disconnected peer"
            );
            error(result, "rank thread disconnected");
        } else {
            result.unwrap();
        }
        for (index, receiver) in receivers.iter().enumerate() {
            if let Some(receiver) = receiver {
                if index < failed_peer {
                    assert!(
                        matches!(receiver.try_recv(), Ok(Command::Commit(Fence(7), 1))),
                        "commit dispatch lost an accepted peer"
                    );
                }
                assert!(
                    matches!(receiver.try_recv(), Err(mpsc::TryRecvError::Empty)),
                    "commit dispatch continued beyond the failed peer or sent twice"
                );
            }
        }
    }
}
