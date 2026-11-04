//! CPU ownership checks for two transfer keys and two or three ranks.
//!
//! Channels inject rank acceptance and replies into the production transfer
//! coordinator. Rank replies occur once per accepted job in each rank's FIFO
//! order. CUDA copies, stream draining, and worker channel lifetimes are outside
//! this fixture; parked threads satisfy the coordinator's worker-liveness check.

use std::collections::BTreeSet;

use super::*;
use flare_selene::testing::geometry;

struct Fixture {
    transfers: Transfers,
    replies: Sender<RankReply>,
    parks: Vec<Sender<()>>,
}

impl Fixture {
    fn new(ranks: usize) -> Self {
        let (replies, receiver) = mpsc::channel();
        let mut parks = Vec::new();
        let ranks = (0..ranks)
            .map(|_| {
                let (park, parked) = mpsc::channel::<()>();
                parks.push(park);
                RankThread {
                    sender: None,
                    thread: Some(std::thread::spawn(move || {
                        let _ = parked.recv();
                    })),
                    outstanding: 0,
                }
            })
            .collect();
        Self {
            transfers: Transfers {
                layout: Layout::new(&[geometry(64, &[("k", 8192)], 4096)]).unwrap(),
                ranks,
                receiver,
                pending: HashMap::new(),
            },
            replies,
            parks,
        }
    }

    fn submit(&mut self, job: Job, accepted: usize, full: bool) -> Result<()> {
        let mut queues = Vec::new();
        for (rank, thread) in self.transfers.ranks.iter_mut().enumerate() {
            let (sender, receiver) = mpsc::sync_channel(1);
            if accepted & (1 << rank) != 0 {
                queues.push((rank, receiver));
            } else if full {
                sender.try_send(job.clone()).unwrap();
                queues.push((rank, receiver));
            }
            thread.sender = Some(sender);
        }
        let result = self.transfers.submit(job.clone());
        for (rank, queue) in queues {
            assert_eq!(queue.try_recv().unwrap(), job);
            assert!(
                queue.try_recv().is_err(),
                "rank {rank} accepted a duplicate job"
            );
        }
        result
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.parks.clear();
    }
}

fn job(window: usize, distinct_ids: bool) -> Job {
    Job {
        id: JobId(if distinct_ids { window as u64 + 1 } else { 1 }),
        window: if distinct_ids { 0 } else { window as u32 },
        direction: if window == 0 {
            Direction::Out
        } else {
            Direction::In
        },
        pages: vec![],
        row: None,
        replication: Default::default(),
    }
}

fn error_code(error: &Error) -> usize {
    let text = error.to_string();
    if let Some(rank) = text.strip_prefix("copy rank ") {
        return 4 + rank.parse::<usize>().unwrap();
    }
    let rank = text
        .strip_prefix("rank ")
        .unwrap()
        .split(' ')
        .next()
        .unwrap();
    1 + rank.parse::<usize>().unwrap()
}

fn projection(
    transfers: &Transfers,
    accepted: [usize; 2],
    done: [usize; 2],
    jobs: &[Job; 2],
    replies: &[Reply],
) -> Vec<usize> {
    let mut result = Vec::new();
    for window in 0..2 {
        let key = (jobs[window].id, jobs[window].window);
        let pending = transfers.pending.get(&key);
        let finished: Vec<_> = replies
            .iter()
            .filter(|reply| (reply.id, reply.window) == key)
            .collect();
        let error = pending
            .and_then(|pending| pending.error.as_ref())
            .or_else(|| {
                finished
                    .first()
                    .and_then(|reply| reply.result.as_ref().err())
            });
        let bytes = pending
            .map(|pending| pending.bytes)
            .or_else(|| finished.first().map(|reply| reply.bytes))
            .unwrap_or(0);
        result.extend([
            accepted[window],
            done[window],
            usize::from(pending.is_some()),
            finished.len(),
            error.map(error_code).unwrap_or(0),
            bytes as usize,
        ]);
    }
    result.extend(transfers.ranks.iter().map(|rank| rank.outstanding));
    result.resize(15, 0);
    result.push(transfers.ranks.len());
    result
}

fn record(
    edges: &mut BTreeSet<Vec<Vec<usize>>>,
    before: Vec<usize>,
    action: [usize; 3],
    after: Vec<usize>,
) {
    edges.insert(vec![before, action.to_vec(), after]);
}

fn schedules(accepted: [usize; 2], ranks: usize) -> Vec<Vec<(usize, usize)>> {
    fn visit(
        accepted: [usize; 2],
        ranks: usize,
        done: [usize; 2],
        path: &mut Vec<(usize, usize)>,
        result: &mut Vec<Vec<(usize, usize)>>,
    ) {
        if done == accepted {
            result.push(path.clone());
            return;
        }
        for window in 0..2 {
            for rank in 0..ranks {
                let bit = 1 << rank;
                if accepted[window] & bit == 0 || done[window] & bit != 0 {
                    continue;
                }
                if window == 1 && accepted[0] & bit != 0 && done[0] & bit == 0 {
                    continue;
                }
                let mut next = done;
                next[window] |= bit;
                path.push((window, rank));
                visit(accepted, ranks, next, path, result);
                path.pop();
            }
        }
    }
    let mut result = Vec::new();
    visit(accepted, ranks, [0; 2], &mut Vec::new(), &mut result);
    result
}

#[test]
fn offload_transfers_match_model() {
    let mut edges = BTreeSet::new();
    let mut cases = 0;
    for ranks in 2..=3 {
        let mut fixture = Fixture::new(ranks);
        for first in 0..1 << ranks {
            for second in 0..1 << ranks {
                let masks = [first, second];
                for path in schedules(masks, ranks) {
                    for failures in 0..1 << path.len() {
                        for distinct_ids in [false, true] {
                            for full in [false, true] {
                                cases += 1;
                                let jobs = [job(0, distinct_ids), job(1, distinct_ids)];
                                let mut accepted = [0; 2];
                                let mut done = [0; 2];
                                let mut replies = Vec::new();
                                let mut bytes = [0; 2];
                                let mut errors = [0; 2];
                                assert_eq!(fixture.transfers.in_flight(), 0);
                                for window in 0..2 {
                                    let before = projection(
                                        &fixture.transfers,
                                        accepted,
                                        done,
                                        &jobs,
                                        &replies,
                                    );
                                    let result =
                                        fixture.submit(jobs[window].clone(), masks[window], full);
                                    assert_eq!(result.is_ok(), masks[window] != 0);
                                    if result.is_ok() {
                                        accepted[window] = masks[window];
                                        errors[window] = (0..ranks)
                                            .find(|rank| masks[window] & (1 << rank) == 0)
                                            .map_or(0, |rank| 1 + rank);
                                    }
                                    record(
                                        &mut edges,
                                        before,
                                        [1, window, masks[window]],
                                        projection(
                                            &fixture.transfers,
                                            accepted,
                                            done,
                                            &jobs,
                                            &replies,
                                        ),
                                    );
                                    if result.is_ok() {
                                        let before = projection(
                                            &fixture.transfers,
                                            accepted,
                                            done,
                                            &jobs,
                                            &replies,
                                        );
                                        let error = fixture
                                            .transfers
                                            .submit(jobs[window].clone())
                                            .unwrap_err();
                                        assert!(error.to_string().contains("already in flight"));
                                        record(
                                            &mut edges,
                                            before,
                                            [4, window, 0],
                                            projection(
                                                &fixture.transfers,
                                                accepted,
                                                done,
                                                &jobs,
                                                &replies,
                                            ),
                                        );
                                    }
                                }
                                for (index, &(window, rank)) in path.iter().enumerate() {
                                    let failed = failures & (1 << index) != 0;
                                    let before = projection(
                                        &fixture.transfers,
                                        accepted,
                                        done,
                                        &jobs,
                                        &replies,
                                    );
                                    let reply = RankReply {
                                        id: jobs[window].id,
                                        window: jobs[window].window,
                                        rank,
                                        result: if failed {
                                            Err(invalid(format!("copy rank {rank}")))
                                        } else {
                                            Ok(())
                                        },
                                        bytes: if failed { 0 } else { rank as u64 + 1 },
                                    };
                                    if index % 2 == 0 {
                                        fixture.replies.send(reply).unwrap();
                                        fixture.transfers.poll(&mut replies);
                                    } else {
                                        fixture.transfers.absorb(reply, &mut replies);
                                    }
                                    done[window] |= 1 << rank;
                                    if failed && errors[window] == 0 {
                                        errors[window] = 4 + rank;
                                    }
                                    if !failed {
                                        bytes[window] += rank as u64 + 1;
                                    }
                                    for w in 0..2 {
                                        let key = (jobs[w].id, jobs[w].window);
                                        let remaining =
                                            (accepted[w] & !done[w]).count_ones() as usize;
                                        assert_eq!(
                                            fixture.transfers.pending.contains_key(&key),
                                            remaining != 0,
                                            "accepted rank replies must retain pending ownership until the final reply"
                                        );
                                        if let Some(pending) = fixture.transfers.pending.get(&key) {
                                            assert_eq!(pending.remaining, remaining);
                                        }
                                        let aggregate: Vec<_> = replies
                                            .iter()
                                            .filter(|reply| (reply.id, reply.window) == key)
                                            .collect();
                                        assert_eq!(
                                            aggregate.len(),
                                            usize::from(accepted[w] != 0 && remaining == 0),
                                            "exactly one aggregate reply follows all accepted rank replies"
                                        );
                                        if let Some(reply) = aggregate.first() {
                                            assert_eq!(reply.direction, jobs[w].direction);
                                            assert_eq!(reply.bytes, bytes[w]);
                                            assert_eq!(
                                                reply
                                                    .result
                                                    .as_ref()
                                                    .err()
                                                    .map(error_code)
                                                    .unwrap_or(0),
                                                errors[w]
                                            );
                                        }
                                    }
                                    for rank in 0..ranks {
                                        assert_eq!(
                                            fixture.transfers.ranks[rank].outstanding,
                                            (0..2)
                                                .filter(
                                                    |&w| accepted[w] & !done[w] & (1 << rank) != 0
                                                )
                                                .count()
                                        );
                                    }
                                    record(
                                        &mut edges,
                                        before,
                                        [2, window, rank + 3 * usize::from(failed)],
                                        projection(
                                            &fixture.transfers,
                                            accepted,
                                            done,
                                            &jobs,
                                            &replies,
                                        ),
                                    );
                                }
                                let before =
                                    projection(&fixture.transfers, accepted, done, &jobs, &replies);
                                fixture.transfers.poll(&mut replies);
                                record(
                                    &mut edges,
                                    before,
                                    [3, 0, 0],
                                    projection(&fixture.transfers, accepted, done, &jobs, &replies),
                                );
                                assert_eq!(
                                    replies.len(),
                                    masks.iter().filter(|&&mask| mask != 0).count()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "offload transfer scenarios {cases}; distinct edges {}",
        edges.len()
    );
    for edge in edges {
        println!("OFFLOAD_EDGE {edge:?}");
    }
}
