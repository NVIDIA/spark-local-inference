/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Windows disk readers share a fixed staging and queue budget across workers.

use super::{Aligned, CHUNK, DEPTH, DiskTimings, Fragment, Read, Stats};
use crate::{Result, os::ioring::ReadRing};
use std::{
    fs::File,
    sync::{Arc, atomic::Ordering},
};

pub(super) struct Readers {
    lanes: Vec<Lane>,
}

struct Lane {
    // Drain before releasing the staging allocation and file.
    ring: ReadRing,
    staging: Aligned,
    #[cfg(test)]
    depth: usize,
}

#[derive(Clone, Copy)]
struct Output {
    address: usize,
    len: usize,
}

impl Output {
    // SAFETY: The planner assigns each destination byte to exactly one fragment,
    // and each read (with its fragments) belongs to one worker. The scoped join
    // finishes every worker before the caller can access or release output.
    unsafe fn copy(self, fragment: &Fragment, source: *const u8) {
        assert!(fragment.destination <= self.len);
        assert!(fragment.nbytes <= self.len - fragment.destination);
        unsafe {
            std::ptr::copy_nonoverlapping(
                source,
                (self.address as *mut u8).add(fragment.destination),
                fragment.nbytes,
            );
        }
    }
}

impl Readers {
    pub(super) fn new(sources: &[Arc<File>], workers: usize) -> Result<Self> {
        if !matches!(workers, 1 | 2 | 4 | 8) {
            return Err("embedding table workers must be 1, 2, 4, or 8".into());
        }
        let depth = DEPTH / workers;
        let mut lanes = Vec::with_capacity(workers);
        for _ in 0..workers {
            let files = sources
                .iter()
                .map(|source| crate::os::direct_reopen(source))
                .collect::<std::io::Result<Vec<_>>>()
                .map_err(|e| format!("open Engram direct readers: {e}"))?;
            let staging = Aligned::new(depth * CHUNK)?;
            // SAFETY: Lane drops its ring before the registered staging allocation.
            let ring = unsafe { ReadRing::new(files, staging.0, depth, CHUNK as u32) }
                .map_err(|e| format!("Engram I/O ring: {e}"))?;
            lanes.push(Lane {
                ring,
                staging,
                #[cfg(test)]
                depth,
            });
        }
        Ok(Self { lanes })
    }

    pub(super) fn gather(
        &mut self,
        reads: &[Read],
        fragments: &[Fragment],
        output: &mut [u8],
        stats: &Stats,
    ) -> Result<()> {
        let output = Output {
            address: output.as_mut_ptr() as usize,
            len: output.len(),
        };
        // Small gathers stay on the existing delivery thread. Large gathers use
        // scoped helpers; their startup cost is included in measured gather time.
        if self.lanes.len() == 1 || reads.len() <= DEPTH {
            return self.lanes[0].gather(reads, fragments, output, stats);
        }
        let chunk = reads.len().div_ceil(self.lanes.len());
        std::thread::scope(|scope| {
            let (local, helpers) = self.lanes.split_first_mut().unwrap();
            let mut jobs = reads.chunks(chunk);
            let first = jobs.next().unwrap();
            let mut joins = Vec::with_capacity(helpers.len());
            for (index, (lane, reads)) in helpers.iter_mut().zip(jobs).enumerate() {
                joins.push(
                    std::thread::Builder::new()
                        .name(format!("flare-table-io-{}", index + 1))
                        .spawn_scoped(scope, move || lane.gather(reads, fragments, output, stats))
                        .map_err(|e| format!("start embedding read worker: {e}"))?,
                );
            }
            let mut result = local.gather(first, fragments, output, stats);
            for worker in joins {
                let next = worker
                    .join()
                    .unwrap_or_else(|_| Err("embedding read worker panicked".into()));
                if result.is_ok() {
                    result = next;
                }
            }
            result
        })
    }
}

impl Lane {
    fn gather(
        &mut self,
        reads: &[Read],
        fragments: &[Fragment],
        output: Output,
        stats: &Stats,
    ) -> Result<()> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.read(reads, fragments, output, stats)
        }))
        .unwrap_or_else(|_| Err("embedding read worker panicked".into()));
        if result.is_err() {
            self.ring.cancel_and_drain();
        }
        result
    }

    fn read(
        &mut self,
        reads: &[Read],
        fragments: &[Fragment],
        output: Output,
        stats: &Stats,
    ) -> Result<()> {
        let mut timing = DiskTimings {
            stats,
            plan: 0,
            submit: 0,
            wait: 0,
            complete: 0,
        };
        // Refill each lane's fixed window as slots complete across the full gather.
        for batch in [reads] {
            let mut failure = None;
            let result = self.ring.read_all(
                batch
                    .iter()
                    .map(|read| (read.file, read.offset, read.nbytes as u32)),
                |index, slot, result| {
                    let read = &batch[index];
                    match result {
                        Ok(count) if count >= read.required => {
                            for fragment in &fragments[read.fragments.clone()] {
                                // SAFETY: This lane exclusively owns each fragment;
                                // read_all waits until copying ends before slot reuse.
                                unsafe {
                                    output.copy(
                                        fragment,
                                        self.staging
                                            .at(slot)
                                            .add((fragment.source - read.offset) as usize),
                                    );
                                }
                            }
                        }
                        result => {
                            failure.get_or_insert_with(|| {
                                format!(
                                    "Engram read at {} requires {} bytes: {result:?}",
                                    read.offset, read.required,
                                )
                            });
                        }
                    }
                },
            );
            timing.submit += self.ring.timings.submit;
            timing.wait += self.ring.timings.wait;
            timing.complete += self.ring.timings.complete;
            result.map_err(|e| format!("Engram I/O ring read: {e}"))?;
            stats
                .disk_reads
                .fetch_add(batch.len() as u64, Ordering::Relaxed);
            stats.disk_bytes.fetch_add(
                batch.iter().map(|read| read.nbytes as u64).sum(),
                Ordering::Relaxed,
            );
            if let Some(error) = failure {
                return Err(error.into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_share_the_same_queue_and_staging_budget() {
        let path =
            std::env::temp_dir().join(format!("flare-worker-budget-{}.bin", std::process::id()));
        std::fs::write(&path, [0; 4096]).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        for workers in [1, 2, 4, 8] {
            let readers = Readers::new(std::slice::from_ref(&file), workers).unwrap();
            assert_eq!(
                readers.lanes.iter().map(|lane| lane.depth).sum::<usize>(),
                DEPTH
            );
            assert_eq!(
                readers
                    .lanes
                    .iter()
                    .map(|lane| lane.staging.1)
                    .sum::<usize>(),
                DEPTH * CHUNK
            );
        }
        assert!(Readers::new(std::slice::from_ref(&file), 0).is_err());
        assert!(Readers::new(std::slice::from_ref(&file), 3).is_err());
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
