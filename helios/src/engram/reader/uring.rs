/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! io_uring disk readers share a fixed staging and queue budget across workers.

use super::{Aligned, CHUNK, DEPTH, DiskTimings, Fragment, Read, Stats};
use crate::Result;
use io_uring::{IoUring, opcode, types};
use std::{
    fs::File,
    os::fd::AsRawFd,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

pub(super) struct Readers {
    lanes: Vec<Lane>,
}

struct Lane {
    // Dropping the ring cancels and drains outstanding I/O before the staging is freed.
    ring: IoUring,
    staging: Aligned,
    files: Vec<Arc<File>>,
    depth: usize,
    // Read index occupying each staging slot, and the reads still owned by the kernel.
    slots: Vec<usize>,
    free: Vec<usize>,
    inflight: usize,
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
    pub(super) fn new(files: &[Arc<File>], workers: usize) -> Result<Self> {
        if !matches!(workers, 1 | 2 | 4 | 8) {
            return Err("embedding table workers must be 1, 2, 4, or 8".into());
        }
        let depth = DEPTH / workers;
        let mut lanes = Vec::with_capacity(workers);
        for _ in 0..workers {
            let staging = Aligned::new(depth * CHUNK)?;
            let ring = IoUring::new(depth as u32).map_err(|e| format!("Engram io_uring: {e}"))?;
            let buffers: Vec<_> = (0..depth)
                .map(|i| libc::iovec {
                    iov_base: staging.at(i).cast(),
                    iov_len: CHUNK,
                })
                .collect();
            unsafe { ring.submitter().register_buffers(&buffers) }
                .map_err(|e| format!("register Engram staging: {e}"))?;
            lanes.push(Lane {
                ring,
                staging,
                files: files.to_vec(),
                depth,
                slots: vec![0; depth],
                free: (0..depth).rev().collect(),
                inflight: 0,
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
            self.drain();
        }
        result
    }

    // Wait out reads the kernel still owns so the next gather never sees their completions.
    fn drain(&mut self) {
        while self.inflight > 0 {
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => return,
            }
            for entry in self.ring.completion() {
                self.free.push(entry.user_data() as usize);
                self.inflight -= 1;
            }
        }
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
        // Refill the lane's fixed window as slots complete across the full gather.
        let mut next = 0;
        let mut failure = None;
        loop {
            let begin = Instant::now();
            let first = next;
            while failure.is_none() && next < reads.len() && self.inflight < self.depth {
                let slot = self.free.pop().unwrap();
                let read = &reads[next];
                let entry = opcode::ReadFixed::new(
                    types::Fd(self.files[read.file].as_raw_fd()),
                    self.staging.at(slot),
                    read.nbytes as u32,
                    slot as u16,
                )
                .offset(read.offset)
                .build()
                .user_data(slot as u64);
                if unsafe { self.ring.submission().push(&entry) }.is_err() {
                    self.free.push(slot);
                    return Err("Engram submission queue full".into());
                }
                self.slots[slot] = next;
                self.inflight += 1;
                next += 1;
            }
            stats
                .disk_reads
                .fetch_add((next - first) as u64, Ordering::Relaxed);
            stats.disk_bytes.fetch_add(
                reads[first..next]
                    .iter()
                    .map(|read| read.nbytes as u64)
                    .sum(),
                Ordering::Relaxed,
            );
            timing.submit += begin.elapsed().as_nanos() as u64;
            if self.inflight == 0 {
                break;
            }
            let begin = Instant::now();
            let result = self.ring.submit_and_wait(1);
            timing.wait += begin.elapsed().as_nanos() as u64;
            match result {
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(format!("submit Engram reads: {error}").into()),
            }
            let begin = Instant::now();
            for entry in self.ring.completion() {
                let slot = entry.user_data() as usize;
                let read = &reads[self.slots[slot]];
                let count = entry.result();
                if count < 0 || (count as usize) < read.required {
                    failure.get_or_insert_with(|| {
                        format!(
                            "Engram read at {} returned {count}, requires {} bytes",
                            read.offset, read.required
                        )
                    });
                } else {
                    for fragment in &fragments[read.fragments.clone()] {
                        // SAFETY: This lane exclusively owns each fragment, and the
                        // slot is not reused until its fragments have been copied.
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
                self.free.push(slot);
                self.inflight -= 1;
            }
            timing.complete += begin.elapsed().as_nanos() as u64;
        }
        match failure {
            Some(error) => Err(error.into()),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_share_the_same_queue_and_staging_budget() {
        let path =
            std::env::temp_dir().join(format!("flare-uring-budget-{}.bin", std::process::id()));
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
