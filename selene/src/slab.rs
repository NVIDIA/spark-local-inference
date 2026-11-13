/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Registered host memory in equal page-record-sized slots.
//!
//! One anonymous mapping, 4 KiB aligned and advised for transparent huge pages,
//! holds every slot. Rows use several slots, with each rank using its page share.
//! Registration with the CUDA driver is a separate step so CPU tests run without a driver. Allocation never blocks: when no
//! slot is free the caller evicts or fails its job.
//!
//! A pool split by rank instead keeps each rank's part of every slot in pinned
//! memory that only that rank's device maps, for stores whose records the CPU
//! never reads. Its slot addresses are identities, and [`RankPart`] locates
//! each rank's part of a slot for the transfer threads.
//!
//! The pool hands out slots from a FIFO free list that starts in index
//! order, and tracks which slots are allocated so that releasing a slot that is
//! free, released twice, or outside the pool is ignored instead of duplicating
//! a free-list entry. The free list is sized at construction and never grows.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use crate::{HostRegistrar, Registration, Result, SplitMapping};

use super::geometry::{BLOCK_BYTES, Layout, invalid};

/// Pages per transfer window.
pub use crate::shared::WINDOW_PAGES;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Slot(pub u32);

struct Slots {
    free: VecDeque<u32>,
    allocated: Vec<bool>,
}

impl Slots {
    fn new(count: usize) -> Self {
        Self {
            free: (0..count as u32).collect(),
            allocated: vec![false; count],
        }
    }
}

fn region_bytes(slot_bytes: u64, count: usize) -> Result<usize> {
    if count == 0 {
        return Err(invalid("a slot pool requires at least one slot"));
    }
    if slot_bytes == 0 || !slot_bytes.is_multiple_of(BLOCK_BYTES) {
        return Err(invalid(format!(
            "slot size of {slot_bytes} bytes is not a nonzero multiple of {BLOCK_BYTES}"
        )));
    }
    u32::try_from(count)
        .ok()
        .and_then(|_| usize::try_from(slot_bytes).ok()?.checked_mul(count))
        .ok_or_else(|| {
            invalid(format!(
                "{count} slots of {slot_bytes} bytes exceed the addressable pool size"
            ))
        })
}

fn slot_count(layout: &Layout, bytes: u64) -> usize {
    (bytes / layout.page_record_bytes) as usize
}

/// Device addresses of one rank's part of every slot.
///
/// A split pool keeps each rank's share of every slot contiguously in its memory.
/// Slot addresses remain identities:
/// where each slot would begin if the pool held whole cross-rank records. They
/// lie inside the pool's reserved range, so they never name another pool's slot.
/// A registered pool instead uses CPU slot addresses as identities and the
/// registration's device alias for copies, keeping cross-rank record strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankPart {
    /// The pool's reserved range.
    range: std::ops::Range<usize>,
    /// Identity of slot 0; identities advance by one cross-rank record.
    identity: usize,
    record_bytes: usize,
    count: usize,
    /// This rank's part of slot 0; parts advance by the rank's share of a record.
    base: usize,
    part_bytes: usize,
}

impl RankPart {
    fn offset(&self, address: usize) -> Option<usize> {
        self.range
            .contains(&address)
            .then(|| address - self.identity)
    }

    /// Resolve a packed share in a registered, contiguous pool.
    pub fn replicated_page(&self, address: usize, rank_offset: u64) -> Result<u64> {
        let offset = self.offset(address).filter(|_| {
            self.part_bytes == self.record_bytes && address.is_multiple_of(BLOCK_BYTES as usize)
        });
        let Some(offset) = offset else {
            return Err(invalid(
                "replicated share is outside a contiguous transfer pool",
            ));
        };
        Ok((self.base + offset) as u64 - rank_offset)
    }

    pub fn record_bytes(&self) -> u64 {
        self.record_bytes as u64
    }

    /// Resolve this rank's part of a slot, rejecting misaligned identities.
    pub fn page(&self, slot: usize) -> Result<Option<u64>> {
        let Some(offset) = self.offset(slot) else {
            return Ok(None);
        };
        let index = offset / self.record_bytes;
        if offset.is_multiple_of(self.record_bytes) && index < self.count {
            Ok(Some((self.base + index * self.part_bytes) as u64))
        } else {
            Err(invalid(format!(
                "address {slot:#x} is not a slot in the pool"
            )))
        }
    }
}

/// Each rank's parts of a pool split by rank and the memory holding them.
struct Split {
    parts: Vec<RankPart>,
    _memory: Box<dyn SplitMapping>,
}

/// Shared slots with immutable mapping geometry and resource ownership.
#[derive(Clone)]
pub struct SlabPool {
    pub(crate) memory: Arc<Memory>,
    slots: Arc<Mutex<Slots>>,
}

pub(crate) struct Memory {
    intake_slots: usize,
    mapping: usize,
    mapping_bytes: usize,
    page_bytes: usize,
    slot_count: usize,
    registration: Option<Box<dyn Registration>>,
    split: Option<Split>,
}

impl SlabPool {
    /// Map equal, page-record-sized slots.
    pub fn new(page_bytes: u64, count: usize, intake_slots: usize) -> Result<Self> {
        let mapping_bytes = region_bytes(page_bytes, count)?;
        let base = flare_eos::map_host(mapping_bytes)
            .map_err(|error| invalid(format!("allocate host slots: {error}")))?;
        Ok(Self {
            memory: Arc::new(Memory {
                intake_slots,
                mapping: base,
                mapping_bytes,
                page_bytes: page_bytes as usize,
                slot_count: count,
                registration: None,
                split: None,
            }),
            slots: Arc::new(Mutex::new(Slots::new(count))),
        })
    }

    /// Round a warm budget down to whole slots.
    pub fn sized(layout: &Layout, bytes: u64, intake_slots: usize) -> Result<Self> {
        Self::new(
            layout.page_record_bytes,
            slot_count(layout, bytes),
            intake_slots,
        )
    }

    /// A pool with the slot counts of [`SlabPool::sized`], split by
    /// rank: each rank's part of every slot is pinned host memory that only the
    /// device of `contexts[rank]` and the host access. Registering one mapping
    /// maps it into every device at about 2 MiB of device memory per GiB; a
    /// split pool costs each device that for its own part only. The CPU cannot
    /// address a slot's record, so the pool suits stores that move records only
    /// between devices and slots; [`SlabPool::rank_parts`] locates the parts.
    pub fn sized_by_rank(
        layout: &Layout,
        bytes: u64,
        intake_slots: usize,
        contexts: &[(usize, i32)],
        registrar: &dyn HostRegistrar,
    ) -> Result<Self> {
        if contexts.len() != layout.ranks.len() {
            return Err(invalid(format!(
                "a pool split by rank needs one context per rank: {} contexts for {} ranks",
                contexts.len(),
                layout.ranks.len()
            )));
        }
        let slot_count = slot_count(layout, bytes);
        let page_region = region_bytes(layout.page_record_bytes, slot_count)?;
        let shares: Vec<usize> = (0..layout.ranks.len())
            .map(|rank| layout.page_share(rank) as usize)
            .collect();
        let requests: Vec<_> = contexts
            .iter()
            .zip(&shares)
            .map(|(&(context, device), page)| (context, device, page * slot_count))
            .collect();
        let memory = registrar.map_split(&requests)?;
        let base = memory.address();
        if memory.nbytes() < page_region {
            return Err(invalid(
                "the reserved range of a pool split by rank is smaller than its records",
            ));
        }
        let range = base..base + memory.nbytes();
        let parts = shares
            .iter()
            .enumerate()
            .map(|(rank, &page)| RankPart {
                range: range.clone(),
                identity: base,
                record_bytes: layout.page_record_bytes as usize,
                count: slot_count,
                base: memory.part_address(rank),
                part_bytes: page,
            })
            .collect();
        Ok(Self {
            memory: Arc::new(Memory {
                intake_slots,
                mapping: base,
                mapping_bytes: memory.nbytes(),
                page_bytes: layout.page_record_bytes as usize,
                slot_count,
                registration: None,
                split: Some(Split {
                    parts,
                    _memory: memory,
                }),
            }),
            slots: Arc::new(Mutex::new(Slots::new(slot_count))),
        })
    }

    /// Each rank's part of the slots, in rank order, for a pool split by rank.
    pub fn rank_parts(&self) -> Option<Vec<RankPart>> {
        self.memory.split.as_ref().map(|split| split.parts.clone())
    }

    /// Resolve GPU copy addresses once at startup. Disk I/O and CPU slices
    /// continue to use the host addresses returned by `slot_address`.
    pub fn device_parts(&self, layout: &Layout) -> Result<Vec<RankPart>> {
        if layout.page_record_bytes != self.page_bytes() as u64 {
            return Err(invalid(
                "slot sizes disagree with the transfer record layout",
            ));
        }
        if let Some(parts) = self.rank_parts() {
            if parts.len() != layout.ranks.len() {
                return Err(invalid("a split pool needs one part per transfer rank"));
            }
            return Ok(parts);
        }
        let registration = self
            .memory
            .registration
            .as_ref()
            .ok_or_else(|| invalid("transfer slots must be registered with CUDA"))?;
        Ok(self.registered_parts(layout, registration.device_address()))
    }

    pub fn registered_parts(&self, layout: &Layout, device: u64) -> Vec<RankPart> {
        layout
            .ranks
            .iter()
            .map(|rank| RankPart {
                range: self.memory.mapping..self.memory.mapping + self.memory.mapping_bytes,
                identity: self.memory.mapping,
                record_bytes: self.page_bytes(),
                count: self.slot_count(),
                base: (device + rank.page_offset) as usize,
                part_bytes: self.page_bytes(),
            })
            .collect()
    }

    /// Register the mapping through `registrar` as portable so every rank's context
    /// can address it, with `context` made current on the calling thread.
    /// A pool that is already registered is not registered again.
    pub fn register(&mut self, registrar: &dyn HostRegistrar, context: usize) -> Result<()> {
        if self.registered() {
            return Err(invalid(
                "slot pool is already registered with the CUDA driver",
            ));
        }
        // The mapping is page aligned and stays mapped at this size for the
        // pool's lifetime; `Drop` releases the registration before unmapping.
        let memory = Arc::get_mut(&mut self.memory)
            .ok_or_else(|| invalid("cannot register a shared slot pool"))?;
        memory.registration =
            Some(registrar.register(context, memory.mapping, memory.mapping_bytes)?);
        Ok(())
    }

    /// Whether the driver can copy the slots: after `register`, and always for a pool split by rank.
    pub fn registered(&self) -> bool {
        self.memory.registration.is_some() || self.memory.split.is_some()
    }

    pub fn intake_slots(&self) -> usize {
        self.memory.intake_slots
    }

    pub fn page_bytes(&self) -> usize {
        self.memory.page_bytes
    }
    pub fn slot_count(&self) -> usize {
        self.memory.slot_count
    }
    pub fn free_slots(&self) -> usize {
        self.lock().free.len()
    }
    pub fn bytes(&self) -> u64 {
        self.memory.mapping_bytes as u64
    }

    pub fn allocate(&self, count: usize) -> Option<Vec<Slot>> {
        let mut slots = self.lock();
        if slots.free.len() < count {
            return None;
        }
        Some(
            (0..count)
                .map(|_| {
                    let index = slots.free.pop_front().unwrap();
                    slots.allocated[index as usize] = true;
                    Slot(index)
                })
                .collect(),
        )
    }

    /// Return allocated slots once; ignore free, repeated and out-of-range slots.
    pub fn release(&self, released: impl IntoIterator<Item = Slot>) {
        let mut slots = self.lock();
        for slot in released {
            if let Some(allocated) = slots.allocated.get_mut(slot.0 as usize)
                && *allocated
            {
                *allocated = false;
                slots.free.push_back(slot.0);
            }
        }
    }

    pub fn addresses(&self, slots: &[Slot]) -> Vec<usize> {
        slots.iter().map(|&slot| self.slot_address(slot)).collect()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn allocate_page(&self) -> Option<Slot> {
        self.allocate(1).map(|slots| slots[0])
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn release_page(&self, slot: Slot) {
        self.release([slot]);
    }
    pub fn slot_address(&self, slot: Slot) -> usize {
        assert!(
            (slot.0 as usize) < self.slot_count(),
            "slot index outside the pool"
        );
        self.memory.mapping + slot.0 as usize * self.page_bytes()
    }
    /// Failed drain: retain the mapping permanently and prevent slot reuse.
    pub(crate) fn leak(&self) {
        let mut slots = self.lock();
        slots.free.clear();
        slots.allocated.fill(false);
        std::mem::forget(Arc::clone(&self.memory));
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, Slots> {
        self.slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    /// The caller owns the slot and keeps it pinned through every reader.
    #[cfg(any(test, feature = "test-support"))]
    pub fn page(&self, slot: Slot) -> &[u8] {
        self.assert_contiguous();
        // SAFETY: The mapping outlives the pool and the slot lies inside it.
        unsafe {
            std::slice::from_raw_parts(self.slot_address(slot) as *const u8, self.page_bytes())
        }
    }
    #[cfg(any(test, feature = "test-support"))]
    pub fn page_mut(&mut self, slot: Slot) -> &mut [u8] {
        self.assert_contiguous();
        // SAFETY: The caller owns the slot exclusively while mutating its bytes.
        unsafe {
            std::slice::from_raw_parts_mut(self.slot_address(slot) as *mut u8, self.page_bytes())
        }
    }
    #[cfg(any(test, feature = "test-support"))]
    fn assert_contiguous(&self) {
        assert!(
            self.memory.split.is_none(),
            "the slots of a pool split by rank hold no contiguous record"
        );
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        self.registration.take();
        if self.split.take().is_none() {
            // SAFETY: The last owner dropped after both copiers drained.
            unsafe { flare_eos::unmap_host(self.mapping, self.mapping_bytes) };
        }
    }
}

/// Fixture for tests of code that consumes a pool's parts.
#[cfg(any(test, feature = "test-support"))]
pub fn rank_part(
    range: std::ops::Range<usize>,
    (record_bytes, count, base, part_bytes): (usize, usize, usize, usize),
) -> RankPart {
    RankPart {
        identity: range.start,
        record_bytes,
        count,
        base,
        part_bytes,
        range,
    }
}

/// Row pieces in persisted record order, with offsets into the contiguous row record.
pub fn row_pieces<'a>(
    layout: &'a Layout,
    slots: &'a [usize],
) -> impl Iterator<Item = (usize, u64, usize)> + 'a {
    layout
        .ranks
        .iter()
        .enumerate()
        .flat_map(move |(rank, part)| {
            let share = layout.page_share(rank);
            let bytes = layout.row_share(rank);
            slots.iter().enumerate().filter_map(move |(piece, &slot)| {
                let offset = piece as u64 * share;
                (offset < bytes).then(|| {
                    (
                        slot + part.page_offset as usize,
                        part.row_offset + offset,
                        (bytes - offset).min(share) as usize,
                    )
                })
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{record, testing::geometry};

    #[test]
    fn slots_are_aligned_disjoint_and_reused_in_release_order() {
        let mut pool = SlabPool::new(8192, 4, 0).unwrap();
        let slots: Vec<_> = (0..4).map(|_| pool.allocate_page().unwrap()).collect();
        assert_eq!(pool.allocate_page(), None);
        for (index, &slot) in slots.iter().enumerate() {
            assert_eq!(pool.slot_address(slot) % BLOCK_BYTES as usize, 0);
            pool.page_mut(slot).fill(index as u8);
            assert!(pool.page(slot).iter().all(|&byte| byte == index as u8));
        }
        assert_eq!(
            pool.slot_address(slots[3]) - pool.slot_address(slots[0]),
            3 * 8192
        );
        for &slot in &slots {
            pool.release_page(slot);
        }
        pool.release_page(slots[0]);
        pool.release_page(Slot(99));
        assert_eq!(pool.free_slots(), 4);
        assert_eq!(pool.allocate_page(), Some(slots[0]));
    }

    #[test]
    fn shared_handles_use_one_free_list_and_retain_the_registration() {
        let registrar = crate::testing::TrackingRegistrar::default();
        let mut pool = SlabPool::new(4096, 2, 0).unwrap();
        pool.register(&registrar, 0).unwrap();
        let other = pool.clone();
        let first = pool.allocate_page().unwrap();
        let second = other.allocate_page().unwrap();
        assert_ne!(first, second);
        assert_eq!(pool.allocate_page(), None);
        other.release_page(first);
        assert_eq!(pool.allocate_page(), Some(first));
        pool.page_mut(first).fill(42);
        drop(pool);
        assert_eq!(registrar.calls.lock().unwrap().len(), 1);
        assert_eq!(*registrar.drops.lock().unwrap(), 0);
        assert!(other.page(first).iter().all(|&byte| byte == 42));
        drop(other);
        assert_eq!(*registrar.drops.lock().unwrap(), 1);
    }

    #[test]
    fn copy_addresses_and_slices_do_not_wait_for_the_free_list() {
        let layout = Layout::new(&[geometry(64, &[("kv", 4096)], 4096)]).unwrap();
        let registrar = crate::testing::TrackingRegistrar::default();
        let mut pool = SlabPool::new(4096, 2, 0).unwrap();
        pool.register(&registrar, 0).unwrap();
        let slot = pool.allocate_page().unwrap();
        let other = pool.clone();
        let locked = other.slots.lock().unwrap();
        let (done, completion) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let address = pool.slot_address(slot);
            pool.page_mut(slot).fill(42);
            assert!(pool.page(slot).iter().all(|&byte| byte == 42));
            assert_eq!(pool.page_bytes(), 4096);
            assert_eq!(pool.slot_count(), 2);
            assert_eq!(pool.bytes(), 8192);
            assert!(pool.registered());
            assert!(pool.rank_parts().is_none());
            assert_eq!(
                pool.registered_parts(&layout, address as u64)[0]
                    .page(address)
                    .unwrap(),
                Some(address as u64)
            );
            assert_eq!(
                pool.device_parts(&layout).unwrap()[0]
                    .page(address)
                    .unwrap(),
                Some(address as u64)
            );
            done.send(()).unwrap();
        });
        let result = completion.recv_timeout(std::time::Duration::from_secs(5));
        drop(locked);
        worker.join().unwrap();
        result.unwrap();
    }

    #[test]
    fn registration_is_frozen_while_the_mapping_is_shared() {
        let registrar = crate::testing::TrackingRegistrar::default();
        let mut pool = SlabPool::new(4096, 2, 0).unwrap();
        let other = pool.clone();
        assert!(pool.register(&registrar, 0).is_err());
        assert!(registrar.calls.lock().unwrap().is_empty());
        assert!(!other.registered());
        drop(other);
        pool.register(&registrar, 0).unwrap();
        assert!(pool.register(&registrar, 0).is_err());
        assert_eq!(registrar.calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn allocation_and_release_preserve_total_and_unique_ownership() {
        let pool = SlabPool::new(4096, 37, 0).unwrap();
        let mut owned = Vec::new();
        let mut seed = 71u64;
        for _ in 0..10_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if seed & 8 != 0 {
                let count = (seed as usize % 8) + 1;
                let free = pool.free_slots();
                match pool.allocate(count) {
                    Some(slots) => {
                        assert_eq!(slots.len(), count);
                        for slot in slots {
                            assert!(!owned.contains(&slot));
                            owned.push(slot);
                        }
                    }
                    None => assert_eq!(pool.free_slots(), free),
                }
            } else if !owned.is_empty() {
                let count = (seed as usize % owned.len()) + 1;
                pool.release(owned.drain(..count));
            }
            assert_eq!(pool.free_slots() + owned.len(), pool.slot_count());
        }
    }

    #[test]
    fn warm_sizes_round_down_without_a_separate_row_minimum() {
        let layout = Layout::new(&[geometry(64, &[("kv", 8192)], 4096)]).unwrap();
        let pool = SlabPool::sized(&layout, 3 * 8192 + 4095, 0).unwrap();
        assert_eq!(pool.bytes(), 3 * 8192);
        assert_eq!(pool.slot_count(), 3);
        assert!(SlabPool::new(8192, 0, 0).is_err());
        assert!(SlabPool::sized(&layout, 4096, 0).is_err());
        assert!(
            SlabPool::sized_by_rank(
                &layout,
                4096,
                0,
                &[(0, 0)],
                &crate::testing::TrackingRegistrar::default()
            )
            .is_err()
        );
        assert!(SlabPool::new(4000, 1, 0).is_err());
        assert!(SlabPool::new(0, 1, 0).is_err());
        assert!(SlabPool::new(8192, usize::MAX, 0).is_err());
    }

    #[test]
    fn unequal_rank_rows_gather_and_hash_in_the_original_record_order() {
        for piece_blocks in [1, 3, 33] {
            let ranks = [
                geometry(
                    64,
                    &[("kv", 4096 * piece_blocks)],
                    4096 * (piece_blocks * 3 + 1),
                ),
                geometry(
                    64,
                    &[("kv", 8192 * piece_blocks)],
                    4096 * (piece_blocks + 1),
                ),
            ];
            let layout = Layout::new(&ranks).unwrap();
            assert_eq!(layout.row_slots(), 4);
            let pool = SlabPool::new(layout.page_record_bytes, 8, 0).unwrap();
            let slots: Vec<_> = (0..4)
                .map(|_| {
                    let slot = pool.allocate_page().unwrap();
                    let gap = pool.allocate_page().unwrap();
                    pool.release_page(gap);
                    pool.slot_address(slot)
                })
                .collect();
            let original: Vec<u8> = (0..layout.row_record_bytes)
                .map(|i| (i * 31 + 7) as u8)
                .collect();
            for (address, offset, len) in row_pieces(&layout, &slots) {
                assert_eq!(address % 4096, 0);
                assert_eq!(offset % 4096, 0);
                assert_eq!(len % 4096, 0);
                unsafe { std::slice::from_raw_parts_mut(address as *mut u8, len) }
                    .copy_from_slice(&original[offset as usize..offset as usize + len]);
            }
            let gathered: Vec<u8> = row_pieces(&layout, &slots)
                .flat_map(|(address, _, len)| {
                    unsafe { std::slice::from_raw_parts(address as *const u8, len) }.to_vec()
                })
                .collect();
            assert_eq!(gathered, original);
            assert_eq!(
                record::row_checksum(&layout, &slots),
                record::checksum(&original)
            );
            let parts = pool.registered_parts(&layout, 0x1000_0000);
            for (rank, part) in parts.iter().enumerate() {
                for &slot in &slots {
                    assert_eq!(
                        part.page(slot).unwrap(),
                        Some(
                            0x1000_0000
                                + (slot - pool.memory.mapping) as u64
                                + layout.ranks[rank].page_offset
                        )
                    );
                }
            }
        }
    }

    #[test]
    fn split_parts_use_only_their_ranks_share_for_every_row_slot() {
        let start = 0x1000_0000;
        let part = rank_part(start..start + 20 * 12288, (12288, 20, 0x2000_0000, 4096));
        for index in 0..20 {
            assert_eq!(
                part.page(start + index * 12288).unwrap(),
                Some(0x2000_0000 + index as u64 * 4096)
            );
        }
        assert!(part.page(start + 4096).is_err());
        assert_eq!(part.page(start - 4096).unwrap(), None);
    }
}
