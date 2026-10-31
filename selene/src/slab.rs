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
//! slab is free the caller evicts or fails its job.
//!
//! A pool split by rank instead keeps each rank's part of every slab in pinned
//! memory that only that rank's device maps, for stores whose records the CPU
//! never reads. Its slab addresses are identities, and [`RankPart`] locates
//! each rank's part of a slab for the transfer threads.
//!
//! The pool hands out slabs from a FIFO free list that starts in index
//! order, and tracks which slabs are allocated so that releasing a slab that is
//! free, released twice, or outside the pool is ignored instead of duplicating
//! a free-list entry. The free lists are sized at construction and never grow.

use std::{
    collections::VecDeque,
    mem::ManuallyDrop,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{HostRegistrar, Registration, Result, SplitMapping};

use super::geometry::{BLOCK_BYTES, Layout, invalid};

/// Page slabs in a staging pool: two outbound and two inbound windows.
pub const STAGING_PAGE_SLABS: usize = 256;
/// Row slabs in a staging pool.
pub const STAGING_ROW_SLABS: usize = 4;
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

/// Bytes a region of `count` slabs of `slab_bytes` occupies. A zero count
/// occupies nothing and ignores the slab size; otherwise the size must be a
/// nonzero multiple of [`BLOCK_BYTES`] and the region must fit the address space.
fn region_bytes(kind: &str, slab_bytes: u64, count: usize) -> Result<usize> {
    if count == 0 {
        return Ok(0);
    }
    if slab_bytes == 0 || !slab_bytes.is_multiple_of(BLOCK_BYTES) {
        return Err(invalid(format!(
            "{kind} slab size of {slab_bytes} bytes is not a nonzero multiple of {BLOCK_BYTES}"
        )));
    }
    u32::try_from(count)
        .ok()
        .and_then(|_| usize::try_from(slab_bytes).ok()?.checked_mul(count))
        .ok_or_else(|| {
            invalid(format!(
                "{count} {kind} slabs of {slab_bytes} bytes exceed the addressable pool size"
            ))
        })
}

/// Device addresses of one rank's part of every slab.
///
/// A split pool keeps each rank's share of every slot contiguously in its memory.
/// Slot addresses remain identities:
/// where each slab would begin if the pool held whole cross-rank records. They
/// lie inside the pool's reserved range, so they never name another pool's slab.
/// A registered pool instead uses CPU slab addresses as identities and the
/// registration's device alias for copies, keeping cross-rank record strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankPart {
    /// The pool's reserved range.
    range: std::ops::Range<usize>,
    pages: PartRegion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PartRegion {
    /// Identity of slab 0; identities advance by one cross-rank record.
    identity: usize,
    record_bytes: usize,
    count: usize,
    /// This rank's part of slab 0; parts advance by the rank's share of a record.
    base: usize,
    part_bytes: usize,
}

impl PartRegion {
    fn locate(&self, slab: usize) -> Option<u64> {
        let offset = slab.checked_sub(self.identity)?;
        let index = offset / self.record_bytes;
        (offset.is_multiple_of(self.record_bytes) && index < self.count)
            .then(|| (self.base + index * self.part_bytes) as u64)
    }
}

impl RankPart {
    /// This rank's part of the page slab at `slab`, or `None` when `slab` lies
    /// outside the pool. An address inside the pool that is not one of its page
    /// slabs is rejected.
    pub fn page(&self, slab: usize) -> Result<Option<u64>> {
        self.locate(&self.pages, slab, "page")
    }

    fn locate(&self, region: &PartRegion, slab: usize, kind: &str) -> Result<Option<u64>> {
        if !self.range.contains(&slab) {
            return Ok(None);
        }
        region.locate(slab).map(Some).ok_or_else(|| {
            invalid(format!(
                "address {slab:#x} inside a slab pool is not one of its {kind} slabs"
            ))
        })
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
    memory: Arc<Memory>,
    slots: Arc<Mutex<Slots>>,
}

struct Memory {
    mapping: usize,
    mapping_bytes: usize,
    page_bytes: usize,
    page_count: usize,
    retain: bool,
    registration: ManuallyDrop<Option<Box<dyn Registration>>>,
    split: ManuallyDrop<Option<Split>>,
    leaked: AtomicBool,
}

impl SlabPool {
    /// Map equal, page-record-sized slots.
    pub fn new(page_bytes: u64, count: usize, retain: bool) -> Result<Self> {
        let mapping_bytes = region_bytes("slot", page_bytes, count)?;
        if mapping_bytes == 0 {
            return Err(invalid("a slot pool requires at least one slot"));
        }
        let base = flare_eos::map_host(mapping_bytes)
            .map_err(|error| invalid(format!("allocate host slots: {error}")))?;
        Ok(Self {
            memory: Arc::new(Memory {
                mapping: base,
                mapping_bytes,
                page_bytes: page_bytes as usize,
                page_count: count,
                retain,
                registration: ManuallyDrop::new(None),
                split: ManuallyDrop::new(None),
                leaked: AtomicBool::new(false),
            }),
            slots: Arc::new(Mutex::new(Slots::new(count))),
        })
    }

    /// Round a warm budget down to whole slots.
    pub fn sized(layout: &Layout, bytes: u64) -> Result<Self> {
        Self::new(
            layout.page_record_bytes,
            (bytes
                .checked_div(layout.page_record_bytes)
                .filter(|_| layout.row_record_bytes != 0)
                .ok_or_else(|| invalid("record layout declares zero-byte page or row records"))?)
                as usize,
            true,
        )
    }

    /// A retaining pool with the slab counts of [`SlabPool::sized`], split by
    /// rank: each rank's part of every slab is pinned host memory that only the
    /// device of `contexts[rank]` and the host access. Registering one mapping
    /// maps it into every device at about 2 MiB of device memory per GiB; a
    /// split pool costs each device that for its own part only. The CPU cannot
    /// address a slab's record, so the pool suits stores that move records only
    /// between devices and slabs; [`SlabPool::rank_parts`] locates the parts.
    pub fn sized_by_rank(
        layout: &Layout,
        bytes: u64,
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
        let page_count = (bytes
            .checked_div(layout.page_record_bytes)
            .filter(|_| layout.row_record_bytes != 0)
            .ok_or_else(|| invalid("record layout declares zero-byte page or row records"))?)
            as usize;
        let page_region = region_bytes("page", layout.page_record_bytes, page_count)?;
        let shares: Vec<usize> = (0..layout.ranks.len())
            .map(|rank| layout.page_share(rank) as usize)
            .collect();
        let requests: Vec<_> = contexts
            .iter()
            .zip(&shares)
            .map(|(&(context, device), page)| (context, device, page * page_count))
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
                pages: PartRegion {
                    identity: base,
                    record_bytes: layout.page_record_bytes as usize,
                    count: page_count,
                    base: memory.part_address(rank),
                    part_bytes: page,
                },
            })
            .collect();
        Ok(Self {
            memory: Arc::new(Memory {
                mapping: base,
                mapping_bytes: memory.nbytes(),
                page_bytes: layout.page_record_bytes as usize,
                page_count,
                retain: true,
                registration: ManuallyDrop::new(None),
                split: ManuallyDrop::new(Some(Split {
                    parts,
                    _memory: memory,
                })),
                leaked: AtomicBool::new(false),
            }),
            slots: Arc::new(Mutex::new(Slots::new(page_count))),
        })
    }

    /// Each rank's part of the slabs, in rank order, for a pool split by rank.
    pub fn rank_parts(&self) -> Option<Vec<RankPart>> {
        self.memory.split.as_ref().map(|split| split.parts.clone())
    }

    /// Resolve GPU copy addresses once at startup. Disk I/O and CPU slices
    /// continue to use the host addresses returned by `page_address`.
    pub fn device_parts(&self, layout: &Layout) -> Result<Vec<RankPart>> {
        if layout.page_record_bytes != self.page_bytes() as u64 {
            return Err(invalid(
                "slab sizes disagree with the transfer record layout",
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
            .ok_or_else(|| invalid("transfer slabs must be registered with CUDA"))?;
        Ok(self.registered_parts(layout, registration.device_address()))
    }

    pub fn registered_parts(&self, layout: &Layout, device: u64) -> Vec<RankPart> {
        layout
            .ranks
            .iter()
            .map(|rank| RankPart {
                range: self.memory.mapping..self.memory.mapping + self.memory.mapping_bytes,
                pages: PartRegion {
                    identity: self.memory.mapping,
                    record_bytes: self.page_bytes(),
                    count: self.page_count(),
                    base: (device + rank.page_offset) as usize,
                    part_bytes: self.page_bytes(),
                },
            })
            .collect()
    }

    /// Non-retaining slots for two inbound and two outbound windows.
    pub fn staging(layout: &Layout) -> Result<Self> {
        Self::new(
            layout.page_record_bytes,
            STAGING_PAGE_SLABS + STAGING_ROW_SLABS * layout.row_slots(),
            false,
        )
    }

    /// Register the mapping through `registrar` as portable so every rank's context
    /// can address it, with `context` made current on the calling thread.
    /// A pool that is already registered is not registered again.
    pub fn register(&mut self, registrar: &dyn HostRegistrar, context: usize) -> Result<()> {
        if self.registered() {
            return Err(invalid(
                "slab pool is already registered with the CUDA driver",
            ));
        }
        // The mapping is page aligned and stays mapped at this size for the
        // pool's lifetime; `Drop` releases the registration before unmapping.
        let memory = Arc::get_mut(&mut self.memory)
            .ok_or_else(|| invalid("cannot register a shared slab pool"))?;
        *memory.registration =
            Some(registrar.register(context, memory.mapping, memory.mapping_bytes)?);
        Ok(())
    }

    /// Whether the driver can copy the slabs: after `register`, and always for a pool split by rank.
    pub fn registered(&self) -> bool {
        self.memory.registration.is_some() || self.memory.split.is_some()
    }

    pub fn retain(&self) -> bool {
        self.memory.retain
    }

    pub fn page_bytes(&self) -> usize {
        self.memory.page_bytes
    }
    pub fn page_count(&self) -> usize {
        self.memory.page_count
    }
    pub fn free_pages(&self) -> usize {
        self.lock().free.len()
    }
    pub fn bytes(&self) -> u64 {
        self.memory.mapping_bytes as u64
    }

    pub fn allocate_page(&mut self) -> Option<Slot> {
        let mut slots = self.lock();
        let index = slots.free.pop_front()?;
        slots.allocated[index as usize] = true;
        Some(Slot(index))
    }
    /// Return an allocated slot once; ignore free, repeated and out-of-range slots.
    pub fn release_page(&mut self, slab: Slot) {
        let mut slots = self.lock();
        if let Some(allocated) = slots.allocated.get_mut(slab.0 as usize)
            && *allocated
        {
            *allocated = false;
            slots.free.push_back(slab.0);
        }
    }
    pub fn page_address(&self, slab: Slot) -> usize {
        assert!(
            (slab.0 as usize) < self.page_count(),
            "slab index outside the pool"
        );
        self.memory.mapping + slab.0 as usize * self.page_bytes()
    }
    /// Failed drain: retain the mapping permanently and prevent slot reuse.
    pub(crate) fn leak(&self) {
        let mut slots = self.lock();
        slots.free.clear();
        slots.allocated.fill(false);
        self.memory.leaked.store(true, Ordering::Relaxed);
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, Slots> {
        self.slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    /// The caller owns the slot and keeps it pinned through every reader.
    pub fn page(&self, slab: Slot) -> &[u8] {
        self.assert_contiguous();
        // SAFETY: The mapping outlives the pool and the slab lies inside it.
        unsafe {
            std::slice::from_raw_parts(self.page_address(slab) as *const u8, self.page_bytes())
        }
    }
    pub fn page_mut(&mut self, slab: Slot) -> &mut [u8] {
        self.assert_contiguous();
        // SAFETY: The caller owns the slot exclusively while mutating its bytes.
        unsafe {
            std::slice::from_raw_parts_mut(self.page_address(slab) as *mut u8, self.page_bytes())
        }
    }
    fn assert_contiguous(&self) {
        assert!(
            self.memory.split.is_none(),
            "the slabs of a pool split by rank hold no contiguous record"
        );
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        if self.leaked.load(Ordering::Relaxed) {
            return;
        }
        // SAFETY: The last handle dropped and both copiers drained; unregister before unmapping.
        unsafe {
            ManuallyDrop::drop(&mut self.registration);
            if self.split.is_none() {
                flare_eos::unmap_host(self.mapping, self.mapping_bytes);
            }
            ManuallyDrop::drop(&mut self.split);
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
        pages: PartRegion {
            identity: range.start,
            record_bytes,
            count,
            base,
            part_bytes,
        },
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
        let mut pool = SlabPool::new(8192, 4, true).unwrap();
        let slots: Vec<_> = (0..4).map(|_| pool.allocate_page().unwrap()).collect();
        assert_eq!(pool.allocate_page(), None);
        for (index, &slot) in slots.iter().enumerate() {
            assert_eq!(pool.page_address(slot) % BLOCK_BYTES as usize, 0);
            pool.page_mut(slot).fill(index as u8);
            assert!(pool.page(slot).iter().all(|&byte| byte == index as u8));
        }
        assert_eq!(
            pool.page_address(slots[3]) - pool.page_address(slots[0]),
            3 * 8192
        );
        for &slot in &slots {
            pool.release_page(slot);
        }
        pool.release_page(slots[0]);
        pool.release_page(Slot(99));
        assert_eq!(pool.free_pages(), 4);
        assert_eq!(pool.allocate_page(), Some(slots[0]));
    }

    #[test]
    fn shared_handles_use_one_free_list_and_retain_the_registration() {
        let registrar = crate::testing::TrackingRegistrar::default();
        let mut pool = SlabPool::new(4096, 2, true).unwrap();
        pool.register(&registrar, 0).unwrap();
        let mut other = pool.clone();
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
        let mut pool = SlabPool::new(4096, 2, true).unwrap();
        pool.register(&registrar, 0).unwrap();
        let slot = pool.allocate_page().unwrap();
        let other = pool.clone();
        let locked = other.slots.lock().unwrap();
        let (done, completion) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let address = pool.page_address(slot);
            pool.page_mut(slot).fill(42);
            assert!(pool.page(slot).iter().all(|&byte| byte == 42));
            assert_eq!(pool.page_bytes(), 4096);
            assert_eq!(pool.page_count(), 2);
            assert_eq!(pool.bytes(), 8192);
            assert!(pool.retain() && pool.registered());
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
        let mut pool = SlabPool::new(4096, 2, true).unwrap();
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
        let mut pool = SlabPool::new(4096, 37, true).unwrap();
        let mut owned = Vec::new();
        let mut seed = 71u64;
        for _ in 0..10_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if seed & 8 != 0 {
                if let Some(slot) = pool.allocate_page() {
                    assert!(!owned.contains(&slot));
                    owned.push(slot);
                }
            } else if !owned.is_empty() {
                let slot = owned.swap_remove(seed as usize % owned.len());
                pool.release_page(slot);
            }
            assert_eq!(pool.free_pages() + owned.len(), pool.page_count());
        }
    }

    #[test]
    fn warm_sizes_round_down_without_a_separate_row_minimum() {
        let layout = Layout::new(&[geometry(64, &[("kv", 8192)], 4096)]).unwrap();
        let pool = SlabPool::sized(&layout, 3 * 8192 + 4095).unwrap();
        assert_eq!(pool.bytes(), 3 * 8192);
        assert_eq!(pool.page_count(), 3);
        assert!(SlabPool::new(4000, 1, true).is_err());
        assert!(SlabPool::new(0, 1, true).is_err());
        assert!(SlabPool::new(8192, usize::MAX, true).is_err());
        let mut invalid = layout.clone();
        invalid.row_record_bytes = 0;
        assert!(SlabPool::sized(&invalid, 8192).is_err());
        invalid.page_record_bytes = 0;
        assert!(SlabPool::sized(&invalid, 8192).is_err());
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
            let mut pool = SlabPool::new(layout.page_record_bytes, 8, true).unwrap();
            let slots: Vec<_> = (0..4)
                .map(|_| {
                    let slot = pool.allocate_page().unwrap();
                    let gap = pool.allocate_page().unwrap();
                    pool.release_page(gap);
                    pool.page_address(slot)
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
