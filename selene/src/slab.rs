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

use std::collections::VecDeque;

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

struct SubPool {
    base: usize,
    slab_bytes: usize,
    count: usize,
    free: VecDeque<u32>,
    allocated: Vec<bool>,
}

impl SubPool {
    fn new(base: usize, slab_bytes: usize, count: usize) -> Self {
        let mut free = VecDeque::with_capacity(count);
        free.extend(0..count as u32);
        Self {
            base,
            slab_bytes,
            count,
            free,
            allocated: vec![false; count],
        }
    }

    /// Slab indices are produced only by `allocate`, so an index outside the
    /// pool is a caller defect; the panic keeps `page` and `row` inside the mapping.
    fn address(&self, index: u32) -> usize {
        assert!(
            (index as usize) < self.count,
            "slab index {index} outside a pool of {} slabs",
            self.count
        );
        self.base + index as usize * self.slab_bytes
    }

    fn allocate(&mut self) -> Option<u32> {
        let index = self.free.pop_front()?;
        self.allocated[index as usize] = true;
        Some(index)
    }

    /// Return `index` to the back of the free list. A slab that is not
    /// allocated, including one released twice or an index outside the pool,
    /// is ignored so the free list never holds a slab twice.
    fn release(&mut self, index: u32) {
        if let Some(allocated) = self.allocated.get_mut(index as usize)
            && *allocated
        {
            *allocated = false;
            self.free.push_back(index);
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

/// Equal slots inside one registered mapping, or split by rank.
pub struct SlabPool {
    mapping: usize,
    mapping_bytes: usize,
    pages: SubPool,
    retain: bool,
    registration: Option<Box<dyn Registration>>,
    split: Option<Split>,
}

// SAFETY: The mapping is plain process memory; the pool is used from one thread
// at a time and slab addresses handed to other threads outlive their transfers.
unsafe impl Send for SlabPool {}

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
            mapping: base,
            mapping_bytes,
            pages: SubPool::new(base, page_bytes as usize, count),
            retain,
            registration: None,
            split: None,
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
            mapping: base,
            mapping_bytes: memory.nbytes(),
            pages: SubPool::new(base, layout.page_record_bytes as usize, page_count),
            retain: true,
            registration: None,
            split: Some(Split {
                parts,
                _memory: memory,
            }),
        })
    }

    /// Each rank's part of the slabs, in rank order, for a pool split by rank.
    pub fn rank_parts(&self) -> Option<Vec<RankPart>> {
        self.split.as_ref().map(|split| split.parts.clone())
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
                range: self.mapping..self.mapping + self.mapping_bytes,
                pages: PartRegion {
                    identity: self.pages.base,
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
        let registration = registrar.register(context, self.mapping, self.mapping_bytes)?;
        self.registration = Some(registration);
        Ok(())
    }

    /// Release the driver registration; a pool that is not registered is unchanged.
    pub fn unregister(&mut self) -> Result<()> {
        self.registration.take();
        Ok(())
    }

    /// Whether the driver can copy the slabs: after `register`, and always for a pool split by rank.
    pub fn registered(&self) -> bool {
        self.registration.is_some() || self.split.is_some()
    }

    pub fn retain(&self) -> bool {
        self.retain
    }

    pub fn page_bytes(&self) -> usize {
        self.pages.slab_bytes
    }
    pub fn page_count(&self) -> usize {
        self.pages.count
    }
    pub fn free_pages(&self) -> usize {
        self.pages.free.len()
    }
    pub fn bytes(&self) -> u64 {
        self.mapping_bytes as u64
    }

    pub fn allocate_page(&mut self) -> Option<Slot> {
        self.pages.allocate().map(Slot)
    }
    /// Returns `slab` to the free list; a slab that is not allocated is ignored.
    pub fn release_page(&mut self, slab: Slot) {
        self.pages.release(slab.0);
    }
    pub fn page_address(&self, slab: Slot) -> usize {
        self.pages.address(slab.0)
    }
    /// The record bytes of a pool that is not split by rank; a split pool has
    /// no contiguous record, so asking for one is a caller defect.
    pub fn page(&self, slab: Slot) -> &[u8] {
        self.assert_contiguous();
        // SAFETY: The mapping outlives the pool and the slab lies inside it.
        unsafe {
            std::slice::from_raw_parts(self.page_address(slab) as *const u8, self.pages.slab_bytes)
        }
    }
    pub fn page_mut(&mut self, slab: Slot) -> &mut [u8] {
        self.assert_contiguous();
        // SAFETY: As above, with exclusive access through `&mut self`.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.page_address(slab) as *mut u8,
                self.pages.slab_bytes,
            )
        }
    }
    fn assert_contiguous(&self) {
        assert!(
            self.split.is_none(),
            "the slabs of a pool split by rank hold no contiguous record"
        );
    }
}

impl Drop for SlabPool {
    fn drop(&mut self) {
        self.registration.take();
        // A split pool's memory releases its own mappings and reservation.
        if self.split.take().is_none() && self.mapping != 0 {
            // SAFETY: The mapping was created by this pool and is no longer registered.
            unsafe {
                flare_eos::unmap_host(self.mapping, self.mapping_bytes);
            }
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
                                + (slot - pool.mapping) as u64
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
