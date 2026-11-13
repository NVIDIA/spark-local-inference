/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Mutually exclusive ownership of snapshot aliases and KV page ids.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SnapshotQuote {
    pub(super) snapshots: Vec<SnapshotId>,
    victims: Vec<CheckpointId>,
    planned_rows: usize,
    witness: Vec<SnapshotId>,
    keep_pages: usize,
    rank: crate::cache::RetentionRank,
}

fn select_snapshots(
    backing: &[bool],
    protected_refs: &[u32],
    ranges: &[std::ops::Range<u32>],
    candidates: &BTreeSet<SnapshotId>,
    keep_pages: usize,
    count: usize,
) -> Option<Vec<SnapshotId>> {
    let available = backing.iter().filter(|free| **free).count();
    if available < keep_pages {
        return None;
    }
    if count == 0 {
        return Some(Vec::new());
    }
    let candidates: Vec<_> = candidates
        .iter()
        .copied()
        .filter_map(|id| {
            let mut cost = Vec::new();
            if let Some(range) = ranges.get(id.0 as usize) {
                for page in range.clone() {
                    if backing[page as usize] {
                        cost.push(page as usize);
                    } else if protected_refs[page as usize] == 0 {
                        return None;
                    }
                }
            }
            Some((id, cost))
        })
        .collect();
    if count > 2 {
        // Future capture capacity uses a conservative witness without enumerating subsets.
        let mut selected = BTreeSet::<SnapshotId>::new();
        let mut excluded = BTreeSet::<usize>::new();
        for _ in 0..count {
            let (id, pages) = candidates
                .iter()
                .filter(|(id, _)| !selected.contains(id))
                .min_by_key(|(id, pages)| {
                    (
                        pages
                            .iter()
                            .filter(|page| !excluded.contains(*page))
                            .count(),
                        *id,
                    )
                })?;
            excluded.extend(pages);
            if excluded.len() > available - keep_pages {
                return None;
            }
            selected.insert(*id);
        }
        return Some(selected.into_iter().collect());
    }
    fn choose(
        candidates: &[(SnapshotId, Vec<usize>)],
        start: usize,
        count: usize,
        selected: &mut Vec<SnapshotId>,
        excluded: &BTreeSet<usize>,
        budget: usize,
        best: &mut Option<(usize, Vec<SnapshotId>)>,
    ) {
        if excluded.len() > budget {
            return;
        }
        // Combinations visit snapshot IDs in lexicographic order.
        if best.as_ref().is_some_and(|best| best.0 == 0) {
            return;
        }
        if count == 0 {
            let candidate = (excluded.len(), selected.clone());
            if best.as_ref().is_none_or(|best| candidate < *best) {
                *best = Some(candidate);
            }
            return;
        }
        if candidates.len().saturating_sub(start) < count {
            return;
        }
        for index in start..=candidates.len() - count {
            let (id, pages) = &candidates[index];
            let mut union = excluded.clone();
            union.extend(pages);
            if best.as_ref().is_some_and(|best| union.len() > best.0) {
                continue;
            }
            selected.push(*id);
            choose(
                candidates,
                index + 1,
                count - 1,
                selected,
                &union,
                budget,
                best,
            );
            selected.pop();
        }
    }
    let mut best = None;
    choose(
        &candidates,
        0,
        count,
        &mut Vec::new(),
        &BTreeSet::new(),
        available - keep_pages,
        &mut best,
    );
    best.map(|(_, selected)| selected)
}

impl RequestScheduler {
    pub fn share_snapshot_pages(&mut self, ranges: &[std::ops::Range<u32>]) -> Result<()> {
        if self.submitted != Fence(0)
            || self.slots.iter().any(|s| s.request.is_some())
            || ranges.len() != self.limits.snapshots as usize
            || ranges
                .iter()
                .any(|r| r.start > r.end || r.end > self.limits.pages)
        {
            return Err(
                "snapshot aliases must cover valid KV ids and be configured before admission"
                    .into(),
            );
        }
        self.snapshot_pages = ranges.to_vec();
        Ok(())
    }

    pub(super) fn available_snapshot(&self, keep_pages: usize) -> Option<SnapshotId> {
        let backing: Vec<_> = (0..self.limits.pages)
            .map(|page| self.free_pages.contains(&PageId(page)))
            .collect();
        select_snapshots(
            &backing,
            &self.page_snapshot_refs,
            &self.snapshot_pages,
            &self.free_snapshots,
            keep_pages,
            1,
        )
        .and_then(|ids| ids.first().copied())
    }

    fn reserve_snapshot_id(&mut self, id: SnapshotId) {
        assert!(self.free_snapshots.remove(&id));
        if let Some(range) = self.snapshot_pages.get(id.0 as usize) {
            for page in range.clone() {
                self.free_pages.remove(&PageId(page));
                if self.page_snapshot_refs[page as usize] == 0 {
                    self.stats.snapshot_backing_pages += 1;
                }
                self.page_snapshot_refs[page as usize] += 1;
            }
        }
    }

    pub(super) fn reserve_snapshot(&mut self, keep_pages: usize) -> Option<SnapshotId> {
        let id = self.available_snapshot(keep_pages)?;
        self.reserve_snapshot_id(id);
        Some(id)
    }

    pub(super) fn quote_snapshots(
        &self,
        rows: usize,
        keep_pages: usize,
        rank: crate::cache::RetentionRank,
    ) -> Option<SnapshotQuote> {
        self.quote_snapshots_preserving(rows, rows, keep_pages, rank)
    }

    pub(super) fn quote_snapshots_preserving(
        &self,
        rows: usize,
        planned_rows: usize,
        keep_pages: usize,
        rank: crate::cache::RetentionRank,
    ) -> Option<SnapshotQuote> {
        if rows > planned_rows {
            return None;
        }
        if rows == planned_rows {
            return self.simulate_snapshot_quote(rows, keep_pages, rank, None);
        }
        let future = self.simulate_snapshot_quote(planned_rows, keep_pages, rank, None)?;
        let candidates = future.snapshots.iter().copied().collect();
        let mut current =
            self.simulate_snapshot_quote(rows, keep_pages, rank, Some(&candidates))?;
        current.planned_rows = planned_rows;
        current.witness = future.snapshots;
        Some(current)
    }

    fn simulate_snapshot_quote(
        &self,
        rows: usize,
        keep_pages: usize,
        rank: crate::cache::RetentionRank,
        allowed: Option<&BTreeSet<SnapshotId>>,
    ) -> Option<SnapshotQuote> {
        let keep_pages = keep_pages.max(self.decode_growth_pages());
        let mut cache = self.cache.clone();
        let mut free_pages = self.free_pages.clone();
        let mut free_snapshots = self.free_snapshots.clone();
        let mut refs = self.page_snapshot_refs.clone();
        let mut victims = Vec::new();
        loop {
            let backing: Vec<_> = (0..self.limits.pages)
                .map(|page| free_pages.contains(&PageId(page)))
                .collect();
            let candidates =
                allowed.map(|allowed| free_snapshots.intersection(allowed).copied().collect());
            if let Some(snapshots) = select_snapshots(
                &backing,
                &refs,
                &self.snapshot_pages,
                candidates.as_ref().unwrap_or(&free_snapshots),
                keep_pages,
                rows,
            ) {
                return Some(SnapshotQuote {
                    witness: snapshots.clone(),
                    snapshots,
                    victims,
                    planned_rows: rows,
                    keep_pages,
                    rank,
                });
            }
            let id = cache.eviction_candidate(Some(rank), true)?;
            cache.evict_checkpoint(id).ok()?;
            victims.push(id);
            let reusable = cache.advance_completed(self.completed).ok()?.reusable;
            free_pages.extend(reusable.pages);
            for snapshot in reusable.snapshots {
                free_snapshots.insert(snapshot);
                if let Some(range) = self.snapshot_pages.get(snapshot.0 as usize) {
                    for page in range.clone() {
                        refs[page as usize] -= 1;
                        if refs[page as usize] == 0 {
                            free_pages.insert(PageId(page));
                        }
                    }
                }
            }
        }
    }

    pub(super) fn apply_snapshot_quote(
        &mut self,
        quote: &SnapshotQuote,
    ) -> Result<Option<Vec<SnapshotId>>> {
        if self
            .quote_snapshots_preserving(
                quote.snapshots.len(),
                quote.planned_rows,
                quote.keep_pages,
                quote.rank,
            )
            .as_ref()
            != Some(quote)
        {
            return Ok(None);
        }
        let mut report = EvictionReport::default();
        for id in &quote.victims {
            let retired = self.cache.evict_checkpoint(*id)?;
            report.pages_scheduled += retired.pages.len();
            report.snapshots_scheduled += 1;
            report.retired.push(retired);
        }
        self.record_evictions(&report);
        self.stats.snapshot_evictions += report.retired.len() as u64;
        let update = self.cache.advance_completed(self.completed)?;
        self.apply_completion(update);
        for id in &quote.snapshots {
            self.reserve_snapshot_id(*id);
        }
        Ok(Some(quote.snapshots.clone()))
    }

    pub(super) fn retain_snapshot(&mut self) -> Result<Option<SnapshotId>> {
        loop {
            if let Some(id) = self.reserve_snapshot(0) {
                return Ok(Some(id));
            }
            if self.revoke_unused_anchor_guards() {
                continue;
            }
            let evicted = self.evict(EvictionTarget {
                pages: 0,
                snapshots: 1,
            });
            self.stats.snapshot_evictions += evicted.retired.len() as u64;
            let update = self.cache.advance_completed(self.completed)?;
            self.apply_completion(update);
            if evicted.retired.is_empty() {
                return Ok(None);
            }
        }
    }

    /// Backing that is free or becomes free once every retired owner and `extra` reaches
    /// its fence, with the snapshots those owners release.
    pub(super) fn pending_backing(&self, extra: &Reclaimable) -> (Vec<bool>, Vec<SnapshotId>) {
        let mut available = vec![false; self.limits.pages as usize];
        let mut retiring_refs = vec![0; available.len()];
        let mut snapshots = extra.snapshots.clone();
        for page in self.free_pages.iter().chain(&extra.pages) {
            available[page.0 as usize] = true;
        }
        for retirement in self.cache.pending_retirement() {
            for page in &retirement.pages {
                available[page.0 as usize] = true;
            }
            snapshots.push(retirement.snapshot);
        }
        for retirement in &self.retired {
            for page in &retirement.pages {
                available[page.0 as usize] = true;
            }
            snapshots.extend(retirement.snapshots.iter().flatten().copied());
        }
        for snapshot in &snapshots {
            if let Some(range) = self.snapshot_pages.get(snapshot.0 as usize) {
                for page in range.clone() {
                    retiring_refs[page as usize] += 1;
                }
            }
        }
        for (page, refs) in retiring_refs.into_iter().enumerate() {
            if refs > 0 && refs == self.page_snapshot_refs[page] {
                available[page] = true;
            }
        }
        (available, snapshots)
    }

    /// Project free backing after every already-retired owner reaches its fence.
    pub(super) fn pending_page_count(&self) -> usize {
        let (available, _) = self.pending_backing(&Reclaimable::default());
        available.into_iter().filter(|free| *free).count()
    }

    pub(super) fn pending_resources(&self, pages: usize, snapshots: usize) -> bool {
        if self.cache.pending_retirement().is_empty() && self.retired.is_empty() {
            return false;
        }
        let (backing, retiring) = self.pending_backing(&Reclaimable::default());
        self.projected_fits(&backing, &retiring, pages, snapshots)
    }

    /// Whether `pages` pages and `snapshots` snapshot ids are obtainable once `backing` is
    /// free and the `retiring` snapshots are released, honouring snapshot aliases.
    pub(super) fn projected_fits(
        &self,
        backing: &[bool],
        retiring: &[SnapshotId],
        pages: usize,
        snapshots: usize,
    ) -> bool {
        let mut candidates = self.free_snapshots.clone();
        let mut refs = self.page_snapshot_refs.clone();
        let retiring: BTreeSet<_> = retiring.iter().copied().collect();
        for snapshot in retiring {
            candidates.insert(snapshot);
            if let Some(range) = self.snapshot_pages.get(snapshot.0 as usize) {
                for page in range.clone() {
                    refs[page as usize] -= 1;
                }
            }
        }
        select_snapshots(
            backing,
            &refs,
            &self.snapshot_pages,
            &candidates,
            pages,
            snapshots,
        )
        .is_some()
    }

    pub(super) fn release_snapshot(&mut self, id: SnapshotId) {
        assert!(self.free_snapshots.insert(id), "snapshot released twice");
        if let Some(range) = self.snapshot_pages.get(id.0 as usize) {
            for page in range.clone() {
                let count = &mut self.page_snapshot_refs[page as usize];
                *count = count.checked_sub(1).expect("snapshot alias ownership");
                if *count == 0 {
                    self.stats.snapshot_backing_pages -= 1;
                    assert!(self.free_pages.insert(PageId(page)));
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn capture_rank() -> crate::cache::RetentionRank {
        crate::cache::RetentionRank { use_epoch: 2, role: crate::cache::RetentionRole::Message, consumed_tokens: 2 }
    }

    #[test]
    fn snapshot_selection_minimizes_newly_excluded_pages_before_id() {
        let mut s = super::super::tests::scheduler(1, 4, 2);
        s.share_snapshot_pages(&[0..3, 2..3]).unwrap();
        assert_eq!(s.reserve_snapshot(1), Some(SnapshotId(1)));
        assert_eq!(s.free_page_count(), 3);
    }

    #[test]
    fn joint_snapshot_quote_checks_union_cost_without_evicting_impossible_pairs() {
        let mut s = super::super::tests::scheduler(1, 4, 2);
        s.share_snapshot_pages(&[0..3, 1..4]).unwrap();
        let (backing, retiring) = s.pending_backing(&Reclaimable::default());
        assert!(s.projected_fits(&backing, &retiring, 1, 1));
        assert!(!s.projected_fits(&backing, &retiring, 1, 2));
        assert!(s.quote_snapshots(2, 1, capture_rank()).is_none());
        assert_eq!(s.free_page_count(), 4);
        assert_eq!(s.free_snapshot_count(), 2);
    }

    #[test]
    fn impossible_quote_preserves_removable_checkpoints_and_aliases() {
        let mut s = super::super::tests::scheduler(1, 4, 2);
        s.share_snapshot_pages(&[0..2, 2..4]).unwrap();
        let a = s.reserve_snapshot(0).unwrap();
        let b = s.reserve_snapshot(0).unwrap();
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&[1, 2], &[]).unwrap();
        let mut parent = None;
        for (endpoint, snapshot) in [(1, a), (2, b)] {
            parent = Some(s.cache.publish(&history, &Publication {
                parent, consumed_tokens: endpoint, tail_pages: Vec::new(),
                snapshot: StateSnapshot { id: snapshot, consumed_tokens: endpoint }, ready_after: Fence(0),
            }).unwrap().id());
        }
        assert!(s.quote_snapshots(2, 1, capture_rank()).is_none());
        assert_eq!(s.cache.stats().published_checkpoints, 2);
        assert_eq!(s.free_page_count(), 0);
        assert_eq!(s.stats().snapshot_backing_pages, 4);
    }

    #[test]
    fn large_future_capture_witness_reserves_only_the_immediate_pair() {
        let mut s = super::super::tests::scheduler(1, 64, 64);
        s.share_snapshot_pages(&(0..64).map(|page| page..page + 1).collect::<Vec<_>>()).unwrap();
        let quote = s.quote_snapshots_preserving(2, 20, 44, capture_rank()).unwrap();
        assert_eq!(quote.snapshots, [SnapshotId(0), SnapshotId(1)]);
        assert_eq!(quote.witness.len(), 20);
        assert_eq!(s.apply_snapshot_quote(&quote).unwrap(), Some(vec![SnapshotId(0), SnapshotId(1)]));
        assert_eq!(s.free_page_count(), 62);
        assert_eq!(s.free_snapshot_count(), 62);
    }

    #[test]
    fn future_capture_witness_does_not_evict_checkpoints_needed_only_by_future_rows() {
        let mut s = super::super::tests::scheduler(1, 4, 3);
        let snapshot = SnapshotId(2);
        s.reserve_snapshot_id(snapshot);
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&[1], &[]).unwrap();
        let id = s.cache.publish(&history, &Publication {
            parent: None, consumed_tokens: 1, tail_pages: Vec::new(),
            snapshot: StateSnapshot { id: snapshot, consumed_tokens: 1 }, ready_after: Fence(0),
        }).unwrap().id();
        let quote = s.quote_snapshots_preserving(2, 3, 4, capture_rank()).unwrap();
        assert_eq!(quote.snapshots, [SnapshotId(0), SnapshotId(1)]);
        assert!(quote.victims.is_empty());
        s.apply_snapshot_quote(&quote).unwrap().unwrap();
        assert!(s.cache.checkpoint(id).is_ok());
        assert_eq!(s.stats().snapshot_evictions, 0);
    }

    #[test]
    fn future_capture_witness_is_revalidated_before_any_current_eviction() {
        let mut s = super::super::tests::scheduler(1, 4, 3);
        let snapshot = SnapshotId(2);
        s.reserve_snapshot_id(snapshot);
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&[1], &[]).unwrap();
        let id = s.cache.publish(&history, &Publication {
            parent: None, consumed_tokens: 1, tail_pages: Vec::new(),
            snapshot: StateSnapshot { id: snapshot, consumed_tokens: 1 }, ready_after: Fence(0),
        }).unwrap().id();
        let quote = s.quote_snapshots_preserving(1, 3, 4, capture_rank()).unwrap();
        s.cache.pin(id).unwrap();
        assert!(s.apply_snapshot_quote(&quote).unwrap().is_none());
        assert_eq!(s.free_snapshot_count(), 2);
        assert!(s.cache.checkpoint(id).is_ok());
    }

    #[test]
    fn optional_quote_revalidates_live_decode_headroom_before_reserving_aliases() {
        let mut s = super::super::tests::scheduler(1, 4, 1);
        s.share_snapshot_pages(&[3..4]).unwrap();
        let mut admission = super::super::tests::spec(60, 400);
        admission.growable_output = true;
        let handle = s.admit(admission).unwrap();
        let quote = s.quote_snapshots(1, 0, capture_rank()).unwrap();
        let request = s.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.scheduled_consumed = 191;
        assert!(s.apply_snapshot_quote(&quote).unwrap().is_none());
        assert_eq!(s.free_page_count(), 2);
        assert_eq!(s.free_snapshot_count(), 1);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
    }

    #[test]
    fn joint_quote_reserves_overlapping_rows_and_rejects_stale_ownership() {
        let mut s = super::super::tests::scheduler(1, 4, 3);
        s.share_snapshot_pages(&[0..3, 2..3, 2..3]).unwrap();
        let quote = s.quote_snapshots(2, 3, capture_rank()).unwrap();
        assert_eq!(quote.snapshots, [SnapshotId(1), SnapshotId(2)]);
        assert_eq!(s.apply_snapshot_quote(&quote).unwrap().unwrap(), quote.snapshots);
        assert_eq!(s.free_page_count(), 3);
        assert!(s.apply_snapshot_quote(&quote).unwrap().is_none());
        s.release_snapshot(SnapshotId(1));
        assert_eq!(s.free_page_count(), 3);
        s.release_snapshot(SnapshotId(2));
        assert_eq!(s.free_page_count(), 4);
    }

    #[test]
    fn optional_quote_cannot_wait_for_gpu_reads_or_replace_higher_demand_rank() {
        for (use_epoch, fence) in [(1, Fence(3)), (3, Fence(0)), (1, Fence(0))] {
            let mut s = super::super::tests::scheduler(1, 4, 1);
            let snapshot = s.reserve_snapshot(0).unwrap();
            let mut history = PrefixHistory::new([9; 32]);
            history.append(&[1], &[]).unwrap();
            let id = s.cache.publish_with_retention(&history, &Publication {
                parent: None, consumed_tokens: 1, tail_pages: Vec::new(),
                snapshot: StateSnapshot { id: snapshot, consumed_tokens: 1 }, ready_after: Fence(0),
            }, crate::cache::RetentionRole::Endpoint, use_epoch).unwrap().id();
            s.cache.mark_used(id, fence).unwrap();
            let quote = s.quote_snapshots(1, 4, capture_rank());
            if use_epoch == 1 && fence == Fence(0) {
                let quote = quote.unwrap();
                assert!(s.cache.checkpoint(id).is_ok());
                s.cache.pin(id).unwrap();
                assert!(s.apply_snapshot_quote(&quote).unwrap().is_none());
                assert!(s.cache.checkpoint(id).is_ok());
                s.cache.unpin(id).unwrap();
                assert_eq!(s.apply_snapshot_quote(&quote).unwrap(), Some(vec![snapshot]));
                assert!(s.cache.checkpoint(id).is_err());
            } else {
                assert!(quote.is_none());
                assert!(s.cache.checkpoint(id).is_ok());
            }
        }
    }

    #[test]
    fn all_two_snapshot_alias_layouts_preserve_page_ownership_and_reservation_budget() {
        fn check(s: &RequestScheduler, ranges: &[std::ops::Range<u32>; 2], live: &[bool; 2]) {
            let refs: Vec<_> = (0..4)
                .map(|page| (0..2).filter(|&id| live[id] && ranges[id].contains(&page)).count())
                .collect();
            for (page, &count) in refs.iter().enumerate() {
                assert_eq!(s.page_snapshot_refs[page] as usize, count);
                assert_eq!(s.free_pages.contains(&PageId(page as u32)), count == 0);
            }
            assert_eq!(s.stats().snapshot_backing_pages as usize,
                refs.iter().filter(|&&count| count != 0).count());
            for (id, &allocated) in live.iter().enumerate() {
                assert_eq!(s.free_snapshots.contains(&SnapshotId(id as u32)), !allocated);
            }
        }

        fn reserve(s: &mut RequestScheduler, ranges: &[std::ops::Range<u32>; 2],
                   live: &mut [bool; 2], keep: usize) {
            let eligible: Vec<_> = (0..2).filter(|&id| {
                !live[id] && (0..4).filter(|page| {
                    !ranges[id].contains(page)
                        && !(0..2).any(|owner| live[owner] && ranges[owner].contains(page))
                }).count() >= keep
            }).collect();
            match s.reserve_snapshot(keep) {
                Some(id) => {
                    assert!(eligible.contains(&(id.0 as usize)));
                    live[id.0 as usize] = true;
                    assert!(s.free_page_count() >= keep);
                }
                None => assert!(eligible.is_empty()),
            }
            check(s, ranges, live);
        }

        let ranges: Vec<_> = (0..=4).flat_map(|start| (start..=4).map(move |end| start..end)).collect();
        for a in &ranges {
            for b in &ranges {
                let aliases = [a.clone(), b.clone()];
                for first_keep in 0..=5 {
                    for second_keep in 0..=5 {
                        for release_first in 0..2 {
                            let mut s = super::super::tests::scheduler(1, 4, 2);
                            s.share_snapshot_pages(&aliases).unwrap();
                            let mut live = [false; 2];
                            reserve(&mut s, &aliases, &mut live, first_keep);
                            reserve(&mut s, &aliases, &mut live, second_keep);
                            if live[release_first] {
                                s.release_snapshot(SnapshotId(release_first as u32));
                                live[release_first] = false;
                                check(&s, &aliases, &live);
                            }
                            reserve(&mut s, &aliases, &mut live, first_keep);
                            for id in [1 - release_first, release_first] {
                                if live[id] {
                                    s.release_snapshot(SnapshotId(id as u32));
                                    live[id] = false;
                                    check(&s, &aliases, &live);
                                }
                            }
                            assert_eq!(s.free_page_count(), 4);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn snapshots_in_execution_alignment_padding_do_not_exclude_pages() {
        let mut s = super::super::tests::scheduler(1, 16, 1);
        s.share_snapshot_pages(&[0..0]).unwrap();
        let snapshot = s.reserve_snapshot(16).unwrap();
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
        s.release_snapshot(snapshot);
        assert_eq!(s.free_page_count(), 16);
    }

    #[test]
    fn overlapping_snapshot_aliases_release_pages_only_after_the_last_owner() {
        let mut s = super::super::tests::scheduler(2, 16, 2);
        s.share_snapshot_pages(&[8..12, 10..14]).unwrap();
        let a = s.reserve_snapshot(0).unwrap();
        let b = s.reserve_snapshot(0).unwrap();
        assert_eq!(s.free_page_count(), 10);
        assert_eq!(s.stats().snapshot_backing_pages, 6);
        s.release_snapshot(a);
        assert_eq!(s.free_page_count(), 12);
        assert_eq!(s.stats().snapshot_backing_pages, 4);
        assert!(!s.free_pages.contains(&PageId(10)));
        s.release_snapshot(b);
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.stats().snapshot_backing_pages, 0);
    }

    #[test]
    fn request_page_reservations_exclude_snapshot_aliases() {
        let mut s = super::super::tests::scheduler(2, 16, 2);
        s.share_snapshot_pages(&[0..8, 8..16]).unwrap();
        let handle = s.admit(super::super::tests::spec(64 * 12, 1)).unwrap();
        assert_eq!(s.free_snapshot_count(), 2);
        assert!(s.reserve_snapshot(0).is_none());
        s.cancel(handle).unwrap();
        assert!(s.reserve_snapshot(8).is_some());
        assert!(s.reserve_snapshot(1).is_none());
    }

    #[test]
    fn cancellation_keeps_snapshot_backing_until_the_last_gpu_fence() {
        let mut s = super::super::tests::scheduler(1, 16, 1);
        s.share_snapshot_pages(&[8..16]).unwrap();
        let handle = s.admit(super::super::tests::spec(64, 1)).unwrap();
        let snapshot = s.reserve_snapshot(0).unwrap();
        s.request_mut(handle).unwrap().terminal_snapshot = Some(snapshot);
        let step = s.plan().unwrap();
        s.commit(&step, Fence(1)).unwrap();
        s.cancel(handle).unwrap();
        assert_eq!(s.stats().snapshot_backing_pages, 8);
        assert!(s.reserve_snapshot(0).is_none());
        s.complete(
            Fence(1),
            &vec![CompletionRecord::default(); REQUEST_LANES],
            Default::default(),
        )
        .unwrap();
        assert_eq!(s.stats().snapshot_backing_pages, 0);
        assert_eq!(s.free_page_count(), 16);
        assert_eq!(s.free_snapshot_count(), 1);
    }
    #[test]
    fn page_pressure_compacts_aliases_with_free_snapshot_ids_and_preserves_fences() {
        for delayed in [false, true] {
            let mut s = super::super::tests::scheduler(2, 16, 4);
            s.share_snapshot_pages(&[8..12, 10..14, 0..0, 0..0])
                .unwrap();
            let mut history = PrefixHistory::new([9; 32]);
            history.append(&(0..128).collect::<Vec<_>>(), &[]).unwrap();
            let (a, b) = (SnapshotId(0), SnapshotId(1));
            s.reserve_snapshot_id(a);
            s.reserve_snapshot_id(b);
            let mut parent = None;
            for (endpoint, page, snapshot) in [(64, 0, a), (128, 1, b)] {
                s.free_pages.remove(&PageId(page));
                parent = Some(
                    s.cache
                        .publish(
                            &history,
                            &Publication {
                                parent,
                                consumed_tokens: endpoint,
                                tail_pages: vec![PageId(page)],
                                snapshot: StateSnapshot {
                                    id: snapshot,
                                    consumed_tokens: endpoint,
                                },
                                ready_after: Fence(0),
                            },
                        )
                        .unwrap()
                        .id(),
                );
            }
            let child = parent.unwrap();
            s.cache.pin(child).unwrap();
            if delayed {
                s.cache.mark_used(child, Fence(7)).unwrap();
            }
            let held = std::mem::take(&mut s.free_pages);
            assert_eq!(s.free_snapshot_count(), 2);
            let mut admission = super::super::tests::spec(64, 1);
            admission.prompt.fill(99);
            let result = s.admit(admission);
            if delayed {
                assert!(result.is_err());
                assert_eq!(s.free_page_count(), 0);
                assert_eq!(s.pending_page_count(), 2);
                assert_eq!(s.stats().snapshot_backing_pages, 6);
                let update = s.cache.advance_completed(Fence(7)).unwrap();
                s.apply_completion(update);
            } else {
                s.cancel(result.unwrap()).unwrap();
            }
            assert_eq!(s.free_page_count(), 2);
            assert_eq!(s.stats().snapshot_backing_pages, 4);
            assert_eq!(s.page_snapshot_refs[10], 1);
            assert_eq!(s.cache.page_chain(child).unwrap(), [PageId(0), PageId(1)]);
            s.free_pages.extend(held);
        }
    }

    #[test]
    fn decoder_growth_waits_for_pending_alias_reclamation() {
        let mut s = super::super::tests::scheduler(1, 16, 4);
        s.share_snapshot_pages(&[8..12, 10..14, 0..0, 0..0])
            .unwrap();
        let mut spec = super::super::tests::spec(60, 400);
        spec.growable_output = true;
        let handle = s.admit(spec).unwrap();
        let request = s.request_mut(handle).unwrap();
        request.phase = RequestPhase::Decode;
        request.scheduled_consumed = 127;
        let mut history = PrefixHistory::new([9; 32]);
        history.append(&vec![7; 128], &[]).unwrap();
        let (a, b) = (SnapshotId(0), SnapshotId(1));
        s.reserve_snapshot_id(a);
        s.reserve_snapshot_id(b);
        let mut parent = None;
        for (endpoint, page, snapshot) in [(64, 2, a), (128, 3, b)] {
            s.free_pages.remove(&PageId(page));
            parent = Some(
                s.cache
                    .publish(
                        &history,
                        &Publication {
                            parent,
                            consumed_tokens: endpoint,
                            tail_pages: vec![PageId(page)],
                            snapshot: StateSnapshot {
                                id: snapshot,
                                consumed_tokens: endpoint,
                            },
                            ready_after: Fence(0),
                        },
                    )
                    .unwrap()
                    .id(),
            );
        }
        let child = parent.unwrap();
        s.cache.pin(child).unwrap();
        s.cache.mark_used(child, Fence(7)).unwrap();
        let held = std::mem::take(&mut s.free_pages);
        s.refill_reservations().unwrap();
        assert_eq!(s.stats().preemptions, 0);
        assert!(!s.fits_next_rounds(s.request(handle).unwrap()));
        assert_eq!(s.free_page_count(), 0);
        assert_eq!(s.pending_page_count(), 2);
        let update = s.cache.advance_completed(Fence(7)).unwrap();
        s.apply_completion(update);
        s.refill_reservations().unwrap();
        assert!(s.fits_next_rounds(s.request(handle).unwrap()));
        assert_eq!(s.stats().preemptions, 0);
        s.free_pages.extend(held);
    }
}
