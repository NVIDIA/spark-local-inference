# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Logical state checks for exact captured checkpoint transfers."""

from __future__ import annotations


def _equal(left, right):
    import torch

    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                            right.contiguous().reshape(-1).view(torch.uint8)))


def checkpoint_transfer_checks(pool, caches, *, slot, snapshot, pages, consumed, accepted=1):
    """Compare committed live state with its normalized immutable snapshot.

    The same comparison applies after Snapshot and Restore. Callers separately
    check unchanged source storage and full shared pages. Uncommitted KV rows,
    incomplete compressed groups, and invalid raw-ring entries are excluded.
    """
    import torch

    if (type(slot) is not int or not 0 <= slot < 4
            or type(snapshot) is not int or snapshot < 0
            or type(consumed) is not int or consumed <= 0
            or type(accepted) is not int or not 1 <= accepted <= 8):
        raise ValueError("checkpoint transfer requires valid slots, endpoint, and accepted count")
    # Rounds commit accepted recurrent state into the live row; histories keep accepted columns.
    live, frozen = slot + 1, 5 + snapshot
    recurrent = live
    checks = {}
    for layer, state in pool.gdn.items():
        name = f"gdn.{layer}."
        checks[name + "recurrent"] = _equal(state.recurrent[recurrent].cpu(), state.recurrent[frozen].cpu())
        source = state.convolution[live, :, accepted - 1:accepted + 2].cpu()
        saved = state.convolution[frozen].cpu()
        checks[name + "convolution"] = _equal(source, saved[:, :3])
        checks[name + "convolution_tail_zero"] = not bool(saved[:, 3:].ne(0).any())
    source = pool.ple_convolution[live, :, accepted - 1:accepted + 8].cpu()
    saved = pool.ple_convolution[frozen].cpu()
    checks["ple_convolution"] = _equal(source, saved[:, :9])
    checks["ple_convolution_tail_zero"] = not bool(saved[:, 9:].ne(0).any())
    regions = {region.name for region in pool.layout.regions}
    for name in ("last_hidden", "consumed_tokens", "ple_history", "rope_delta",
                 "target_residual", "target_residual_position"):
        if name in regions:
            checks[name] = _equal(pool.region(name)[live].cpu(), pool.region(name)[frozen].cpu())
    checks["exact_consumed"] = int(pool.consumed_tokens[frozen].cpu().item()) == consumed
    for layer, cache in caches.items():
        prefix = f"qsa.{layer}."
        positions = cache.raw_logical_positions[live].cpu()
        saved_positions = pool.region(prefix + "raw_positions")[frozen].cpu()
        valid = (positions >= 0) & (positions < consumed)
        expected = torch.where(valid, positions, -1)
        checks[prefix + "raw_positions"] = _equal(expected, saved_positions)
        for region, attribute in (("raw_keys", "raw_k_ring"), ("raw_rope", "raw_rope_positions")):
            source = getattr(cache, attribute)[live].cpu()
            saved = pool.region(prefix + region)[frozen].cpu()
            checks[prefix + region] = _equal(source[valid], saved[valid])
        checks[prefix + "raw_anchor"] = int(pool.region(prefix + "raw_anchor")[frozen].cpu().item()) == consumed - 1
        tail = consumed % 64
        if not tail:
            continue
        page = pages[consumed // 64]
        if not 0 <= page < cache.main_k_cache.shape[0]:
            raise ValueError("checkpoint tail page lies outside the prepared cache")
        for region, attribute, count in (
                ("boundary_key", "main_k_cache", tail),
                ("boundary_value", "main_v_cache", tail),
                ("boundary_selector", "compressed_k_cache", tail // 4)):
            source = getattr(cache, attribute)[page, :count].cpu()
            saved = pool.region(prefix + region)[frozen, :count].cpu()
            checks[prefix + region] = _equal(source, saved)
    return checks
