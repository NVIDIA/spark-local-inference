# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel GLM placement and selection kernels against direct references."""

import os

import pytest
import torch

from ignition.recipes.glm53.dcp import ContextShards, merge_candidates, merge_expand_local, pack_candidates

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def owner_local(t, world):
    return (t // 4) % world, (t // (4 * world)) * 4 + t % 4


def test_slots_lengths_merge_and_expansion_match_references():
    torch.manual_seed(0)
    device = torch.device("cuda")
    for world in (1, 2, 4):
        for rank in range(world):
            shards = ContextShards.on(rank, world, device)
            # Slot mapping and local lengths.
            blocks = 64
            table = torch.randperm(4096, device=device)[:blocks].int()
            positions = torch.randint(-1, blocks * 256 * world, (5000,), device=device)
            slots = shards.token_slots(positions, table.expand(positions.numel(), -1))
            for p, s in zip(positions.tolist()[:2000], slots.tolist()[:2000]):
                o, l = owner_local(p, world)
                want = -1 if p < 0 or o != rank else int(table[l // 256]) * 256 + l % 256
                assert s == want, (world, rank, p, s, want)
            lengths = torch.arange(0, 3000, device=device)
            local = shards.local_tokens(lengths)
            brute = torch.tensor([sum(1 for t in range(n) if owner_local(t, world)[0] == rank) for n in range(3000)],
                                 device=device)
            assert torch.equal(local, brute), "local_tokens"
            pools = torch.arange(0, 700, device=device)
            brute = torch.tensor([sum(1 for q in range(n) if q % world == rank) for n in range(700)], device=device)
            assert torch.equal(shards.local_pools(pools), brute), "local_pools"

        # Candidate pack and merge: the top-512 set by score, lower pool id on ties.
        rows = 7
        local_ids = [torch.full((rows, 512), -1, dtype=torch.int32, device=device) for _ in range(world)]
        scores = [torch.full((rows, 512), -float("inf"), device=device) for _ in range(world)]
        packed = []
        for rank in range(world):
            count = torch.randint(0, 513, (rows,)).tolist()
            # A short context: every rank holds a few candidates, fewer than 512 between them.
            count[0] = 3 + rank
            for r, c in enumerate(count):
                local_ids[rank][r, :c] = torch.randperm(5000, device=device)[:c].int()
                scores[rank][r, :c] = torch.randint(0, 50, (c,), device=device).float() / 7
            out = torch.empty((rows, 512, 2), device=device)
            pack_candidates(local_ids[rank], scores[rank], out, ContextShards.on(rank, world, device))
            packed.append(out)
        gathered = torch.cat(packed, dim=1).contiguous()
        merged = torch.empty((rows, 512), dtype=torch.int32, device=device)
        merge_candidates(gathered, merged)
        for r in range(rows):
            candidates = []
            for rank in range(world):
                for l, s in zip(local_ids[rank][r].tolist(), scores[rank][r].tolist()):
                    if l >= 0:
                        candidates.append((-s, l * world + rank))
            candidates.sort()
            want = [g for _, g in candidates[:512]] + [-1] * max(0, 512 - len(candidates))
            # Expansion reads the first min(pools, 512) entries, so the selected pools lead in any order.
            got, present = merged[r].tolist(), min(len(candidates), 512)
            assert sorted(got[:present]) == sorted(want[:present]) and got[present:] == want[present:], \
                ("merge", world, r)

        # Expansion: global pools plus the incomplete tail, owned records compacted to the front.
        seqs, blocks = 3, 40
        tables = torch.stack([torch.randperm(1000, device=device)[:blocks] for _ in range(seqs)]).int()
        rows = 9
        requests = torch.randint(0, seqs, (rows,), dtype=torch.int32, device=device)
        positions = torch.randint(-1, blocks * 256 * world - 1, (rows,), device=device)
        # Short contexts, whose pools do not fill the selection.
        positions[:2] = torch.tensor([32, 1001], device=device)
        pools = torch.full((rows, 512), -1, dtype=torch.int32, device=device)
        for r in range(rows):
            complete = (int(positions[r]) + 1) // 4
            take = min(complete, 512)
            if take:
                pools[r, :take] = torch.randperm(complete, device=device)[:take].int()
        # Candidates whose merge yields exactly these pools, in descending score order: all on the first rank, or
        # each on the rank that owns it.
        for split in (False, True):
            candidates = torch.zeros((rows, world * 512, 2), dtype=torch.float32, device=device)
            candidates[:, :, 0] = -float("inf")
            candidates[:, :, 1] = -1
            for r in range(rows):
                listed = [0] * world
                for i, pool in enumerate(pools[r].tolist()):
                    if pool >= 0:
                        owner = pool % world if split else 0
                        candidates[r, owner * 512 + listed[owner]] = torch.tensor([1000.0 - i, pool])
                        listed[owner] += 1
            for rank in range(world):
                shards = ContextShards.on(rank, world, device)
                output = torch.empty((rows, 2051), dtype=torch.int32, device=device)
                counts = torch.empty(rows, dtype=torch.int32, device=device)
                merged = torch.empty((rows, 512), dtype=torch.int32, device=device)
                merge_expand_local(candidates, merged, positions, requests, tables, output, counts, shards)
                for r in range(rows):
                    # The selected pools lead in any order; records follow the merged order.
                    take = int((pools[r] >= 0).sum())
                    selected = merged[r].tolist()
                    assert sorted(selected[:take]) == sorted(pools[r, :take].tolist()), ("merge", world, rank, split, r)
                    assert all(v == -1 for v in selected[take:]), ("merge tail", world, rank, split, r)
                    length = int(positions[r]) + 1
                    complete = length // 4
                    tokens = [pool * 4 + k for pool in selected[:take] for k in range(4)]
                    tokens += list(range(complete * 4, length))
                    want = []
                    for t in tokens:
                        o, l = owner_local(t, world)
                        if o == rank:
                            want.append(int(tables[int(requests[r]), l // 256]) * 256 + l % 256)
                    assert counts[r].item() == len(want), ("count", world, rank, split, r, counts[r].item(), len(want))
                    got = output[r].tolist()
                    assert got[:len(want)] == want and all(v == -1 for v in got[len(want):]), \
                        ("expand", world, rank, split, r)
