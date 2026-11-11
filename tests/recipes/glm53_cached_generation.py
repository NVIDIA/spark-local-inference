# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-model cached-prefix branch execution through the native rank coordinator."""

import time

import torch

from ignition.components.sampling import AdmissionBuffers, RequestAdmission
from ignition.components.state_transfer import StateCommands
from ignition.recipes.glm53.batch import BatchInputs


def run_cached_generation(worker, bundle, pool, requests, scratch, resume, tokens, eos, capacity, progress, *, resume_readback):
    host = BatchInputs.allocate(capacity, 64, "cpu")
    host.request_state_rows[:, 0].copy_(torch.arange(1, 5))
    admission = AdmissionBuffers.allocate("cpu", vocab_size=154880)
    commands = StateCommands.allocate("cpu")
    staging = dict(host.buffers(), **commands.buffers(), **{
        name: getattr(admission, name) for name in admission.__dataclass_fields__})
    device = pool.backing.device

    def submit(mode):
        uploads = [(name, 0, staging[name].numpy().tobytes()) for name in bundle.signatures[mode].inputs]
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_stats(device)
        ticket = worker.submit(bundle.graph_indices[mode], uploads)
        completion, deadline = None, time.monotonic() + 120
        while completion is None and time.monotonic() < deadline:
            completion = worker.poll()
            if completion is None:
                time.sleep(.001)
        assert ticket is not None and completion is not None and completion.fence == ticket
        after = torch.cuda.memory_stats(device)
        for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
            assert before[key] == after[key], key
        return completion

    def transfer(rows):
        commands.request_slots.fill_(-1)
        commands.source_slots.fill_(-1)
        commands.destination_slots.fill_(-1)
        commands.expected_consumed.fill_(-1)
        commands.operation.zero_()
        for row, (slot, source, destination, operation, endpoint) in enumerate(rows):
            commands.request_slots[row] = slot
            commands.source_slots[row], commands.destination_slots[row] = source, destination
            commands.operation[row], commands.expected_consumed[row] = operation, endpoint
        commands.num_ops.fill_(len(rows))

    admission.admission_count.fill_(1)
    admission.stage(0, 0, RequestAdmission(1, 13, max_new_tokens=32, eos_token_ids=tuple(eos)), tokens)
    transfer([(0, -1, 1, 1, 0)])
    result = submit("init")
    assert result.records[0][:7] == [1, result.fence, 0, 0, -1, 1, 0], result.records[0][:7]

    host.row_slot_ids[0], host.row_generations[0] = 0, 1
    host.live_rows.fill_(1)
    host.query_lengths[0] = len(tokens)
    host.token_ids[:len(tokens)].copy_(torch.tensor(tokens))
    host.page_tables[0].copy_(torch.arange(63, -1, -1, dtype=torch.int32))
    result = submit("prefill")
    assert result.records[0][:7] == [1, result.fence, len(tokens), 0, -1, 1, 0]
    assert not requests.rng_counter.count_nonzero()
    prefix_hidden = pool.last_hidden[1].clone()
    prefix_logits = scratch["prefill"].logits.clone()
    prefix_pages = (len(tokens) + 255) // 256
    physical = host.page_tables[0, :prefix_pages].to(device=device, dtype=torch.int64)
    original_cache = {name: cache[physical].clone() for name, cache in pool.partial_pages()}
    transfer([(0, 1, 5, 0, len(tokens))])
    result = submit("snapshot")
    assert result.records[0][:7] == [1, result.fence, len(tokens), 0, -1, 1, 0]
    immutable = pool.backing[5].clone()

    admission.admission_count.fill_(2)
    for row, slot in enumerate((1, 2)):
        admission.stage(row, slot, RequestAdmission(1, 13, max_new_tokens=32, eos_token_ids=tuple(eos)), tokens)
        host.page_tables[slot].copy_(host.page_tables[0])
        host.page_tables[slot, len(tokens) // 256] = row
    transfer([(1, 5, 2, 0, len(tokens)), (2, 5, 3, 0, len(tokens))])
    result = submit("restore")
    for slot in (1, 2):
        assert result.records[slot][:7] == [1, result.fence, len(tokens), 0, -1, 1, 0]
        assert torch.equal(pool.backing[slot + 1], immutable)
        for name, cache in pool.partial_pages():
            assert torch.equal(cache[slot - 1], pool.region(name)[5].reshape(-1))
    progress(f"restored exact {len(tokens)}-token prefix into two private tails")

    generated = {}
    for slot in (1, 2):
        host.row_slot_ids[0] = slot
        result = submit("resume")
        record = result.records[slot]
        assert record[:4] == [1, result.fence, len(tokens), 1]
        assert record[6] == 0 and record[5] in (1, 2)
        torch.testing.assert_close(resume_readback[0][0], prefix_hidden, rtol=0, atol=0)
        torch.testing.assert_close(resume_readback[1], prefix_logits, rtol=0, atol=0)
        assert torch.equal(pool.backing[slot + 1], immutable)
        generated[slot] = [record[4]]
        host.sample_enabled[0] = 1
        while record[5] == 1 and len(generated[slot]) < 32:
            host.token_ids.zero_()
            host.token_ids[0] = generated[slot][-1]
            host.query_lengths[0] = 1
            result = submit("decode")
            record = result.records[slot]
            assert record[:4] == [1, result.fence, len(tokens) + len(generated[slot]), len(generated[slot]) + 1]
            assert record[6] == 0 and record[5] in (1, 2)
            assert torch.isfinite(scratch["decode"].hidden).all()
            torch.testing.assert_close(pool.last_hidden[slot + 1], scratch["decode"].hidden[0], rtol=0, atol=0)
            generated[slot].append(record[4])
        assert record[5] == 2
        assert all(0 <= token < 154880 for token in generated[slot])
        assert requests.rng_counter[slot] == len(generated[slot])
        assert torch.equal(pool.backing[5], immutable)
        for name, cache in pool.partial_pages():
            assert torch.equal(cache[physical], original_cache[name])
        progress(f"cached branch {slot} sampled {len(generated[slot])} tokens")
    assert generated[1][0] == generated[2][0]
    return generated
