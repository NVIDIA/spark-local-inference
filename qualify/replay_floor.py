# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure autoregressive replay of prepared decode graphs without scheduling."""

from __future__ import annotations

import hashlib
import time
from dataclasses import fields

from .provenance import write_json


def decode_groups(corpus):
    cases = {case['id']: case for case in corpus['cases']}
    groups = []
    for group in corpus['groups']:
        if not group['kind'].startswith('decode-b') or not group['measured']:
            continue
        rows = [cases[name] for name in group['cases']]
        if len(rows) != group['batch'] or group['batch'] not in (1, 2, 4):
            raise ValueError('decode group has an invalid batch extent')
        for row in rows:
            request = row['request']
            if (row['output_tokens'] != 256 or row['expected_cache'] != 'cold'
                    or request.get('temperature') != 1
                    or request.get('top_k') != 20 or request.get('top_p') != .95
                    or not row['prompt_tokens']):
                raise ValueError('replay floor requires cold 256-token cases with temperature=1, top_k=20, top_p=0.95')
        groups.append((group, rows))
    if not groups:
        raise ValueError('performance corpus has no measured decode groups')
    return groups


def _byte_view(tensor):
    import torch

    if tensor.device.type != 'cpu':
        raise ValueError('replay state comparison requires fenced CPU readback')
    value = tensor.contiguous().reshape(-1)
    if value.stride(0) != 1:
        value = value.clone(memory_format=torch.contiguous_format)
    return value.view(torch.uint8)


def _hash(tensor):
    value = _byte_view(tensor)
    return hashlib.sha256(memoryview(value.numpy())).hexdigest()


class ReplayCase:
    def __init__(self, model):
        import torch

        from ignition.components.sampling import AdmissionBuffers
        from ignition.recipes.qwen38.decoder import DecoderScratch, PreparedDecoder
        from ignition.recipes.qwen38.sampling import DeviceRequestState
        from ignition.recipes.qwen38.state import HybridStatePool

        self.model = model
        self.state = next(value for value in model.resources if isinstance(value, HybridStatePool))
        self.requests = next(value for value in model.resources if isinstance(value, DeviceRequestState))
        self.scratch = next(value for value in model.resources if isinstance(value, DecoderScratch))
        decoders = next(value for value in model.resources if isinstance(value, list)
                        and value and isinstance(value[0], PreparedDecoder))
        self.cache = decoders[0].qsa_states
        self.host = {name: tensor.cpu().clone() for name, tensor in model.buffers.items()}
        self.pristine_inputs = {name: tensor.clone() for name, tensor in self.host.items()}
        self.pristine_requests = {field.name: getattr(self.requests, field.name).cpu().clone()
                                  for field in fields(self.requests)}
        self.admission = AdmissionBuffers(**{field.name: self.host[field.name]
                                            for field in fields(AdmissionBuffers)})
        self.stream = torch.cuda.Stream(device=model.device)
        self.batch = 0
        self.page_count = 0
        self.touched_pages = 0

    def upload(self, names):
        for name in names:
            self.model.buffers[name].copy_(self.host[name])

    def execute(self, name):
        import torch

        self.upload(self.model.signatures[name].inputs)
        self.model.buffers['frame_index'].zero_()
        self.model.buffers['submission_sequence'].fill_(1)
        self.stream.wait_stream(torch.cuda.current_stream(self.model.device))
        with torch.cuda.stream(self.stream):
            self.model.graphs[name].replay()
        self.stream.synchronize()

    def records(self):
        records = self.model.buffers['completion_records'][0].cpu().tolist()
        for slot, row in enumerate(records[:self.batch]):
            if row[0] != 1 or row[1] != 1 or row[6:] != [0, 0]:
                raise RuntimeError(f'invalid replay completion for slot {slot}: {row}')
        return records

    def rows(self, slots):
        h = self.host
        h['row_slot_ids'].fill_(-1)
        h['row_generations'].zero_()
        h['live_rows'].fill_(len(slots))
        h['query_lengths'].zero_()
        h['sample_enabled'].fill_(1)
        h['vision_token_indices'].fill_(-1)
        for row, slot in enumerate(slots):
            h['row_slot_ids'][row] = slot
            h['row_generations'][row] = 1

    def prepare(self, cases):
        import torch

        from ignition.recipes.qwen38.sampling import RequestAdmission

        self.batch = len(cases)
        self.state.reset_for_preparation()
        self.requests.generation.zero_()
        self.requests.status.zero_()
        self.requests.error_code.zero_()
        self.model.buffers['completion_records'].zero_()
        h = self.host
        h['page_tables'].fill_(-1)
        self.page_count = 0
        for slot, case in enumerate(cases):
            count = (len(case['prompt_tokens']) + case['output_tokens'] + 63) // 64
            h['page_tables'][slot, :count] = torch.arange(self.page_count, self.page_count + count)
            self.page_count += count
        if self.page_count > self.model.memory_report['pages']:
            raise ValueError('replay cases exceed the prepared KV page pool')
        self.touched_pages = max(self.touched_pages, self.page_count)
        for cache in self.cache.values():
            for field in fields(cache):
                tensor = getattr(cache, field.name)
                if field.name.endswith('_cache'):
                    tensor[:self.page_count].zero_()
                elif tensor.dtype.is_floating_point:
                    tensor.zero_()
                else:
                    tensor.fill_(-1)
        h['admission_count'].fill_(self.batch)
        h['admission_slots'].fill_(-1)
        h['state_num_ops'].fill_(self.batch)
        for name in ('source_slots', 'destination_slots', 'operation', 'accepted_tokens'):
            h['state_' + name].zero_()
        h['state_qsa_slots'].fill_(-1)
        h['state_expected_consumed'].fill_(-1)
        for slot, case in enumerate(cases):
            request = case['request']
            admission = RequestAdmission(generation=1, seed=request.get('seed', 0),
                max_new_tokens=case['output_tokens'], temperature=request.get('temperature', 1),
                top_k=request.get('top_k', 20), top_p=request.get('top_p', .95),
                min_p=request.get('min_p', 0),
                repetition_penalty=request.get('repetition_penalty', 1),
                presence_penalty=request.get('presence_penalty', 0),
                frequency_penalty=request.get('frequency_penalty', 0))
            self.admission.stage(slot, slot, admission, case['prompt_tokens'])
            h['state_source_slots'][slot] = slot + 1
            h['state_destination_slots'][slot] = slot + 1
            h['state_operation'][slot] = 1
            h['state_qsa_slots'][slot] = slot + 1
            h['state_accepted_tokens'][slot] = 1
            h['state_expected_consumed'][slot] = 0
        self.execute(f'init{self.batch}')
        initial = [[] for _ in cases]
        chunk = self.model.memory_report['prefill_chunk_size']
        capacities = self.model.memory_report['prefill_capacities']
        for slot, case in enumerate(cases):
            prompt = case['prompt_tokens']
            for begin in range(0, len(prompt), chunk):
                end = min(begin + chunk, len(prompt))
                capacity = min(value for value in capacities if value >= end - begin)
                self.rows([slot])
                h['query_lengths'][0] = end - begin
                h['sample_enabled'].fill_(int(end == len(prompt)))
                h['token_ids'].fill_(248044)
                h['token_ids'][:end - begin] = torch.tensor(prompt[begin:end])
                h['prefill_positions'].zero_()
                h['prefill_positions'][:end - begin] = torch.arange(begin, end)[:, None]
                self.execute(f'prefill{capacity}')
                record = self.model.buffers['completion_records'][0, slot].cpu().tolist()
                expected_generated = int(end == len(prompt))
                if (record[:4] != [1, 1, end, expected_generated]
                        or record[5:] != [1, 0, 0]):
                    raise RuntimeError(f'prefill failed or terminated early: {record}')
                if expected_generated:
                    initial[slot].append(record[4])
        self.rows(range(self.batch))
        self.upload(self.model.signatures[f'decode{self.batch}'].inputs)
        torch.cuda.synchronize(self.model.device)
        return initial

    def snapshot(self, cases):
        result = {}
        for slot in range(self.batch):
            result[f'hybrid.live.{slot}'] = self.state.backing[slot + 1].cpu()
        for field in fields(self.requests):
            result['request.' + field.name] = getattr(self.requests, field.name)[:self.batch].cpu()
        for index, cache in self.cache.items():
            for field in fields(cache):
                tensor = getattr(cache, field.name)
                rows = tensor[:self.page_count] if field.name.endswith('_cache') else tensor[:self.batch + 1]
                result[f'qsa.{index}.{field.name}'] = rows.cpu()
        result['logits'] = self.scratch.logits[:self.batch].cpu()
        records = self.records()
        for slot, case in enumerate(cases):
            expected = [1, 1, len(case['prompt_tokens']) + 255, 256]
            if records[slot][:4] != expected or records[slot][5:] != [2, 0, 0]:
                raise RuntimeError(f'decode did not execute all 256 tokens: {records[slot]}')
        return result, records

    def restore_for_server(self):
        import torch

        self.stream.synchronize()
        self.state.reset_for_preparation()
        for name, tensor in self.pristine_requests.items():
            getattr(self.requests, name).copy_(tensor)
        for name, tensor in self.pristine_inputs.items():
            self.model.buffers[name].copy_(tensor)
        for cache in self.cache.values():
            for field in fields(cache):
                tensor = getattr(cache, field.name)
                if field.name.endswith('_cache'):
                    tensor[:self.touched_pages].zero_()
                elif tensor.dtype.is_floating_point:
                    tensor.zero_()
                else:
                    tensor.fill_(-1)
        torch.cuda.synchronize(self.model.device)
        if self.model.consumed or self.requests.generation.count_nonzero().item():
            raise RuntimeError('replay measurement did not restore native admission ownership')


def measure(model, corpus, output):
    """Compare isolated and queued replay, then retain the model for native serving."""
    import torch

    from ignition.preparation.graphs import PreparedGraphBundle

    output.mkdir(parents=True, exist_ok=False)
    runner = ReplayCase(model)
    report = {'schema': 'flare-autoregressive-replay-floor-v1',
        'prefill_chunk_size': model.memory_report['prefill_chunk_size'],
        'timing': 'Native enqueue through synchronized completion of 255 decode graph launches; prefill and state readback excluded.',
        'scope': 'Preallocated pages and fixed packed lanes; scheduling, uploads, completion downloads, cache publication and frontend delivery excluded.',
        'checks': [], 'groups': []}
    allocation_names = ('allocation.all.allocated', 'allocation.all.freed',
                        'segment.all.allocated', 'segment.all.freed')
    for group, cases in decode_groups(corpus):
        name = f'decode{len(cases)}'
        tokens = runner.prepare(cases)
        for index in range(255):
            with torch.cuda.stream(runner.stream):
                model.graphs[name].replay()
            runner.stream.synchronize()
            for slot, record in enumerate(runner.records()[:len(cases)]):
                if record[3] != index + 2 or record[4] < 0:
                    raise RuntimeError('isolated replay stopped before its output budget')
                tokens[slot].append(record[4])
        expected, expected_records = runner.snapshot(cases)
        first = runner.prepare(cases)
        if [row[0] for row in tokens] != [row[0] for row in first]:
            raise RuntimeError('prefill changed before the queued replay measurement')
        bundle = PreparedGraphBundle(device=model.device, graphs={name: model.graphs[name]},
                                     buffers=model.buffers, resources=(model,))
        worker = bundle.start_worker()
        try:
            before = torch.cuda.memory_stats(model.device)
            started = time.perf_counter_ns()
            ticket = worker.submit(0, repetitions=255)
            completion = None
            while completion is None:
                completion = worker.poll(timeout_ms=1000)
            elapsed = time.perf_counter_ns() - started
            if completion.ticket != ticket or completion.repetitions != 255 or completion.error:
                raise RuntimeError(f'bare replay failed: {completion.error}')
            after = torch.cuda.memory_stats(model.device)
        finally:
            try:
                worker.close()
            except BaseException:
                model._startup_undrained = True
                raise
        actual, actual_records = runner.snapshot(cases)
        mismatches = [key for key in expected if not torch.equal(
            _byte_view(expected[key]), _byte_view(actual[key]))]
        stable = all(before[key] == after[key] for key in allocation_names)
        entry = {'id': group['id'], 'cases': [case['id'] for case in cases],
            'batch': len(cases), 'repetition': group['repetition'], 'decode_steps': 255,
            'elapsed_ns': elapsed, 'ns_per_decode_step': elapsed / 255,
            'tokens_per_second': len(cases) * 255e9 / elapsed,
            'isolated_generated_tokens': tokens,
            'final_records': actual_records, 'state_mismatches': mismatches,
            'equal_final_records': actual_records == expected_records,
            'allocator_before': {key: before[key] for key in allocation_names},
            'allocator_after': {key: after[key] for key in allocation_names},
            'state_sha256': {key: _hash(value) for key, value in actual.items()}}
        report['groups'].append(entry)
        passed = not mismatches and stable and actual_records == expected_records
        report['checks'].append({'group': group['id'], 'passed': passed,
            'compared_state_tensors': len(expected), 'allocator_stable': stable})
        write_json(output / 'replay-floor.json', report)
        if not passed:
            torch.save({'expected': {key: expected[key] for key in mismatches},
                        'actual': {key: actual[key] for key in mismatches}}, output / 'mismatch.pt')
            raise RuntimeError(f'queued autoregressive replay qualification failed: {group["id"]}')
        print(f'replay floor {group["id"]}: {elapsed / 255e6:.3f} ms/step', flush=True)
        del expected, actual, bundle, worker
    runner.restore_for_server()
    report['restored_for_native_admission'] = True
    write_json(output / 'replay-floor.json', report)
    return report
