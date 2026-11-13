# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare selected attention partitions using captured 262144-context operands."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

from qualify_qsa_fixed_partitions import (
    ROOT, GPU, SPLITS, CapturedAttention, ImmutableCache, comparison,
    gpu_state, quantiles, save, sha, tensor_info, tensor_sha,
)

ARTIFACT_SHA = 'd988ee3ec3c2b3be0fc0afc3e780178045b8bcf9535267eba7df0e81d8fc4e2c'
CAPACITIES = (1, 2, 4, 16, 64, 128, 512)
WORKLOADS = (('prefill504', 'last_prefill512', 504), ('decode1', 'last_decode1', 1))
BASELINES = ('prefill504-capacity512-direct1', 'decode1-capacity1-split64')


def load_inputs(path, receipt_path):
    provenance = json.loads(receipt_path.read_text())
    if sha(path) != ARTIFACT_SHA or provenance['tensor_artifact']['sha256'] != ARTIFACT_SHA:
        raise ValueError('long-context operand SHA does not match its durable capture receipt')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    records, inventory = {}, {}
    for workload, source, count in WORKLOADS:
        record = dict(saved[source])
        record['cache_after'] = record.pop('cache_after_request')
        ids = record['metadata']['request_ids']
        if int((ids >= 0).sum()) != count or not bool((ids[:count] == 0).all()) or not bool((ids[count:] == -1).all()):
            raise ValueError(f'{workload}: invalid contiguous active rows')
        selected = record['tensors']['selected_positions'][:count]
        positions = record['metadata']['query_positions'][:count]
        valid = selected >= 0
        support = valid.sum(1)
        if int(support.max()) != 2051 or bool((valid & (selected > positions[:, None])).any()):
            raise ValueError(f'{workload}: missing full support or noncausal selection')
        if record['constants']['position_capacity'] != 262144:
            raise ValueError('unexpected position capacity')
        table = record['metadata']['main_block_table']
        used = set(table[0, selected[valid].long() // 64].tolist())
        if min(used) < 0:
            raise ValueError('selection references an unmapped page')
        k, v = (record['cache_after'][field] for field in ('main_k_cache', 'main_v_cache'))
        if k['page_ids'] != v['page_ids'] or len(set(k['page_ids'])) != len(k['page_ids']):
            raise ValueError('K/V recorded page order differs or repeats')
        if used != set(k['page_ids']):
            raise ValueError('recorded page set differs from exact selected page coverage')
        for row in range(count):
            row_positions = selected[row][valid[row]]
            if row_positions.unique().numel() != row_positions.numel():
                raise ValueError('selection repeats a logical position')
        records[workload] = record
        inventory[workload] = {
            'observation': source, 'active_rows': count, 'query_positions': positions.tolist(),
            'selected_count_per_row': support.tolist(), 'full_2051_rows': int((support == 2051).sum()),
            'used_physical_pages': sorted(used),
            'tensors': {key: tensor_info(value) for key, value in record['tensors'].items()},
            'metadata': {key: tensor_info(value) for key, value in record['metadata'].items()},
            'cache': {key: {'page_ids': value['page_ids'], 'values': tensor_info(value['values']),
                            'pool_shape': value['pool_shape'], 'pool_strides': value['pool_strides']}
                      for key, value in record['cache_after'].items()},
        }
    return records, inventory, provenance


def independent_reference(record, reference):
    """Compact physical pages only for the exact CPU reference implementation."""
    saved_k, saved_v = (record['cache_after'][field] for field in ('main_k_cache', 'main_v_cache'))
    page_ids = saved_k['page_ids']
    original_table = record['metadata']['main_block_table']
    table = torch.full_like(original_table, -1)
    for compact, physical in enumerate(page_ids):
        table[original_table == physical] = compact
    count = int((record['metadata']['request_ids'] >= 0).sum())
    return reference(
        query=record['tensors']['query'][:count], key_cache=saved_k['values'],
        value_cache=saved_v['values'], block_table=table,
        request_ids=record['metadata']['request_ids'][:count],
        selected_positions=record['tensors']['selected_positions'][:count],
        query_positions=record['metadata']['query_positions'][:count], softmax_scale=1 / 16,
    )


def execute(args, records, capture_receipt, receipt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU or os.environ.get('CUDA_HOME') != '/opt/cuda':
        raise RuntimeError('assigned GPU UUID and CUDA_HOME=/opt/cuda must be explicit')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader'],
                               check=True, text=True, capture_output=True).stdout
    if any(line.startswith(GPU) for line in processes.splitlines()):
        raise RuntimeError(f'assigned GPU occupied before preparation: {processes}')
    import b12x.attention.paged._selected_forward as kernels
    import b12x.attention.qsa.reference as reference_module
    import flare_heliopause
    from ignition.preparation.graphs import PreparedGraphBundle

    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    native = next(Path(flare_heliopause.__file__).parent.glob('*.so'))
    root = Path(kernels.__file__).resolve().parents[2]
    receipt['installed_native'] = {'path': str(native), 'sha256': sha(native)}
    receipt['versions'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'flashinfer': importlib.metadata.version('flashinfer-python'),
        'cutlass_dsl': importlib.metadata.version('nvidia-cutlass-dsl')}
    receipt['b12x_sources'] = {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*.py'))}
    receipt['selected_forward'] = {'path': kernels.__file__, 'sha256': sha(kernels.__file__)}
    receipt['independent_reference'] = {'path': reference_module.__file__, 'sha256': sha(reference_module.__file__),
        'callable': 'sparse_paged_gqa_reference', 'device': 'cpu', 'rtol': 0, 'atol': .02,
        'arithmetic': 'FP32 QK, FP32 softmax, FP32 PV, BF16 output; exact installed callable.'}
    receipt['sampling_sources'] = {str(path): sha(path) for path in (
        ROOT / 'ignition/recipes/qwen38/sampling.py',
        Path(importlib.util.find_spec('flashinfer.sampling').origin),
    )}
    if receipt['b12x_sources'] != capture_receipt['b12x_sources']:
        raise AssertionError('installed b12x source differs from operand capture')
    if receipt['installed_native']['sha256'] != capture_receipt['native_extension']['sha256']:
        raise AssertionError('native extension differs from operand capture')
    shutil.copyfile(kernels.__file__, args.artifacts / 'installed-selected-forward.py')
    shutil.copyfile(reference_module.__file__, args.artifacts / 'installed-reference.py')
    references = {}
    for workload, record in records.items():
        started = time.monotonic()
        references[workload] = independent_reference(record, reference_module.sparse_paged_gqa_reference)
        print(f'CPU REFERENCE {workload}: {time.monotonic() - started:.2f}s', flush=True)
    torch.save(references, args.artifacts / 'independent-reference.pt')
    receipt['reference_tensors'] = {'path': str(args.artifacts / 'independent-reference.pt'),
        'sha256': sha(args.artifacts / 'independent-reference.pt'),
        'tensors': {key: tensor_info(value) for key, value in references.items()}}
    torch.cuda.set_device(0)
    receipt['device_before'] = gpu_state()
    stream = torch.cuda.Stream()
    caches = {name: ImmutableCache(record) for name, record in records.items()}
    owners = []
    for workload in records:
        for splits in SPLITS:
            for capacity in CAPACITIES:
                print(f'CAPTURE {workload} capacity={capacity} splits={splits}', flush=True)
                owners.append(CapturedAttention(workload, records[workload], caches[workload], capacity, splits, stream, kernels))
    by_name = {owner.name: owner for owner in owners}
    graphs = {owner.name: owner.graph for owner in owners}
    buffers = {owner.name + '.' + key: value for owner in owners for key, value in owner.buffers.items()}
    buffers.update({name + '.cache.' + key: value for name, cache in caches.items() for key, value in cache.buffers.items()})
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers=buffers, resources=(*owners, caches, stream, kernels))
    receipt['cases'] = {owner.name: owner.description() for owner in owners}
    receipt['cuda_memory_after_capture'] = {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
        'max_allocated': torch.cuda.max_memory_allocated()}
    receipt['cache_storage'] = {name: {key: tensor_info(value) for key, value in cache.buffers.items()} for name, cache in caches.items()}
    receipt['observations'], receipt['baseline_checks'] = [], {}
    outputs, samples, reference_hashes = {}, {owner.name: [] for owner in owners}, {}
    worker = None

    def replay(owner):
        before = torch.cuda.memory_stats()['allocation.all.allocated']
        ticket = worker.submit(bundle.graph_indices[owner.name])
        result = worker.poll(timeout_ms=60_000)
        if result is None or result.ticket != ticket or result.error is not None:
            raise RuntimeError(f'native replay failed: {owner.name}: {result}')
        after = torch.cuda.memory_stats()['allocation.all.allocated']
        elapsed = owner.start_event.elapsed_time(owner.end_event) * 1000
        observed, padding_zero = owner.readback()
        details = {'allocation_count_before': before, 'allocation_count_after': after, 'elapsed_us': elapsed,
            'output_sha256': tensor_sha(observed), 'inactive_output_zero': padding_zero}
        if before != after or not padding_zero or not bool(torch.isfinite(observed).all()) or not math.isfinite(elapsed) or elapsed <= 0:
            raise AssertionError(f'replay contract failed: {owner.name}: {details}')
        return observed, details

    try:
        worker = bundle.start_worker()
        for name in BASELINES:
            owner = by_name[name]
            observed, details = replay(owner)
            expected = records[owner.workload]['tensors']['attended'][:owner.count]
            result = comparison(observed, expected)
            receipt['baseline_checks'][name] = {**details, **result}
            save(args.output, receipt)
            if not result['bitwise_equal']:
                torch.save({'actual': observed, 'expected': expected}, args.artifacts / (name + '-baseline-failure.pt'))
                raise AssertionError(f'recorded default baseline reproduction failed: {name}')
        print('RECORDED BASELINES: prefill direct512 and decode S64 capacity1 bitwise exact', flush=True)
        for iteration in range(23):
            order = owners if iteration % 2 == 0 else list(reversed(owners))
            for owner in order:
                observed, details = replay(owner)
                stable = reference_hashes.setdefault(owner.name, details['output_sha256']) == details['output_sha256']
                receipt['observations'].append({'case': owner.name, 'iteration': iteration,
                    'warmup': iteration < 3, 'repeat_bitwise_equal': stable, **details})
                outputs.setdefault(owner.name, observed)
                if not stable:
                    torch.save({'actual': observed, 'first': outputs[owner.name]}, args.artifacts / (owner.name + '-repeat-failure.pt'))
                    raise AssertionError(f'captured replay output changed: {owner.name}')
                if iteration >= 3:
                    samples[owner.name].append(details['elapsed_us'])
            save(args.output, receipt)
            print(f'REPLAY {iteration + 1}/23: {len(owners)} captured variants', flush=True)
        receipt['timing_us'] = {name: {'raw': values, **quantiles(values)} for name, values in samples.items()}
        receipt['comparisons'] = {}
        for owner in owners:
            variant = 'direct1' if owner.splits == 1 else f'split{owner.splits}'
            same = outputs[f'{owner.workload}-capacity1-{variant}']
            direct = outputs[f'{owner.workload}-capacity512-direct1']
            result = comparison(outputs[owner.name], references[owner.workload])
            result['rtol0_atol002_passed'] = bool(torch.allclose(outputs[owner.name], references[owner.workload], rtol=0, atol=.02))
            receipt['comparisons'][owner.name] = {
                'same_partition_across_capacity': comparison(outputs[owner.name], same),
                'versus_direct512': comparison(outputs[owner.name], direct), 'versus_fp32_sparse_reference': result}
        receipt['immutable_input_checks'] = {
            owner.name: all(tensor_sha(owner.buffers[key]) == value for key, value in owner.immutable.items()) for owner in owners}
        receipt['immutable_input_checks'].update({name + '.cache': all(tensor_sha(cache.buffers[key]) == value for key, value in cache.initial_hashes.items())
            for name, cache in caches.items()})
        receipt['fixed_partition_capacity_invariant'] = all(value['same_partition_across_capacity']['bitwise_equal'] for value in receipt['comparisons'].values())
        receipt['sparse_reference_gate_passed'] = all(value['versus_fp32_sparse_reference']['rtol0_atol002_passed'] for value in receipt['comparisons'].values())
        receipt['passed'] = receipt['fixed_partition_capacity_invariant'] and receipt['sparse_reference_gate_passed'] and all(receipt['immutable_input_checks'].values())
        torch.save(outputs, args.artifacts / 'attention-outputs.pt')
        receipt['output_tensors'] = {'path': str(args.artifacts / 'attention-outputs.pt'), 'sha256': sha(args.artifacts / 'attention-outputs.pt')}
        receipt['device_after'] = gpu_state()
    finally:
        if worker is not None:
            worker.close()
            receipt['native_worker_closed'] = True
        for graph in graphs.values():
            graph.reset()
        receipt['graph_handles_released'] = True
        save(args.output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=Path('/tmp/sparkgap-qsa-long-context/attention-operands.pt'))
    parser.add_argument('--input-receipt', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, default=Path('/tmp/sparkgap-qsa-long-partitions'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/sparkgap-qsa-long-partitions/receipt.json'))
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    records, inventory, capture_receipt = load_inputs(args.artifact, args.input_receipt)
    if args.validate_only:
        print(json.dumps({'inputs': {key: {'rows': value['active_rows'], 'full_2051_rows': value['full_2051_rows'],
            'pages': len(value['used_physical_pages'])} for key, value in inventory.items()}, 'cases': 56,
            'cuda_initialized': torch.cuda.is_initialized()}, indent=2))
        return
    if args.output.exists() or (args.artifacts / 'attention-outputs.pt').exists():
        parser.error('output already exists; preserve it and choose another destination')
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, args.artifacts / Path(__file__).name)
    base = Path(__file__).with_name('qualify_qsa_fixed_partitions.py')
    shutil.copyfile(base, args.artifacts / base.name)
    receipt = {'schema': 'flare.qsa-long-partitions.v1', 'passed': False,
        'script': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'shared_capture_helper': {'path': str(base), 'sha256': sha(base)}, 'command': sys.argv,
        'input_artifact': {'path': str(args.artifact), 'sha256': ARTIFACT_SHA},
        'input_receipt': {'path': str(args.input_receipt), 'sha256': sha(args.input_receipt)},
        'gpu_uuid': GPU, 'warmups': 3, 'repetitions': 20, 'inputs': inventory,
        'scope': 'Attention only on actual layer3 queries, selected positions and referenced main KV pages; no selector, projections, cache writes, model, or HTTP.',
        'timing_scope': 'Captured CUDA events bracket all attention groups for the complete saved workload. Native driver replay. Preparation, CPU readback and reference excluded.',
        'cache_reconstruction': 'Kernel inputs preserve original physical page IDs and block table. CPU reference alone remaps recorded pages bijectively into compact storage.',
        'direct1': 'Direct BF16 output kernel, without split partial/merge; not the fixed-S1 public candidate.',
        'padding': 'Inactive query zero; request IDs/query positions/selected positions -1; output must be zero.',
        'baseline_policy': 'Recorded direct512 prefill and S64 capacity1 decode must reproduce bitwise before alternatives.'}
    started = time.monotonic()
    try:
        save(args.output, receipt)
        execute(args, records, capture_receipt, receipt)
    except BaseException:
        receipt['failure'] = traceback.format_exc()
        raise
    finally:
        receipt['wall_seconds'] = time.monotonic() - started
        save(args.output, receipt)
    print(json.dumps({'passed': receipt['passed'], 'capacity_invariant': receipt['fixed_partition_capacity_invariant'],
        'reference_gate': receipt['sparse_reference_gate_passed'], 'cases': len(receipt['cases']),
        'native_replays': len(receipt['observations']) + len(receipt['baseline_checks']), 'receipt': str(args.output)}, indent=2))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
