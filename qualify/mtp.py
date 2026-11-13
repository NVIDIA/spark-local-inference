# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced exact-checkpoint MTP component comparison.

Capture Flare MTP outputs from stored target residuals and shifted token IDs.
Compare saved outputs on matching contexts and explicit request-tail rows.
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from qualify.provenance import plain

SNAPSHOT = Path('/data/checkpoints-masters/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD')
MODEL_ID = 'local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD'
REVISION = 'dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2'


class _ObservedCall:
    def __init__(self, operation, before=None, after=None):
        self.operation, self.before, self.after = operation, before, after

    def __getattr__(self, name):
        return getattr(self.operation, name)

    def __call__(self, *args, **kwargs):
        if self.before is not None:
            self.before(args, kwargs)
        result = self.operation(*args, **kwargs)
        if self.after is not None:
            self.after(args, kwargs, result)
        return result


class StageTrace:
    """Copies existing operands into separate storage; never substitutes math."""

    def __init__(self):
        self.buffers, self.handles, self.originals = {}, [], []

    def record(self, name, value):
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            raise TypeError(f'{name}: stage requires row tensor')
        value = value.reshape(value.shape[0], -1)
        prior = self.buffers.get(name)
        if prior is None or prior.shape != value.shape or prior.dtype != value.dtype:
            if value.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('stage storage must be allocated by eager priming')
            prior = torch.empty_like(value)
            self.buffers[name] = prior
        prior.copy_(value)

    def wrap(self, owner, name, *, before=None, after=None):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        setattr(owner, name, _ObservedCall(original, before, after))

    def cpu(self, rows):
        return {name: (value if name.startswith('history_') else value[:rows]).cpu().clone()
                for name, value in self.buffers.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)

    def install_flare(self, op):
        self.wrap(op, 'feedback',
            before=lambda args, kwargs: self.record('embedding', op.scratch.embedding),
            after=lambda args, kwargs, result: self.record('feedback', op.scratch.residual))
        for prefix, hc in (('attn', op.attn_hc), ('mlp', op.mlp_hc), ('final', op.final_mixer)):
            def mix(args, kwargs, result, prefix=prefix, hc=hc):
                for field, value in hc.trace_intermediates().items():
                    self.record(f'{prefix}_{field}', value)
                self.record(f'{prefix}_mix', result)
            self.wrap(hc, 'mix', after=mix)
            if hc.injection is not None:
                for method in ('combine', 'combine_norm'):
                    self.wrap(hc, method, after=lambda args, kwargs, result, prefix=prefix:
                              self.record(f'{prefix}_residual', result))
        def attention(args, kwargs, result):
            self.record('attn_output', result)
            for name, value in (('qsa_query', op.scratch.qsa.query), ('qsa_key', op.scratch.qsa.key),
                                ('qsa_attended', op.scratch.qsa.attended), ('qsa_gated', op.scratch.qsa.gated),
                                ('qsa_selected', op.scratch.qsa.selected_positions),
                                ('history_keys', op.attention.state.main_k_cache[1, :8]),
                                ('history_values', op.attention.state.main_v_cache[1, :8])):
                self.record(name, value)
            for key, value in (('q_projected', op.scratch.qsa.projected_q),
                               ('k_projected', op.scratch.qsa.projected_k),
                               ('v_projected', op.scratch.qsa.projected_v),
                               ('index_projected', op.scratch.qsa.projected_index)):
                self.record(key, value)
        self.wrap(op, 'attention', after=attention)
        def moe(args, kwargs, result):
            self.record('moe_router_logits', op.moe.router_logits)
            self.record('moe_ids', op.moe.router.ids)
            self.record('moe_weights', op.moe.router.weights)
            self.record('moe_shared', op.moe.shared.output)
            self.record('moe_routed', op.moe.routed_output)
            self.record('mlp_output', result)
        self.wrap(op, 'moe', after=moe)


@contextmanager
def observe_functional_linear(model):
    """Record actual functional.linear invocations without changing their operands."""
    import torch.nn.functional as functional
    names = {}
    for name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            names.setdefault(weight.data_ptr(), []).append(name)
    original, calls = functional.linear, []
    def observed(source, weight, bias=None):
        result = original(source, weight, bias)
        calls.append({'modules': names.get(weight.data_ptr(), []), 'source_shape': list(source.shape),
                      'weight_shape': list(weight.shape), 'source_dtype': str(source.dtype),
                      'weight_dtype': str(weight.dtype), 'output_dtype': str(result.dtype)})
        return result
    functional.linear = observed
    try:
        yield calls
    finally:
        functional.linear = original


def fixture_from_trace(trace):
    if trace['model_id'] != MODEL_ID or trace['revision'] != REVISION:
        raise ValueError('MTP feedback source must identify the exact checkpoint')
    prompt, teachers = trace['prompt_tokens'], trace['teacher_tokens']
    layers = trace['layer_outputs']
    if len(layers) != len(teachers) + 1 or not prompt or not teachers:
        raise ValueError('MTP source requires one target context per teacher token plus the final context')
    cases = []
    for step, teacher in enumerate(teachers):
        residual = layers[step][-1]
        expected_rows = len(prompt) if step == 0 else 1
        if residual.shape != (expected_rows, 10240) or not torch.isfinite(residual).all():
            raise ValueError('MTP feedback requires finite full four-stream target residuals')
        if residual.device.type != 'cpu' or not torch.equal(residual.bfloat16().float(), residual.float()):
            raise ValueError('MTP target residual source must preserve exact BF16 values on CPU')
        ids = prompt[1:] + [teacher] if step == 0 else [teacher]
        start = 0 if step == 0 else len(prompt) + step - 1
        cases.append({'step': step, 'target_context': prompt + teachers[:step],
            'hidden': residual.bfloat16().contiguous(),
            'token_ids': torch.tensor(ids, dtype=torch.int64),
            'positions': torch.arange(start, start + expected_rows, dtype=torch.int64),
            'is_prefilling': step == 0})
    return {'schema_version': 1, 'model_id': MODEL_ID, 'revision': REVISION,
            'cases': cases, 'unconsumed_target_contexts': 1}


def fixture(args):
    trace = torch.load(args.trace, map_location='cpu', weights_only=True)
    data = fixture_from_trace(trace)
    torch.save(data, args.output / 'fixture.pt')
    return {'fixture': str(args.output / 'fixture.pt'),
        'cases': [{k: plain(v) if not isinstance(v, torch.Tensor) else
                   {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in c.items()} for c in data['cases']],
        'feedback': 'target residual h[i], token x[i+1], unshifted position i',
        'final_target_context_excluded': 'no following teacher token is stored for that context'}


def device():
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('assign a CUDA device with CUDA_VISIBLE_DEVICES')
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) not in ((12, 0), (12, 1)):
        raise RuntimeError('MTP capture requires one SM120/121 device')
    return torch.device('cuda:0')


def inputs(args):
    value = torch.load(args.output / 'fixture.pt', map_location='cpu', weights_only=True)
    if value['model_id'] != MODEL_ID or value['revision'] != REVISION:
        raise ValueError('fixture checkpoint identity mismatch')
    return value['cases']


@torch.no_grad()
def flare(args):
    from b12x.preparation import PreparationSession

    from ignition.preparation import inspect_qwen38_checkpoint
    from ignition.preparation.qwen38 import TEXT_PREFIX
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.attention import QSAMetadata, QSAState, RotaryTables
    from ignition.recipes.qwen38.mtp import MTPScratch, PreparedMTP
    from ignition.recipes.qwen38.mtp_weights import MTPWeights
    from ignition.recipes.qwen38.weights import QwenWeightLoader

    dev = device()
    manifest = inspect_qwen38_checkpoint(args.snapshot)
    cases = inputs(args)
    with QwenWeightLoader(manifest, dev, prefixes=(f'{TEXT_PREFIX}.embed_tokens.', 'lm_head.')) as loader:
        embedding = loader.load_tensor(f'{TEXT_PREFIX}.embed_tokens.weight')
        head = loader.load_tensor('lm_head.weight')
    weights = MTPWeights.load(manifest, dev, embedding_weight=embedding, lm_head_weight=head)
    cache = QSAState.allocate(pages=2, slots=1, device=dev, draft_tokens=7)
    rotary = RotaryTables.allocate(weights.module.config, 128, dev)
    arena = ScratchArena(dev)
    graphs, owners, outputs, allocation_checks = [], [], [], []
    with PreparationSession(device=dev, autotune=False, compile_workers=0) as session:
        try:
            for capacity in (8, 1):
                scratch = MTPScratch.allocate(capacity, 1, dev)
                hidden = torch.zeros((capacity, 10240), dtype=torch.bfloat16, device=dev)
                tokens = torch.zeros(capacity, dtype=torch.int64, device=dev)
                selected = torch.zeros(1, dtype=torch.int64, device=dev)
                metadata = QSAMetadata.allocate(tokens=capacity, batch=1, context=128, device=dev)
                op = PreparedMTP(weights, scratch, tokens, hidden, cache, metadata, rotary, arena,
                    output_indices=selected, session=session, name=f'mtp.{capacity}', max_cache_pages=2)
                trace = StageTrace() if args.trace_stages else None
                if trace is not None:
                    trace.install_flare(op)
                owners.append((op, tokens, hidden, selected, metadata, trace))
            arena.finalize(); arena.freeze()
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            for op, *_ in owners:
                graph = torch.cuda.CUDAGraph(); graphs.append(graph)
                with torch.cuda.stream(stream):
                    op()
                stream.synchronize()
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    op()
            session.freeze()
            for name in cache.__dataclass_fields__:
                value = getattr(cache, name)
                value.fill_(-1 if value.dtype == torch.int64 else 0)
            for case in cases:
                index = 0 if case['is_prefilling'] else 1
                op, tokens, hidden, selected, metadata, trace = owners[index]
                count, start = len(case['token_ids']), int(case['positions'][0])
                tokens.zero_(); hidden.zero_()
                tokens[:count].copy_(case['token_ids']); hidden[:count].copy_(case['hidden'])
                selected.fill_(count - 1)
                metadata.main_block_table[0].copy_(torch.tensor([1, 0], dtype=torch.int32, device=dev))
                metadata.compressed_block_table.copy_(metadata.main_block_table)
                metadata.raw_state_slot_ids.fill_(0)
                metadata.request_ids.fill_(-1); metadata.request_ids[:count].fill_(0)
                metadata.query_positions.fill_(-1); metadata.query_positions[:count].copy_(case['positions'])
                metadata.rope_positions.fill_(-1)
                metadata.rope_positions[:count].copy_(case['positions'][:, None].expand(-1, 3))
                metadata.sequence_lengths.fill_(start + count)
                metadata.query_start_loc[-1] = count
                metadata.num_accepted_tokens.fill_(1)
                metadata.is_prefilling.fill_(case['is_prefilling'])
                metadata.slot_mapping.fill_(-1)
                metadata.slot_mapping[:count].copy_(case['positions'] + 64)
                arena.tensor.fill_(0xA5)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()['allocation.all.allocated']
                graphs[index].replay(); torch.cuda.synchronize()
                after = torch.cuda.memory_stats()['allocation.all.allocated']
                allocation_checks.append({'step': case['step'], 'before': before, 'after': after, 'equal': before == after})
                outputs.append({'step': case['step'], 'residual': op.scratch.residual[:count].cpu().clone(),
                    'hidden': op.scratch.hidden[:count].cpu().clone(), 'logits': op.scratch.logits.cpu().clone(),
                    'stages': {} if trace is None else trace.cpu(count),
                    'attention_constants': {'q_norm': weights.module.layers[0].self_attn.q_norm.weight.cpu().clone(),
                        'k_norm': weights.module.layers[0].self_attn.k_norm.weight.cpu().clone(),
                        'rope_cos': rotary.cos[:8].cpu().clone(), 'rope_sin': rotary.sin[:8].cpu().clone()}})
            if any(not check['equal'] for check in allocation_checks):
                raise RuntimeError('Flare MTP graph replay allocated')
            torch.save(outputs, args.output / 'flare.pt')
            return {'stored_mtp_bytes': weights.stored_bytes,
                'shared_embedding_head': True, 'arena_bytes': arena.nbytes,
                'capacities': [8, 1], 'kv_dtype': 'bfloat16', 'cache_page_size': 64,
                'allocation_checks': allocation_checks, 'result': str(args.output / 'flare.pt')}

        finally:
            torch.cuda.synchronize()
            for graph in graphs:
                graph.reset()


def compare_output(actual, reference):
    from qualify.layer_comparison import _measurement
    from qualify.text_oracle import _metrics

    rows = reference['selected_rows']
    if rows != [actual['residual'].shape[0] - 1]:
        raise ValueError('MTP reference must select the exact request-tail row')
    result = {}
    for name in ('residual', 'hidden', 'logits'):
        value = actual[name] if name == 'logits' else actual[name][rows]
        if not torch.isfinite(value).all() or not torch.isfinite(reference[name]).all():
            raise ValueError('MTP outputs must be finite')
        result[name] = _measurement(value, reference[name])
    result['vocabulary'] = _metrics(actual['logits'][0].float(), reference['logits'][0].float())
    a_stages, r_stages = actual.get('stages', {}), reference.get('stages', {})
    result['stage_coverage'] = {'missing_in_flare': sorted(r_stages.keys() - a_stages.keys()),
                                'missing_in_reference': sorted(a_stages.keys() - r_stages.keys())}
    stages = {}
    for name in sorted(a_stages.keys() & r_stages.keys()):
        value, expected = a_stages[name], r_stages[name]
        selected = None
        if value.shape[0] != expected.shape[0]:
            if expected.shape[0] != 1 or value.shape[0] != actual['residual'].shape[0]:
                raise ValueError(f'{name}: unexplained MTP stage row geometry')
            selected, value = rows, value[rows]
        if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
            raise ValueError(f'{name}: MTP stage is nonfinite')
        stages[name] = {'selected_actual_rows': selected, **_measurement(value, expected)}
    result['stages'] = stages
    return result


def compare(args):
    a = torch.load(args.output / 'flare.pt', map_location='cpu', weights_only=True)
    b = torch.load(args.reference, map_location='cpu', weights_only=True)
    if len(a) != len(b):
        raise ValueError('MTP step count differs')
    records = []
    for actual, reference in zip(a, b, strict=True):
        if actual['step'] != reference['step']:
            raise ValueError('MTP steps differ')
        records.append({'step': actual['step'], 'reference_selected_rows': reference['selected_rows'],
            **compare_output(actual, reference)})
    return {'cases': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fixture', 'flare', 'compare'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--trace', type=Path, help='teacher-forced target residuals and token IDs')
    parser.add_argument('--reference', type=Path, help='MTP outputs with explicit selected_rows')
    parser.add_argument('--trace-stages', action='store_true')
    args = parser.parse_args()
    if args.command == 'fixture' and args.trace is None:
        parser.error('fixture requires --trace')
    if args.command == 'compare' and args.reference is None:
        parser.error('compare requires --reference')
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f'{args.command}.json'
    tensor_output = args.output / f'{args.command}.pt'
    if destination.exists() or (args.command != 'compare' and tensor_output.exists()):
        raise FileExistsError(f'output already exists: {destination}')
    result = globals()[args.command](args)
    destination.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
