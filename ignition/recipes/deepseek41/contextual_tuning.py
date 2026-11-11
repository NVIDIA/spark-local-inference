# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Restore bounded inputs for joint projection tuning in prepared decoder layers."""

from dataclasses import dataclass
from functools import partial

import torch
from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference
from b12x.preparation import PreparedCall

from ignition.preparation.execution import GraphSignature


@dataclass(frozen=True)
class _Case:
    seed: int
    window: torch.Tensor
    inputs: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    readonly: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    starts: tuple[int, ...]
    lengths: tuple[int, ...]


class _LayerTrial:
    def __init__(self, fixture, layer):
        self.fixture, self.layer = fixture, layer

    def __call__(self):
        f = self.fixture
        indices = f.indices(f.metadata, replay_start=f.replay_start)
        incoming, _, _, _ = self.layer(f.incoming, f.pre_mix, f.metadata, indices, f.selection)
        f.output.copy_(incoming[0])


class _LayerObservation:
    def __init__(self, fixture, layer):
        self.fixture, self.layer = fixture, layer

    def __call__(self):
        f = self.fixture
        indices = f.indices(f.metadata, replay_start=f.replay_start)
        *_, observations = self.layer._run(f.incoming, f.pre_mix, f.metadata, indices, f.selection)
        for name, destination in f.observations.items():
            source = observations[name]
            if isinstance(destination, tuple):
                for target, value in zip(destination, source, strict=True):
                    target.copy_(value)
            else:
                destination.copy_(source)


def _reference_quantize(source):
    blocks = source.float().view(source.shape[0], -1, 32)
    ratio = blocks.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = ratio.view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7fffff) != 0).int()
    scale = (exponent << 23).view(torch.float32)
    values = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.view_as(source), exponent.to(torch.uint8)



class CatalogLayerFixture:
    """Measure a catalog layer with packed inputs, cache state, and TP collectives.

    Inputs use checkpoint embeddings, valid mHC seam values, and bounded packed
    cache history. Two cases exercise full and partial request occupancy. Each
    replay restores mutable history outside its measured interval; a third seed
    checks the selected tuple. Untimed observation graphs expose local numeric
    intermediates; measured graphs retain only their final layer output. No
    collective counter is restored.
    """

    def __init__(self, name, signature, decoder, layer_index):
        self.target_name, self.target_signature = name, signature
        self.target_decoder, self.layer_index = decoder, layer_index
        self.layer = decoder.layers[layer_index]
        attention = self.layer.attention
        self.metadata, self.pool = self.layer.metadata, attention.pool
        self.scratch = self.layer.scratch
        self.indices, self.replay_start = attention.window_metadata, attention.replay_start
        scratch = self.scratch
        self.pre_mix = scratch.pre_mix
        self.incoming = ((scratch.mlp_output, scratch.ffn_hc.residual, scratch.ffn_hc.post, scratch.ffn_hc.comb)
                         if self.layer.carry else (scratch.residual,))
        self.selection = (scratch.attention.selected, scratch.attention.candidates, scratch.attention.candidate_counts)
        self.output = torch.empty_like(scratch.residual if self.layer.post is not None else scratch.mlp_output)
        self.observations = None
        self.tokens = self.metadata.positions.numel()
        self.batch = self.metadata.starts.numel()
        self.width = self.metadata.max_query_tokens
        self.label = f"{name}.layer{layer_index}"
        kind = "prefill" if attention.mode == "prefill" else "decode" if attention.mode == "decode" else "speculative"
        self.trial_signature = GraphSignature(kind, self.batch, self.width,
            packed_token_capacity=self.tokens if kind == "speculative" else 0)
        if (self.layer.engram is not None or self.layer.boundary is not None
                or attention.global_kv is not None):
            raise ValueError("joint tuning requires a layer with independently restorable inputs")
        if (self.layer.attention_reduce.runtime is None or self.layer.mlp_reduce.runtime is None
                or self.batch < 1 or not self.batch <= self.tokens <= self.batch * self.width):
            raise ValueError("joint tuning requires prepared request geometry and TP collectives")
        if int(self.pool.memory_capacity[0].item()) <= self.batch:
            raise ValueError("joint tuning request rows exceed physical preparation backing")
        pages = int(self.pool.memory_capacity[1].item())
        # The cache prefix is shared read-only across requests; every window has its own row.
        prefix = min(4096, (min(pages, self.metadata.page_table.shape[1]) * 128 - self.width) // 128 * 128)
        if prefix < 128:
            raise ValueError("joint tuning requires a backed history page before the query")
        self.start = prefix
        self.required_pages = (self.start + self.width + 127) // 128
        self.window = attention.window[1:self.batch + 1]
        self.tune_seeds, self.holdout_seed = (7101, 7102), 7103
        self.cases = ()
        self.active_case = None

    def _lengths(self, partial):
        live = self.batch - 1 if partial and self.batch > 1 else self.batch
        count = min(self.tokens, live * self.width)
        if partial and self.batch == 1 and count > 1:
            count -= 1
        quotient, remainder = divmod(count, live)
        return tuple(quotient + (row < remainder) if row < live else 0 for row in range(self.batch))

    def _set_metadata(self, seed, *, partial=False):
        metadata = self.metadata
        device = metadata.token_ids.device
        lengths = self._lengths(partial)
        starts = (self.start,) * self.batch
        pages = torch.full_like(metadata.page_table, -1)
        pages[:, :self.required_pages] = torch.arange(self.required_pages, device=device)
        vocabulary = min(int(self.target_decoder.weights.config.text["vocab_size"]),
                         self.target_decoder.weights.embedding.shape[0])
        vision = getattr(self.target_decoder, "vision", None)
        if vision is not None:
            vocabulary = min(vocabulary, vision.token_base)
        if vocabulary <= 3:
            raise ValueError("joint tuning requires ordinary text embedding IDs")
        tokens = torch.randint(3, vocabulary, (self.batch, self.width),
            generator=torch.Generator().manual_seed(seed), dtype=torch.int64).to(device)
        metadata.set(torch.tensor(starts, dtype=torch.int64, device=device),
            torch.tensor(lengths, dtype=torch.int32, device=device),
            torch.arange(1, self.batch + 1, device=device), pages, tokens,
            torch.tensor([sum(length > 0 for length in lengths)], dtype=torch.int32, device=device))
        if self.replay_start is not None:
            self.replay_start.zero_()
        return starts, lengths

    @staticmethod
    def _snapshot(tensors):
        seen, snapshots = set(), []
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.data_ptr() in seen:
                continue
            if any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, tensor.stride())):
                continue
            seen.add(tensor.data_ptr())
            snapshots.append((tensor, tensor.clone()))
        return tuple(snapshots)

    def _cache_inputs(self, generator):
        attention = self.layer.attention
        device = self.window.device
        history = (torch.randn((self.batch * 128, 512), generator=generator) * .125).bfloat16().to(device)
        self.window.copy_(pack_deepseek_v41_cache_reference(history,
            page_size=128, cache_kind="swa").reshape_as(self.window))
        if not attention.ratio:
            return (), ()
        from b12x.attention import dsa_indexer as dsa

        source = attention.source.layer
        page_size = 128 // attention.ratio
        latent = self.pool.pages[f"layer{source}.latent_page"][:self.required_pages]
        values = (torch.randn((self.required_pages * page_size, 512),
            generator=generator) * .125).bfloat16().to(device)
        latent.copy_(pack_deepseek_v41_cache_reference(values,
            page_size=page_size, cache_kind="indexed").reshape_as(latent))
        source_index = self.target_decoder.weights.config.index_source(self.layer_index)
        indexer = self.target_decoder.layers[source_index].attention.selectors[0]
        index_keys = indexer.cache[:self.required_pages]
        keys = (torch.randn((self.required_pages * page_size, 128),
            generator=generator) * .125).bfloat16().to(device)
        slots = torch.arange(keys.shape[0], device=device, dtype=torch.int64)
        for first in range(0, keys.shape[0], min(self.tokens, 128)):
            last = min(first + min(self.tokens, 128), keys.shape[0])
            dsa.quantize_write_index_k_mxfp4(indexer.plan, keys[first:last],
                index_k_cache=indexer.cache, slot_mapping=slots[first:last])
        scratch = self.scratch.attention
        count = min(self.start // attention.ratio, scratch.candidates.shape[1])
        scratch.candidates.fill_(-1)
        scratch.candidates[:, :count] = torch.arange(count, dtype=torch.int32, device=device)
        scratch.candidate_counts.fill_(count)
        scratch.selected.fill_(-1)
        selected = min(count, scratch.selected.shape[1])
        scores = torch.randn((self.tokens, count), generator=generator)
        scratch.selected[:, :selected].copy_(scores.topk(selected, dim=1).indices.to(device, torch.int32))
        indices = (scratch.candidates, scratch.candidate_counts, scratch.selected)
        readonly = (latent, index_keys, *indices) if not attention.selectors else (latent, index_keys)
        return readonly, indices

    def initialize(self):
        if self.cases:
            raise RuntimeError("joint tuning inputs are already initialized")
        layer, metadata = self.layer, self.metadata
        scratch = self.scratch
        cases = []
        for ordinal, seed in enumerate((*self.tune_seeds, self.holdout_seed)):
            starts, lengths = self._set_metadata(seed, partial=ordinal == 1)
            readonly, indices = self._cache_inputs(torch.Generator().manual_seed(seed))
            embedding = self.target_decoder.weights.embedding[metadata.token_ids]
            scratch.residual.copy_(embedding[:, None, :])
            scratch.pre_mix.zero_()
            scratch.pre_mix[:, 0].fill_(1)
            incoming = [scratch.residual, scratch.pre_mix]
            if layer.carry:
                scratch.mlp_output.zero_()
                scratch.ffn_hc.residual.copy_(scratch.residual)
                scratch.ffn_hc.post.fill_(1)
                scratch.ffn_hc.comb.copy_(torch.eye(4, device=embedding.device))
                incoming.extend((scratch.mlp_output, scratch.ffn_hc.residual,
                                 scratch.ffn_hc.post, scratch.ffn_hc.comb))
            cases.append(_Case(seed, self.window.clone(), self._snapshot((
                *(getattr(metadata, name) for name in metadata.__dataclass_fields__),
                *incoming, *indices, self.replay_start)),
                self._snapshot(readonly), starts, lengths))
        self.cases = tuple(cases)

    def close(self):
        self.cases = ()
        self.active_case = None

    def _check_layer(self, layer):
        reference = self.layer
        if (layer.attention.pool is not self.pool or layer.carry != reference.carry
                or layer.post is not reference.post or layer.engram is not reference.engram
                or layer.boundary is not reference.boundary
                or layer.attention.window_metadata is not reference.attention.window_metadata
                or layer.attention_reduce.runtime is not reference.attention_reduce.runtime
                or layer.mlp_reduce.runtime is not reference.mlp_reduce.runtime):
            raise ValueError("joint tuning must retain the catalog layer's cache and execution context")

    def trial_forward(self, layer):
        self._check_layer(layer)
        return _LayerTrial(self, layer)

    def observation_forward(self, layer):
        self._check_layer(layer)
        if self.observations is None:
            scratch, attention = self.scratch, layer.attention
            self.observations = {
                "attention_hc.normalized": torch.empty_like(scratch.attention_hc.normalized),
                "attention.projected": torch.empty_like(scratch.attention.projected),
                "attention.query_low": torch.empty_like(scratch.attention.query_low),
                "attention.query": torch.empty_like(scratch.attention.query),
                "attention.window_values": torch.empty_like(scratch.attention.window_values),
                "attention.output": torch.empty_like(scratch.attention.output),
                "router.ids": torch.empty_like(scratch.router.ids),
                "output": torch.empty_like(self.output),
            }
            if attention.ratio:
                self.observations["attention.selected"] = torch.empty_like(scratch.attention.selected)
            qkv = attention.projections["qkv_a"]
            self.observations["attention.qkv_encoded_bases"] = tuple(
                torch.empty(shape, dtype=dtype, device=self.output.device) for shape, dtype in qkv.encoded_layout)
            self.observations["attention.query_encoded_bases"] = tuple(
                torch.empty(shape, dtype=dtype, device=self.output.device) for shape, dtype in attention.norm_quant.layouts[2:])
        return _LayerObservation(self, layer)

    def _poison(self, layer):
        self.output.fill_(float("nan"))


    def _restore(self, case, layer):
        self.window.copy_(case.window)
        for target, saved in (*case.inputs, *case.readonly):
            target.copy_(saved)
        self.active_case = case

    def validate_result(self, layer, *, window=None, readonly=None):
        case = self.active_case
        if case is None:
            raise RuntimeError("joint tuning has no active input case")
        observations = self.observations
        if (not bool(torch.isfinite(self.output).all())
                or not bool(torch.isfinite(observations["attention.output"]).all())):
            raise AssertionError("joint tuning produced nonfinite layer output")
        expected = case.window.clone().view(self.batch, 128, 528)
        if layer.attention.mode in ("decode", "prefill"):
            records = pack_deepseek_v41_cache_reference(observations["attention.window_values"],
                page_size=128, cache_kind="swa").view(-1, 528)[:self.tokens]
            offset = 0
            for row, (start, length) in enumerate(zip(case.starts, case.lengths, strict=True)):
                for column in range(max(0, length - 128), length):
                    expected[row, (start + column) % 128].copy_(records[offset + column])
                offset += length
        window = self.window if window is None else window
        if not torch.equal(window, expected.reshape_as(window)):
            raise AssertionError("joint tuning changed cache history outside the layer's declared writes")
        readonly = case.readonly if readonly is None else readonly
        if any(not torch.equal(target, saved) for target, saved in readonly):
            raise AssertionError("joint tuning changed read-only compressed history")
        if layer.attention.ratio:
            selected = observations["attention.selected"]
            live = self.metadata.positions >= 0
            lengths = (self.metadata.positions[live] + 1) // layer.attention.ratio
            values = selected[live]
            if not bool(((values >= -1) & (values < lengths[:, None])).all()):
                raise AssertionError("joint tuning selected an index outside the valid cache prefix")
        ids = observations["router.ids"]
        if not bool(((ids >= 0) & (ids < layer.router.shape[1])).all()):
            raise AssertionError("joint tuning produced an invalid routed-expert ID")

    def validate_numeric(self, layer):
        """Check actual QKV and BF16-seam encoding outside the timed race."""
        from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
        from b12x.quantization import mxfp8

        self._check_layer(layer)
        observations = self.observations
        source = observations["attention_hc.normalized"]
        projection = layer.attention.projections["qkv_a"]
        encoded = mxfp8.dense_rows_from_bases(*observations["attention.qkv_encoded_bases"])
        if encoded is None or projection.plan.query.allow_bf16_atomic_reduction:
            raise ValueError("contextual QKV requires prepared FP32-reduced MXFP8 rows")
        expected_values, expected_scales = _reference_quantize(source)
        if (not torch.equal(encoded.values.view(torch.uint8), expected_values.view(torch.uint8))
                or not torch.equal(encoded.scale_rows.view(torch.uint8),
                                   expected_scales.unsqueeze(0))):
            raise AssertionError("QKV activation differs from the declared BF16-to-MXFP8 encoding")
        lhs = dequantize_mxfp8_rows_torch(encoded.values, encoded.scale_rows).double()
        packed = projection.weight.weight
        rhs = dequantize_mxfp8_rows_torch(packed.values, packed.scale_rows).double()
        exact = lhs @ rhs.T
        magnitude = lhs.abs() @ rhs.abs().T
        slices = projection.plan.selection.config.split_k_slices or 1
        operations = source.shape[1] + slices
        u = operations * 2.0 ** -24
        error = (u / (1.0 - u)) * magnitude
        actual = observations["attention.projected"]
        below = torch.nextafter(actual, torch.full_like(actual, -float("inf"))).double()
        above = torch.nextafter(actual, torch.full_like(actual, float("inf"))).double()
        error += .5 * torch.maximum(actual.double() - below,
                                     above - actual.double())
        deviation = (actual.double() - exact).abs()
        if not bool(torch.isfinite(actual).all()) or not bool((deviation <= error).all()):
            raise AssertionError("QKV exceeds FP32 accumulation and BF16 output rounding bounds")
        q = actual[:, :1280].double()
        weight = layer.attention.weights["q_norm"].double()
        expected_query = (q * torch.rsqrt(q.square().mean(-1, keepdim=True)
                            + layer.attention.eps) * weight).bfloat16()
        actual_query = observations["attention.query_low"]
        torch.testing.assert_close(actual_query, expected_query, rtol=2**-7, atol=0)
        query_rows = mxfp8.dense_rows_from_bases(*observations["attention.query_encoded_bases"])
        expected_values, expected_scales = _reference_quantize(actual_query)
        if (not torch.equal(query_rows.values.view(torch.uint8),
                            expected_values.view(torch.uint8))
                or not torch.equal(query_rows.scale_rows.view(torch.uint8),
                                   expected_scales.unsqueeze(0))):
            raise AssertionError("query expansion differs from BF16-seam MXFP8 encoding")
        return dict(qkv_max_abs_vs_fp64=float(deviation.max()),
                    query_max_abs_vs_fp64=float((actual_query.float()
                        - expected_query.float()).abs().max()))


    def restore_tune(self, layer, case_index=0):
        if not 0 <= case_index < len(self.tune_seeds) or len(self.cases) != 3:
            raise ValueError("joint tuning input case is unavailable")
        self._check_layer(layer)
        self._poison(layer)
        self._restore(self.cases[case_index], layer)

    def restore_holdout(self, layer):
        if len(self.cases) != 3:
            raise RuntimeError("joint tuning held-out input is unavailable")
        self._check_layer(layer)
        self._poison(layer)
        self._restore(self.cases[2], layer)

    def validate_trial(self, layer, observation_graph):
        case = self.active_case
        if case is None:
            raise RuntimeError("joint tuning has no active input case")
        window = self.window.clone()
        readonly = tuple((target.clone(), saved) for target, saved in case.readonly)
        self._restore(case, layer)
        observation_graph.replay()
        torch.cuda.synchronize(self.output.device)
        self.validate_result(layer, window=window, readonly=readonly)
        return self.validate_numeric(layer)

    def prepared_call(self, name, graph, compiled, layer, observation_graph, observation_compiled):
        self._check_layer(layer)
        producers = tuple(partial(self._restore, case, layer) for case in self.cases[:2])
        return PreparedCall(run=graph.replay, output=self.output,
            reset=partial(self._poison, layer), produce=producers[0], benchmark_producers=producers,
            validate=partial(self.validate_trial, layer, observation_graph),
            owners=(self, graph, compiled, layer, observation_graph, observation_compiled),
            close=graph.reset, capture_safe=True)

    def release_examples(self):
        self.close()
        self.scratch = self.metadata = self.replay_start = self.incoming = self.pre_mix = None
        self.selection = self.output = self.observations = self.target_decoder = None
