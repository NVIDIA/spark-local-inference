# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared BF16 MLA normalization with MXFP8 query encoding."""

import torch
from b12x.preparation import PreparedCall
from b12x.quantization import mla_norm_mxfp8, mxfp8

from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_context_plan, register_finalizer


class PreparedMLANormQuant:
    """Normalize query and KV slices, then encode the rounded BF16 query."""

    def __init__(self, projected, weights, query, kv, *, eps, session, name, buffers=None,
                 release_dependents=False, examples=None):
        rows = projected.shape[0]
        if (projected.shape != (rows, 1792) or projected.dtype != torch.bfloat16
                or not projected.is_contiguous() or query.shape != (rows, 1280)
                or kv.shape != (rows, 512) or query.dtype != torch.bfloat16
                or kv.dtype != torch.bfloat16 or not query.is_contiguous()
                or not kv.is_contiguous() or eps != 1e-20):
            raise ValueError("MLA normalization requires contiguous BF16 1280+512 rows and model epsilon")
        self.weights, self.query, self.kv = tuple(weights), query, kv
        for weight, width in zip(self.weights, (1280, 512), strict=True):
            if (weight.shape != (width,) or weight.dtype not in (torch.bfloat16, torch.float32)
                    or weight.device != projected.device or not weight.is_contiguous()):
                raise ValueError("MLA normalization affine weights must match their output slices")
            record_numeric_tensor(weight)
        def allocate(role, shape, dtype):
            if examples is not None:
                return declaration_tensor(examples, "mla.query." + role, shape, dtype, projected.device)
            return (torch.zeros(shape, dtype=dtype, device=projected.device) if buffers is None
                    else buffers.zeros(shape, dtype))

        values_shape, rows_shape, mma_shape = mxfp8.dense_rows_storage_shapes(rows, 1280)
        self.encoded_bases = (allocate("values", values_shape, torch.float8_e4m3fn),
                              allocate("scale_rows", rows_shape, torch.uint8),
                              allocate("scale_mma", mma_shape, torch.uint8))
        self.encoded = mxfp8.dense_rows_from_bases(*self.encoded_bases)
        self.plan = mla_norm_mxfp8.plan(mla_norm_mxfp8.Query(
            rows=rows, query_weight_dtype=str(weights[0].dtype).removeprefix("torch."),
            kv_weight_dtype=str(weights[1].dtype).removeprefix("torch."),
            query_features=1280, kv_features=512, epsilon=eps, min_amax=1e-4,
            affine_prefetch_ready=True, release_dependents=release_dependents))
        representative = input_factory(projected, projected,
            storage=getattr(session, "benchmark_storage", None))

        encoded_bases, encoded = self.encoded_bases, self.encoded

        def prepare(state, fixture=None):
            source = projected if fixture is None else fixture.source
            return PreparedCall(run=lambda: state.run(source, *self.weights, query, kv,
                *encoded_bases), produce=None if fixture is None else fixture.produce,
                output=(query, kv, *encoded_bases), owners=(fixture, encoded))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: prepare(state),
            benchmark_call=lambda state: prepare(state, representative())),))
        register_context_plan(session, name, self)

    def functional(self, *, session):
        self.layouts = tuple((tuple(t.shape), t.dtype) for t in (self.query, self.kv, *self.encoded_bases))
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.query = self.kv = self.encoded_bases = self.encoded = None

    def __call__(self, projected):
        if self.query is None:
            query, kv, *bases = (torch.zeros(shape, dtype=dtype, device=projected.device)
                                 for shape, dtype in self.layouts)
            mla_norm_mxfp8.run(projected, *self.weights, query, kv, *bases, plan=self.plan)
            return query, kv, tuple(bases)
        mla_norm_mxfp8.run(projected, *self.weights, self.query, self.kv,
                            *self.encoded_bases, plan=self.plan)
        return self.query, self.kv
