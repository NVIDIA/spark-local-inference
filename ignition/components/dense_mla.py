# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared dense attention over paged compressed latent records."""

import torch
from b12x.attention import dense_mla
from b12x.preparation import PreparedCall, plan_from_handle

from ignition.preparation.persistent import persistent_empty
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer


@torch.library.custom_op("flare::dense_mla", mutates_args=("output", "scratch", "lse"))
def _attend(q: torch.Tensor, cache: torch.Tensor, output: torch.Tensor, pages: torch.Tensor,
            lengths: torch.Tensor, starts: torch.Tensor, scratch: list[torch.Tensor], handle: int, scale: float,
            lse: torch.Tensor | None, kv_scale: torch.Tensor | None, q_scale: torch.Tensor | None) -> None:
    plan = plan_from_handle(handle)
    binding = dense_mla.bind(plan, q=q, kv_cache=cache, output=output, page_table=pages,
        cache_seqlens=lengths, cu_seqlens_q=starts, scratch=scratch, sm_scale=scale,
        kv_scale=kv_scale, q_scale=q_scale)
    _, logsumexp = dense_mla.run(binding=binding)
    if lse is not None:
        lse.copy_(logsumexp)


@_attend.register_fake
def _fake_attend(q, cache, output, pages, lengths, starts, scratch, handle, scale, lse, kv_scale, q_scale):
    pass


class PreparedDenseMLA:
    """Dense MLA over a paged cache; an E4M3 cache defaults to unit query and cache scales.

    With ``lse``, each query head's natural-log LSE [rows, heads] is copied out beside the output.
    """

    def __init__(self, query, cache, output, page_table, lengths, starts, *, caps, scale, session, name, workspace,
                 lse=None, kv_scale=None, q_scale=None):
        if cache.dtype == torch.float8_e4m3fn:
            if kv_scale is None:
                kv_scale = persistent_empty((1,), torch.float32, cache.device).fill_(1)
            if q_scale is None:
                q_scale = persistent_empty((1,), torch.float32, query.device).fill_(1)
        self.arguments = dict(q=query, kv_cache=cache, output=output, page_table=page_table,
                              cache_seqlens=lengths, cu_seqlens_q=starts, kv_scale=kv_scale, q_scale=q_scale)
        self.plan = dense_mla.plan(caps, invocation=dense_mla.invocation_from_tensors(caps, **self.arguments))
        self.output, self.scale = output, float(scale)
        self.lse = lse

        declaration = self.arguments

        def prepare(state, arguments):
            scratch = planned_scratch(state.scratch_specs(), workspace)
            binding = state.bind(**arguments, sm_scale=scale, scratch=scratch)
            state.prime(binding)
            return PreparedCall(run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding))

        def benchmark(state):
            def inputs():
                arguments = dict(declaration)
                for field in ("q", "output", "page_table", "cache_seqlens", "cu_seqlens_q"):
                    value = arguments[field]
                    arguments[field] = torch.empty_strided(value.shape, value.stride(),
                        dtype=value.dtype, device=value.device)
                generator = torch.Generator(device=query.device).manual_seed(410528)
                if query.dtype == torch.float8_e4m3fn:
                    arguments["q"].copy_(torch.randn(query.shape, device=query.device, generator=generator) * .2)
                else:
                    arguments["q"].normal_(std=.2, generator=generator)
                arguments["page_table"].zero_()
                arguments["cache_seqlens"].fill_(min(max(4096, query.shape[0]), caps.max_cache_tokens))
                arguments["cu_seqlens_q"].copy_(torch.arange(starts.numel(), device=query.device,
                    dtype=starts.dtype) * query.shape[0] // lengths.numel())
                # E4M3 queries cannot be sampled directly; seed in FP32 and convert.
                cache_seed = (torch.randn(cache[0].shape, device=cache.device, generator=generator) * .2).to(
                    cache.dtype)
                return arguments, arguments["q"].clone(), cache_seed

            arguments, query_seed, cache_seed = benchmark_resource(session, ("dense_mla", name), inputs)
            call = prepare(state, arguments)

            def produce():
                arguments["q"].copy_(query_seed)
                cache[0].copy_(cache_seed)

            return PreparedCall(run=call.run, produce=produce, output=arguments["output"],
                owners=(*call.owners, arguments, query_seed, cache_seed))

        declare(session, (self.plan.request(name=name, prepare_call=lambda state: prepare(state, declaration),
                                           benchmark_call=benchmark),))

        def finalize():
            if self.output is None:
                return
            scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            self.scratch = list(scratch) if isinstance(scratch, tuple) else [scratch]
            dense_mla.bind(self.plan, **declaration, sm_scale=scale, scratch=self.scratch)

        register_finalizer(session, finalize, workspace=workspace)

    def functional(self, *, session, cache=...):
        if cache is not ...:
            self.arguments = {**self.arguments, "kv_cache": cache}
        self.output_layout = tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype
        self.return_lse = self.lse is not None
        self.lse_layout = None if self.lse is None else (tuple(self.lse.shape), tuple(self.lse.stride()))
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.arguments = {key: value for key, value in self.arguments.items()
                          if key in ("kv_cache", "kv_scale", "q_scale")}
        self.output = self.lse = self.scratch = None

    def __call__(self, query=None, page_table=None, lengths=None, starts=None, *, cache=None,
                 kv_scale=None, q_scale=None):
        a = self.arguments
        if self.output is not None:
            _attend(a["q"], a["kv_cache"], a["output"], a["page_table"], a["cache_seqlens"],
                    a["cu_seqlens_q"], self.scratch, self.plan.handle, self.scale, self.lse,
                    a["kv_scale"], a["q_scale"])
            return self.output
        shape, stride, dtype = self.output_layout
        output = torch.empty_strided(shape, stride, dtype=dtype, device=query.device).zero_()
        lse = torch.empty_strided(*self.lse_layout, dtype=torch.float32, device=query.device).zero_() if self.return_lse else None
        scratch = [torch.empty(shape, dtype=dtype, device=query.device) for shape, dtype in self.scratch_layout]
        _attend(query, a["kv_cache"] if cache is None else cache, output, page_table, lengths, starts,
                scratch, self.plan.handle, self.scale, lse,
                a["kv_scale"] if kv_scale is None else kv_scale, a["q_scale"] if q_scale is None else q_scale)
        return (output, lse) if self.return_lse else output
