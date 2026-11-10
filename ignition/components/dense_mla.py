# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared dense attention over paged compressed latent records."""

import torch
from b12x.attention import dense_mla
from b12x.preparation import PreparedCall, plan_from_handle

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
                kv_scale = torch.ones(1, dtype=torch.float32, device=cache.device)
            if q_scale is None:
                q_scale = torch.ones(1, dtype=torch.float32, device=query.device)
        self.arguments = dict(q=query, kv_cache=cache, output=output, page_table=page_table,
                              cache_seqlens=lengths, cu_seqlens_q=starts, kv_scale=kv_scale, q_scale=q_scale)
        self.plan = dense_mla.plan(caps, invocation=dense_mla.invocation_from_tensors(caps, **self.arguments))
        self.scale = float(scale)
        self.lse = lse

        def prepare(state, arguments):
            scratch = planned_scratch(state.scratch_specs(), workspace)
            binding = state.bind(**arguments, sm_scale=scale, scratch=scratch)
            state.prime(binding)
            return PreparedCall(run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding))

        def benchmark(state):
            def inputs():
                arguments = dict(self.arguments)
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

        declare(session, (self.plan.request(name=name, prepare_call=lambda state: prepare(state, self.arguments),
                                           benchmark_call=benchmark),))

        def finalize():
            scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            self.scratch = list(scratch) if isinstance(scratch, tuple) else [scratch]
            dense_mla.bind(self.plan, **self.arguments, sm_scale=scale, scratch=self.scratch)

        register_finalizer(session, finalize, workspace=workspace)

    def __call__(self, query, output):
        a = self.arguments
        _attend(query, a["kv_cache"], output, a["page_table"], a["cache_seqlens"],
                a["cu_seqlens_q"], self.scratch, self.plan.handle, self.scale, self.lse,
                a["kv_scale"], a["q_scale"])
        return output
