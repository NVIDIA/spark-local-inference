# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-token Kimi sparse indexing over rank-local FP8 key pages."""

import torch

from ignition.components.candidate_selection import PreparedCandidateSelection
from ignition.components.fp8_indexer import PreparedFP8Indexer, write_fp8_index_keys
from ignition.components.linear import UnquantizedProjection
from ignition.preparation.tuning import register_finalizer
from ignition.preparation.scratch import buffer_rows

from .dcp import TOPK, PREFILL_ROWS, index_pages


class SharedBuffers:
    """Declaration examples shared by serial layers and query slices."""

    def __init__(self, values, device, prefix=""):
        self.values, self.device, self.prefix = values, device, prefix
        self.ordinal = 0

    def buffer(self, name, shape, dtype):
        name = self.prefix + name
        value = self.values.get(name)
        if value is None or value.shape[0] < shape[0]:
            value = torch.zeros(shape, dtype=dtype, device=self.device)
            self.values[name] = value
        if value.shape[1:] != shape[1:] or value.dtype != dtype:
            raise ValueError("Kimi declaration tensors require matching non-row dimensions and dtype")
        return buffer_rows(self.values, name, shape[0])

    def zeros(self, shape, dtype):
        name = str(self.ordinal)
        self.ordinal += 1
        return self.buffer(name, shape, dtype)


class PreparedIndexSelection:
    def __init__(self, query, weights, cache, metadata, shards, context, buffers, *, start, end,
                 mode, session, name, workspace, latent_cache):
        rows, width = end - start, metadata.token_page_table.shape[1]
        self.rows, self.width, self.latent_cache = rows, width, latent_cache
        self.shards, self.mode = shards, mode
        storage = SharedBuffers(buffers, query.device)
        self.pages = storage.buffer("pages", (1 if mode == "prefill" else rows, width * 2), torch.int32)
        self.lengths = storage.buffer("local_lengths", (rows,), torch.int32)
        self.local = storage.buffer("local_ids", (rows, TOPK), torch.int32)
        self.scores = storage.buffer("local_scores", (rows, TOPK), torch.float32)
        self.selected = storage.buffer("selected", (rows, TOPK), torch.int32)
        self.candidate_counts = storage.buffer("candidate_counts", (rows,), torch.int32)
        pages = self.pages.expand(rows, -1) if mode == "prefill" else self.pages
        self.indexer = PreparedFP8Indexer(query[start:end].view(rows, 32, 128), weights[start:end],
            cache, pages, self.lengths, None, self.local, mode="prefill" if mode == "prefill" else "decode", session=session,
            name=name + ".local", workspace=workspace,
            buffers=SharedBuffers(buffers, query.device, prefix="fp8."), topk=TOPK, scores=self.scores,
            prefill_records_per_call=131072, head_weight_scale=1.0, hadamard=False, key_power_of_two=False)
        runtime = None if context is None else context.candidates
        if shards.world > 1 and runtime is None:
            raise ValueError("Kimi sparse selection requires a candidate exchange for every context rank")
        tuning = None if context is None else getattr(context, "selection_tuning", None)
        tune = tuning is not None and mode != "prefill" and metadata.starts.numel() == 1
        self.merge = PreparedCandidateSelection(self.local, self.scores, metadata.positions[start:end],
            shards.index, self.selected, world=shards.world, gather_runtime=runtime,
            owner_runtime=context.candidate_owners if tune else None,
            reduction_runtime=tuning.reduction if tune else None, session=session,
            coordinator=None if context is None else context.coordinator, name=name + ".merge",
            buffers=SharedBuffers(buffers, query.device, prefix="merge."),
            dependencies=(latent_cache,), alternatives=("replicated", "owner") if tune else ("replicated",),
            local_counts=self.candidate_counts if tune else None)
        if tune:
            tuning.register(self.merge)
        else:
            self.merge.bind("replicated")

        self.indexer.functional(session=session)
        self.merge.functional(session=session)
        register_finalizer(session, self._release_examples)

    def _release_examples(self):
        self.pages = self.lengths = self.local = self.scores = self.selected = self.candidate_counts = None

    def prepare_local(self, query, weights, positions, tables):
        return self.prepare_local_result(query, weights, positions, tables)[0]

    def prepare_local_result(self, query, weights, positions, tables):
        pages = torch.empty((1 if self.mode == "prefill" else self.rows, self.width * 2),
                            dtype=torch.int32, device=query.device)
        index_pages(tables[:1] if self.mode == "prefill" else tables, pages)
        lengths = self.shards.local_tokens((positions + 1).clamp(min=0)).to(torch.int32)
        (indices, scores), quantized, scales, scaled_weights = self.indexer._run_observed(
            query.view(query.shape[0], 32, 128), weights,
            pages.expand(self.rows, -1) if self.mode == "prefill" else pages, lengths)
        packed = self.merge.pack(indices=indices, scores=scores)
        return packed, indices, scores, quantized, scales, scaled_weights

    def exchange(self, packed, positions, *, dependencies=()):
        return self.exchange_result(packed, positions, dependencies=dependencies)[0]

    def exchange_result(self, packed, positions, *, dependencies=()):
        return self.merge.exchange_with_scores(packed=packed, positions=positions,
                                   dependencies=(self.latent_cache, *dependencies))

    def __call__(self, query, weights, positions, tables):
        packed = self.prepare_local(query, weights, positions, tables)
        return self.exchange(packed, positions)


class PreparedDSAIndexer:
    """BF16 projections and head normalization, FP32 LayerNorm, and global top-2048 selection."""

    def __init__(self, weights, source, query_latent, metadata, cache, shards, *, context, eps, mode,
                 session, name, workspace, buffers, latent_cache):
        rows = source.shape[0]
        expected = {"indexer.wq_b": (4096, 1536), "indexer.wk": (128, 7168),
            "indexer.weights_proj": (32, 7168), "indexer.k_norm.weight": (128,),
            "indexer.k_norm.bias": (128,)}
        if source.shape != (rows, 7168) or query_latent.shape != (rows, 1536):
            raise ValueError("Kimi sparse indexing requires hidden[rows,7168] and normalized query[rows,1536]")
        for field, shape in expected.items():
            if weights[field].shape != shape or weights[field].dtype != torch.bfloat16:
                raise ValueError(f"Kimi sparse index weight {field} requires BF16{shape}")
        if cache.dtype != torch.uint8 or cache.numel() % (64 * 132):
            raise ValueError("Kimi sparse index cache requires planar 64-record uint8 pages")
        storage = SharedBuffers(buffers, source.device)
        self.query = storage.buffer("query", (rows, 4096), torch.bfloat16)
        self.key = storage.buffer("key", (rows, 128), torch.bfloat16)
        self.normalized_key = storage.buffer("normalized_key", (rows, 128), torch.bfloat16)
        self.head_weights = storage.buffer("weights", (rows, 32), torch.bfloat16)
        self.weights, self.metadata, self.shards = weights, metadata, shards
        self.eps = eps
        self.slots = storage.buffer("slots", (rows,), torch.int64)
        self.projections = tuple(UnquantizedProjection(x, weights[field], output,
            session=session, name=name + "." + field) for field, x, output in (
                ("indexer.wq_b", query_latent, self.query), ("indexer.wk", source, self.key),
                ("indexer.weights_proj", source, self.head_weights)))
        self.query_latent = query_latent
        step = PREFILL_ROWS if mode == "prefill" else rows
        self.bounds = tuple((start, min(start + step, rows)) for start in range(0, rows, step))
        selection_buffers = buffers.setdefault("selection", {})
        cache_pages = cache.view(-1, 64 * 132)
        self.cache, self.sharded = cache_pages, None
        if mode == "prefill" and rows >= 4096:
            from .query_selection import PreparedQuerySelection

            self.sharded = PreparedQuerySelection(self.query, self.head_weights, cache_pages,
                metadata, shards, context, buffers.setdefault("query_selection", {}),
                session=session, name=name + ".queries", workspace=workspace)
        self.selections = {}
        for start, end in (() if self.sharded is not None else self.bounds):
            rows = end - start
            if rows not in self.selections:
                self.selections[rows] = PreparedIndexSelection(self.query, self.head_weights, cache_pages,
                    metadata, shards, context, selection_buffers, start=start, end=end, mode=mode,
                    session=session, name=f"{name}.slice{start}", workspace=workspace,
                    latent_cache=latent_cache)

        for projection in self.projections:
            projection.functional(session=session)
        register_finalizer(session, self._release_examples)

    def _release_examples(self):
        self.query = self.key = self.normalized_key = self.head_weights = None
        self.query_latent = self.metadata = self.slots = None

    @property
    def local_plans(self):
        """Projection, query-quantization, and index-scoring plans with no peer exchange."""
        indexers = ([part.indexer for part in self.selections.values()] if self.sharded is None
                    else [part[0] for part in self.sharded.indexers.values()])
        return (*[projection.plan for projection in self.projections],
                *[plan for indexer in indexers for plan in (indexer.query_plan, indexer.selection.plan)])

    def project(self, source, query_latent, metadata):
        query, _, _, head_weights = self.project_result(source, query_latent, metadata)
        return query, head_weights

    def project_result(self, source, query_latent, metadata):
        query = self.projections[0](query_latent)
        key = self.projections[1](source).float()
        head_weights = self.projections[2](source) * (32 ** -.5)
        centered = key - key.mean(-1, keepdim=True)
        variance = centered.square().mean(-1, keepdim=True)
        normalized_key = (centered * torch.rsqrt(variance + self.eps)
            * self.weights["indexer.k_norm.weight"].float()
            + self.weights["indexer.k_norm.bias"].float()).to(source.dtype)
        slots = self.shards.token_slots(metadata.positions, metadata.token_page_table)
        write_fp8_index_keys(normalized_key, slots, self.cache, hadamard=False, power_of_two=False)
        return query, key.to(source.dtype), normalized_key, head_weights
