# SPDX-License-Identifier: Apache-2.0

"""Kimi MLA DFlash2 with bounded accepted-context rings and sharded projections."""

from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from b12x.attention import dense_mla

from ignition.components.collectives import PreparedAllReduce, PreparedChunkedAllGather
from ignition.components.dense_mla import PreparedDenseMLA
from ignition.components.dflash2 import PooledPreparedDraft as BlockDraft
from ignition.components.kda_speculation import KDARecords
from ignition.components.linear import BatchedUnquantizedProjection, prepare_projection
from ignition.components.mlp import MLPScratch
from ignition.components.speculation import SpeculativeRngState
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.kimi3.decoder import PreparedDecoder
from ignition.recipes.kimi3.dcp import write_records
from ignition.recipes.kimi3.pointwise import rms_norm_columns


@triton.jit
def _rotate_yarn(Values, Positions, Frequencies, row_stride, head_stride,
                 OFFSET: tl.constexpr, SCALE: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    pair = tl.arange(0, 32)
    angle = tl.maximum(tl.load(Positions + row), 0).to(tl.float32) * tl.load(Frequencies + pair)
    # Full-range reduction matters at positions near one million; approximate sin.approx is not sufficient.
    cos, sin = libdevice.cos(angle) * SCALE, libdevice.sin(angle) * SCALE
    address = Values + row * row_stride + head * head_stride + OFFSET + 2 * pair
    first, second = tl.load(address).to(tl.float32), tl.load(address + 1).to(tl.float32)
    tl.store(address, first * cos - second * sin)
    tl.store(address + 1, second * cos + first * sin)


class YarnRotary:
    """DeepSeek YaRN frequencies with interleaved FP32 rotation and BF16 outputs."""

    def __init__(self, config, device):
        from transformers import PretrainedConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        c = config.draft.text
        rope = dict(c["rope_parameters"])
        # The serving reference selects DeepSeek YaRN, which derives magnitude from
        # mscale/mscale_all_dim and does not consume the legacy attn_factor field.
        rope.pop("attn_factor", None)
        canonical = PretrainedConfig(rope_parameters=rope, hidden_size=c["hidden_size"],
            num_attention_heads=c["num_attention_heads"], head_dim=c["qk_rope_head_dim"],
            max_position_embeddings=c["max_position_embeddings"])
        self.inverse, self.scale = ROPE_INIT_FUNCTIONS["yarn"](canonical, device=device)

    def apply(self, tensor, positions):
        if tensor.ndim not in (2, 3) or tensor.shape[-1] < 64 or tensor.stride(-1) != 1:
            raise ValueError("YaRN rotary inputs require contiguous 64-channel tails")
        torch.library.wrap_triton(_rotate_yarn)[(tensor.shape[0], tensor.shape[1] if tensor.ndim == 3 else 1)](
            tensor, positions, self.inverse, tensor.stride(0), tensor.stride(1) if tensor.ndim == 3 else 0,
            tensor.shape[-1] - 64, self.scale, num_warps=1, enable_fp_fusion=False)


def stage_latent_window(history, latent, output, metadata):
    """Materialize a private chronological window; rejected candidates never mutate accepted history."""
    m, window, rows = metadata, history.shape[1], metadata.lengths.numel()
    capacity = output.shape[0] * output.shape[1] // rows
    offsets = torch.arange(capacity, device=latent.device)[None]
    prefix = m.starts[:, None].clamp(max=window)
    positions = m.starts[:, None] - prefix + offsets
    previous = history[m.state_rows[:, None], positions.remainder(window)].to(torch.bfloat16)
    indices = (m.query_start_loc[:-1, None] + offsets - prefix).clamp(0, latent.shape[0] - 1).long()
    valid = (m.lengths[:, None] > 0) & (offsets < prefix + m.lengths[:, None])
    values = torch.where((offsets < prefix)[..., None], previous, latent[indices])
    output.view(rows, capacity, 576).copy_(torch.where(valid[..., None], values, 0).float().clamp(-448., 448.))


@dataclass
class DraftScratch:
    local_embedding: torch.Tensor
    embedding: torch.Tensor
    normalized: torch.Tensor
    output: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    gathered_logits: torch.Tensor
    logits: torch.Tensor
    convolution: dict
    attention: dict
    block_size: int
    retained: list = field(default_factory=list)
    selectors: dict = field(default_factory=dict)

    @classmethod
    def allocate(cls, config, shard, tokens, block, device, buffers):
        if tokens <= 0 or tokens % block:
            raise ValueError("MLA DFlash2 requires complete proposal blocks")
        cursor = buffers.cursor()
        h, vocab = config.hidden, config.vocabulary
        def zeros(columns, dtype=torch.bfloat16):
            return cursor.zeros((tokens, columns), dtype)
        return cls(zeros(h//shard.world_size), zeros(h), zeros(h), zeros(h),
            MLPScratch.allocate(tokens, h, config.draft.text["intermediate_size"]//shard.world_size, device, buffers=cursor),
            zeros(h), zeros(vocab//shard.world_size), zeros(vocab), zeros(vocab, torch.float32),
            dict(coefficients=zeros(4*h//16), before=zeros(h), after=zeros(h),
                 local_coefficients=zeros(4*h//16//shard.world_size)), {}, block)

    def rows(self, tokens, samples):
        if samples != tokens or not 0 < tokens <= self.embedding.shape[0] or tokens % self.block_size:
            raise ValueError("MLA DFlash2 scratch requires complete proposal blocks")
        return type(self)(*(getattr(self, name) if name in ("attention", "convolution", "block_size", "retained", "selectors")
            else self.mlp.rows(tokens) if name == "mlp" else getattr(self, name)[:tokens]
            for name in self.__dataclass_fields__))

    def scratch_tensors(self):
        return tuple(getattr(self, name) for name in self.__dataclass_fields__ if name not in ("selectors", "block_size"))


class PreparedContext:
    """Project only accepted target features and retain one 1024-token latent ring per draft layer."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace,
                 runtime, coordinator, latent_runtime, buffers):
        self.metadata, self.weights = metadata, weights
        c, shard = weights.config.draft, weights.shard
        rows, h = auxiliary.shape[0], weights.config.hidden
        self.window, self.world, self.eps = c.window, shard.world_size, c.text["rms_norm_eps"]
        cursor = buffers.cursor() if isinstance(buffers, GraphBufferPool) else buffers.fork()
        self.local = cursor.zeros((rows, h//shard.world_size), torch.bfloat16)
        self.projected = cursor.zeros((rows, h), torch.bfloat16)
        self.normalized = cursor.zeros((rows, h), torch.bfloat16)
        local_width = (c.layers*576//shard.world_size + 7)//8*8
        self.kv_local = cursor.zeros((rows, local_width), torch.bfloat16)
        self.kv = cursor.zeros((rows, local_width*shard.world_size), torch.bfloat16)
        self.latent = cursor.zeros((rows, 576), torch.bfloat16)
        self.slots = cursor.full((rows,), -1, torch.int64)
        self.project = prepare_projection(auxiliary, weights.draft.projection, self.local,
            session=session, name=name+".project", workspace=workspace, activation_mode="a16")
        self.gather = PreparedChunkedAllGather(self.local, self.projected, runtime=runtime,
            coordinator=coordinator, session=session, name=name+".gather")
        self.project_kv = prepare_projection(self.normalized, weights.draft.context_kv, self.kv_local,
            session=session, name=name+".kv", workspace=workspace, activation_mode="a16")
        self.gather_kv = PreparedChunkedAllGather(self.kv_local, self.kv, runtime=latent_runtime,
            coordinator=coordinator, session=session, name=name+".kv_gather")
        self.rotary = YarnRotary(weights.config, auxiliary.device)
        self.histories = tuple(pool.reserved_region(f"draft{i}.latent") for i in range(c.layers))

    def __call__(self, auxiliary, counts):
        m, c = self.metadata, self.weights.config.draft
        offset = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        live = (m.positions >= 0) & (offset >= 0) & (offset < count) & (offset >= (count-self.window).clamp_min(0))
        self.slots.copy_(torch.where(live, m.state_rows[m.requests]*self.window + m.positions.remainder(self.window), -1))
        self.gather(self.project(auxiliary))
        rms_norm_columns(self.projected, self.weights.draft.hidden_norm, self.normalized, self.eps)
        self.gather_kv(self.project_kv(self.normalized))
        # Each rank contributes the same channel slice of every layer's latent projection.
        kv = self.kv.view(-1, self.world, self.kv_local.shape[1])[:, :, :c.layers*576//self.world]
        kv = kv.view(-1, self.world, c.layers, 576//self.world)
        for index, layer in enumerate(self.weights.draft.layers):
            self.latent.view(-1, self.world, 576//self.world).copy_(kv[:, :, index])
            rms_norm_columns(self.latent[:, :512], layer.attention["kv_norm"], self.latent[:, :512], self.eps)
            self.rotary.apply(self.latent, m.positions)
            write_records(self.latent, self.slots, self.histories[index])

    def scratch_tensors(self):
        return (self.local, self.projected, self.normalized, self.kv_local, self.kv, self.latent, self.slots)


class DraftAttention:
    def __init__(self, index, weights, source, config, shard, scratch, metadata, pool, *,
                 runtime, coordinator, session, name, workspace, buffers, latent_runtime):
        self.metadata, self.weights = metadata, weights.attention
        self.window, self.world, self.eps = config.window, shard.world_size, config.text["rms_norm_eps"]
        n, h = source.shape
        heads, width = config.heads//shard.world_size, (2112//shard.world_size + 7)//8*8
        cursor = buffers.fork()
        def storage(key, shape, dtype=torch.bfloat16):
            if key not in scratch.attention:
                capacity = scratch.convolution["before"].shape[0]
                scratch.attention[key] = cursor.zeros((shape[0]*capacity//n, *shape[1:]), dtype)
            value = scratch.attention[key]
            if value.shape[0] < shape[0] or value.shape[1:] != shape[1:] or value.dtype != dtype:
                raise ValueError("MLA draft attention scratch must be prepared at one complete proposal capacity")
            return value[:shape[0]]
        self.local = storage("local", (n, width))
        self.gathered = storage("gathered", (n, width*shard.world_size))
        self.q_latent = storage("q_latent", (n, 1536))
        self.q_norm = storage("q_norm", (n, 1536))
        self.latent = storage("latent", (n, 576))
        self.query = storage("query", (n, heads, 192))
        self.absorbed = storage("absorbed", (n, heads, 576))
        self.attended = storage("attended", (n, heads, 512))
        self.expanded = storage("expanded", (n, heads*128))
        self.partial = storage("partial", (n, h))
        pages = (config.window + scratch.block_size + 127)//128
        rows = metadata.lengths.numel()
        self.staged = storage("staged", (rows*pages, 128, 576), torch.float8_e4m3fn)
        self.history = pool.reserved_region(f"draft{index}.latent")
        self.page_table = torch.arange(rows*pages, dtype=torch.int32, device=source.device).view(rows, pages)
        self.lengths = storage("lengths", (rows,), torch.int32)
        self.project = prepare_projection(source, self.weights["latent_gate"], self.local,
            session=session, name=name+".qkv", workspace=workspace, activation_mode="a16")
        self.gather = PreparedChunkedAllGather(self.local, self.gathered, runtime=latent_runtime,
            session=session, coordinator=coordinator, name=name+".qkv_gather")
        self.query_project = prepare_projection(self.q_norm, self.weights["q_b"], self.query.view(n, -1),
            session=session, name=name+".query", workspace=workspace, activation_mode="a16")
        self.absorb = BatchedUnquantizedProjection(self.query[:, :, :128].transpose(0, 1),
            self.weights["key"], self.absorbed[:, :, :512].transpose(0, 1),
            major="n", session=session, name=name+".absorb")
        self.expand = BatchedUnquantizedProjection(self.attended.transpose(0, 1), self.weights["value"],
            self.expanded.view(n, heads, 128).transpose(0, 1), major="n", session=session, name=name+".expand")
        self.rotary = YarnRotary(SimpleNamespace(draft=config), source.device)
        caps = dense_mla.Caps(mode="verify", device=source.device, q_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn, num_q_heads=heads, max_total_q=n, max_batch=rows,
            max_cache_tokens=pages*128, max_page_table_width=pages, num_cache_pages=rows*pages,
            page_size=128, use_cuda_graph=True, partial_dtype=torch.float32, window_size=config.window, causal=True)
        self.attention = PreparedDenseMLA(self.absorbed, self.staged, self.attended, self.page_table,
            self.lengths, metadata.query_start_loc, caps=caps, scale=config.attention_scale,
            session=session, name=name+".attention", workspace=workspace)
        self.project_output = prepare_projection(self.expanded, self.weights["output"], self.partial,
            session=session, name=name+".output", workspace=workspace, activation_mode="a16")
        self.reduce = PreparedAllReduce(self.partial, scratch.output, runtime=runtime,
            coordinator=coordinator, session=session, name=name+".reduce")

    def __call__(self, source):
        m, w = self.metadata, self.weights
        gathered = self.gather(self.project(source)).view(source.shape[0], self.world, -1)
        q = 1536//self.world
        self.q_latent.view(-1, self.world, q).copy_(gathered[:, :, :q])
        self.latent.view(-1, self.world, 576//self.world).copy_(gathered[:, :, q:q+576//self.world])
        rms_norm_columns(self.q_latent, w["query_norm"], self.q_norm, self.eps)
        rms_norm_columns(self.latent[:, :512], w["kv_norm"], self.latent[:, :512], self.eps)
        self.query_project(self.q_norm)
        self.rotary.apply(self.query, m.positions)
        self.rotary.apply(self.latent, m.positions)
        self.absorb(self.query[:, :, :128].transpose(0, 1))
        self.absorbed[:, :, 512:].copy_(self.query[:, :, 128:])
        stage_latent_window(self.history, self.latent, self.staged, m)
        self.lengths.copy_(torch.where(m.lengths > 0, m.starts.clamp(max=self.window)+m.lengths, 0).int())
        self.attention(self.absorbed, self.attended)
        self.expand(self.attended.transpose(0, 1))
        return self.reduce(self.project_output(self.expanded))


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, scratch, *, completion, frame, sequence, session, workspace,
                 collectives, decoder_buffers, draft_tokens, context_runtime, page_tokens,
                 latent_runtime, context_latent_runtime, convolution_runtime, context_buffers=None):
        self.page_tokens = page_tokens
        self.weights, self.tables, self.pool = weights, None, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        coordinator = collectives["coordinator"]
        self.collectives = {key: collectives[key] for key in ("runtime", "vocabulary_runtime", "coordinator")}
        self.markov_runtime = None
        layers = {i: w["attention"] for i, w in enumerate(weights.layers) if weights.config.layers[i] == "kda"}
        heads = weights.config.text["linear_attn_config"]["num_heads"]//weights.shard.world_size
        self.kda_records = KDARecords.allocate(layers, requests.capacity+1, heads, pool.backing.device)
        def decoder(weights, tables, scratch, metadata, pool, *, mode, **kwargs):
            return PreparedDecoder(weights, scratch, metadata, pool, mode=mode, buffers=decoder_buffers,
                kda_records=self.kda_records if mode == "verify" else None, **{**collectives, **kwargs})
        self.block_size, self.logit_offset = draft_tokens+1, 1
        self.context_buffers = (GraphBufferPool(pool.backing.device, chunk_bytes=2 << 20)
                                if context_buffers is None else context_buffers)
        self.draft_buffers = GraphBufferPool(pool.backing.device, chunk_bytes=2 << 20)
        self.draft_scratch = DraftScratch.allocate(weights.config, weights.shard,
            requests.capacity*self.block_size, self.block_size, pool.backing.device, self.draft_buffers)
        # Attention scratch is separate from activations, logits and convolution inputs that remain live.
        self.attention_buffers = GraphBufferPool(pool.backing.device, chunk_bytes=2 << 20)
        self.forwards = SimpleNamespace(decoder=decoder,
            context=partial(PreparedContext, runtime=context_runtime, coordinator=coordinator,
                latent_runtime=context_latent_runtime, buffers=self.context_buffers),
            draft=partial(BlockDraft, embedding=weights.embedding,
                head=weights.head if weights.draft.head is None else weights.draft.head,
                embedding_runtime=collectives["embedding_runtime"], output_multiplier=1., final_logit_softcapping=None,
                buffers=self.attention_buffers.cursor(), attention_factory=partial(DraftAttention, latent_runtime=latent_runtime),
                convolution_runtime=convolution_runtime), markov=None)
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
