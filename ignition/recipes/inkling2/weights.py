# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Role, precision and tensor-parallel placement of every Inkling 2 checkpoint tensor.

`checkpoint_layout` derives, from the validated configuration alone, each tensor's
semantic role, stored shape and dtype, and the extent each rank reads:

- Query, relative-query, output-gate and sink projections and the query and
  relative clip scales split by query head; key and value projections, the K/V
  short convolutions and the global layers' dynamic-convolution projection split
  by KV head. The attention output projection splits its input columns.
- Each expert's `w13` rows interleave gate and up rows (row `2i` gates intermediate
  `i`, row `2i + 1` is its up projection), so a contiguous block of `2 * I / world`
  rows holds a rank's gate/up pairs; `w2` splits its input columns the same way.
  Block scales follow their values; per-expert amax and the outer-scale
  denominator are replicated.
- The embedding and unembedding split their vocabulary rows into equal
  64-aligned widths; rows past the table are zero on the last rank.
- Router, normalization, output-convolution, attention-residual query, vision and
  Markov-head tensors are replicated. Drafter attention and MLP split like the target's.

Placements keep the stored dtypes that the precision groups of
`hf_quant_config.json` declare; no tensor is converted or requantized.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.persistent import persistent_empty, persistent_zeros

from .config import DRAFT_FILE, EXPERT_ALIGNMENT, TEXT_PREFIX, VISION_PREFIX, VOCABULARY_ALIGNMENT, ModelConfig

DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
_GROUP_DTYPES = {"float32": "F32", "bfloat16": "BF16", "bfloat16_float32_output": "BF16"}


@dataclass(frozen=True)
class Placement:
    """One checkpoint tensor's role and the extent of it that each rank holds.

    `axis` is the partitioned dimension, or None when every rank holds the whole
    tensor. Each rank holds `width` consecutive elements of that axis starting at
    `rank * width`; indices past the stored extent are zero-filled.
    """

    name: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    precision: str
    layer: int | None = None
    axis: int | None = None
    width: int = 0

    def __post_init__(self):
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"{self.name}: unsupported stored dtype {self.dtype}")
        if self.axis is not None and not (0 <= self.axis < len(self.shape) and self.width > 0):
            raise ValueError(f"{self.name}: invalid partition")

    def rows(self, rank: int) -> tuple[int, int]:
        """Stored indices along `axis` that `rank` reads; the remainder of its width is zero."""
        if self.axis is None:
            raise ValueError(f"{self.name} is replicated")
        first = rank * self.width
        return min(first, self.shape[self.axis]), min(first + self.width, self.shape[self.axis])

    def extent(self, rank: int) -> tuple[slice, ...]:
        """Index of the stored tensor that `rank` reads."""
        if self.axis is None:
            return tuple(slice(None) for _ in self.shape)
        return tuple(slice(*self.rows(rank)) if axis == self.axis else slice(None) for axis in range(len(self.shape)))

    @property
    def local_shape(self) -> tuple[int, ...]:
        if self.axis is None:
            return self.shape
        return tuple(self.width if axis == self.axis else size for axis, size in enumerate(self.shape))

    @property
    def stored_bytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]

    @property
    def rank_bytes(self) -> int:
        return math.prod(self.local_shape) * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class CheckpointLayout:
    """Placements of the indexed model tensors and of the drafter file's tensors."""

    world_size: int
    model: Mapping[str, Placement]
    draft: Mapping[str, Placement]

    def layer(self, layer: int) -> dict[str, Placement]:
        """Placements a decoder layer consumes, keyed by role, including its residual queries."""
        result = {}
        for placement in self.model.values():
            if placement.layer == layer and not placement.role.startswith("vision_"):
                if placement.role in result:
                    raise ValueError(f"layer {layer} assigns role {placement.role} twice")
                result[placement.role] = placement
        return result

    def validate(self, tensors: Mapping[str, tuple[tuple[int, ...], str]],
                 draft: Mapping[str, tuple[tuple[int, ...], str]] | None = None):
        """Require exactly the mapped names with their mapped shapes and dtypes."""
        for label, placements, stored in (("model", self.model, tensors), (DRAFT_FILE, self.draft, draft)):
            if stored is None:
                continue
            missing, unmapped = placements.keys() - stored.keys(), stored.keys() - placements.keys()
            if missing or unmapped:
                raise ValueError(f"{label} tensors differ from the layout: missing {sorted(missing)[:4]}, "
                                 f"unmapped {sorted(unmapped)[:4]}")
            for name, (shape, dtype) in stored.items():
                placement = placements[name]
                if tuple(shape) != placement.shape or dtype != placement.dtype:
                    raise ValueError(f"{name}: stored {dtype} {tuple(shape)}, layout requires "
                                     f"{placement.dtype} {placement.shape}")

    def rank_bytes(self, *, draft: bool = True, vision: bool = True) -> int:
        """Device bytes of one rank's weights. Every rank holds the same number of bytes."""
        placements = list(self.model.values()) + (list(self.draft.values()) if draft else [])
        return sum(p.rank_bytes for p in placements if vision or not p.role.startswith("vision_"))


def checkpoint_layout(config: ModelConfig, world_size: int) -> CheckpointLayout:
    """Placements of every tensor of the checkpoint `config` describes, at `world_size` ranks."""
    builder = _Builder(config, world_size)
    builder.text()
    builder.vision()
    model = builder.finish(config.quantization.formats, "model")
    builder = _Builder(config, world_size)
    builder.drafter()
    draft = builder.finish(config.quantization.draft_formats, DRAFT_FILE)
    return CheckpointLayout(world_size, model, draft)


class _Builder:
    def __init__(self, config, world_size):
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world size must be a positive integer")
        self.config, self.world = config, world_size
        self.placements: dict[str, Placement] = {}
        # Scale tensors absent from the precision groups, mapped to the listed weight whose group they follow.
        self.companions: dict[str, str] = {}

    def add(self, name, role, shape, dtype, precision, *, layer=None, axis=None, unit=1, padded=False,
            listed=None):
        if name in self.placements:
            raise ValueError(f"{name} is mapped twice")
        width = 0
        if axis is not None:
            extent = shape[axis]
            if padded:
                width = -(-extent // (self.world * unit)) * unit
            else:
                if extent % (self.world * unit):
                    raise ValueError(f"{name}: axis {axis} of {extent} does not split into {self.world} "
                                     f"rank extents aligned to {unit}")
                width = extent // self.world
            if self.world == 1:
                axis, width = None, 0
        self.placements[name] = Placement(name, role, tuple(shape), dtype, precision, layer, axis, width)
        if listed is not None:
            self.companions[name] = listed

    def plain(self, name, role, shape, *, layer=None, axis=None, unit=1, padded=False, precision="bfloat16"):
        self.add(name, role, shape, _GROUP_DTYPES[precision], precision, layer=layer, axis=axis, unit=unit,
                 padded=padded)

    def text(self):
        t, p = self.config.text, TEXT_PREFIX
        hidden = t.hidden
        self.plain(f"{p}.embed.weight", "embedding", (t.vocabulary, hidden), axis=0,
                   unit=VOCABULARY_ALIGNMENT, padded=True)
        self.plain(f"{p}.embed_norm.weight", "embedding_norm", (hidden,))
        self.plain(f"{p}.norm.weight", "final_norm", (hidden,))
        self.plain(f"{p}.unembed.weight", "unembedding", (t.vocabulary, hidden), axis=0,
                   unit=VOCABULARY_ALIGNMENT, padded=True, precision="bfloat16_float32_output")
        self.plain(f"{p}.attn_res.output_query", "output_query", (hidden,), precision="float32")
        for layer in range(t.layers):
            if layer:
                shape = (hidden,) if t.local(layer) else (3, hidden)
                self.plain(f"{p}.attn_res.attn_queries.{layer - 1}", "attention_query", shape, layer=layer,
                           precision="float32")
            self.plain(f"{p}.attn_res.mlp_queries.{layer}", "mlp_query", (hidden,), layer=layer, precision="float32")
            self.decoder(layer)

    def decoder(self, layer):
        t = self.config.text
        prefix, attention = f"{TEXT_PREFIX}.layers.{layer}", f"{TEXT_PREFIX}.layers.{layer}.attn"
        hidden, heads, kv, dim, width = t.hidden, t.heads, t.kv_heads, t.head_dim, t.convolution_width
        by_head = dict(layer=layer, axis=0, unit=dim)
        self.plain(f"{attention}.wq_du.weight", "query", (heads * dim, hidden), **by_head)
        self.plain(f"{attention}.gate_attn.weight", "output_gate", (heads * dim, hidden), **by_head)
        self.plain(f"{attention}.wr_du.weight", "relative_query", (heads * t.relative_width, hidden),
                   layer=layer, axis=0, unit=t.relative_width)
        self.plain(f"{attention}.attn_sink_proj.weight", "sink", (heads, hidden), layer=layer, axis=0)
        self.plain(f"{attention}.q_l2_clip.scale", "query_clip_scale", (heads, dim), layer=layer, axis=0)
        self.plain(f"{attention}.rel_l2_clip.scale", "relative_clip_scale", (heads, t.relative_width),
                   layer=layer, axis=0)
        self.plain(f"{attention}.rel_logits_proj.proj", "relative_projection",
                   (t.relative_width, t.extent(layer)), layer=layer)
        self.plain(f"{attention}.wk_dv.weight", "key", (kv * dim, hidden), **by_head)
        self.plain(f"{attention}.wv_dv.weight", "value", (kv * dim, hidden), **by_head)
        self.plain(f"{attention}.wo_ud.weight", "output", (hidden, heads * dim), layer=layer, axis=1, unit=dim)
        for stream in ("k", "v"):
            role = "key" if stream == "k" else "value"
            convolution = f"{attention}.{stream}_sconv"
            if t.local(layer):
                # Depthwise taps ordered oldest first; the last tap multiplies the current token.
                self.plain(f"{convolution}.weight", f"{role}_convolution", (kv * dim, 1, width), **by_head)
            else:
                # Static taps ordered by lag (lag 0 is the current token) and per-token lag weights
                # projected from the normalized stream, row `head * width + lag`.
                self.plain(f"{convolution}.static_w", f"{role}_convolution_static", (width, kv * dim),
                           layer=layer, axis=1, unit=dim)
                self.plain(f"{convolution}.weight_proj", f"{role}_convolution_dynamic", (kv * width, hidden),
                           layer=layer, axis=0, unit=width)
        norms = ("attn_norm",) if t.local(layer) else ("q_input_norm", "k_input_norm", "v_input_norm")
        roles = {"attn_norm": "attention_norm", "q_input_norm": "query_norm", "k_input_norm": "key_norm",
                 "v_input_norm": "value_norm"}
        for norm in (*norms, "mlp_norm"):
            self.plain(f"{prefix}.{norm}.weight", roles.get(norm, norm), (hidden,), layer=layer)
        self.plain(f"{prefix}.attn_sconv.weight", "attention_convolution", (hidden, 1, width), layer=layer)
        self.plain(f"{prefix}.mlp_sconv.weight", "mlp_convolution", (hidden, 1, width), layer=layer)
        self.plain(f"{prefix}.mlp.gate.weight", "router", (t.experts, hidden), layer=layer)
        self.plain(f"{prefix}.mlp.gate.bias", "router_bias", (t.experts,), layer=layer, precision="float32")
        self.experts(layer)

    def experts(self, layer):
        t = self.config.text
        precision = self.config.quantization.expert_format(layer)
        prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts"
        experts, hidden, intermediate = t.experts, t.hidden, t.intermediate
        if precision == "nvfp4":
            values, scales, block, packing = "U8", "F8_E4M3", 16, 2
        else:
            values, scales, block, packing = "F8_E4M3", "U8", 32, 1
        # w13 interleaves gate and up rows, so whole pairs keep each rank's rows contiguous.
        rows = dict(layer=layer, axis=1, unit=2 * EXPERT_ALIGNMENT)
        self.add(f"{prefix}.w13_weight", "w13", (experts, 2 * intermediate, hidden // packing), values, precision,
                 **rows)
        self.add(f"{prefix}.w13_weight_scale", "w13_scale", (experts, 2 * intermediate, hidden // block), scales,
                 precision, listed=f"{prefix}.w13_weight", **rows)
        self.add(f"{prefix}.w2_weight", "w2", (experts, hidden, intermediate // packing), values, precision,
                 layer=layer, axis=2, unit=EXPERT_ALIGNMENT // packing)
        self.add(f"{prefix}.w2_weight_scale", "w2_scale", (experts, hidden, intermediate // block), scales,
                 precision, layer=layer, axis=2, unit=EXPERT_ALIGNMENT // block, listed=f"{prefix}.w2_weight")
        if precision == "nvfp4":
            for name in ("w13", "w2"):
                listed = f"{prefix}.{name}_weight"
                self.add(f"{listed}_amax", f"{name}_amax", (experts,), "F32", precision, layer=layer, listed=listed)
                self.add(f"{listed}_outer_scale_denominator", f"{name}_denominator", (1,), "F32", precision,
                         layer=layer, listed=listed)

    def vision(self):
        v, p = self.config.vision, VISION_PREFIX
        hidden, patch = v.hidden, v.patch
        backbone = f"{p}.backbone"
        self.plain(f"{backbone}.embeddings.patch_embedding.weight", "vision_patch_embedding",
                   (hidden, 3 * patch * patch))
        self.plain(f"{backbone}.embeddings.position_embedding", "vision_position_embedding",
                   (v.position_grid * v.position_grid, hidden))
        self.plain(f"{backbone}.embeddings.class_embedding", "vision_class_embedding", (hidden,))
        self.plain(f"{backbone}.embeddings.cls_position_embedding", "vision_class_position", (hidden,))
        for suffix in ("weight", "bias"):
            self.plain(f"{backbone}.ln_pre.{suffix}", f"vision_pre_norm_{suffix}", (hidden,))
        for layer in range(v.layers):
            block = f"{backbone}.layers.{layer}"
            for norm in ("ln_1", "ln_2"):
                for suffix in ("weight", "bias"):
                    self.plain(f"{block}.{norm}.{suffix}", f"vision_{norm}_{suffix}", (hidden,), layer=layer)
            for scale in ("ls_1", "ls_2"):
                self.plain(f"{block}.{scale}.gamma", f"vision_{scale}", (hidden,), layer=layer)
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                self.plain(f"{block}.self_attn.{projection}.weight", f"vision_{projection}", (hidden, hidden),
                           layer=layer)
                self.plain(f"{block}.self_attn.{projection}.bias", f"vision_{projection}_bias", (hidden,),
                           layer=layer)
            self.plain(f"{block}.mlp.c_fc.weight", "vision_fc", (v.mlp_hidden, hidden), layer=layer)
            self.plain(f"{block}.mlp.c_fc.bias", "vision_fc_bias", (v.mlp_hidden,), layer=layer)
            self.plain(f"{block}.mlp.c_proj.weight", "vision_fc_output", (hidden, v.mlp_hidden), layer=layer)
            self.plain(f"{block}.mlp.c_proj.bias", "vision_fc_output_bias", (hidden,), layer=layer)
        self.plain(f"{p}.proj_in.weight", "vision_projector_in", (v.projection_hidden, hidden * v.merge * v.merge))
        self.plain(f"{p}.proj_out.weight", "vision_projector_out", (self.config.text.hidden, v.projection_hidden))

    def drafter(self):
        d, t = self.config.draft, self.config.text
        hidden, heads, kv, dim = d.hidden, d.heads, d.kv_heads, d.head_dim
        by_head = dict(axis=0, unit=dim)
        self.plain("fuse_weight", "draft_tap_weights", (d.layers + 1, len(d.taps)))
        self.plain("mask_embed", "draft_mask_embedding", (hidden,))
        self.plain("norm.weight", "draft_final_norm", (hidden,))
        for layer in range(d.layers):
            p, a = f"layers.{layer}", f"layers.{layer}.self_attn"
            for name, role in (("input_layernorm", "draft_input_norm"),
                               ("post_attention_layernorm", "draft_post_attention_norm"),
                               ("dflare_norm", "draft_context_norm")):
                self.plain(f"{p}.{name}.weight", role, (hidden,), layer=layer)
            for name, role in (("q_norm", "draft_query_norm"), ("k_norm", "draft_key_norm"),
                               ("k_norm_ctx", "draft_context_key_norm")):
                self.plain(f"{a}.{name}.weight", role, (dim,), layer=layer)
            self.plain(f"{a}.q_proj.weight", "draft_query", (heads * dim, hidden), layer=layer, **by_head)
            for name, role in (("k_proj", "draft_key"), ("v_proj", "draft_value"),
                               ("k_proj_ctx", "draft_context_key"), ("v_proj_ctx", "draft_context_value")):
                self.plain(f"{a}.{name}.weight", role, (kv * dim, hidden), layer=layer, **by_head)
            self.plain(f"{a}.o_proj.weight", "draft_output", (hidden, heads * dim), layer=layer, axis=1, unit=dim)
            for name, role in (("gate_proj", "draft_gate"), ("up_proj", "draft_up")):
                self.plain(f"{p}.mlp.{name}.weight", role, (d.intermediate, hidden), layer=layer, axis=0,
                           unit=EXPERT_ALIGNMENT)
            self.plain(f"{p}.mlp.down_proj.weight", "draft_down", (hidden, d.intermediate), layer=layer, axis=1,
                       unit=EXPERT_ALIGNMENT)
        rank, markov = d.markov_rank, "markov_head"
        # The recurrent cell reads [state, previous-token embedding, draft hidden, fused taps].
        self.plain(f"{markov}.markov_w1.weight", "markov_embedding", (t.vocabulary, rank))
        self.plain(f"{markov}.rnn_proj.weight", "markov_recurrent", (3 * rank, 2 * rank + 2 * hidden))
        self.plain(f"{markov}.rnn_proj.bias", "markov_recurrent_bias", (3 * rank,))
        self.plain(f"{markov}.hidden_out.weight", "markov_output", (hidden, rank))
        self.plain(f"{markov}.hidden_out.bias", "markov_output_bias", (hidden,))
        self.plain(f"{markov}.dflare_norm.weight", "markov_context_norm", (hidden,))

    def finish(self, formats: Mapping[str, str], label: str) -> Mapping[str, Placement]:
        """Check each placement's precision against the checkpoint's precision groups."""
        for name, placement in self.placements.items():
            declared = formats.get(self.companions.get(name, name))
            if declared != placement.precision:
                raise ValueError(f"{name}: {label} precision groups declare {declared}, the layout requires "
                                 f"{placement.precision}")
        listed = {self.companions.get(name, name) for name in self.placements}
        extra = formats.keys() - listed
        if extra:
            raise ValueError(f"{label} precision groups list unmapped tensors: {sorted(extra)[:4]}")
        return MappingProxyType(dict(self.placements))


def stored_tensors(entries: Iterable) -> dict[str, tuple[tuple[int, ...], str]]:
    """Shapes and dtypes of `TensorMetadata` header entries, for `CheckpointLayout.validate`."""
    return {entry.name: (tuple(entry.shape), entry.dtype) for entry in entries}


_TORCH_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8}


class PlacedWeightLoader(DirectWeightLoader):
    """Direct checkpoint reads of the indexed tensors `names`, as placed by `layout`.

    Within the loader's context, `load(name, rank=r)` copies rank `r`'s extent of a
    stored tensor to the device without converting its bytes, zero-filling the
    vocabulary tail past the table; `rank=None` copies the whole stored tensor.
    """

    def __init__(self, manifest, device, *, layout: CheckpointLayout, names: Iterable[str], progress=None):
        selected = frozenset(names)
        unmapped = selected - layout.model.keys()
        if unmapped:
            raise ValueError(f"tensors outside the layout: {sorted(unmapped)[:4]}")
        super().__init__(manifest, device, progress=progress)
        self.layout, self.names = layout, selected

    def _selected_names(self):
        return set(self.names)

    def load(self, name: str, *, rank: int | None = None) -> torch.Tensor:
        self._require_open()
        placement, source = self.layout.model[name], self._sources[name]
        if tuple(source.shape) != placement.shape or source.dtype != _TORCH_DTYPES[placement.dtype]:
            raise ValueError(f"{name}: stored {source.dtype} {tuple(source.shape)} differs from the layout")
        if rank is None or placement.axis is None:
            return self.load_tensor(name)
        first, last = placement.rows(rank)
        allocate = persistent_empty if last - first == placement.width else persistent_zeros
        target = allocate(placement.local_shape, dtype=source.dtype, device=self.device)
        if last > first:
            self._copy(target.narrow(placement.axis, 0, last - first), source[placement.extent(rank)])
        self._session.flush()
        return target
