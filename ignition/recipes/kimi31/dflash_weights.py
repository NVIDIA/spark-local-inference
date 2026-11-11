# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel MLA DFlash2 weights with explicit online MXFP8 conversion."""

from dataclasses import dataclass
from functools import partial

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights, quantize_bf16_linear_as_mxfp8
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.persistent import persistent_cat, persistent_empty


@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights
    attention_conv: dict
    mlp_conv: dict


@dataclass(frozen=True)
class DraftWeights:
    projection: object
    hidden_norm: torch.Tensor
    context_kv: object
    norm: torch.Tensor
    layers: tuple[LayerWeights, ...]
    selector: dict
    head: object = None

    @classmethod
    def load(cls, snapshot, config, device, shard, *, target_head, progress=None,
             online_mxfp8=False, draft_head_quantization=None):
        manifest = inspect_indexed_checkpoint(snapshot)
        validate_metadata(manifest, config.draft)
        d, h = config.draft, config.hidden
        d.validate_shard(shard)
        h0, h1 = shard.partition(d.heads)
        q_range, kv_range = shard.partition(1536), shard.partition(576)
        intermediate = shard.partition(d.text["intermediate_size"], alignment=32)
        conv_range = shard.partition(4 * h // 16, alignment=8)
        pack = (partial(quantize_bf16_linear_as_mxfp8, retain_scale_rows=False)
                if online_mxfp8 else lambda weight: weight)
        cat = torch.cat if online_mxfp8 else persistent_cat
        with DirectWeightLoader(manifest, device, progress=progress, progress_label="Load draft weights") as loader:
            def matrix(name, rows=None, columns=None, *, temporary=False):
                return loader.load_tensor(name + ".weight", extent=(slice(*(rows or (None, None))),
                    slice(*(columns or (None, None)))), temporary=temporary)
            project = pack(matrix("fc", rows=shard.partition(h, alignment=64), temporary=online_mxfp8))
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            layers, context = [], []
            for i in range(d.layers):
                p = f"layers.{i}"
                a = p + ".self_attn"
                qa = matrix(a + ".q_a_proj", rows=q_range, temporary=True)
                ka = matrix(a + ".kv_a_proj_with_mqa", rows=kv_range, temporary=True)
                context.append(ka)
                fused = cat((qa, ka, qa.new_zeros((-(qa.shape[0] + ka.shape[0]) % 8, h))))
                kv = matrix(a + ".kv_b_proj", rows=(h0 * 256, h1 * 256), temporary=True).view(h1 - h0, 256, 512)
                attention = dict(latent_gate=pack(fused), q_b=pack(matrix(a + ".q_b_proj", rows=(h0*192, h1*192), temporary=online_mxfp8)),
                    key=persistent_empty((h1 - h0, 128, 512), dtype=kv.dtype, device=kv.device).copy_(kv[:, :128]),
                    value=persistent_empty((h1 - h0, 512, 128), dtype=kv.dtype, device=kv.device).copy_(kv[:, 128:].transpose(1, 2)),
                    query_norm=loader.load_tensor(a + ".q_a_layernorm.weight"),
                    kv_norm=loader.load_tensor(a + ".kv_a_layernorm.weight"),
                    output=pack(matrix(a + ".o_proj", columns=(h0*128, h1*128), temporary=online_mxfp8)))
                mlp = MLPWeights(pack(cat([matrix(p + ".mlp." + n, rows=intermediate, temporary=True)
                    for n in ("gate_proj", "up_proj")])), pack(matrix(p + ".mlp.down_proj", columns=intermediate, temporary=online_mxfp8)))
                def conv(kind):
                    return dict(base=loader.load_tensor(p + "." + kind + ".base_kernel"),
                        projection=matrix(p + "." + kind + ".kernel_projection", rows=conv_range))
                layers.append(LayerWeights(attention, loader.load_tensor(p + ".input_layernorm.weight"),
                    loader.load_tensor(p + ".post_attention_layernorm.weight"), mlp,
                    conv("attention_conv"), conv("mlp_conv")))
            values = loader.load_tensors(("candidate_selector.hidden_projection.weight",
                "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"))
        head = None
        if draft_head_quantization is not None:
            from ignition.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
            head = {"mxfp8": mxfp8_draft_head, "nvfp4": nvfp4_draft_head}[draft_head_quantization](target_head)
        context = cat((*context, context[0].new_zeros((-sum(weight.shape[0] for weight in context) % 8, h))))
        return cls(project, hidden_norm, pack(context), norm, tuple(layers),
                   dict(zip(("projection", "predecessor", "successor"), values, strict=True)), head)


def validate_metadata(manifest, config):
    c, d = config.text, config.text["dflash_config"]
    h, i, heads = c["hidden_size"], c["intermediate_size"], c["num_attention_heads"]
    expected = {"fc.weight": (h, h * len(config.auxiliary_layers)), "hidden_norm.weight": (h,), "norm.weight": (h,),
        "candidate_selector.hidden_projection.weight": (d["selector_rank"], h),
        "candidate_selector.predecessor_codebook": (c["vocab_size"], d["selector_rank"]),
        "candidate_selector.successor_codebook": (c["vocab_size"], d["selector_rank"])}
    for layer in range(config.layers):
        p = f"layers.{layer}."
        for name, shape in {"self_attn.q_a_proj": (1536, h), "self_attn.kv_a_proj_with_mqa": (576, h),
                "self_attn.q_b_proj": (heads*192, 1536), "self_attn.kv_b_proj": (heads*256, 512),
                "self_attn.q_a_layernorm": (1536,), "self_attn.kv_a_layernorm": (512,),
                "self_attn.o_proj": (h, heads*128), "input_layernorm": (h,), "post_attention_layernorm": (h,),
                "mlp.gate_proj": (i, h), "mlp.up_proj": (i, h), "mlp.down_proj": (h, i)}.items():
            expected[p + name + ".weight"] = shape
        for kind in ("attention_conv", "mlp_conv"):
            expected[p + kind + ".base_kernel"] = (2, 2, h)
            expected[p + kind + ".kernel_projection.weight"] = (4 * h // 16, h)
    if set(manifest.tensors) != set(expected):
        raise ValueError("MLA DFlash2 checkpoint tensor names disagree with its architecture")
    for name, shape in expected.items():
        if manifest.tensors[name].shape != shape or manifest.tensors[name].dtype != "BF16":
            raise ValueError(f"{name}: MLA DFlash2 requires BF16 storage with shape {shape}")
