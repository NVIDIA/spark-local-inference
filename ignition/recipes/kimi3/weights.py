# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 direct transfers with sharded attention, latent projections, and experts."""

from dataclasses import dataclass, replace
import sys

import torch

from ignition.components.weights import (DirectWeightLoader, SharedColumnReads, fp4_down_staging_bytes,
                                         quantize_bf16_linear_as_mxfp8)

from .config import DSparkConfig, ModelConfig
from .residual import score_weight

PREFIX = "language_model.model"
# Routed experts keep BF16 activations; MXFP4 weights expand in the expert kernels.
EXPERT_ACTIVATION = "a16"


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    output_res_score: torch.Tensor
    layers: tuple[dict, ...]
    scale_scratch: tuple[torch.Tensor, ...]
    allocated_bytes: int
    vision: object = None
    draft: object = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, draft_checkpoint=None, online_mxfp8=False,
             peers=None, vision=True):
        """``peers``, a host's Gloo group and peer storage, reads column-sharded projections once per host.

        ``draft_checkpoint`` loads the target model's DSpark checkpoint.
        Without ``vision``, the vision encoder stays on disk.
        """
        before = torch.cuda.memory_allocated(device)
        with KimiWeightLoader(manifest, device, shard=shard, progress=progress, online_mxfp8=online_mxfp8,
                              peers=peers) as loader:
            config = loader.config
            embedding = loader.linear(f"{PREFIX}.embed_tokens", columns=shard.partition(config.hidden, alignment=64))
            norm, res_norm, res_proj = loader.load_tensors(tuple(f"{PREFIX}.{n}.weight" for n in
                ("norm", "output_attn_res_norm", "output_attn_res_proj")))
            head = loader.load_tensor("language_model.lm_head.weight",
                extent=slice(*shard.partition(config.vocabulary, alignment=64)))
            layers = tuple(loader.load_layer(i) for i in range(len(config.layers)))
        if vision:
            from .vision import VisionWeights
            vision = VisionWeights.load(manifest, device, progress=progress)
        else:
            vision = None
        draft = None
        if draft_checkpoint is not None:
            if config.dsa is None:
                from .dspark_weights import DraftWeights

                draft_config, draft_args = DSparkConfig, {}
            else:
                from ignition.recipes.kimi31.dspark_config import DSparkConfig as draft_config
                from ignition.recipes.kimi31.dspark_weights import DraftWeights

                draft_args = dict(target_embedding=embedding)
            config = replace(config, draft=draft_config.load(draft_checkpoint, config))
            draft = DraftWeights.load(draft_checkpoint, config, device, shard, target_head=head, progress=progress,
                                      online_mxfp8=online_mxfp8, **draft_args)
        return cls(config, shard, embedding, norm, head, score_weight(res_norm, res_proj), layers,
                   loader.scale_scratch, torch.cuda.memory_allocated(device) - before, vision=vision, draft=draft)


class KimiWeightLoader(DirectWeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None, online_mxfp8=False, peers=None):
        super().__init__(manifest, device, prefixes=prefixes, progress=progress)
        self.peers, self.shared_columns = peers, None
        self.config, self.shard = ModelConfig.load(manifest.snapshot), shard
        # Online conversion stores attention, shared-expert, and latent projections as E4M3 values
        # with E8M0 block-32 scales; their activations stay BF16.
        self.online_mxfp8 = online_mxfp8
        self.config.validate_shard(shard)
        self.scale_scratch = None

    def _selected_names(self):
        return {n for n in super()._selected_names() if n.startswith("language_model.")}

    def __exit__(self, *exc):
        shared, self.shared_columns = self.shared_columns, None
        if exc[0] is None and shared is not None:
            try:
                shared.close()
            except BaseException:
                super().__exit__(*sys.exc_info())
                raise
        return super().__exit__(*exc)

    def _shared_reads(self):
        """Read column-sharded projections once per host; staging holds this rank's whole expert down projections."""
        if self.peers is None:
            return None
        if self.shared_columns is None:
            group, peer_storage = self.peers
            c = self.config.text
            capacity = -(-c["num_experts"] // self.shard.world_size) * fp4_down_staging_bytes(
                c["routed_expert_hidden_size"], c["moe_intermediate_size"], 32)
            self.shared_columns = SharedColumnReads(group, peer_storage, self.device, capacity)
        return self.shared_columns

    def linear(self, prefix, *, rows=None, columns=None):
        source = self._sources[prefix + ".weight"]
        if source.ndim != 2 or source.dtype != torch.bfloat16:
            raise ValueError(f"{prefix}: expected unquantized BF16 projection")
        shared = self._shared_reads() if rows is None and columns is not None else None
        if shared is not None:
            out = torch.empty((source.shape[0], columns[1] - columns[0]), dtype=source.dtype, device=self.device)
            return shared.read_columns(self, source, columns, out)
        return self.load_tensor(prefix + ".weight", extent=(slice(*(rows or (0, source.shape[0]))),
            slice(*(columns or (0, source.shape[1])))))

    def fused(self, prefixes, ranges):
        values = [self.linear(p, rows=r) for p, r in zip(prefixes, ranges, strict=True)]
        return torch.cat(values)

    def mlp(self, prefix, intermediate, *, online_mxfp8=False):
        interval = self.shard.partition(intermediate, alignment=32)
        weights = dict(gate_up=self.fused([prefix + ".gate_proj", prefix + ".up_proj"], [interval] * 2),
                       down=self.linear(prefix + ".down_proj", columns=interval))
        return {key: quantize_bf16_linear_as_mxfp8(weight) for key, weight in weights.items()} if online_mxfp8 else weights

    def load_attention(self, index):
        c, shard = self.config.text, self.shard
        prefix = f"{PREFIX}.layers.{index}"
        a = prefix + ".self_attn"
        heads = c["linear_attn_config"]["num_heads"] if self.config.layers[index] == "kda" else c["num_attention_heads"]
        h0, h1 = shard.partition(heads)
        head_rows = h0 * 128, h1 * 128
        attention = {}
        if self.config.layers[index] == "kda":
            # Every projection of the layer input in one GEMM: query, key, value, forget-gate input, output gate and
            # beta. Each output segment starts on a 16-byte boundary and zero rows pad the width to eight.
            projected = self.fused([a + "." + n for n in ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_proj", "b_proj")],
                [head_rows] * 3 + [None, head_rows, (h0, h1)])
            padding = -projected.shape[0] % 8
            projected = torch.cat((projected, projected.new_zeros((padding, projected.shape[1]))))
            attention["input"] = projected
            attention["forget"] = self.linear(a + ".f_b_proj", rows=head_rows)
            conv = [self.load_tensor(a + "." + n + ".weight", extent=slice(*head_rows))
                    for n in ("q_conv1d", "k_conv1d", "v_conv1d")]
            attention["convolution"] = torch.cat(conv).squeeze(1)
            log = self._sources[a + ".A_log"]
            if tuple(log.shape) not in ((heads,), ((heads + 127) // 128 * 128,)) or log.dtype != torch.float32:
                raise ValueError(f"{a}.A_log must cover the logical heads, optionally padded to 128")
            attention["A_log"] = self.load_tensor(a + ".A_log", extent=slice(h0, h1))
            attention["dt_bias"] = self.load_tensor(a + ".dt_bias", extent=slice(*head_rows))
            attention["norm"] = self.load_tensor(a + ".o_norm.weight")
        else:
            attention["q_b"] = self.linear(a + ".q_b_proj", rows=(h0 * 192, h1 * 192))
            kv = self.linear(a + ".kv_b_proj", rows=(h0 * 256, h1 * 256)).view(h1 - h0, 256, 512)
            attention["key"] = kv[:, :128].contiguous()
            attention["value"] = kv[:, 128:].transpose(1, 2).contiguous()
            attention["query_norm"] = self.load_tensor(a + ".q_a_layernorm.weight")
            attention["kv_norm"] = self.load_tensor(a + ".kv_a_layernorm.weight")
        if self.config.layers[index] != "kda":
            # This rank's query and KV latent rows beside its output gate; the ranks gather the latents, as
            # vLLM's VLLM_KIMI_SHARD_QKV_A does. Zero rows pad the latent shard to the exchange's 16-byte width.
            latent = (self.linear(a + ".q_a_proj", rows=shard.partition(c["q_lora_rank"])),
                      self.linear(a + ".kv_a_proj_with_mqa",
                                  rows=shard.partition(c["kv_lora_rank"] + c["qk_rope_head_dim"])))
            gate = self.linear(a + ".g_proj", rows=head_rows)
            width = sum(x.shape[0] for x in latent)
            attention["latent_gate"] = torch.cat((*latent, gate.new_zeros((-width % 8, gate.shape[1])), gate))
        if self.config.layers[index] == "dsa":
            d = self.config.dsa
            shapes = {"wq_b": (d.heads * d.head_dim, c["q_lora_rank"]),
                      "wk": (d.head_dim, c["hidden_size"]),
                      "weights_proj": (d.heads, c["hidden_size"])}
            for key, shape in shapes.items():
                name = a + ".indexer." + key
                source = self._sources[name + ".weight"]
                if tuple(source.shape) != shape or source.dtype != torch.bfloat16:
                    raise ValueError(f"{name}: expected BF16 indexer projection {shape}")
                attention["indexer." + key] = self.linear(name)
            for key in ("weight", "bias"):
                name = a + ".indexer.k_norm." + key
                source = self._sources[name]
                if tuple(source.shape) != (d.head_dim,) or source.dtype != torch.bfloat16:
                    raise ValueError(f"{name}: expected BF16 LayerNorm vector")
                attention["indexer.k_norm." + key] = self.load_tensor(name)
        attention["output"] = self.linear(a + ".o_proj", columns=head_rows)
        if self.online_mxfp8:
            for key in ("input", "forget", "latent_gate", "q_b", "output"):
                if key in attention:
                    attention[key] = quantize_bf16_linear_as_mxfp8(attention[key])
        return attention

    def load_layer(self, index):
        c, shard = self.config.text, self.shard
        prefix = f"{PREFIX}.layers.{index}"
        result = {name: self.load_tensor(f"{prefix}.{name}.weight") for name in
            ("input_layernorm", "post_attention_layernorm")}
        for score, norm, projection in (("attention_res_score", "self_attention_res_norm", "self_attention_res_proj"),
                                        ("mlp_res_score", "mlp_res_norm", "mlp_res_proj")):
            result[score] = score_weight(*self.load_tensors((f"{prefix}.{norm}.weight", f"{prefix}.{projection}.weight")))
        result["attention"] = self.load_attention(index)
        if index < c["first_k_dense_replace"]:
            result["mlp"] = self.mlp(prefix + ".mlp", c["intermediate_size"])
            return result
        m = prefix + ".block_sparse_moe"
        result["shared"] = self.mlp(m + ".shared_experts", c["num_shared_experts"] * c["moe_intermediate_size"],
                                    online_mxfp8=self.online_mxfp8)
        result["router"] = self.linear(m + ".gate", rows=shard.partition(c["num_experts"], alignment=8))
        result["correction"] = self.load_tensor(m + ".gate.e_score_correction_bias")
        result["latent_down"] = self.linear(m + ".routed_expert_down_proj", columns=shard.partition(c["hidden_size"]))
        # Decode rows project their own latent rows and gather them with the router shards.
        result["latent_down_rows"] = self.linear(m + ".routed_expert_down_proj",
                                                 rows=shard.partition(c["routed_expert_hidden_size"]))
        result["latent_up"] = self.linear(m + ".routed_expert_up_proj", columns=shard.partition(c["routed_expert_hidden_size"]))
        if self.online_mxfp8:
            for key in ("latent_down", "latent_down_rows", "latent_up"):
                result[key] = quantize_bf16_linear_as_mxfp8(result[key])
        # Each rank applies the latent norm's weights to the columns its up-projection shard reads.
        result["latent_norm"] = self.load_tensor(m + ".routed_expert_norm.weight",
                                                 extent=slice(*shard.partition(c["routed_expert_hidden_size"])))
        prefixes = [f"{m}.experts.{i}" for i in range(c["num_experts"])]
        formats = {self.config.projection_format(f"{p}.{projection}") for p in prefixes for projection in ("w1", "w2", "w3")}
        if len(formats) != 1 or not formats <= {"mxfp4", "mxfp4_csf"}:
            raise ValueError(f"{m}: expert storage disagrees with quantization configuration")
        hidden, intermediate = c["routed_expert_hidden_size"], c["moe_intermediate_size"] // shard.world_size
        if self.scale_scratch is None:
            def buffer(rows, columns):
                return torch.empty((c["num_experts"], (rows + 255) // 256 * 256,
                                    (columns + 7) // 8 * 8), dtype=torch.uint8, device=self.device)
            self.scale_scratch = (buffer(2 * intermediate, hidden // 32), buffer(hidden, intermediate // 32))
        if formats == {"mxfp4"}:
            # Native E8M0 block scales are compressed while the experts are prepared, so both
            # checkpoint encodings keep the same resident form; gate/up follow b12x's w31 order.
            result["experts"] = self.load_fp4_experts(prefixes, hidden_size=hidden,
                intermediate_size=c["moe_intermediate_size"], format="mxfp4", activation_mode=EXPERT_ACTIVATION,
                intermediate_range=shard.partition(c["moe_intermediate_size"], alignment=32),
                projection_names=("w1", "w3", "w2"), weight_name="weight_packed", activation="situ",
                w13_layout="w31", scale_compression="csf", scale_scratch=self.scale_scratch,
                shared_columns=self._shared_reads())
            return result
        result["experts"] = self.load_mxfp4_csf_experts(prefixes, hidden_size=hidden,
            intermediate_size=c["moe_intermediate_size"], intermediate_range=shard.partition(c["moe_intermediate_size"], alignment=32),
            projection_names=("w1", "w3", "w2"), weight_name="weight_packed",
            activation="situ", activation_mode=EXPERT_ACTIVATION, scale_scratch=self.scale_scratch)
        return result
