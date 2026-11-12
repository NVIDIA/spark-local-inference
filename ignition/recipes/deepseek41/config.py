# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint geometry and source ownership for DeepSeek V4.1 Flash."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from ignition.components.parallel import TensorParallelShard


@dataclass(frozen=True)
class CacheSource:
    layer: int
    ratio: int

    def records(self, tokens):
        if tokens % self.ratio:
            raise ValueError("cache page tokens must be divisible by the compression ratio")
        return tokens // self.ratio


@dataclass(frozen=True)
class DSparkConfig:
    layers: int
    block_size: int
    noise_token: int
    target_layers: tuple[int, ...]
    markov_rank: int
    experts: int
    top_k: int

    @classmethod
    def from_text(cls, text):
        result = cls(text["num_nextn_predict_layers"], text["dspark_block_size"],
            text["dspark_noise_token_id"], tuple(text["dspark_target_layer_ids"]),
            text["dspark_markov_rank"], text["dspark_n_routed_experts"],
            text["dspark_num_experts_per_tok"])
        if (result.layers != 3 or result.block_size < 1 or result.markov_rank != 256
                or result.experts != 128 or result.top_k != 3
                or not 0 <= result.noise_token < text["vocab_size"]
                or not result.target_layers or tuple(sorted(set(result.target_layers))) != result.target_layers
                or any(not 0 <= i < text["num_hidden_layers"] for i in result.target_layers)):
            raise ValueError("unsupported DeepSeek V4.1 Flash DSpark geometry")
        start = text["num_hidden_layers"]
        if text["compress_ratios"][start:start + result.layers] != [0] * result.layers:
            raise ValueError("DSpark stages require uncompressed sliding-window attention")
        return result


# The geometry the DeepSeek V4.1 Flash checkpoint must declare. Weight-layout and memory tests read
# these values instead of repeating them, so a geometry change fails the config check first.
TEXT_GEOMETRY = {
    "hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 64,
    "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64,
    "q_lora_rank": 1280, "o_lora_rank": 1024, "o_groups": 8,
    "vocab_size": 129280, "moe_intermediate_size": 2304, "n_routed_experts": 384,
    "n_shared_experts": 1, "num_experts_per_tok": 6, "scoring_func": "sqrtsoftplus",
    "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
    "candidate_topk_blocks": 2048, "candidate_block_size": 8, "hc_mult": 4,
    "sliding_window": 128, "engram_max_ngram_size": 4, "engram_n_heads": 8,
    "engram_head_dim": 256,
}


def attention_weight_extents(text, shard):
    """Checkpoint row extents for the attention projections, per rank and replicated.

    A tensor-parallel rank packs its share of the eight head blocks and eight output groups. A
    row-sharded prefill rank instead owns every head and every output group, so `all_heads`,
    `all_groups`, and `all_sink` name the wholesale extents it replicates beside the partitioned ones.
    """
    first, last = shard.padded_partition(8)
    return SimpleNamespace(
        heads=(first * 8 * text["head_dim"], last * 8 * text["head_dim"]),
        groups=(first * text["o_lora_rank"], last * text["o_lora_rank"]),
        sink=(first * 8, last * 8),
        all_heads=(0, text["num_attention_heads"] * text["head_dim"]),
        all_groups=(0, text["o_groups"] * text["o_lora_rank"]),
        all_sink=(0, text["num_attention_heads"]))


# The backbone schedule that decides which layers a prepared prefill can put on a row shard:
# two uncompressed layers, eighteen at ratio 2, and twenty at ratio 1 whose KV arrives from layer
# 20, the two Engram layers, and the three DSpark target layers. Planning sizes the replicated
# attention charge before a checkpoint is downloaded, so it reads them from this declaration, and
# a row-sharded attention load refuses a checkpoint whose decoder boundary or layer sets differ.
QUALIFIED_SCHEDULE = {
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20,
    "kv_source_layer_ids": [2, 8, 14, 20],
    "engram_layer_ids": [1, 14],
    "dspark_target_layer_ids": [37, 38, 39],
}


def replicated_layer_bytes(world_size):
    """Device bytes one layer's replicated attention projections add per rank.

    The layer loads all `num_attention_heads` head rows of `wq_b` and all `o_groups` output groups
    of `wo_a` and `wo_b` at MXFP8 packing, one value byte per weight and one scale byte per 32
    values, beside the partition every rank keeps for decode.
    """
    text, extent = TEXT_GEOMETRY, attention_weight_extents(TEXT_GEOMETRY, TensorParallelShard(0, world_size))
    group_width = text["num_attention_heads"] * text["head_dim"] // text["o_groups"]
    values = ((extent.all_heads[1] - extent.all_heads[0]) * text["q_lora_rank"]
              + (extent.all_groups[1] - extent.all_groups[0]) * (group_width + text["hidden_size"]))
    return -(-values * 33 // 32)


def replicated_attention_bytes(world_size, *, prompt_logprobs=False, mtp_only=False):
    """Device bytes per rank that row-sharded attention replicates beyond its partition.

    Only the layers a prepared prefill can put on a row shard load the replicated projections.
    A draft target layer shards only in a prefill graph that omits the draft, and `--mtp-only`
    prepares no such graph, so its plan replicates three fewer layers than a scored plan does.
    """
    config = ModelConfig({**TEXT_GEOMETRY, **QUALIFIED_SCHEDULE}, {})
    aux = QUALIFIED_SCHEDULE["dspark_target_layer_ids"] if mtp_only else ()
    return len(config.shardable_attention_layers(config.prefill_stop(prompt_logprobs), aux)) * \
        replicated_layer_bytes(world_size)


@dataclass(frozen=True)
class ModelConfig:
    text: dict
    quantization: dict
    vision: dict | None = None

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        if (config.get("model_type") != "deepseek_v41"
                or config.get("architectures") != ["DeepseekV41ForCausalLM"]):
            raise ValueError("DeepSeek V4.1 requires DeepseekV41ForCausalLM")
        text, quant = config["text_config"], config["quantization_config"]
        for name, value in TEXT_GEOMETRY.items():
            if text.get(name) != value:
                raise ValueError(f"DeepSeek V4.1 Flash {name} requires {value}, got {text.get(name)}")
        if (quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [32, 32]
                or quant.get("scale_fmt") != "ue8m0" or quant.get("expert_dtype") != "fp4"
                or quant.get("activation_scheme") != "dynamic"):
            raise ValueError("DeepSeek V4.1 requires block-32 FP8 weights and native MXFP4 experts")
        vision = config.get("vision_config")
        if vision is not None:
            for key, expected in {"num_hidden_layers": 32, "hidden_size": 1024,
                    "num_attention_heads": 16, "intermediate_size": 2816, "patch_size": 14,
                    "rope_theta": 10000, "downsample_ratio": 3, "max_image_tokens": 1024,
                    "min_pixels": 295936, "max_wh_ratio": None}.items():
                if vision.get(key) != expected:
                    raise ValueError(f"unsupported DeepSeek V4.1 vision {key}: {vision.get(key)}")
            if config.get("image_token_id") != 129264:
                raise ValueError("DeepSeek V4.1 image placeholder requires token 129264")
        result = cls(text, quant, vision)
        ratios = text["compress_ratios"]
        if len(ratios) < result.layers or any(r not in (0, 1, 2) for r in ratios):
            raise ValueError("unsupported DeepSeek compression schedule")
        for name in ("kv_source_layer_ids", "index_source_layer_ids"):
            ids = text[name]
            if not ids or ids != sorted(set(ids)) or any(not 0 <= i < result.layers for i in ids):
                raise ValueError(f"{name} must be ordered unique backbone layers")
        if not set(text["kv_source_layer_ids"]) <= set(text["index_source_layer_ids"]):
            raise ValueError("KV source layers must produce index keys")
        for layer in range(result.layers):
            source = result.source(layer)
            if ratios[layer] and (source is None or source.ratio != ratios[layer]):
                raise ValueError("shared KV consumers must match their source's compression ratio")
            if layer in text["kv_source_layer_ids"] and not ratios[layer]:
                raise ValueError("KV sources require compressed attention")
        candidate = text["candidate_source_layer_id"]
        if candidate not in text["index_source_layer_ids"]:
            raise ValueError("candidate selection must belong to an index source")
        if any(result.source(i) != result.source(candidate)
               for i in text["index_source_layer_ids"] if i > candidate):
            raise ValueError("candidate consumers must use the candidate source's KV history")
        return result

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def draft(self):
        return DSparkConfig.from_text(self.text)

    @property
    def sources(self):
        return tuple(CacheSource(i, self.text["compress_ratios"][i])
                     for i in self.text["kv_source_layer_ids"])

    @property
    def decoder_start(self):
        ratios = self.text["compress_ratios"]
        return next((source.layer for source in self.sources if source.ratio == 1
                     and any(ratio > 1 for ratio in ratios[:source.layer])), None)

    def prefill_stop(self, scored):
        """First backbone layer a prepared prefill leaves on full rows.

        Packed prefill stops at the decoder boundary, whose suffix a replay restores from full
        rows. Scored prefill replays no suffix, so only the last layer stays full-row. A prefill of
        128 rows or fewer never shards, because a row shard needs at least 512 rows per rank on at
        least two ranks, so the stop only names the layer where sharding would end.
        """
        return (self.layers if scored or self.decoder_start is None else self.decoder_start) - 1

    def shardable_attention_layers(self, stop, aux_layers=()):
        """Backbone layers whose attention a row-sharded prefill can run on this rank's rows.

        A prefill leaves the layers from `stop` on full rows: the packed prefill stops at the
        decoder boundary whose suffix a replay restores from full rows, and scored prefill, which
        replays no suffix, stops below the last layer. An Engram layer reads and writes full rows,
        and a layer in `aux_layers` writes a full-row draft feedback column, so neither can shard.
        """
        engram, aux = set(self.text["engram_layer_ids"]), set(aux_layers)
        return tuple(index for index in range(min(stop, self.layers))
                     if index not in engram and index not in aux)

    def require_planned_attention_schedule(self):
        """Refuse a checkpoint whose schedule the row-sharded attention charge does not describe.

        `replicated_attention_bytes` sizes the charge from `QUALIFIED_SCHEDULE` before a checkpoint
        is read, and the loader replicates the layers this config names, so another decoder boundary,
        Engram set, or draft target set would load weights the plan did not charge. A plan without
        the option reads none of these layer sets, so it accepts the checkpoint as it stands.
        """
        qualified = ModelConfig({**TEXT_GEOMETRY, **QUALIFIED_SCHEDULE}, {})
        if (self.decoder_start != qualified.decoder_start
                or self.text["engram_layer_ids"] != qualified.text["engram_layer_ids"]
                or tuple(self.text.get("dspark_target_layer_ids", ())) != tuple(qualified.text["dspark_target_layer_ids"])):
            raise ValueError("row-sharded attention requires the qualified decoder boundary,"
                             " Engram layers, and DSpark target layers")

    def source(self, layer):
        if not self.text["compress_ratios"][layer]:
            return None
        return next((source for source in reversed(self.sources) if source.layer <= layer), None)

    def index_source(self, layer):
        if self.source(layer) is None:
            return None
        return next((i for i in reversed(self.text["index_source_layer_ids"]) if i <= layer), None)
