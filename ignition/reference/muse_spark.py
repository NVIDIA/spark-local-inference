# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer-streamed Muse Spark text reference with checkpoint-native NVFP4 arithmetic.

Attention, rotary positions, normalization, routing and residual gates use the
installed Transformers forwards. Experts read one checkpoint expert at a time;
NVFP4 projections independently decode checkpoint weights and quantize activations
with their calibrated scales. No prepared Flare operators execute in this path.
Attention defaults to SDPA with FlashAttention as its sole enabled backend.
Explicit eager selection retains BF16 attention score and probability rounding.

Run ``python -m ignition.reference.muse_spark SNAPSHOT INPUT_JSON OUTPUT_PT``
with an assigned CUDA device. The JSON input contains ``token_ids``; the output
stores those IDs, final normalized hidden states, and BF16 logits. Logit row j
predicts the token after input positions zero through j.
"""

import json
from contextlib import nullcontext
from pathlib import Path

import torch
from safetensors import safe_open

from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.reference.qwen38_quant import Nvfp4Projection, nvfp4_expert_reference


class CheckpointReader:
    def __init__(self, manifest, device):
        self.manifest, self.device = manifest, device

    def read(self, name, extent=None):
        metadata = self.manifest.tensors[name]
        with safe_open(self.manifest.snapshot / metadata.shard, framework="pt", device="cpu") as file:
            value = file.get_tensor(name) if extent is None else file.get_slice(name)[extent]
            # Complete the transfer before releasing the mapped checkpoint storage.
            return value.to(self.device, non_blocking=False)

    def install(self, module, prefix):
        for name, _ in tuple(module.named_parameters()):
            if name.startswith("mlp.experts."):
                continue
            path, _, attribute = name.rpartition(".")
            target = module.get_submodule(path) if path else module
            value = self.read(prefix + "." + name)
            if "residual_gate" in name or name == "mlp.gate.weight":
                value = value.float()
            setattr(target, attribute, torch.nn.Parameter(value, requires_grad=False))
        for name, _ in tuple(module.named_buffers()):
            full = prefix + "." + name
            if full in self.manifest.tensors:
                path, _, attribute = name.rpartition(".")
                target = module.get_submodule(path) if path else module
                setattr(target, attribute, self.read(full))


class StreamingExperts(torch.nn.Module):
    def __init__(self, canonical, reader, prefix, *, quantized, scale_math):
        super().__init__()
        self.reader, self.prefix = reader, prefix
        self.quantized, self.scale_math = quantized, scale_math
        self.norm, self.act = canonical.post_expert_norm, canonical.act_fn
        self.norm.weight = torch.nn.Parameter(reader.read(prefix + ".post_expert_norm.weight"), requires_grad=False)

    def projection(self, expert, name, section=None):
        if name == "down_proj":
            section = slice(None)
        extent = (expert, section, slice(None))
        packed = self.reader.read(self.prefix + "." + name, extent)
        scales = self.reader.read(self.prefix + "." + name + "_weight_scale", extent)
        gain = self.reader.read(self.prefix + "." + name + "_weight_scale_2", expert)
        activation = self.reader.read(self.prefix + "." + name + "_input_scale", expert)
        if name == "gate_up_proj":
            half = 0 if section.start == 0 else 1
            gain, activation = gain[half], activation[half]
        return Nvfp4Projection(packed, scales, gain, activation)

    def forward(self, source, ids, weights):
        output = torch.zeros_like(source)
        for expert in ids.unique(sorted=True).tolist():
            rows, slots = torch.where(ids == expert)
            current = source[rows]
            if self.quantized:
                gate = self.projection(expert, "gate_up_proj", slice(0, 4096))
                up = self.projection(expert, "gate_up_proj", slice(4096, 8192))
                down = self.projection(expert, "down_proj")
                projected = nvfp4_expert_reference(current, gate, up, down,
                    quant_scale_math=self.scale_math).down.bfloat16()
            else:
                gate_up = self.reader.read(self.prefix + ".gate_up_proj", expert)
                down = self.reader.read(self.prefix + ".down_proj", expert)
                gate, up = torch.nn.functional.linear(current, gate_up).chunk(2, -1)
                projected = torch.nn.functional.linear(self.act(gate) * up, down)
            contribution = self.norm(projected) * weights[rows, slots, None]
            output.index_add_(0, rows, contribution.to(output.dtype))
        return output


@torch.no_grad()
def text_logits(snapshot, token_ids, *, device="cuda:0", scale_math="dynamic_fast", progress=None,
                attention_implementation="sdpa"):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.muse_spark.configuration_muse_spark import MuseSparkConfig
    from transformers.models.muse_spark.modeling_muse_spark import (
        MuseSparkRMSNorm, MuseSparkTextDecoderLayer, MuseSparkTextRotaryEmbedding,
    )

    if not isinstance(token_ids, (list, tuple)) or not token_ids or any(type(token) is not int for token in token_ids):
        raise ValueError("reference token IDs must be a nonempty integer sequence")
    if attention_implementation not in ("sdpa", "eager"):
        raise ValueError("reference attention implementation must be sdpa or eager")
    snapshot, device = Path(snapshot), torch.device(device)
    if device.type != "cuda":
        raise ValueError("Muse Spark text reference requires an NVIDIA CUDA device")
    configuration = json.loads((snapshot / "config.json").read_text())
    if configuration.get("architectures") != ["MuseSparkForConditionalGeneration"]:
        raise ValueError("text reference requires MuseSparkForConditionalGeneration")
    config = MuseSparkConfig.from_dict(configuration).text_config
    if config.moe_hidden_size != 4096 or config.moe_intermediate_size != 4096:
        raise ValueError("text reference requires 4096-channel Muse Spark experts")
    config._attn_implementation = attention_implementation
    manifest = inspect_indexed_checkpoint(snapshot)
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device).reshape(1, -1)
    if ids.numel() == 0 or bool(((ids < 0) | (ids >= config.vocab_size)).any()):
        raise ValueError("reference token IDs must be nonempty and inside the checkpoint vocabulary")
    excluded = configuration["quantization_config"]["modules_to_not_convert"]
    reader = CheckpointReader(manifest, device)
    unique, inverse = ids.flatten().unique(sorted=True, return_inverse=True)
    embeddings = torch.stack([reader.read("model.language_model.embed_tokens.weight", int(token))
                              for token in unique.tolist()])
    hidden = MuseSparkRMSNorm(eps=config.rms_norm_eps, with_scale=False)(embeddings[inverse].reshape(1, -1, config.hidden_size))
    position_ids = torch.arange(ids.shape[1], device=device).reshape(1, -1)
    rotary = MuseSparkTextRotaryEmbedding(config).to(device)
    position_embeddings = rotary(hidden, position_ids)
    mask_options = dict(config=config, inputs_embeds=hidden, attention_mask=None,
                        past_key_values=None, position_ids=position_ids)
    masks = {"full_attention": create_causal_mask(**mask_options),
             "sliding_attention": create_sliding_window_causal_mask(**mask_options)}
    for index in range(config.num_hidden_layers):
        with torch.device("meta"):
            layer = MuseSparkTextDecoderLayer(config, index)
        prefix = f"model.language_model.layers.{index}"
        reader.install(layer, prefix)
        expert_prefix = prefix + ".mlp.experts"
        quantized = not any(expert_prefix == p or expert_prefix.startswith(p + ".") for p in excluded)
        layer.mlp.experts = StreamingExperts(layer.mlp.experts, reader, expert_prefix,
                                             quantized=quantized, scale_math=scale_math)
        attention = (sdpa_kernel([SDPBackend.FLASH_ATTENTION])
                     if attention_implementation == "sdpa" else nullcontext())
        with attention:
            hidden = layer.eval()(hidden, attention_mask=masks[config.layer_types[index]],
                position_embeddings=position_embeddings if config.layer_rope_theta[index] else None,
                position_ids=position_ids)
        if progress is not None:
            progress(index, hidden)
        del layer
    final_norm = MuseSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device)
    final_norm.weight = torch.nn.Parameter(reader.read("model.language_model.norm.weight"), requires_grad=False)
    hidden = final_norm(hidden)
    logits = torch.nn.functional.linear(hidden, reader.read("lm_head.weight"))
    logits = logits * config.output_multiplier
    logits = logits / config.final_logit_softcapping
    logits = torch.tanh(logits)
    logits = logits * config.final_logit_softcapping
    hidden, logits = hidden.squeeze(0).cpu(), logits.squeeze(0).cpu()
    if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Muse Spark reference produced nonfinite hidden states or logits")
    return hidden, logits


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("input", type=Path, help="JSON containing token_ids")
    parser.add_argument("output", type=Path, help="PyTorch token_ids, hidden states and logits fixture")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale-math", choices=("dynamic_fast", "dynamic_precise", "micro"), default="dynamic_fast")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa",
                        help="sdpa requires the FlashAttention backend; eager retains BF16 score/probability rounding")
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    ids = value["token_ids"]
    started = time.monotonic()
    def progress(layer, hidden):
        torch.cuda.synchronize(args.device)
        print(f"layer={layer} tokens={hidden.shape[1]} elapsed_seconds={time.monotonic() - started:.3f}", flush=True)
    hidden, logits = text_logits(args.snapshot, ids, device=args.device, scale_math=args.scale_math, progress=progress,
                                 attention_implementation=args.attention_implementation)
    torch.save(dict(token_ids=torch.tensor(ids), hidden=hidden, logits=logits,
                    attention_implementation=args.attention_implementation,
                    attention_backend="FLASH_ATTENTION" if args.attention_implementation == "sdpa" else "eager"),
               args.output)
    scores, tokens = logits[-1].float().topk(10)
    print(dict(output=str(args.output), top_tokens=tokens.tolist(), top_logits=scores.tolist(),
        elapsed_seconds=time.monotonic() - started, peak_tensor_bytes=torch.cuda.max_memory_allocated()), flush=True)


if __name__ == "__main__":
    main()
