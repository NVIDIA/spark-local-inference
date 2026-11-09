# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer-streamed Muse Glimmer text reference with checkpoint-native NVFP4 arithmetic.

The installed Transformers modules carry the model: ``MuseGlimmerTextDecoderLayer`` (scale-less
q/k RMS norm, ``qk_scale_factor``, rope on sliding layers and none on full layers, the sigmoid
gate on the raw attended value, dense SwiGLU, four centered RMS norms), the rotary embedding,
the causal and sliding masks, the normed embedding, the final norm and the softcapped head.
Each of the eight NVFP4 projections per layer is replaced by an ``Nvfp4Projection`` evaluated
through ``nvfp4_linear_reference`` (FP32 decode of the packed checkpoint weight, calibrated
activation quantization) and rounded to BF16 as the prepared kernels round. Layers stream one
at a time so the reference fits beside a prepared decoder; ``layers`` bounds the run for
layer-wise diagnostics. The vision tower is not composed. No prepared Flare operators execute
in this path. SDPA leaves the backend choice to torch because prompts past the sliding window
need a materialized mask; eager retains BF16 attention score and probability rounding.
``history=Fp8History()`` rounds keys and values to E4M3 at the canonical cache seam, which is
where the prepared FP8 rings and pages store them; the default keeps BF16 histories.

Run ``python -m ignition.reference.muse_glimmer SNAPSHOT INPUT_JSON OUTPUT_PT`` with an
assigned CUDA device. The JSON input contains ``token_ids``; the output stores those IDs,
final normalized hidden states, and BF16 logits. Logit row j predicts the token after input
positions zero through j.
"""

import json
from pathlib import Path

import torch

from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.muse_glimmer_nvfp4.config import PREFIX, PROJECTIONS
from ignition.reference.muse_spark import CheckpointReader
from ignition.reference.qwen38_quant import Nvfp4Projection, nvfp4_linear_reference


class Fp8History:
    """Keys and values as the prepared FP8 rings and pages hold them: E4M3-rounded after norm and rope.

    Stands in for ``past_key_values`` at the canonical attention's ``update`` call and caches nothing.
    """

    def update(self, key, value, layer_idx, **kwargs):
        return tuple(x.clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype) for x in (key, value))

    def get_seq_length(self, layer_idx=0):
        return 0


class Nvfp4Linear(torch.nn.Module):
    """A checkpoint NVFP4 projection in a canonical linear's place; it registers no parameters."""

    def __init__(self, reader, name, scale_math):
        super().__init__()
        self.projection = Nvfp4Projection(reader.read(name + ".weight"), reader.read(name + ".weight_scale"),
                                          reader.read(name + ".weight_scale_2"), reader.read(name + ".input_scale"))
        self.scale_math = scale_math

    def forward(self, values):
        return nvfp4_linear_reference(values, self.projection, quant_scale_math=self.scale_math).to(values.dtype)


def quantized_layer(config, index, reader, scale_math):
    """The canonical decoder layer with its eight linears swapped for checkpoint NVFP4 projections."""
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextDecoderLayer

    with torch.device("meta"):
        layer = MuseGlimmerTextDecoderLayer(config, index)
    prefix = f"{PREFIX}.layers.{index}"
    for name in PROJECTIONS:
        parent, _, attribute = name.rpartition(".")
        setattr(layer.get_submodule(parent), attribute, Nvfp4Linear(reader, f"{prefix}.{name}", scale_math))
    reader.install(layer, prefix)
    return layer.eval()


@torch.no_grad()
def text_logits(snapshot, token_ids, *, device="cuda:0", scale_math="dynamic_fast", progress=None,
                attention_implementation="sdpa", layers=None, history=None):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerRMSNorm, MuseGlimmerTextRotaryEmbedding

    if not isinstance(token_ids, (list, tuple)) or not token_ids or any(type(token) is not int for token in token_ids):
        raise ValueError("reference token IDs must be a nonempty integer sequence")
    if attention_implementation not in ("sdpa", "eager"):
        raise ValueError("reference attention implementation must be sdpa or eager")
    snapshot, device = Path(snapshot), torch.device(device)
    if device.type != "cuda":
        raise ValueError("Muse Glimmer text reference requires an NVIDIA CUDA device")
    configuration = json.loads((snapshot / "config.json").read_text())
    if configuration.get("architectures") != ["MuseGlimmerForConditionalGeneration"]:
        raise ValueError("text reference requires MuseGlimmerForConditionalGeneration")
    config = MuseGlimmerConfig.from_dict(configuration).text_config
    config._attn_implementation = attention_implementation
    count = config.num_hidden_layers if layers is None else layers
    if type(count) is not int or not 0 < count <= config.num_hidden_layers:
        raise ValueError("layers must name a leading run of the decoder layers")
    manifest = inspect_indexed_checkpoint(snapshot)
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device).reshape(1, -1)
    if bool(((ids < 0) | (ids >= config.vocab_size)).any()):
        raise ValueError("reference token IDs must be inside the checkpoint vocabulary")
    reader = CheckpointReader(manifest, device)
    unique, inverse = ids.flatten().unique(sorted=True, return_inverse=True)
    embeddings = torch.stack([reader.read(PREFIX + ".embed_tokens.weight", int(token)) for token in unique.tolist()])
    hidden = MuseGlimmerRMSNorm(eps=config.rms_norm_eps, with_scale=False)(
        embeddings[inverse].reshape(1, -1, config.hidden_size))
    position_ids = torch.arange(ids.shape[1], device=device).reshape(1, -1)
    position_embeddings = MuseGlimmerTextRotaryEmbedding(config).to(device)(hidden, position_ids)
    mask_options = dict(config=config, inputs_embeds=hidden, attention_mask=None,
                        past_key_values=None, position_ids=position_ids)
    masks = {"full_attention": create_causal_mask(**mask_options),
             "sliding_attention": create_sliding_window_causal_mask(**mask_options)}
    for index in range(count):
        layer = quantized_layer(config, index, reader, scale_math)
        hidden = layer(hidden, attention_mask=masks[config.layer_types[index]],
                       position_embeddings=position_embeddings if config.layer_rope_theta[index] else None,
                       position_ids=position_ids, past_key_values=history)
        if progress is not None:
            progress(index, hidden)
        del layer
    final_norm = MuseGlimmerRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device)
    final_norm.weight = torch.nn.Parameter(reader.read(PREFIX + ".norm.weight"), requires_grad=False)
    hidden = final_norm(hidden)
    logits = torch.nn.functional.linear(hidden, reader.read("lm_head.weight"))
    logits = logits * config.output_multiplier
    logits = logits / config.final_logit_softcapping
    logits = torch.tanh(logits)
    logits = logits * config.final_logit_softcapping
    hidden, logits = hidden.squeeze(0).cpu(), logits.squeeze(0).cpu()
    if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Muse Glimmer reference produced nonfinite hidden states or logits")
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
                        help="sdpa lets torch choose a masked backend; eager retains BF16 score/probability rounding")
    args = parser.parse_args()
    ids = json.loads(args.input.read_text())["token_ids"]
    started = time.monotonic()

    def progress(layer, hidden):
        torch.cuda.synchronize(args.device)
        print(f"layer={layer} tokens={hidden.shape[1]} elapsed_seconds={time.monotonic() - started:.3f}", flush=True)

    hidden, logits = text_logits(args.snapshot, ids, device=args.device, scale_math=args.scale_math, progress=progress,
                                 attention_implementation=args.attention_implementation)
    torch.save(dict(token_ids=torch.tensor(ids), hidden=hidden, logits=logits,
                    attention_implementation=args.attention_implementation), args.output)
    scores, tokens = logits[-1].float().topk(10)
    print(dict(output=str(args.output), top_tokens=tokens.tolist(), top_logits=scores.tolist(),
               elapsed_seconds=time.monotonic() - started, peak_tensor_bytes=torch.cuda.max_memory_allocated()), flush=True)


if __name__ == "__main__":
    main()
