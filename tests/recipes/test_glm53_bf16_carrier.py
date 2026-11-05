# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 Flash BF16 carrier: every projection the release stores in BF16 runs in BF16.

The TrellisMX release ships a pinned carrier whose only quantized tensors are the routed experts, and
those payloads live in the release's sidecars rather than in the carrier. Flare's glm53 recipe
therefore has to run every other projection in the checkpoint's own BF16 storage, with no conversion
to MXFP8, block-FP8 or anything else. The CPU tests read the carrier's headers and prove the storage
per group; the GPU tests build one dense MLP layer, one KDA layer and one MLA layer from the carrier's
real BF16 weights and compare each against a plain torch reference forward with the same weights.

Bound: both paths consume the same BF16 operands and accumulate in FP32, so they differ only in the
order of the FP32 reduction (and, for the fused activations, in the last bit of an fp32 sigmoid or
rsqrt). After the final BF16 store, which keeps eight mantissa bits, a differing value can therefore
be at most one BF16 ulp -- 2**-8 of the tensor's magnitude -- from the reference. The tests assert
that bound against the tensor's largest magnitude and report the share of bitwise-equal elements.
"""

import json

import pytest
import torch
from safetensors import safe_open

from ignition.preparation.quantization import linear_format, quantized_layers
from tests.recipes.glm53_trellismx_release import CARRIER, requires_cuda, requires_release

CHECKPOINT = "GLM-5.3-Flash-TrellisMX-carrier"
TEXT_PREFIX = "model.language_model"
# Layers 0-2 are dense MLP, 3 and every fourth layer after it are sparse MLA, 45 is the MTP layer.
DENSE_LAYER = 0
MLA_LAYER = 3
MTP_LAYER = 45
ROUTED_LAYERS = tuple(range(3, 46))
# One BF16 ulp at the tensor's magnitude.
BF16_ULP = 2.0**-8
# Projections whose checkpoint tensor carries no `.weight` suffix.
UNSUFFIXED = ("self_attn.indexer.index_kpool_compress_gate",)

# Projections the carrier stores in BF16, with their stored shapes: (layer, tensor suffix, shape).
# Every suffix but the index pool gate ends in `.weight`, which is the projection prefix's tensor.
PROJECTIONS = (
    (DENSE_LAYER, "mlp.gate_proj", (12288, 4096)),
    (DENSE_LAYER, "mlp.up_proj", (12288, 4096)),
    (DENSE_LAYER, "mlp.down_proj", (4096, 12288)),
    (DENSE_LAYER, "self_attn.q_proj", (8192, 4096)),
    (DENSE_LAYER, "self_attn.k_proj", (8192, 4096)),
    (DENSE_LAYER, "self_attn.v_proj", (8192, 4096)),
    (DENSE_LAYER, "self_attn.b_proj", (64, 4096)),
    (DENSE_LAYER, "self_attn.f_a_proj", (128, 4096)),
    (DENSE_LAYER, "self_attn.f_b_proj", (8192, 128)),
    (DENSE_LAYER, "self_attn.g_a_proj", (128, 4096)),
    (DENSE_LAYER, "self_attn.g_b_proj", (8192, 128)),
    (DENSE_LAYER, "self_attn.o_proj", (4096, 8192)),
    (MLA_LAYER, "self_attn.q_a_proj", (1536, 4096)),
    (MLA_LAYER, "self_attn.kv_a_proj_with_mqa", (512, 4096)),
    (MLA_LAYER, "self_attn.q_b_proj", (16384, 1536)),
    (MLA_LAYER, "self_attn.kv_b_proj", (32768, 512)),
    (MLA_LAYER, "self_attn.o_proj", (4096, 16384)),
    (MLA_LAYER, "self_attn.indexer.wq_b", (4096, 1536)),
    (MLA_LAYER, "self_attn.indexer.wk", (128, 4096)),
    (MLA_LAYER, "self_attn.indexer.weights_proj", (32, 4096)),
    # This one carries no `.weight` suffix in the checkpoint.
    (MLA_LAYER, "self_attn.indexer.index_kpool_compress_gate", (128, 4096)),
    (MLA_LAYER, "mlp.gate", (288, 4096)),
    (MLA_LAYER, "mlp.shared_experts.gate_proj", (2048, 4096)),
    (MLA_LAYER, "mlp.shared_experts.up_proj", (2048, 4096)),
    (MLA_LAYER, "mlp.shared_experts.down_proj", (4096, 2048)),
    (MTP_LAYER, "eh_proj", (4096, 8192)),
    (MTP_LAYER, "self_attn.q_a_proj", (1536, 4096)),
    (MTP_LAYER, "self_attn.q_b_proj", (16384, 1536)),
    (MTP_LAYER, "self_attn.kv_b_proj", (32768, 512)),
    (MTP_LAYER, "self_attn.o_proj", (4096, 16384)),
)

# Non-projection BF16 parameters: norms, the mHC hyper-connection functions, index pooling.
PARAMETERS = (
    (DENSE_LAYER, "input_layernorm.weight", (4096,)),
    (DENSE_LAYER, "post_attention_layernorm.weight", (4096,)),
    (DENSE_LAYER, "hc_attn_fn", (24, 16384)),
    (DENSE_LAYER, "hc_ffn_fn", (24, 16384)),
    (DENSE_LAYER, "self_attn.o_norm.weight", (128,)),
    (MLA_LAYER, "self_attn.q_a_layernorm.weight", (1536,)),
    (MLA_LAYER, "self_attn.kv_a_layernorm.weight", (512,)),
    (MLA_LAYER, "self_attn.indexer.k_norm.weight", (128,)),
    (MLA_LAYER, "self_attn.indexer.k_norm.bias", (128,)),
    (MLA_LAYER, "self_attn.indexer.index_kpool_compress_ape", (4, 128)),
    (MTP_LAYER, "enorm.weight", (4096,)),
    (MTP_LAYER, "hnorm.weight", (4096,)),
    (MTP_LAYER, "shared_head.norm.weight", (4096,)),
)

# FP32 scalars and per-expert calibration values.
FP32_PARAMETERS = (
    (DENSE_LAYER, "hc_attn_base", (24,)),
    (DENSE_LAYER, "hc_attn_scale", (3,)),
    (DENSE_LAYER, "self_attn.A_log", (64,)),
    (DENSE_LAYER, "self_attn.dt_bias", (8192,)),
    (MLA_LAYER, "mlp.gate.e_score_correction_bias", (288,)),
)


def _name(layer, suffix):
    return f"{TEXT_PREFIX}.layers.{layer}.{suffix}"


pytestmark = requires_release


@pytest.fixture(scope="module")
def carrier():
    """The release's own carrier: wherever the release is located, these tests run against it."""
    assert (CARRIER / "model.safetensors.index.json").is_file(), f"{CARRIER} holds no carrier index"
    return CARRIER


@pytest.fixture(scope="module")
def headers(carrier):
    """Every tensor's stored dtype and shape, read from the safetensors headers only."""
    index = json.loads((carrier / "model.safetensors.index.json").read_text())
    stored = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(carrier / shard), framework="pt") as handle:
            for name in handle.keys():
                slice_ = handle.get_slice(name)
                stored[name] = (str(slice_.get_dtype()), tuple(slice_.get_shape()))
    return stored


def test_quantization_declarations_cover_only_routed_experts(carrier):
    """The carrier's ModelOpt metadata must not declare any projection these tests expect in BF16."""
    quantization = quantized_layers(json.loads((carrier / "config.json").read_text()))
    declared = set(quantization)
    expected = {f"{TEXT_PREFIX}.layers.{layer}.mlp.experts" for layer in ROUTED_LAYERS}
    assert declared == expected
    assert not any(".self_attn" in prefix or ".shared_experts" in prefix or ".gate" in prefix
                   for prefix in declared)
    # The expert declarations parse to the formats the overlay replaces, so the missing payload is a
    # storage gap, not a metadata gap.
    assert linear_format(quantization, f"{TEXT_PREFIX}.layers.3.mlp.experts") == "nvfp4"
    assert linear_format(quantization, f"{TEXT_PREFIX}.layers.45.mlp.experts") == "mxfp8"


def test_carrier_stores_every_projection_in_bf16(carrier, headers):
    """Per group, the format the weight loader resolves equals the checkpoint's plain storage."""
    quantization = quantized_layers(json.loads((carrier / "config.json").read_text()))
    for layer, suffix, shape in PROJECTIONS:
        name = _name(layer, suffix if suffix in UNSUFFIXED else suffix + ".weight")
        assert name in headers, f"{name} is missing from the carrier index"
        dtype, stored = headers[name]
        assert (dtype, stored) == ("BF16", shape), f"{name}: stored {dtype} {stored}, expected BF16 {shape}"
        # components/weights.py::DirectWeightLoader.linear_format: a declared prefix takes its
        # quantized_layers format; otherwise the stored dtype is the format, provided the projection
        # has no sibling scale tensors. Nothing else may be declared for these prefixes.
        assert suffix not in quantization
        assert f"{_name(layer, suffix)}.weight_scale" not in headers
        assert f"{_name(layer, suffix)}.input_scale" not in headers
    for layer, suffix, shape in PARAMETERS:
        name = _name(layer, suffix)
        assert headers.get(name) == ("BF16", shape), f"{name}: {headers.get(name)}"
    for layer, suffix, shape in FP32_PARAMETERS:
        name = _name(layer, suffix)
        assert headers.get(name) == ("F32", shape), f"{name}: {headers.get(name)}"


def test_carrier_holds_no_packed_or_float8_storage(carrier, headers):
    """No MXFP8, block-FP8, NVFP4 or other packed payload exists in any carrier safetensors file.

    Every file is read, the index's shards and the unindexed calibration files alike, headers only.
    """
    files = sorted(carrier.glob("*.safetensors"))
    assert files, f"{carrier} holds no safetensors files"
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            stored = {name: str(handle.get_slice(name).get_dtype()) for name in handle.keys()}
        assert set(stored.values()) <= {"BF16", "F32"}, f"{path.name} stores {sorted(set(stored.values()))}"
        assert not any(name.endswith((".weight_scale", ".weight_scale_2")) for name in stored), \
            f"{path.name} carries block scales"
    # The loader reads the index's shards, whose only per-expert tensors are the calibration
    # scalars the expert overlay consumes.
    scales = {name.rsplit(".", 1)[1] for name in headers if "mlp.experts." in name}
    assert scales == {"input_scale"}, f"unexpected expert tensors {sorted(scales)}"


def test_carrier_routed_experts_have_no_payload(headers):
    """The routed expert weights come from the release sidecars; the carrier holds only calibration."""
    for layer in range(3, 45):
        prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts.0"
        assert f"{prefix}.gate_proj.input_scale" in headers
        for projection in ("gate_proj", "up_proj", "down_proj"):
            assert f"{prefix}.{projection}.weight" not in headers
            assert f"{prefix}.{projection}.weight_scale" not in headers
    # The MXFP8 draft layer's experts come from its own sidecar, with no carrier tensors at all.
    assert not any(f"{TEXT_PREFIX}.layers.45.mlp.experts." in name for name in headers)


# --------------------------------------------------------------------------------------------------
# GPU: build the carrier's real BF16 layers and compare them with plain torch reference forwards.
# --------------------------------------------------------------------------------------------------

cuda = requires_cuda


def assert_bf16_bound(actual, expected, label):
    """Each element within one BF16 ulp of its own reference value; report bitwise agreement.

    Every reference below consumes the layer's own upstream BF16 buffer, so a comparison isolates one
    projection or activation and the only permitted divergence is that single BF16 store.
    """
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert bool(torch.isfinite(actual.float()).all()), f"{label}: non-finite output"
    error = (actual.float() - expected.float()).abs()
    # One BF16 ulp of the element plus one of the tensor's largest magnitude: the two paths sum the
    # same exactly-representable products in FP32 in different orders, so a value far below the
    # tensor's accumulation scale can land a rounding step away from its own magnitude.
    bound = BF16_ULP * (expected.float().abs() + expected.float().abs().max())
    worst = float((error / bound.clamp_min(torch.finfo(torch.float32).tiny)).max())
    share = (actual == expected).float().mean().item()
    print(f"  {label}: bitwise={share:.6f} worst={worst:.3f} of bound")
    assert worst <= 1.0, (
        f"{label}: {int((error > bound).sum())} of {error.numel()} elements exceed the BF16 bound "
        f"(worst {worst:.3f} of bound)")
    return share


def project(source, weight):
    """Reference projection: BF16 operands, FP32 accumulation, one BF16 store."""
    return (source.float() @ weight.float().t()).to(torch.bfloat16)


def rms_norm_reference(source, weight, eps):
    """flashinfer.triton.norm.rms_norm with round_before_weight: round, then weight, then store."""
    normalized = source.float() * torch.rsqrt(source.float().square().mean(-1, keepdim=True) + eps)
    return (weight.float() * normalized.to(torch.bfloat16).float()).to(torch.bfloat16)


def _loader(carrier, shard, prefixes):
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.glm53.weights import GlmWeightLoader

    manifest = inspect_indexed_checkpoint(carrier)
    return GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(shard[0], shard[1]),
                           prefixes=prefixes)


@cuda
@pytest.mark.parametrize("tokens", [1, 8, 32, 256])
@torch.no_grad()
def test_dense_mlp_layer_runs_in_carrier_bf16(carrier, tokens):
    """Dense MLP layer 0: gate/up/down BF16 projections and the capped SiLU activation."""
    from ignition.components.mlp import PreparedMLP
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.weights import TEXT_PREFIX as prefix_root

    device = torch.device("cuda", 0)
    shard = TensorParallelShard(0, 1)
    prefix = f"{prefix_root}.layers.{DENSE_LAYER}.mlp"
    limit = 10.0
    with _loader(carrier, (0, 1), (prefix + ".",)) as loader:
        weights = loader.load_mlp(DENSE_LAYER)
        gate_up_weight = loader.load_linear_group(
            (prefix + ".gate_proj", prefix + ".up_proj"),
            row_ranges=(shard.partition(12288), shard.partition(12288)))
        down_weight = loader.load_linear(prefix + ".down_proj", columns=shard.partition(12288))
    assert isinstance(weights.gate_up, torch.Tensor) and weights.gate_up.dtype == torch.bfloat16
    assert torch.equal(weights.gate_up, gate_up_weight)
    assert torch.equal(weights.down, down_weight)

    source = (torch.randn((tokens, 4096), generator=torch.Generator().manual_seed(53)) * .1).to(
        torch.bfloat16).to(device)
    gate_up = torch.empty((tokens, 2 * 12288), dtype=torch.bfloat16, device=device)
    middle = torch.empty((tokens, 12288), dtype=torch.bfloat16, device=device)
    output = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=device)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint=CHECKPOINT,
                                    workload_schema="glm53-bf16-mlp-v1",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        mlp = PreparedMLP(source, weights, gate_up, middle, output, session=session, name="dense",
                          workspace=arena, swiglu_limit=limit)
        session.tune()
        mlp(source)
    torch.cuda.synchronize()

    reference_gate_up = project(source, gate_up_weight)
    # The activation consumes the layer's own gate_up buffer, and the down projection its own
    # intermediate, so each comparison isolates one stage instead of compounding roundings.
    gate = torch.minimum(gate_up[:, :12288].float(), torch.tensor(limit))
    up = gate_up[:, 12288:].float().clamp(-limit, limit)
    reference_middle = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    reference_output = project(middle, down_weight)

    shares = [assert_bf16_bound(actual, expected, label) for label, actual, expected in (
        (f"gate_up M={tokens}", gate_up, reference_gate_up),
        (f"intermediate M={tokens}", middle, reference_middle),
        (f"output M={tokens}", output, reference_output))]
    assert min(shares) > .4, f"M={tokens}: bitwise agreement {shares}"


@cuda
@pytest.mark.parametrize("tokens", [8, 256])
@torch.no_grad()
def test_dense_mlp_layer_is_graph_captured(carrier, tokens):
    """The BF16 projections capture into a CUDA graph and replay bitwise identically."""
    from ignition.components.mlp import PreparedMLP
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.weights import TEXT_PREFIX as prefix_root

    device = torch.device("cuda", 0)
    prefix = f"{prefix_root}.layers.{DENSE_LAYER}.mlp"
    with _loader(carrier, (0, 1), (prefix + ".",)) as loader:
        weights = loader.load_mlp(DENSE_LAYER)
    source = (torch.randn((tokens, 4096), generator=torch.Generator().manual_seed(59)) * .1).to(
        torch.bfloat16).to(device)
    gate_up = torch.empty((tokens, 2 * 12288), dtype=torch.bfloat16, device=device)
    middle = torch.empty((tokens, 12288), dtype=torch.bfloat16, device=device)
    output = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=device)
    arena = ScratchArena(device)
    graph = torch.cuda.CUDAGraph()
    with RecordedPreparationSession(device=device, checkpoint=CHECKPOINT,
                                    workload_schema="glm53-bf16-mlp-v1",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        mlp = PreparedMLP(source, weights, gate_up, middle, output, session=session, name="dense",
                          workspace=arena, swiglu_limit=10.0)
        session.tune()
        mlp(source)
        torch.cuda.synchronize()
        expected = output.clone()
        compiled = compile_forward(lambda: mlp(source))
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        torch.cuda.synchronize()
        output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected), "captured BF16 projection replay differs from eager"


@cuda
@pytest.mark.parametrize("tokens", [65, 256])
@torch.no_grad()
def test_kda_layer_runs_in_carrier_bf16(carrier, tokens):
    """KDA layer 0: fused q/k/v/b/f_a, f_b, g_a/g_b and o_proj BF16 projections, then the recurrence."""
    from b12x.sequence.kda_prefill.reference import prefill_kda

    from ignition.components.recurrent import DeltaMetadata, DeltaState
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.kda import KDAScratch, PreparedGlmKDA
    from ignition.recipes.glm53.model import create_model_tree
    from ignition.recipes.glm53.weights import TEXT_PREFIX as prefix_root

    device = torch.device("cuda", 0)
    heads, width = 64, 64 * 128
    module = create_model_tree(carrier).model.language_model.layers[DENSE_LAYER].self_attn
    prefix = f"{prefix_root}.layers.{DENSE_LAYER}.self_attn"
    with _loader(carrier, (0, 1), (prefix + ".",)) as loader:
        weights = loader.load_attention(DENSE_LAYER)
        input_weight = loader.load_mixed_linear_group(
            tuple(f"{prefix}.{name}" for name in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj")),
            row_ranges=((0, 8192), (0, 8192), (0, 8192), (0, 64), None))
        f_b_weight = loader.load_linear(prefix + ".f_b_proj", rows=(0, 8192))
        g_a_weight = loader.load_linear(prefix + ".g_a_proj")
        g_b_weight = loader.load_linear(prefix + ".g_b_proj", rows=(0, 8192))
        o_weight = loader.load_linear(prefix + ".o_proj", columns=(0, 8192))
        conv_weight = torch.cat([
            loader.load_tensor(f"{prefix}.{name}_conv1d.weight", extent=slice(0, 8192))
            for name in ("q", "k", "v")]).squeeze(1)
        a_log = loader.load_tensor(f"{prefix}.A_log", extent=slice(0, 64))
        dt_bias = loader.load_tensor(f"{prefix}.dt_bias", extent=slice(0, 8192))
    assert isinstance(weights["input_proj"], torch.Tensor), "KDA input projections must fuse as BF16"
    assert weights["input_proj"].dtype == torch.bfloat16

    state = DeltaState.allocate(4, device, heads=heads, channels=3 * width)
    metadata = DeltaMetadata.allocate(1, device)
    capacity = 1 << (tokens - 1).bit_length()
    source = (torch.randn((tokens, 4096), generator=torch.Generator().manual_seed(7)) * .1).to(
        torch.bfloat16).to(device)
    working = torch.zeros((capacity, 4096), dtype=torch.bfloat16, device=device)
    working[:tokens].copy_(source)
    scratch = KDAScratch.allocate(capacity, heads, device)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint=CHECKPOINT,
                                    workload_schema="glm53-bf16-kda-v1",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        operator = PreparedGlmKDA(module, weights, working, scratch, state, metadata, mode="prefill",
                                  session=session, name="kda", workspace=arena, checkpoint_export=False)
        session.tune()
        metadata.query_start_loc.copy_(torch.tensor([0, tokens], dtype=torch.int32))
        metadata.convolution_slots.copy_(torch.tensor([1], dtype=torch.int64))
        metadata.fresh.fill_(True)
        metadata.state_indices[0, 0] = 1
        metadata.initial_state_indices[0] = 0
        metadata.final_state_indices[0] = 1
        metadata.checkpoint_offsets[0] = 0
        metadata.num_seqs.fill_(1)
        metadata.num_tokens.fill_(tokens)
        operator(working)
        torch.cuda.synchronize()
        # The BF16 projections capture into a graph and replay bitwise identically.
        expected = scratch.output[:tokens].clone()
        graph = torch.cuda.CUDAGraph()
        compiled = compile_forward(lambda: operator(working))
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        torch.cuda.synchronize()
        scratch.output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(scratch.output[:tokens], expected), "captured KDA replay differs"
    torch.cuda.synchronize()

    # Each reference consumes the layer's own upstream BF16 buffer, so a comparison isolates one
    # projection, the convolution or the recurrence instead of compounding roundings.
    projected = project(source, input_weight)
    gate_input = project(source, g_a_weight)
    z = project(scratch.gate_input[:tokens], g_b_weight)
    raw_g = project(scratch.forget_input[:tokens], f_b_weight)
    # b12x's short conv accumulates in FP32, rounds to BF16 and then applies SiLU
    # (b12x/sequence/_short_conv.py::_short_conv_kernel, ROUND_CONV=True).
    padded = torch.zeros((tokens + 3, 3 * width), dtype=torch.float32, device=device)
    padded[3:] = scratch.projected_qkv[:tokens].float()
    accumulated = torch.zeros((tokens, 3 * width), dtype=torch.float32, device=device)
    for tap in range(4):
        accumulated += padded[tap:tap + tokens] * conv_weight[:, tap].float()
    rounded = accumulated.to(torch.bfloat16).float()
    convolved = (rounded * torch.sigmoid(rounded)).to(torch.bfloat16)
    q, k, v = (view.view(tokens, heads, 128) for view in scratch.convolved_qkv[:tokens].split(width, dim=-1))
    recurrent = prefill_kda(q, k, v, scratch.raw_g[:tokens].view(tokens, heads, 128), scratch.raw_beta[:tokens],
        a_log, dt_bias.view(heads, 128),
        recurrent_state=torch.zeros((4, heads, 128, 128), dtype=torch.float32, device=device),
        cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=device),
        initial_state_indices=torch.zeros(1, dtype=torch.int64, device=device),
        final_state_indices=torch.ones(1, dtype=torch.int64, device=device),
        checkpoint_state_indices=torch.zeros(1, dtype=torch.int64, device=device),
        checkpoint_offsets=torch.zeros(1, dtype=torch.int32, device=device),
        num_seqs=1, num_tokens=tokens, lower_bound=-5.0)
    normalized = scratch.recurrence[:tokens].view(tokens, heads, 128).float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(-1, keepdim=True) + module.o_norm.variance_epsilon)
    gated = (module.o_norm.weight.float() * normalized
             * torch.sigmoid(scratch.z[:tokens].view(tokens, heads, 128).float())).to(torch.bfloat16)
    reference_output = project(scratch.gated[:tokens], o_weight)

    shares = [assert_bf16_bound(actual, expected, label) for label, actual, expected in (
        (f"input projections M={tokens}", scratch.projected_input[:tokens], projected),
        (f"g_a_proj M={tokens}", scratch.gate_input[:tokens], gate_input),
        (f"g_b_proj M={tokens}", scratch.z[:tokens], z),
        (f"f_b_proj M={tokens}", scratch.raw_g[:tokens], raw_g),
        (f"short conv M={tokens}", scratch.convolved_qkv[:tokens], convolved),
        (f"recurrence M={tokens}", scratch.recurrence[:tokens].view(tokens, width), recurrent.view(tokens, width)),
        (f"gated norm M={tokens}", scratch.gated[:tokens], gated.view(tokens, width)),
        (f"o_proj M={tokens}", scratch.output[:tokens], reference_output))]
    assert min(shares) > .25, f"M={tokens}: bitwise agreement {shares}"


@cuda
@pytest.mark.parametrize("mode,rows", [("prefill", 65), ("decode", 1)])
@torch.no_grad()
def test_mla_layer_runs_in_carrier_bf16(carrier, mode, rows):
    """MLA layer 3: qkv_a, q_b, kv_b absorption/expansion and o_proj BF16 projections.

    The C4 indexer selection and the sparse-MLA attention are b12x kernels, not projections, so the
    reference consumes the layer's own attention output where the value expansion reads it.
    """
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.indexer import IndexMetadata
    from ignition.recipes.glm53.mla import MLAScratch, MLAState, PreparedGlmMLA
    from ignition.recipes.glm53.model import create_model_tree
    from ignition.recipes.glm53.weights import TEXT_PREFIX as prefix_root

    device = torch.device("cuda", 0)
    heads = 64
    module = create_model_tree(carrier).model.language_model.layers[MLA_LAYER].self_attn
    prefix = f"{prefix_root}.layers.{MLA_LAYER}.self_attn"
    with _loader(carrier, (0, 1), (prefix + ".",)) as loader:
        weights = loader.load_attention(MLA_LAYER)
        fused_weight = loader.load_mixed_linear_group(
            tuple(f"{prefix}.{name}" for name in ("q_a_proj", "kv_a_proj_with_mqa", "indexer.wk",
                                                 "indexer.weights_proj")))
        q_b_weight = loader.load_linear(prefix + ".q_b_proj", rows=(0, 64 * 256))
        kv_b_weight = loader.load_linear_group((prefix + ".kv_b_proj",), row_ranges=((0, 64 * 512),))
        o_weight = loader.load_linear(prefix + ".o_proj", columns=(0, 64 * 256))
    assert isinstance(weights["kv_b_proj"], torch.Tensor), "MLA absorption must keep BF16 storage"
    assert weights["kv_b_proj"].dtype == torch.bfloat16

    capacity = 1 << (rows - 1).bit_length()
    sequences = rows if mode == "decode" else 1
    source = torch.zeros((capacity, 4096), dtype=torch.bfloat16, device=device)
    source[:rows].normal_(std=.1, generator=torch.Generator(device=device).manual_seed(11))
    scratch = MLAScratch.allocate(capacity, heads, 64, mode, device)
    state = MLAState.allocate(128, 4, device)
    metadata = IndexMetadata.allocate(capacity, sequences, 64, device)
    arena = ScratchArena(device, overlap=True)
    with RecordedPreparationSession(device=device, checkpoint=CHECKPOINT,
                                    workload_schema="glm53-bf16-mla-v1",
                                    numerical_policy="default") as session:
        session.register_scratch(arena)
        attention = PreparedGlmMLA(module, weights, source, scratch, state, metadata, mode=mode,
                                   session=session, name="mla", workspace=arena)
        # Decode projects and absorbs queries on the auxiliary stream; prefill does not.
        assert (attention.query_chain is not None) is (mode == "decode")
        session.tune()
        state.cache.zero_()
        state.index.cache.zero_()
        state.index.tail.normal_(std=.1)
        metadata.block_table.copy_(
            torch.randperm(128, device=device, dtype=torch.int32).reshape(2, 64)[:sequences])
        metadata.state_slots.copy_(torch.tensor([2, 1][:sequences], dtype=torch.int64, device=device))
        metadata.query_start_loc.copy_(torch.tensor(
            list(range(rows + 1)) if mode == "decode" else [0, rows], dtype=torch.int32, device=device))
        if mode == "decode":
            metadata.requests.copy_(torch.arange(rows, dtype=torch.int32, device=device))
            metadata.positions[:rows].copy_(torch.tensor([255, 3][:rows], dtype=torch.int64, device=device))
        else:
            metadata.positions[:rows].copy_(torch.arange(rows, device=device, dtype=torch.int64) + 3)
        metadata.token_slots[:rows].copy_(
            metadata.block_table[metadata.requests[:rows].long(), metadata.positions[:rows] // 256].long()
            * 256 + metadata.positions[:rows] % 256)
        # The attention output buffer is reused once the expansion has read it, so record it there.
        recorded = torch.zeros((capacity, heads, 512), dtype=torch.bfloat16, device=device)
        expansion = attention.value_expansion

        def record(values):
            recorded.copy_(values.transpose(0, 1))
            return expansion(values)

        attention.value_expansion = record
        attention(source)
        torch.cuda.synchronize()
        # The BF16 projections, including the decode query chain, capture into a graph and replay
        # bitwise identically.
        expected = scratch.output[:rows].clone()
        graph = torch.cuda.CUDAGraph()
        compiled = compile_forward(lambda: attention(source))
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        torch.cuda.synchronize()
        scratch.output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(scratch.output[:rows], expected), "captured MLA replay differs"
    torch.cuda.synchronize()

    # Each reference consumes the layer's own upstream BF16 buffer, isolating one projection or norm.
    projected = project(source[:rows], fused_weight)
    query_residual = rms_norm_reference(scratch.query_input[:rows], module.q_a_layernorm.weight,
                                        module.q_a_layernorm.variance_epsilon)
    query = project(scratch.query_residual[:rows], q_b_weight).view(rows, heads, 256)
    latent = rms_norm_reference(scratch.latent_input[:rows], module.kv_a_layernorm.weight,
                                module.kv_a_layernorm.variance_epsilon)
    values = kv_b_weight.view(heads, 512, 512)
    absorbed = torch.einsum("thk,hko->tho", scratch.query[:rows].float(), values[:, :256].float()).to(torch.bfloat16)
    expanded = torch.einsum("thk,hok->tho", recorded[:rows].float(), values[:, 256:].float()).to(torch.bfloat16)
    reference_output = project(scratch.expanded[:rows].reshape(rows, -1), o_weight)

    shares = [assert_bf16_bound(actual, expected, label) for label, actual, expected in (
        (f"qkv_a_proj {mode} M={rows}", scratch.projected_input[:rows], projected),
        (f"q_a norm {mode} M={rows}", scratch.query_residual[:rows], query_residual),
        (f"q_b_proj {mode} M={rows}", scratch.query[:rows], query),
        (f"kv_a norm {mode} M={rows}", scratch.latent[:rows], latent),
        (f"kv_b absorb {mode} M={rows}", scratch.absorbed_query[:rows], absorbed),
        (f"kv_b expand {mode} M={rows}", scratch.expanded[:rows], expanded),
        (f"o_proj {mode} M={rows}", scratch.output[:rows], reference_output))]
    assert min(shares) > .25, f"{mode} M={rows}: bitwise agreement {shares}"
