# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless DeepSeek expert loading and compressed-scale replay with shared scratch."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes import resolve

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def expanded_scale(read, name, rows, columns):
    fixed = read(name + ".mxfp4_csf_fixed").numpy().reshape(rows // 16, -1)
    base = fixed[:, :16].reshape(rows, 1)
    bits = np.unpackbits(fixed[:, 16:].reshape(rows, -1), axis=1, bitorder="little")[:, :columns]
    result = (base + bits).copy()
    words = read(name + ".mxfp4_csf_exceptions").numpy()
    result.reshape(-1)[words & 0xFFFFFF] = words >> 24
    return torch.from_numpy(result)


@pytest.fixture(scope="module", params=[("deepseek-v4-flash", 4096, 2048, 2),
                                       ("deepseek-v4-flash", 4096, 2048, 4),
                                       ("deepseek-v4.1-flash", 5120, 2304, 4),
                                       pytest.param((None, 5120, 2304, 4), id="native-conversion-tp4")])
def experts(request, tmp_path_factory):
    from b12x.moe import fused_moe as moe

    alias, h, full, world = request.param
    if alias is None:
        root = tmp_path_factory.mktemp("native-mxfp4")
        gen = torch.Generator().manual_seed(192)
        tensors = {}
        for layer in (0, 1):
            for expert in (0, 1):
                for projection in ("w1", "w3", "w2"):
                    rows, columns = (h, full) if projection == "w2" else (full, h)
                    prefix = f"layers.{layer}.ffn.experts.{expert}.{projection}"
                    tensors[prefix + ".weight"] = torch.randint(-128, 128, (rows, columns // 2),
                        dtype=torch.int8, generator=gen)
                    tensors[prefix + ".scale"] = torch.randint(121, 125, (rows, columns // 32),
                        dtype=torch.uint8, generator=gen).view(torch.float8_e8m0fnu)
        save_file(tensors, str(root / "weights.safetensors"))
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": {name: "weights.safetensors" for name in tensors}}))
        (root / "config.json").write_text("{}")
        del tensors
    else:
        model = resolve(alias)
        hub = Path.home() / ".cache/huggingface/hub"
        root = model.snapshot(hub)
        if not model.cached(hub):
            pytest.skip(f"{alias} checkpoint is unavailable")
    manifest = inspect_indexed_checkpoint(root)
    e, n = 2, full // world
    first, last = n, 2 * n
    shared = {}
    owners = []
    for layer in (0, 1):
        prefixes = [f"layers.{layer}.ffn.experts.{i}" for i in range(e)]
        with DirectWeightLoader(manifest, "cuda:0", prefixes=tuple(p + "." for p in prefixes)) as loader:
            compressed = loader.load_mxfp4_experts(prefixes, hidden_size=h, intermediate_size=full,
                intermediate_range=(first, last), scale_scratch=shared, scale_compression="csf",
                projection_names=("w3", "w1", "w2"), scale_name="scale", swiglu_limit=10.)
            assert loader._session.torch_copy_bytes == 0
        buffers = shared[(e, h, n)]
        assert compressed.has_csf_blocks
        assert compressed._impl.w1_blockscale.data_ptr() == buffers[0].data_ptr()
        assert compressed._impl.w2_blockscale.data_ptr() == buffers[1].data_ptr()
        w13, w2, s13, s2 = [], [], [], []
        order = ("w1", "w3") if compressed.plan.source.w13_layout.value == "w31" else ("w3", "w1")
        for prefix in prefixes:
            path = root / manifest.tensors[prefix + ".w1.weight"].shard
            with safe_open(path, framework="pt", device="cpu") as reader:
                read = reader.get_tensor
                def scales(name, rows, columns):
                    return (expanded_scale(read, name, rows, columns)
                            if name in getattr(manifest, "compressed_scales", ())
                            else read(name).view(torch.uint8))
                w13.append(torch.cat([read(prefix + f".{p}.weight")[first:last].view(torch.uint8) for p in order]))
                w2.append(read(prefix + ".w2.weight")[:, first // 2:last // 2].view(torch.uint8))
                s13.append(torch.cat([scales(prefix + f".{p}.scale", full, h // 32)[first:last] for p in order]))
                s2.append(scales(prefix + ".w2.scale", h, full // 32)[:, first // 32:last // 32])
        one = torch.ones(e, device="cuda:0")
        ordinary_plan = moe.plan_weights(source=compressed.plan.source, activation=compressed.plan.activation,
                                         geometry=compressed.plan.geometry)
        ordinary = moe.prepare_weights(plan=ordinary_plan, weights=moe.PackedWeights(
            w13=torch.stack(w13).cuda(), w2=torch.stack(w2).cuda(),
            w13_block_scales=torch.stack(s13).cuda(), w2_block_scales=torch.stack(s2).cuda(),
            w13_global_scales=one, w2_global_scales=one))
        for name in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale"):
            assert torch.equal(getattr(compressed._impl, name), getattr(ordinary._impl, name)), name
        owners.append((compressed, ordinary))
    return owners, buffers, h


@pytest.mark.parametrize("tokens", [1, 32, 256, 4096])
@pytest.mark.parametrize("compiled", [False, True])
@torch.no_grad()
def test_csf_experts_match_expanded_scales_with_poisoned_shared_scratch(experts, tokens, compiled):
    from b12x.preparation import PreparationSession
    from b12x.moe import fused_moe as moe
    from ignition.components.moe import PreparedRoutedExperts

    owners, buffers, hidden = experts
    torch.manual_seed(754 + tokens)
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="cuda:0") * .1
    ids = (torch.arange(tokens, device="cuda:0") % 2).to(torch.int32)[:, None]
    weights = torch.ones((tokens, 1), dtype=torch.float32, device="cuda:0")
    graphs = []
    with PreparationSession(device=x.device, autotune=False, compile_workers=0) as session:
        for layer in owners:
            pair = []
            for owner in layer:
                out = torch.empty_like(x)
                # Atomic output accumulation can vary even between two expanded-scale calls.
                # Deterministic reduction makes exact equality a valid storage-format oracle.
                operation = PreparedRoutedExperts(x, owner, ids, weights, out, session=session,
                    name=f"expert-{len(graphs)}-{len(pair)}", routing=moe.RoutingSpec(deterministic_output=True))
                call = operation
                if compiled:
                    from ignition.preparation.compilation import compile_forward
                    call = compile_forward(operation)
                call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call()
                pair.append((graph, out, call))
            graphs.append(pair)
        for index in (0, 1, 0, 1):
            for buffer in buffers:
                buffer.fill_(255)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            for graph, _, _ in graphs[index]:
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(graphs[index][0][1], graphs[index][1][1], rtol=0, atol=0)
