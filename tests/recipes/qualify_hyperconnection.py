# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualify prepared hyperconnection against exact-checkpoint canonical weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from b12x.preparation import PreparationSession

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.qwen38 import TRANSFORMERS_REVISION
from ignition.preparation.compilation import ForwardTracer, compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.qwen38 import create_model_tree
from ignition.recipes.qwen38.hyperconnection import HyperconnectionScratch, PreparedHyperconnection


def qualify(snapshot: Path) -> list[dict]:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise RuntimeError("hyperconnection qualification requires the assigned SM120/121 GPU")
    manifest = inspect_qwen38_checkpoint(snapshot)
    tree = create_model_tree(manifest)
    prefix = "model.language_model.layers.0.attn_hyper_connection"
    module = tree.get_submodule(prefix)
    following_prefix = "model.language_model.layers.0.mlp_hyper_connection"
    following_module = tree.get_submodule(following_prefix)
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    for path, mixer in ((prefix, module), (following_prefix, following_module)):
        for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
            tensor_name = f"{path}.{name}.weight"
            with safe_open(snapshot / weight_map[tensor_name], framework="pt", device="cpu") as shard:
                weight = shard.get_tensor(tensor_name).to(device)
            mixer.get_submodule(name).weight = torch.nn.Parameter(weight, requires_grad=False)
    generator = torch.Generator().manual_seed(1607)
    reports = []
    with torch.inference_mode(), PreparationSession(device=device, autotune=True, compile_workers=0) as session:
        tracer = ForwardTracer()
        for tokens in (1, 4, 8, 16, 64, 4096):
            scratch = HyperconnectionScratch.allocate(tokens, device)
            source = torch.zeros((tokens, 10240), dtype=torch.bfloat16, device=device)
            prepared = PreparedHyperconnection(module, scratch, priming_state=source,
                session=session, name=f"hyperconnection_{tokens}")
            following = PreparedHyperconnection(following_module, scratch, priming_state=source,
                session=session, name=f"following_{tokens}")
            prepared.prepare_combine_norm(following, priming_state=source,
                session=session, name=f"combine_norm_{tokens}")
            for projection in (prepared.down, prepared.gate):
                assert projection.plan.selection.source in ("tuned", "cached")
            block = torch.zeros((tokens, 2560), dtype=torch.bfloat16, device=device)
            mixed = torch.empty_like(block)
            combined = torch.empty_like(source)
            normalized = torch.empty_like(source)
            following_mixed = torch.empty_like(block)
            unfused_combined = torch.empty_like(source)
            unfused_normalized = torch.empty_like(source)
            unfused_following = torch.empty_like(block)

            def run_unfused():
                prepared.mix(source)
                unfused_combined.copy_(source)
                prepared.combine(unfused_combined, block)
                unfused_following.copy_(following.mix(unfused_combined))
                unfused_normalized.copy_(scratch.normalized)

            def run():
                mixed.copy_(prepared.mix(source))
                prepared.combine_norm(source, block)
                combined.copy_(source)
                normalized.copy_(scratch.normalized)
                following_mixed.copy_(following.mix(source, pre_normalized=True))

            compiled_unfused = compile_forward(run_unfused, tracer=tracer)
            compiled = compile_forward(run, tracer=tracer)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                compiled_unfused()
                compiled()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                compiled()
            unfused_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(unfused_graph, stream=stream):
                compiled_unfused()
            bundle = PreparedGraphBundle(device=0, graphs={"unfused": unfused_graph, "hyperconnection": graph},
                buffers={"source": source, "block": block, "mixed": mixed, "combined": combined,
                         "normalized": normalized, "following_mixed": following_mixed,
                         "unfused_combined": unfused_combined, "unfused_normalized": unfused_normalized,
                         "unfused_following": unfused_following},
                resources=(module, prepared, following, session, stream, compiled_unfused, compiled))
            worker = bundle.start_worker()
            try:
                for iteration in range(2):
                    source.copy_(torch.randn(source.shape, dtype=torch.bfloat16, generator=generator).to(device) * .25)
                    block.copy_(torch.randn(block.shape, dtype=torch.bfloat16, generator=generator).to(device) * .25)
                    expected_mixed, residual, injection = module(source)
                    expected_combined = residual + (block[:, None, :] * injection[:, :, None]).flatten(1)
                    expected_following, _, _ = following_module(expected_combined)
                    scratch.gate_workspace.fill_(float("nan"))
                    torch.cuda.synchronize(device)
                    allocated = torch.cuda.memory_allocated(device)
                    ticket = worker.submit(0)
                    completion = worker.poll(timeout_ms=30000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                    ticket = worker.submit(1)
                    completion = worker.poll(timeout_ms=30000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                    assert torch.cuda.memory_allocated(device) == allocated
                    expected_normalized = following_module.hc_norm(combined)
                    expected_following_component, _, _ = following_module(combined)
                    for name, actual, expected in (("mix", mixed, expected_mixed), ("combine", combined, expected_combined),
                            ("normalized", normalized, expected_normalized),
                            ("following_mix", following_mixed, expected_following_component),
                            ("unfused_combine", combined, unfused_combined),
                            ("unfused_normalized", normalized, unfused_normalized),
                            ("unfused_following_mix", following_mixed, unfused_following)):
                        torch.testing.assert_close(actual, expected, rtol=.02, atol=.02,
                            msg=lambda error: f"{tokens=} {iteration=} {name=}\n{error}")
                        difference = (actual.float() - expected.float())
                        reports.append({"tokens": tokens, "iteration": iteration, "operation": name,
                            "down_backend": prepared.down.plan.selection.config.backend,
                            "gate_implementation": prepared.gate.plan.selection.config.implementation,
                            "max_absolute_error": difference.abs().max().item(),
                            "relative_l2_error": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()})
                    for label, values in (("fused", following_mixed), ("unfused", unfused_following)):
                        delta = values.float() - expected_following.float()
                        reports.append({"tokens": tokens, "iteration": iteration,
                            "operation": f"{label}_chained_canonical_mix", "diagnostic": True,
                            "max_absolute_error": delta.abs().max().item(),
                            "relative_l2_error": (delta.norm() / expected_following.float().norm().clamp_min(1e-12)).item()})
            finally:
                worker.close()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparisons = qualify(args.snapshot)
    report = {
        "component": "Qwen layer 0 attention hyperconnection",
        "snapshot": str(args.snapshot.resolve()),
        "transformers_revision": TRANSFORMERS_REVISION,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "seed": 1607,
        "compiled_forwards": True,
        "poisoned_gate_workspace": True,
        "rtol": .02,
        "atol": .02,
        "comparisons": comparisons,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
