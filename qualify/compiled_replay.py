# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare compiled graph replay with the original prepared model forwards."""

from types import SimpleNamespace

import torch

from ignition.recipes.qwen38.decoder import PreparedDecoder
from qualify.replay_floor import ReplayCase, _byte_view


@torch.no_grad()
def check_compiled_replay(model, groups, output):
    """Check the 512-prompt/256-generation corpus after GPU profiling stops."""
    if any(len(case["prompt_tokens"]) != 512 or case["output_tokens"] != 256
           for group in groups for case in group["cases"]):
        raise ValueError("compiled replay comparison requires the 512/256-token corpus")
    functions = next(value for value in model.resources if isinstance(value, dict)
                     and isinstance(value.get("decode1"), PreparedDecoder))
    reference = SimpleNamespace(resources=model.resources, buffers=model.buffers,
        signatures=model.signatures, device=model.device, memory_report=model.memory_report,
        graphs=dict(model.graphs))
    graphs = {}
    stream = torch.cuda.Stream(device=model.device)
    steps = 0
    try:
        names = {"prefill512", *(f"decode{group['batch']}" for group in groups)}
        for name in sorted(names):
            stream.wait_stream(torch.cuda.current_stream(model.device))
            with torch.cuda.stream(stream):
                functions[name]()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                functions[name]()
            graphs[name] = graph
            reference.graphs[name] = graph
        for group in groups:
            cases, batch = group["cases"], group["batch"]
            expected_logits, expected_records = [], []
            runner = ReplayCase(reference)
            runner.prepare(cases)
            for step in range(256):
                if step:
                    with torch.cuda.stream(runner.stream):
                        reference.graphs[f"decode{batch}"].replay()
                    runner.stream.synchronize()
                expected_logits.append(runner.scratch.logits[:batch].cpu().clone())
                expected_records.append(runner.records())
            expected_state, _ = runner.snapshot(cases)
            runner = ReplayCase(model)
            runner.prepare(cases)
            for step in range(256):
                if step:
                    with torch.cuda.stream(runner.stream):
                        model.graphs[f"decode{batch}"].replay()
                    runner.stream.synchronize()
                actual = runner.scratch.logits[:batch].cpu()
                expected = expected_logits[step]
                location = f"batch={batch} repeat={group['repeat']} step={step}"
                if not torch.equal(_byte_view(actual), _byte_view(expected)):
                    torch.save({"group": group, "step": step, "expected": expected,
                        "actual": actual, "reference_records": expected_records},
                        output / "compiled-logit-mismatch.pt")
                    maximum = (actual.float() - expected.float()).abs().max().item()
                    raise RuntimeError(f"compiled logits differ at {location}: max absolute error={maximum}")
                if runner.records() != expected_records[step]:
                    raise RuntimeError(f"compiled completion records differ at {location}")
                steps += 1
            actual_state, _ = runner.snapshot(cases)
            differences = [name for name in expected_state if not torch.equal(
                _byte_view(actual_state[name]), _byte_view(expected_state[name]))]
            if differences:
                raise RuntimeError(f"compiled final state differs at {location}: {differences}")
            print(f"compiled parity batch={batch} repeat={group['repeat']}: "
                  "256 logits/records and final request/hybrid/KV state match", flush=True)
        return {"steps": steps, "groups": len(groups), "status": "passed"}
    finally:
        torch.cuda.synchronize(model.device)
        for graph in graphs.values():
            graph.reset()
