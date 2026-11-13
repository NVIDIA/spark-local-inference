# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure captured production samplers on saved Qwen checkpoint logits."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD"
REVISION = "dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2"
SLOTS = (2, 0, 3, 1)


def load_inputs(path):
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    logits = artifact["logits"]
    if len(logits) != 4 or any(t.dtype != torch.bfloat16 or t.shape != (248320,)
                              or not bool(t.isfinite().all()) for t in logits):
        raise ValueError("require four finite BF16 full-vocabulary checkpoint logits rows")
    if [int(t.argmax()) for t in logits] != artifact["tokens"]:
        raise ValueError("recorded greedy tokens differ from the saved logits")
    if not artifact["prompt"] or any(type(t) is not int or not 0 <= t < 248320
                                     for t in artifact["prompt"] + artifact["tokens"]):
        raise ValueError("checkpoint prompt and generated tokens must be valid vocabulary IDs")
    return artifact


def cpu_case(artifact, rows, block_steps, policy, *, one_sample=False):
    import torch

    from ignition.components.sampling import philox4x32_reference
    from ignition.recipes.qwen38.sampling import DeviceRequestState, RequestAdmission

    if policy != "checkpoint_defaults":
        raise ValueError("sampling paths require checkpoint_defaults")
    steps = 1 if one_sample else block_steps
    initial = DeviceRequestState.allocate("cpu")
    parameters = []
    for row, slot in enumerate(SLOTS[:rows]):
        prefix = artifact["prompt"] + artifact["tokens"][:row]
        parameter = RequestAdmission(generation=31 + slot, seed=0xF123000000000001 + slot * 0x100000001,
            max_new_tokens=block_steps + 1, temperature=1.,
            top_k=20, top_p=.95)
        initial.admit(slot, parameter, prefix, consumed_tokens=len(prefix), ple_history=tuple(prefix[-2:]))
        initial.rng_counter[slot] = row * 7
        parameters.append(parameter)
    state = {name: getattr(initial, name).clone() for name in initial.__dataclass_fields__}
    expected = {name: value.clone() for name, value in state.items()}
    trace = torch.full((steps, 4), -1, dtype=torch.int64)
    records = torch.full((2, 4, 24), -99, dtype=torch.int64)
    def completion_record(epoch=0, consumed=0, generated=0, token=-1, status=0, *, active=False):
        return torch.tensor([epoch, int(active), consumed, generated, token, status, 0, 0,
            int(token >= 0), 0, 0, 0, -1, max(consumed - 1, 0) if active else 0,
            0, 1, token, *([-1] * 7)], dtype=torch.int64)
    records[0].copy_(completion_record().expand(4, -1))
    emitted = torch.zeros((steps, 4), dtype=torch.int64)
    vector_tokens = torch.full((steps, 4), -1, dtype=torch.int64)
    sampled_rows = [0] * steps
    consumed_rows = [0] * steps
    terminals = []
    for row, slot in enumerate(SLOTS[:rows]):
        scores = artifact["logits"][row].float()
        values, ids = torch.sort(scores, descending=True, stable=True)
        values[values < values[19].clone()] = -torch.inf
        probabilities = values.softmax(0)
        cumulative = probabilities.cumsum(0)
        starts = torch.searchsorted(-values, -values, right=False)
        probabilities[(cumulative - probabilities)[starts] >= torch.tensor(.95, dtype=torch.float32)] = 0
        probabilities /= probabilities.sum()
        probabilities = torch.zeros_like(probabilities).scatter_(0, ids, probabilities)
        cdf = probabilities.cumsum(0).double()
        for step in range(steps):
            if int(expected["status"][slot]) != 1:
                records[0, slot].copy_(completion_record(parameters[row].generation,
                    int(expected["consumed_tokens"][slot]), int(expected["generated_tokens"][slot]),
                    status=int(expected["status"][slot]), active=True))
                continue
            counter = int(expected["rng_counter"][slot].item())
            word = philox4x32_reference(parameters[row].seed, counter)[0]
            uniform = float(torch.tensor(word, dtype=torch.float32) * 2**-32 + 2**-33)
            index = int(torch.searchsorted(cdf, torch.tensor(uniform, dtype=torch.float64), right=False))
            token = index if index < len(cdf) else int(torch.nonzero(probabilities > 0)[-1])
            trace[step, slot] = token
            vector_tokens[step, slot] = token
            emitted[step, slot] = 1
            sampled_rows[step] += 1
            consumed = int(expected["consumed_tokens"][slot])
            expected["generated_tokens"][slot] += 1
            expected["rng_counter"][slot] = counter + 1
            expected["histogram"][slot, token] += 1
            terminal = token in parameters[row].eos_token_ids
            records[0, slot].copy_(completion_record(parameters[row].generation, consumed,
                int(expected["generated_tokens"][slot]), token, 2 if terminal else 1, active=True))
            if terminal:
                expected["status"][slot] = 2
                expected["pending_token"][slot] = token
                terminals.append({"row": row, "slot": slot, "step": step, "token": token,
                    "seed": parameters[row].seed, "rng_counter_before": counter})
                continue
            consumed_rows[step] += 1
            previous = int(expected["ple_history"][slot, 1])
            expected["ple_history"][slot] = torch.tensor([previous, token])
            expected["consumed_tokens"][slot] = consumed + 1
    expected = {"state." + name: value for name, value in expected.items()}
    expected.update(sampled_ids=trace, emitted_counts=emitted, vector_tokens=vector_tokens,
                    completion_records=records)
    return {"name": f"{policy}-b{rows}-n{block_steps}" + ("-one-sample" if one_sample else ""),
        "rows": rows, "steps": steps, "budget_steps": block_steps, "policy": policy,
        "timing_scope": "one_active_sample" if one_sample else "block_with_terminal_semantics",
        "sampled_rows_per_step": sampled_rows, "consumed_rows_per_step": consumed_rows,
        "sampled_lane_tokens": sum(sampled_rows), "consumed_lane_tokens": sum(consumed_rows), "terminals": terminals,
        "initial": state, "expected": expected, "parameters": [asdict(p) for p in parameters]}


class CapturedCase:
    def __init__(self, case, artifact, stream):
        import torch

        from ignition.components.sampling import PreparedSampling
        from ignition.components.completion import CompletionOutputs
        from ignition.components.token_state import PreparedTokenState
        from ignition.recipes.qwen38.sampling import DeviceRequestState

        self.case = case
        self.name = case["name"]
        rows, steps = case["rows"], case["steps"]
        self.state = DeviceRequestState.allocate("cuda")
        self.logits = torch.stack(artifact["logits"][:rows]).cuda()
        self.slots = torch.tensor(SLOTS[:rows], dtype=torch.int32, device="cuda")
        self.epochs = torch.tensor([31 + slot for slot in SLOTS[:rows]], dtype=torch.int64, device="cuda")
        self.live = torch.tensor([rows], dtype=torch.int32, device="cuda")
        self.completion = CompletionOutputs.allocate((2, 4, 24), device="cuda")
        self.records = self.completion.records
        self.records.fill_(-99)
        self.frame = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.sequence = torch.ones(1, dtype=torch.int64, device="cuda")
        self.trace = torch.full((steps, 4), -1, dtype=torch.int64, device="cuda")
        self.emitted = torch.zeros((steps, 4), dtype=torch.int64, device="cuda")
        self.vector_tokens = torch.full((steps, 4), -1, dtype=torch.int64, device="cuda")
        self.tokens = torch.zeros(rows, dtype=torch.int64, device="cuda")
        self.spans = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
        self.sampler = PreparedSampling(self.logits, self.state, self.slots, self.epochs, self.live,
            self.completion, self.frame, self.sequence)
        self.token_state = PreparedTokenState(self.state, self.tokens, self.spans, self.slots, self.epochs, self.live, history=self.state.ple_history, history_pad_id=248044)
        for name, value in case["initial"].items():
            getattr(self.state, name).copy_(value)
        self.sampler.prepare()
        self.token_state.prepare()
        self.buffers = {"state." + name: getattr(self.state, name) for name in self.state.__dataclass_fields__}
        self.buffers.update(self.completion.buffers())
        self.buffers.update(logits=self.logits, slots=self.slots, epochs=self.epochs, live=self.live,
            completion_records=self.records, frame=self.frame, sequence=self.sequence, sampled_ids=self.trace,
            emitted_counts=self.emitted, vector_tokens=self.vector_tokens,
            tokens=self.tokens, spans=self.spans, sample_enabled=self.sampler.sample_enabled)
        for name, tensor in self.sampler.scratch_tensors.items():
            self.buffers["sampler." + name] = tensor
        for name in ("pending_tokens", "positions", "history", "active"):
            self.buffers["token_state." + name] = getattr(self.token_state, name)
        self.initial = {name: value.clone() for name, value in self.buffers.items()}
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        self.start.record()
        self.end.record()
        self.step(0)
        self.reset()
        torch.cuda.synchronize()
        self.reset_graph, self.graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.reset_graph, stream=stream):
            self.reset()
        with torch.cuda.graph(self.graph, stream=stream):
            self.start.record(stream)
            for step in range(steps):
                self.step(step)
            self.end.record(stream)
        torch.cuda.synchronize()

    def reset(self):
        for name, value in self.buffers.items():
            value.copy_(self.initial[name])

    def step(self, step):
        self.sampler()
        self.trace[step].copy_(self.records[0, :, 4])
        self.emitted[step].copy_(self.records[0, :, 8])
        self.vector_tokens[step].copy_(self.records[0, :, 16])
        self.token_state.gather()
        self.tokens.copy_(self.token_state.pending_tokens)
        self.token_state.consume()

    def readback(self):
        return {name: self.buffers[name].cpu() for name in self.case["expected"]}


def quantiles(values):
    ordered = sorted(values)

    def quantile(q):
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"count": len(values), "minimum": ordered[0], "median": quantile(.5),
        "p90": quantile(.9), "p95": quantile(.95), "maximum": ordered[-1], "quantile_method": "linear"}


def check_outputs(expected, observed):
    import torch

    for name, reference in expected.items():
        actual = observed[name]
        assert actual.shape == reference.shape and actual.dtype == reference.dtype, name
        assert torch.equal(actual.contiguous().view(torch.uint8),
                           reference.contiguous().view(torch.uint8)), f"state mismatch: {name}"


def measure(args, artifact, cases):
    import torch

    from ignition.preparation.graphs import PreparedGraphBundle

    if not args.gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("--gpu-uuid must match the single assigned CUDA_VISIBLE_DEVICES UUID")
    if not args.gpu_uuid.startswith("GPU-"):
        raise ValueError("GPU selection must use its explicit UUID")
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if actual_uuid.removeprefix("GPU-") != args.gpu_uuid.removeprefix("GPU-"):
        raise ValueError(f"CUDA UUID {actual_uuid} differs from assigned {args.gpu_uuid}")
    stream = torch.cuda.Stream()
    owners = [CapturedCase(case, artifact, stream) for case in cases]
    graphs = {owner.name + suffix: graph for owner in owners
              for suffix, graph in (("-reset", owner.reset_graph), ("-sample", owner.graph))}
    buffers = {owner.name + "." + name: tensor for owner in owners for name, tensor in owner.buffers.items()}
    bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers=buffers, resources=(*owners, stream))
    timings = {owner.name: [] for owner in owners}
    observations = []
    worker = bundle.start_worker()

    def replay(name):
        ticket = worker.submit(bundle.graph_indices[name])
        result = worker.poll(timeout_ms=60000)
        if result is None or result.ticket != ticket or result.error is not None:
            raise RuntimeError(f"native replay did not complete {name}: {result}")

    try:
        for iteration in range(args.warmup + args.repetitions):
            order = owners if iteration % 2 == 0 else list(reversed(owners))
            for owner in order:
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                replay(owner.name + "-reset")
                replay(owner.name + "-sample")
                after = torch.cuda.memory_stats()["allocation.all.allocated"]
                elapsed = owner.start.elapsed_time(owner.end)
                assert math.isfinite(elapsed) and elapsed > 0
                check_outputs(owner.case["expected"], owner.readback())
                assert before == after, f"replay allocation in {owner.name}: {before} -> {after}"
                observation = dict(name=owner.name, iteration=iteration, warmup=iteration < args.warmup,
                    elapsed_block_ms=elapsed, sampled_lane_tokens=owner.case["sampled_lane_tokens"],
                    consumed_lane_tokens=owner.case["consumed_lane_tokens"])
                observations.append(observation)
                print(json.dumps(observation), flush=True)
                if iteration >= args.warmup:
                    timings[owner.name].append(elapsed)
    finally:
        worker.close()
    result = dict(checkpoint=dict(repository=MODEL, revision=REVISION),
        input=str(args.logits_artifact), gpu_uuid=actual_uuid, observations=observations,
        summary_block_ms={name: quantiles(values) for name, values in timings.items()},
        summary_active_batch_step_ms={owner.name: quantiles(timings[owner.name])
                                     for owner in owners if owner.case["steps"] == 1})
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits-artifact", type=Path, required=True,
        help=f"Saved four-row BF16 decoder logits from {MODEL} at {REVISION}; no generated substitutes.")
    parser.add_argument("--output", type=Path, help="Raw timing measurements as JSON.")
    parser.add_argument("--validate-only", action="store_true",
        help="Load checkpoint logits and construct CPU state oracles without CUDA initialization.")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--blocks", nargs="+", type=int, choices=(32, 64), default=(32, 64))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if not args.logits_artifact.is_file():
        parser.error(f"saved checkpoint logits are unavailable: {args.logits_artifact}")
    if args.output is not None and args.output.exists():
        parser.error("output already exists")
    if args.warmup < 1 or args.repetitions < 5 or len(set(args.blocks)) != len(args.blocks):
        parser.error("require at least one warmup, five repetitions and unique block sizes")
    import torch

    torch.set_num_threads(2)
    artifact = load_inputs(args.logits_artifact)
    cases = [cpu_case(artifact, rows, steps, "checkpoint_defaults", one_sample=one_sample)
             for rows in (1, 2, 4) for steps in args.blocks for one_sample in (False, True)]
    if args.validate_only:
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(passed=True, cases=len(cases), cuda_initialized=False)))
    else:
        measure(args, artifact, cases)


if __name__ == "__main__":
    main()
