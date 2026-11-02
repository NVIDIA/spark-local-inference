# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-context latency measurements of complete captured generation and prefill graphs."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import random
import statistics


@dataclass(frozen=True)
class BenchmarkCase:
    batch: int
    width: int
    packed_tokens: int
    context_tokens: int

    def __post_init__(self):
        if (self.batch not in (1, 2, 4) or self.width != 8
                or self.packed_tokens != self.batch * self.width or self.context_tokens <= 0):
            raise ValueError("round benchmark requires B=1,2,4, W=8, P=B*W and positive context")


@dataclass(frozen=True)
class PrefillCase:
    chunk_tokens: int
    context_tokens: int
    batch: int = 1

    def __post_init__(self):
        if (self.batch != 1 or self.chunk_tokens < 128 or self.chunk_tokens % 128
                or self.context_tokens < 0 or self.context_tokens % 128):
            raise ValueError("prefill benchmark requires batch one, a positive 128-aligned chunk and nonnegative 128-aligned context")


def summarize_case(case, samples):
    """Use all replay samples equally within a case, and retain seed-local raw times."""
    times = [value for sample in samples for value in sample["times_ms"]]
    if not times or any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError("graph latencies must be finite and positive")
    ordered = sorted(times)

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    median = statistics.median(times)
    p25, p75 = quantile(.25), quantile(.75)
    return dict(asdict(case), median_ms=median, min_ms=min(times), max_ms=max(times),
                p25_ms=p25, p75_ms=p75, relative_iqr=(p75 - p25) / median, samples=samples)


_SUBJECTS = (
    "database recovery and write-ahead logging",
    "satellite navigation and orbital mechanics",
    "distributed scheduling and bounded queues",
    "botanical field observations and seasonal change",
    "numerical integration and error estimation",
    "historical maps and maritime trade",
)


def prompt_tokens(tokenizer, length, *, seed, lane):
    """Tokenize a seeded long-form document and use its exact-length prefix."""
    rng = random.Random((seed << 16) + lane)
    subject = _SUBJECTS[rng.randrange(len(_SUBJECTS))]
    paragraphs = [f"Read the following technical notes about {subject}. "
                  "Explain their assumptions and identify the most useful conclusions.\n\n"]
    tokens = []
    while len(tokens) < length:
        for _ in range(max(16, length // 48)):
            index = len(paragraphs)
            count = rng.randrange(3, 97)
            paragraphs.append(
                f"Section {index}: The study of {subject} uses {count} observations. "
                "Each observation includes a recorded condition, a measurement, and an uncertainty. "
                "Independent checks compare the result with the stated assumptions. "
                "A useful explanation follows the dependencies and accounts for boundary cases.\n\n")
        encoded = tokenizer.encode("".join(paragraphs), add_special_tokens=True)
        tokens = [int(token) for token in (encoded.ids if hasattr(encoded, "ids") else encoded)]
        if not tokens:
            raise ValueError("checkpoint tokenizer produced an empty benchmark prompt")
    return tokens[:length]


class StateSnapshot:
    """Same-device copies of explicitly declared persistent tensors, outside replay timing."""

    def __init__(self, tensors, stream):
        import torch

        self.entries = {}
        seen = set()
        with torch.cuda.stream(stream), torch.no_grad():
            for name, tensor in tensors.items():
                if tensor.device.type != "cuda":
                    raise ValueError(f"benchmark state {name} must be a CUDA tensor")
                key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
                if key in seen:
                    continue
                seen.add(key)
                saved = torch.empty_like(tensor)
                saved.copy_(tensor, non_blocking=True)
                self.entries[name] = (tensor, saved)
        stream.synchronize()

    def restore(self, stream):
        import torch

        with torch.cuda.stream(stream), torch.no_grad():
            for tensor, saved in self.entries.values():
                tensor.copy_(saved, non_blocking=True)


def _message(scheduler, step):
    return dict(graph=step.graph, uploads=scheduler.pack(step), frame=step.frame, sequence=step.sequence)


def _complete(model, runner, scheduler, step, elapsed=None):
    confidence = (model.buffers["proposal_confidence"][step.frame].cpu().tolist()
                  if step.graph in scheduler.layout.confidence_widths else None)
    records = runner.records(step.frame)
    scheduler.complete(step, records, elapsed, confidence=confidence)
    errors = [(request.ident, request.error) for request in scheduler.slots
              if request is not None and request.error is not None]
    errors.extend((lane.request.ident, lane.request.error) for lane in step.lanes
                  if lane.request.error is not None)
    if errors:
        raise RuntimeError(f"prepared model reported benchmark device errors: {errors}")
    return records


def _prefill(model, runner, layout, case, tokenizer, seed, generation):
    from ignition.direct.scheduler import Request, SamplingParameters, Scheduler

    scheduler = Scheduler(layout, prefix_cache=False)
    scheduler.generations[:] = [generation - 1] * layout.max_active
    defaults = model.server_config.get("sampling_defaults", {})
    fields = set(SamplingParameters.__dataclass_fields__) - {"seed"}
    prefill = isinstance(case, PrefillCase)
    prompt_length = case.context_tokens + (case.chunk_tokens if prefill else 0)
    output_tokens = 2 if prefill else 2 * case.width
    for lane in range(case.batch):
        sampling = SamplingParameters(seed=(seed + lane) % (1 << 64),
                                      **{key: defaults[key] for key in fields if key in defaults})
        request = Request(lane + 1, prompt_tokens(tokenizer, prompt_length, seed=seed, lane=lane),
                          output_tokens, sampling, ignore_eos=True)
        scheduler.submit(request)
    while (scheduler.queue or any(
            (request.phase != "prefill" or request.consumed < case.context_tokens) if prefill
            else request.phase != "decode" for request in scheduler.active)):
        prefilling = [request for request in scheduler.active if request.phase == "prefill"]
        if prefilling:
            from ignition.direct.scheduler import Lane

            request = prefilling[0]
            count = min(case.context_tokens - request.consumed, layout.prefill_chunk)
            step = scheduler._step("prefill", count, [Lane(request, request.consumed, count)])
        else:
            step = scheduler.plan()
        if step is None or step.kind not in ("init", "prefill", "resume"):
            raise RuntimeError(f"benchmark prefill could not reach context {case.context_tokens}")
        elapsed = runner.run(_message(scheduler, step))
        records = _complete(model, runner, scheduler, step, elapsed)
        completed = [lane.request for lane in step.lanes if lane.request.done]
        if completed:
            state = model.benchmark_requests
            details = dict(case=asdict(case), seed=seed, graph=step.graph, kind=step.kind,
                           sequence=step.sequence, eos_token_ids=list(layout.eos), requests=[])
            for request in completed:
                slot = request.slot
                details["requests"].append(dict(
                    id=request.ident, slot=slot, generation=request.generation,
                    consumed=request.consumed, generated=request.generated,
                    max_new_tokens=request.max_new_tokens, finish_reason=request.finish_reason,
                    completion=records[slot].tolist(),
                    device_status=int(state.status[slot].item()),
                    device_max_new_tokens=int(state.max_new_tokens[slot].item()),
                    device_pending_token=int(state.pending_token[slot].item()),
                    device_eos_count=int(state.eos_count[slot].item()),
                    device_eos_token_ids=state.eos_token_ids[slot].cpu().tolist()))
            raise RuntimeError("benchmark prompt completed before its measured graph: "
                               + json.dumps(details, sort_keys=True))
    if len(scheduler.active) != case.batch or any(request.consumed != case.context_tokens
                                                 for request in scheduler.active):
        raise RuntimeError("real prefill did not produce the requested fixed context")
    return scheduler


def _round(scheduler, case):
    from ignition.direct.scheduler import Lane, Step

    matches = [(shape, graph) for shape, graph in scheduler.layout.graphs.items()
               if len(shape) == 5 and shape[:4] == ("speculative", case.batch, case.width, case.packed_tokens)]
    if len(matches) != 1:
        raise ValueError(f"prepared catalog must contain exactly one complete round for {case}")
    shape, graph = matches[0]
    lanes = [Lane(request, request.consumed, case.width) for request in scheduler.active]
    return scheduler._commit(Step("speculative", graph, case.batch, case.width, lanes, 0,
                                  packed_tokens=case.packed_tokens, draft_tokens=shape[4]))


def _prefill_step(scheduler, case):
    from ignition.direct.scheduler import Lane

    requests = scheduler.active
    if len(requests) != 1 or requests[0].phase != "prefill" or requests[0].consumed != case.context_tokens:
        raise RuntimeError("prefill benchmark state does not match the requested prefix")
    step = scheduler._step("prefill", case.chunk_tokens,
                           [Lane(requests[0], case.context_tokens, case.chunk_tokens)])
    if step.width != case.chunk_tokens:
        raise ValueError("prefill benchmark requires an exact chunk graph without capacity padding")
    return step


def _check_request_state(model, step, records):
    requests = model.benchmark_requests
    for lane in step.lanes:
        request = lane.request
        slot = request.slot
        if (int(requests.consumed_tokens[slot].item()) != request.consumed
                or int(requests.generated_tokens[slot].item()) != len(request.generated)
                or int(requests.generation[slot].item()) != request.generation
                or int(requests.pending_token[slot].item()) != int(records[slot, 4])
                or int(requests.seed[slot].item()) != request.sampling.seed
                or int(requests.rng_counter[slot].item()) != len(request.generated)
                or int(requests.histogram[slot].sum().item()) != len(request.prompt) + len(request.generated)):
            raise RuntimeError(f"request {request.ident} device state disagrees with graph completion")


def _check_metadata(model, step, case):
    if isinstance(case, PrefillCase):
        metadata = model.benchmark_prefills[step.graph]
        if (metadata.live_counts.cpu().tolist() != [case.chunk_tokens, 1]
                or metadata.starts.cpu().tolist() != [case.context_tokens]
                or metadata.lengths.cpu().tolist() != [case.chunk_tokens]
                or metadata.query_start_loc.cpu().tolist() != [0, case.chunk_tokens]):
            raise RuntimeError("prefill metadata does not describe the exact benchmark chunk")
        return
    forward = model.benchmark_rounds[step.graph]
    for role, metadata, width in (("target", forward.metadata, case.width),
                                  ("draft", forward.draft_metadata, forward.draft_inputs.shape[1])):
        expected_starts = [case.context_tokens] * case.batch
        expected_lengths = [width] * case.batch
        expected_offsets = [row * width for row in range(case.batch + 1)]
        if (metadata.live_counts.cpu().tolist() != [case.batch * width, case.batch]
                or metadata.starts.cpu().tolist() != expected_starts
                or metadata.lengths.cpu().tolist() != expected_lengths
                or metadata.query_start_loc.cpu().tolist() != expected_offsets):
            raise RuntimeError(f"{role} metadata does not describe the complete benchmark round")
    owner = forward.owner
    verification = forward.operations[2]
    for lane in step.lanes:
        slot, generation = lane.request.slot, lane.request.generation
        if (int(owner.generation[slot].item()) != generation
                or int(owner.offset[slot].item()) != 4 * step.draft_tokens
                or int(owner.rng.generation[slot].item()) != generation
                or int(owner.rng.target_offset[slot].item()) != verification.draw_reservation):
            raise RuntimeError("speculative sampling streams did not consume their declared draw reservations")


def acceptance_observation(forward):
    """Read one input/RNG draw; restored timing replays are not additional observations."""
    _, _, verification, commit = forward.operations
    acceptance = verification.output
    offered = acceptance.offered_drafts.cpu().tolist()
    accepted = acceptance.accepted_drafts.cpu().tolist()
    committed = commit.accepted_drafts.cpu().tolist()
    emitted = commit.emitted_count.cpu().tolist()
    tokens = commit.emitted_ids.cpu().tolist()
    survival = acceptance.survival.cpu().tolist()
    valid = acceptance.valid.cpu().tolist()
    return dict(
        observations_per_request=1,
        survival_semantics="cumulative acceptance probability along the sampled proposal chain",
        requests=[dict(valid=bool(valid[row]), offered_draft_count=int(count),
                       accepted_draft_count=int(accepted[row]),
                       committed_draft_count=int(committed[row]), emitted_count=int(emitted[row]),
                       emitted_token_ids=tokens[row][:emitted[row]],
                       proposal_survival=survival[row][:count])
                  for row, count in enumerate(offered)],
    )


def measurement_result(cases, measurements, *, warmup, repetitions, seeds, status, failure=None):
    results = []
    for case, samples in zip(cases, measurements):
        if not samples:
            continue
        complete = len(samples) == len(seeds) and all(len(sample["times_ms"]) == repetitions for sample in samples)
        summary = (summarize_case(case, samples) if any(sample["times_ms"] for sample in samples)
                   else dict(asdict(case), samples=samples))
        results.append(dict(summary, complete=complete))
    metric = "cuda_graph_prefill_latency_ms" if isinstance(cases[0], PrefillCase) else "cuda_graph_round_latency_ms"
    result = dict(metric=metric, status=status, ignore_eos=True,
                  warmup=warmup, repetitions=repetitions, seeds=list(seeds), cases=results)
    if status == "complete":
        if len(results) != len(cases) or not all(case["complete"] for case in results):
            raise ValueError("complete benchmark results require every case, seed and repetition")
        result["aggregate_ms"] = statistics.mean(case["median_ms"] for case in results)
    if failure is not None:
        if results:
            results[-1]["complete"] = False
        result["failure"] = failure
    return result


def run_benchmark(model, tokenizer, *, scenarios, warmup=5, repetitions=20, seeds=(0, 1, 2), progress=None):
    """Measure captured graphs at a real prefix, restoring state before every replay."""
    import torch

    from ignition.direct.executor import StepRunner
    from ignition.direct.scheduler import Layout

    cases, seeds = tuple(scenarios), tuple(seeds)
    if not cases or not seeds or warmup < 1 or repetitions < 1:
        raise ValueError("benchmark needs cases, seeds, positive warmup and positive repetitions")
    if any(type(seed) is not int or not 0 <= seed < 1 << 64 for seed in seeds):
        raise ValueError("benchmark seeds must be uint64 integers")
    if not getattr(model, "benchmark_state", None):
        raise ValueError("model preparation did not declare complete benchmark state")
    layout = Layout.of(model)
    for case in cases:
        required = case.chunk_tokens + 2 if isinstance(case, PrefillCase) else 2 * case.width
        if case.context_tokens + required > layout.context or case.batch > layout.max_active:
            raise ValueError(f"benchmark case exceeds prepared capacities: {case}")
        if isinstance(case, PrefillCase):
            if ("prefill", 1, case.chunk_tokens) not in layout.graphs or case.chunk_tokens != layout.prefill_chunk:
                raise ValueError(f"benchmark case has no exact prepared prefill chunk: {case}")
            continue
        _round_shape = ("speculative", case.batch, case.width, case.packed_tokens)
        if sum(shape[:4] == _round_shape for shape in layout.packed_shapes) != 1:
            raise ValueError(f"benchmark case has no unique prepared round: {case}")
    runner = StepRunner(model, eager=False)
    measurements = [[] for _ in cases]

    def result(status, failure=None):
        return measurement_result(cases, measurements, warmup=warmup, repetitions=repetitions,
                                  seeds=seeds, status=status, failure=failure)

    generation = int(model.benchmark_requests.generation.max().item())
    try:
        for case, samples in zip(cases, measurements):
            for seed in seeds:
                times = []
                samples.append(dict(seed=seed, times_ms=times))
                generation += 1
                baseline = _prefill(model, runner, layout, case, tokenizer, seed, generation)
                snapshot = StateSnapshot(model.benchmark_state, runner.stream)
                for iteration in range(-1, warmup + repetitions):
                    scheduler = deepcopy(baseline, {id(layout): layout})
                    snapshot.restore(runner.stream)
                    step = _prefill_step(scheduler, case) if isinstance(case, PrefillCase) else _round(scheduler, case)
                    allocated = torch.cuda.memory_allocated(runner.device)
                    elapsed = runner.run(_message(scheduler, step))
                    if iteration >= warmup:
                        times.append(elapsed / 1e6)
                    if torch.cuda.memory_allocated(runner.device) != allocated:
                        raise RuntimeError("captured benchmark replay changed allocated CUDA memory")
                    records = _complete(model, runner, scheduler, step, elapsed)
                    _check_request_state(model, step, records)
                    if isinstance(case, PrefillCase) and any(
                            lane.request.consumed != case.context_tokens + case.chunk_tokens
                            or len(lane.request.generated) != 1 for lane in step.lanes):
                        raise RuntimeError("prefill replay did not consume the complete chunk and sample its pending token")
                    if iteration == -1:
                        _check_metadata(model, step, case)
                        if not isinstance(case, PrefillCase):
                            samples[-1]["acceptance"] = acceptance_observation(model.benchmark_rounds[step.graph])
                if progress is not None:
                    progress(result("running"))
                snapshot.restore(runner.stream)
                runner.stream.synchronize()
                del snapshot
    except BaseException as error:
        if progress is not None:
            try:
                progress(result("incomplete", failure=f"{type(error).__name__}: {error}"))
            except Exception as output_error:
                error.add_note(f"could not save incomplete benchmark measurements: {output_error}")
        raise
    finally:
        runner.close()
    return result("complete")
