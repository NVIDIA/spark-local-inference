"""Single-request speculative depth measurements outside the native graph catalog."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from ignition.bench.executor import StateSnapshot
from ignition.bench.runner import (BenchmarkCase, _check_metadata, _prefill, _round,
                                   acceptance_observation, measurement_result)


@dataclass(frozen=True)
class StudyCase:
    batch: int
    width: int
    packed_tokens: int
    context_tokens: int

    def __post_init__(self):
        if self.batch != 1 or self.width not in (8, 13, 16) or self.packed_tokens != self.width:
            raise ValueError("speculation study requires B=1 and W=P in 8,13,16")


@dataclass(frozen=True)
class StudySignature:
    width: int
    block_size: int
    kind: str = "speculative"


class SpeculationStudy:
    def __init__(self, owner, width, *, buffers):
        from ignition.components.parallel_draft import PreparedParallelDraftRound

        self.owner, self.width = owner, width
        self.name = f"speculation-study-w{width}-block{owner.block_size}"
        self.signature = StudySignature(width, owner.block_size)
        self.forward = PreparedParallelDraftRound(owner, 1, width, self.name,
                                                  target_token_capacity=width, buffers=buffers)
        self.forward.prepare()
        self.graphs, self.compiled = {}, {}

    def capture(self, *, cache_root, **kwargs):
        from ignition.preparation.capture import capture_forwards

        root = None if cache_root is None else Path(cache_root) / self.name
        self.graphs, self.compiled = capture_forwards(
            {self.name: self.forward}, {self.name: self.signature}, cache_root=root,
            report_catalog=False, label="Capture speculative depth study", **kwargs)

    def close(self):
        import torch

        torch.cuda.synchronize(self.owner.requests.generation.device)
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.compiled.clear()


class StudyModel:
    """Own diagnostic graphs until drained, then close their native setup model."""

    def __init__(self, model, studies):
        self.model, self.studies = model, studies

    def manifest(self, **kwargs):
        raise ValueError("speculation studies cannot produce native package metadata")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        try:
            for study in self.studies.values():
                study.close()
        finally:
            self.model.close()


def run_study(prepared, tokenizer, *, scenarios, warmup, repetitions, seeds, progress=None):
    import torch

    from ignition.direct.executor import StepRunner
    from ignition.direct.scheduler import Layout

    model, studies = prepared.model, prepared.studies
    layout = Layout.of(model)
    runner = StepRunner(model, eager=False)
    cases, seeds = tuple(scenarios), tuple(seeds)
    measurements = [[] for _ in cases]

    def result(status, failure=None):
        value = measurement_result(cases, measurements, warmup=warmup, repetitions=repetitions,
                                   seeds=seeds, status=status, failure=failure)
        value.update(speculation_study=True, shared_prefill=True,
                     sampling_configuration=model.server_config.get("sampling_defaults", {}),
                     metric="cuda_graph_speculation_study_latency_ms",
                     native_completion_publication=False)
        for case in value["cases"]:
            case["effective_draft_block"] = studies[case["width"]].owner.block_size
            for sample in case["samples"]:
                if sample.get("times_ms") and "acceptance" in sample:
                    import statistics

                    emitted = sum(row["emitted_count"] for row in sample["acceptance"]["requests"])
                    sample["useful_tokens_per_gpu_second"] = emitted * 1000 / statistics.median(sample["times_ms"])
        return value

    generation = int(model.benchmark_requests.generation.max().item())
    try:
        for context in dict.fromkeys(case.context_tokens for case in cases):
            selected = [(case, samples) for case, samples in zip(cases, measurements)
                        if case.context_tokens == context]
            if any(case.width not in studies for case, _ in selected):
                raise ValueError("study case must match a captured width")
            prefill_case = max((case for case, _ in selected), key=lambda case: case.width)
            for seed in seeds:
                generation += 1
                baseline = _prefill(model, runner, layout, prefill_case, tokenizer, seed, generation)
                snapshot = StateSnapshot(model.benchmark_state, runner.stream)
                request = baseline.active[0]
                shared_input = dict(pending_token=int(model.benchmark_requests.pending_token[request.slot].item()),
                                    generation=request.generation, seed=request.sampling.seed,
                                    context_tokens=context, max_new_tokens=request.max_new_tokens)
                for case, samples in selected:
                    study = studies[case.width]
                    sample = dict(seed=seed, times_ms=[], shared_input=shared_input.copy())
                    samples.append(sample)
                    for iteration in range(-1, warmup + repetitions):
                        scheduler = deepcopy(baseline, {id(layout): layout})
                        snapshot.restore(runner.stream)
                        step = _round(scheduler, BenchmarkCase(1, 8, 8, case.context_tokens))
                        with torch.cuda.stream(runner.stream), torch.no_grad():
                            for name, offset, data in scheduler.pack(step):
                                runner._upload(name, offset, data)
                            study.owner.inputs.query_lengths[:1].fill_(case.width)
                            allocated = torch.cuda.memory_allocated(runner.device)
                            runner.started.record(runner.stream)
                            study.graphs[study.name].replay()
                            runner.finished.record(runner.stream)
                        runner.finished.synchronize()
                        elapsed = runner.started.elapsed_time(runner.finished)
                        if torch.cuda.memory_allocated(runner.device) != allocated:
                            raise RuntimeError("study replay changed allocated CUDA memory")
                        commit = study.forward.operations[3]
                        emitted = int(commit.emitted_count[0].item())
                        request = scheduler.active[0]
                        slot = request.slot
                        state = model.benchmark_requests
                        if (not bool(study.forward.operations[2].output.valid[0].item())
                                or int(state.error_code[slot].item()) != 0 or not 1 <= emitted <= case.width
                                or int(state.generated_tokens[slot].item()) != len(request.generated) + emitted
                                or int(state.rng_counter[slot].item()) != len(request.generated) + emitted
                                or int(state.generation[slot].item()) != request.generation
                                or int(state.seed[slot].item()) != request.sampling.seed
                                or int(state.histogram[slot].sum().item()) != len(request.prompt) + len(request.generated) + emitted
                                or int(state.consumed_tokens[slot].item()) != case.context_tokens + emitted
                                or int(state.pending_token[slot].item()) != int(commit.emitted_ids[0, emitted - 1].item())):
                            raise RuntimeError("study replay produced invalid committed request state")
                        if iteration == -1:
                            _check_metadata(SimpleNamespace(benchmark_rounds={study.name: study.forward}),
                                SimpleNamespace(graph=study.name, lanes=step.lanes, draft_tokens=case.width - 1), case)
                            sample["acceptance"] = acceptance_observation(study.forward)
                        if iteration >= warmup:
                            sample["times_ms"].append(elapsed)
                    if progress is not None:
                        progress(result("running"))
                snapshot.restore(runner.stream)
                runner.stream.synchronize()
                del snapshot
    except BaseException as error:
        if progress is not None:
            progress(result("incomplete", f"{type(error).__name__}: {error}"))
        raise
    finally:
        runner.close()
    return result("complete")
