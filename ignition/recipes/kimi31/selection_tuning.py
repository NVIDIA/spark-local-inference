# SPDX-License-Identifier: Apache-2.0

"""Choose complete sparse-candidate compositions during model preparation."""

from functools import partial

import torch
from b12x.preparation import PreparedCall

from ignition.components.candidate_selection import PreparedCandidateSelection, CandidateSelectionForward
from ignition.preparation.capture import capture_forwards
from ignition.preparation.composition_tuning import PreparedComposition, plan_identity, race_compositions
from ignition.preparation.execution import GraphSignature
from ignition.preparation.tuning import RecordedPreparationSession

from .indexer import SharedBuffers


def _candidate_pattern(rows, world, topk, rank, seed, count, *, partial_rows=False):
    """Independent finite scores and interleaved IDs with ordinary invalid padding."""
    generator = torch.Generator().manual_seed(seed)
    width = world * topk
    scores = torch.stack([torch.randperm(width, generator=generator) for _ in range(rows)]).float() / width
    local = torch.arange(topk, dtype=torch.int32)[None].expand(rows, -1).clone()
    local[:, torch.arange(topk)*world+rank >= count] = -1
    scores = scores[:, rank*topk:(rank+1)*topk].contiguous()
    scores[local < 0] = -torch.inf
    positions = torch.arange(rows, dtype=torch.int64) + count
    if partial_rows and rows > 1:
        positions[-1] = -1
        local[-1].fill_(-1)
        scores[-1].fill_(-torch.inf)
    return local, scores, positions


class _Fixture:
    def __init__(self, rows, context, reduction, session):
        self.rows, self.context, self.step = rows, context, 0
        device, world = context.shards.index.device, context.shards.world
        self.values = {}
        buffers = SharedBuffers(self.values, device)
        self.indices = buffers.buffer("input_ids", (rows, 2048), torch.int32)
        self.scores = buffers.buffer("input_scores", (rows, 2048), torch.float32)
        self.positions = buffers.buffer("positions", (rows,), torch.int64)
        self.output = buffers.buffer("output", (rows, 2048), torch.int32)
        self.selection = PreparedCandidateSelection(self.indices, self.scores, self.positions,
            context.shards.index, self.output, world=world, gather_runtime=context.candidates,
            owner_runtime=context.candidate_owners, reduction_runtime=reduction,
            session=session, coordinator=context.coordinator, name=f"selection_trial.rows{rows}",
            buffers=SharedBuffers(self.values, device, prefix="merge."))
        self.forwards = {kind: CandidateSelectionForward(self.selection, kind)
                         for kind in self.selection.alternatives}
        self.patterns = tuple(tuple(value.to(device) for value in _candidate_pattern(
            rows, world, 2048, context.shards.rank, seed, count, partial_rows=partial))
            for seed, count, partial in ((101,256,False), (102,8192,False),
                                         (103,32768,True), (207,4096,True)))
        self.expected = []

    def produce(self, index):
        self.step = index
        for target, source in zip((self.indices, self.scores, self.positions), self.patterns[index], strict=True):
            target.copy_(source)

    def validate(self):
        if not torch.equal(self.output, self.expected[self.step]):
            raise AssertionError("candidate composition changed the ordered selected IDs")

    @property
    def nbytes(self):
        tensors = [value for value in self.values.values() if isinstance(value, torch.Tensor)]
        tensors.extend(value for pattern in self.patterns for value in pattern)
        tensors.extend(self.expected)
        return sum({value.untyped_storage()._cdata: value.untyped_storage().nbytes()
                    for value in tensors}.values())


class CandidateSelectionTuning:
    """Bind measured compositions for C1 decode and verification capacities.

    Child plans prepare with the model. Private trials use separate plans and
    tensors, then compare those selected plans with each model binding. Trial
    storage and compiled owners are released before complete-model capture.
    """

    def __init__(self, context, reduction, *, cache_root=None):
        self.context, self.reduction = context, reduction
        self.cache_root = cache_root
        self.selections = {}

    def register(self, selection):
        self.selections.setdefault(selection.rows, []).append(selection)

    def tune(self, session):
        if not self.selections:
            return []
        namespace = session.namespace.to_dict()
        checkpoint, schema, numerics = (namespace.pop(key) for key in (
            "checkpoint", "workload_schema", "numerical_policy"))
        coordinator = self.context.coordinator
        device = self.context.shards.index.device
        report, graphs, compiled = [], {}, {}
        with RecordedPreparationSession(device=device, checkpoint=checkpoint, workload_schema=schema,
                numerical_policy=numerics, namespace=namespace, coordinator=coordinator,
                compile_workers=session.requested_compile_workers,
                autotune_effort=session.autotune_effort, collective_tuning="static",
                headroom_bytes=session.headroom_bytes, lock_gpu_clocks=session.lock_gpu_clocks) as trials:
            fixtures = {rows: _Fixture(rows, self.context, self.reduction, trials)
                        for rows in sorted(self.selections)}
            trials.tune(show_progress=False)
            for rows, fixture in fixtures.items():
                for target in self.selections[rows]:
                    for kind in fixture.selection.alternatives:
                        measured = tuple((role, plan_identity(plan)) for role, plan in fixture.selection.plans(kind))
                        actual = tuple((role, plan_identity(plan)) for role, plan in target.plans(kind))
                        if measured != actual:
                            raise ValueError("candidate trial plans differ from the model's prepared kernels")
            forwards = {f"{kind}{rows}": forward for rows, fixture in fixtures.items()
                        for kind, forward in fixture.forwards.items()}
            signatures = {f"{kind}{rows}": GraphSignature("decode", 1, rows)
                          for rows, fixture in fixtures.items() for kind in fixture.forwards}
            for fixture in fixtures.values():
                fixture.produce(0)
            try:
                graphs, compiled = capture_forwards(forwards, signatures, session=trials, device=device,
                    stream=torch.cuda.Stream(device=device), pool=torch.cuda.graph_pool_handle(),
                    coordinator=coordinator, report_catalog=False, cache_root=self.cache_root, retain_forwards=True)
                for rows, fixture in fixtures.items():
                    for index in range(len(fixture.patterns)):
                        fixture.produce(index)
                        torch.cuda.synchronize(device)
                        coordinator.barrier(f"selection reference {rows}.{index}", coordinator.ranks)
                        compiled[f"replicated{rows}"]()
                        torch.cuda.synchronize(device)
                        fixture.expected.append(fixture.output.clone())
                    candidates = {}
                    for kind in fixture.forwards:
                        forward = compiled[f"{kind}{rows}"]
                        producers = tuple(partial(fixture.produce, index) for index in range(3))
                        call = PreparedCall(run=forward, output=fixture.output,
                            produce=producers[0], benchmark_producers=producers,
                            reset=partial(fixture.output.fill_, -91), validate=fixture.validate,
                            owners=(fixture,), capture_safe=True, benchmark_repetitions=8)
                        candidates[kind] = PreparedComposition(forward, fixture.selection.plans(kind), call)
                    with trials._tuning_clocks():
                        chosen = race_compositions(candidates, operation="candidate-selection",
                            geometry=dict(rows=rows, world_size=self.context.shards.world, local_topk=2048, output_topk=2048),
                            numerics="exact-fp32-words-ordered-ties",
                            workload=dict(version=2, seeds=(101,102,103), counts=(256,8192,32768), partial_last=True),
                            default="replicated", session=trials, coordinator=coordinator,
                            cache_policy="warm", resident_bytes=fixture.nbytes)
                    fixture.produce(3)
                    torch.cuda.synchronize(device)
                    coordinator.barrier(f"selection holdout {rows}", coordinator.ranks)
                    compiled[f"{chosen['implementation']}{rows}"]()
                    torch.cuda.synchronize(device)
                    fixture.validate()
                    for selection in self.selections[rows]:
                        selection.bind(chosen["implementation"])
                    report.append(dict(rows=rows, **chosen))
            finally:
                for graph in graphs.values():
                    graph.reset()
                for forward in compiled.values():
                    forward.retire_after_capture()
                graphs.clear()
                compiled.clear()
        return report
