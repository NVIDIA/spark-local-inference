# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Representative hyperconnection trials own activations and retain model gain."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.qwen38 import hyperconnection
from ignition.recipes.qwen38.hyperconnection import HyperconnectionScratch
from ignition.preparation.fixtures.hyperconnection import HyperconnectionBenchmarkInputs, benchmark_call
from ignition.preparation.fixtures.hyperconnection_gate import HyperconnectionGateInputs


def test_packed_projection_preserves_canonical_weights_without_duplicate_storage():
    module = torch.nn.Module()
    module.input_mix_weight_down = torch.nn.Linear(10240, 320, bias=False, dtype=torch.bfloat16)
    module.block_inject_weight = torch.nn.Linear(10240, 4, bias=False, dtype=torch.bfloat16)
    down, injection = module.input_mix_weight_down.weight.detach().clone(), module.block_inject_weight.weight.detach().clone()
    original = weakref.ref(module.input_mix_weight_down.weight)
    packed = hyperconnection.packed_down_projection(module)
    assert original() is None
    assert hyperconnection.packed_down_projection(module) is packed
    assert packed.shape == (336, 10240) and not packed[324:].count_nonzero()
    for actual, expected in ((module.input_mix_weight_down.weight, down),
                             (module.block_inject_weight.weight, injection)):
        assert torch.equal(actual, expected)
        assert actual.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    scratch = HyperconnectionScratch.allocate(64, torch.device("cpu"))
    for tokens in (1, 4, 16, 64):
        rows = scratch.rows(tokens)
        assert rows.projected_down.stride() == rows.injection_logits.stride() == (336, 1)
        rows.down_and_injection[:, :320].fill_(2)
        rows.down_and_injection[:, 320:324].fill_(3)
        assert torch.all(rows.projected_down == 2) and torch.all(rows.injection_logits == 3)


def test_benchmark_call_retains_private_operands_gain_and_launch_options():
    source = torch.zeros((3, 16), dtype=torch.bfloat16)[:, ::2]
    gain = torch.linspace(.5, 1.5, 8)
    output = torch.zeros_like(source)
    launches = []

    def launch(actual_source, actual_gain, actual_output, **options):
        launches.append((actual_source, actual_gain, actual_output, options))
        actual_output.copy_(actual_source)

    call = benchmark_call(SimpleNamespace(launch=launch), "grouped_rmsnorm",
                          (source, gain, output), {"eps": 1e-6})
    operands, = call.owners
    assert call.output is operands.arguments[-1]
    assert call.reset is None and call.restore is None
    assert launches == []
    call.produce()
    expected = operands.arguments[0].clone()
    call.run()
    assert launches[0][0] is operands.arguments[0]
    assert launches[0][1] is gain and launches[0][2] is call.output
    assert launches[0][3] == {"eps": 1e-6}
    assert operands.arguments[0].stride() == source.stride()
    assert torch.equal(call.output, expected)
    operands.arguments[0].fill_(float("nan"))
    call.produce()
    assert torch.equal(operands.arguments[0], expected)
    assert not source.count_nonzero() and not output.count_nonzero()


@pytest.mark.parametrize("operation", ["grouped_rmsnorm", "scaled_silu", "gate_mean", "combine"])
def test_benchmark_inputs_are_finite_private_and_repeatable(operation):
    tokens = 4
    if operation == "grouped_rmsnorm":
        arguments = (torch.zeros(tokens, 10240, dtype=torch.bfloat16),
            torch.arange(10240, dtype=torch.float32) / 10240 + .5,
            torch.zeros(tokens, 10240, dtype=torch.bfloat16))
    elif operation == "scaled_silu":
        arguments = tuple(torch.zeros(tokens, 320, dtype=torch.bfloat16) for _ in range(2))
    elif operation == "gate_mean":
        arguments = (torch.zeros(tokens, 10240, dtype=torch.bfloat16),
            torch.zeros(tokens, 10240, dtype=torch.bfloat16),
            torch.zeros(tokens, 2560, dtype=torch.bfloat16))
    else:
        arguments = tuple(torch.zeros(tokens, width, dtype=torch.bfloat16)
                          for width in (10240, 2560, 4, 10240))
    original = tuple(tensor.clone() for tensor in arguments)
    fixture = HyperconnectionBenchmarkInputs(operation, arguments)
    fixture.produce()
    for index, (actual, source) in enumerate(zip(fixture.arguments, arguments)):
        assert actual.shape == source.shape and actual.stride() == source.stride()
        assert actual.dtype == source.dtype
        if operation == "grouped_rmsnorm" and index == 1:
            assert actual is source
        else:
            assert actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    before = tuple(target.clone() for target, _ in fixture.inputs)
    for target, _ in fixture.inputs:
        assert torch.isfinite(target).all() and target.count_nonzero() > 0
        target.fill_(float("nan"))
    fixture.produce()
    assert all(torch.equal(target, saved) for (target, _), saved in zip(fixture.inputs, before))
    assert all(torch.equal(actual, saved) for actual, saved in zip(arguments, original))
    assert not fixture.arguments[-1].count_nonzero()


def test_priming_borrows_one_source_across_capacities_and_retains_its_owner():
    scratch = HyperconnectionScratch.allocate(64, torch.device("cpu"))
    source = torch.zeros_like(scratch.normalized)
    gain = torch.ones(10240, dtype=torch.float32)
    reference = weakref.ref(source)
    storage = source.untyped_storage().data_ptr()
    calls = [scratch.rows(tokens).priming_arguments(source[:tokens], gain)
             for tokens in (1, 4, 16, 64)]
    assert {args["grouped_rmsnorm"][0].untyped_storage().data_ptr()
            for args in calls} == {storage}
    del source
    gc.collect()
    assert reference() is not None
    for args in calls:
        borrowed, weight, output = args["grouped_rmsnorm"]
        assert weight is gain
        assert borrowed.stride() == output.stride() == (10240, 1)
        for arguments in args.values():
            arguments[-1].fill_(17)
        assert not borrowed.count_nonzero()
    del borrowed, args, calls, arguments, output
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("operation", ["grouped_rmsnorm", "scaled_silu", "combine"])
def test_borrowed_priming_keeps_exact_private_benchmark_fixture(operation):
    scratch = HyperconnectionScratch.allocate(4, torch.device("cpu"))
    source = torch.zeros_like(scratch.normalized)
    gain = torch.arange(10240, dtype=torch.float32) / 10240 + .5
    arguments = scratch.priming_arguments(source, gain)[operation]
    previous_arguments = ((torch.zeros_like(source), gain, scratch.normalized)
                          if operation == "grouped_rmsnorm" else arguments)
    previous = HyperconnectionBenchmarkInputs(operation, previous_arguments)
    borrowed = HyperconnectionBenchmarkInputs(operation, arguments)
    previous.produce()
    borrowed.produce()
    for before, after in zip(previous.arguments, borrowed.arguments, strict=True):
        assert torch.equal(before, after)
        assert before.stride() == after.stride()
    for target, _ in borrowed.inputs:
        assert target.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    borrowed.arguments[-1].fill_(23)
    assert not source.count_nonzero()


@pytest.mark.parametrize("mismatch", ["shape", "stride", "dtype", "device", "output_alias"])
def test_priming_rejects_incompatible_or_mutable_scratch_storage(mismatch):
    scratch = HyperconnectionScratch.allocate(4, torch.device("cpu"))
    source = {
        "shape": lambda: torch.zeros(3, 10240, dtype=torch.bfloat16),
        "stride": lambda: torch.zeros(4, 20480, dtype=torch.bfloat16)[:, ::2],
        "dtype": lambda: torch.zeros(4, 10240, dtype=torch.float32),
        "device": lambda: torch.zeros(4, 10240, dtype=torch.bfloat16, device="meta"),
        "output_alias": lambda: scratch.normalized[:],
    }[mismatch]()
    with pytest.raises(ValueError, match="priming source"):
        scratch.priming_arguments(source, torch.ones(10240))


def test_combine_priming_reuses_disjoint_existing_storage():
    scratch = HyperconnectionScratch.allocate(64, torch.device("cpu"))
    source = torch.zeros_like(scratch.normalized)
    gain = torch.ones(10240, dtype=torch.float32)
    for tokens in (1, 4, 16, 64):
        live = scratch.rows(tokens)
        arguments = live.priming_arguments(source[:tokens], gain)["combine"]
        assert arguments[1] is live.block_input
        assert arguments[2] is live.injection_logits
        assert arguments[3] is live.normalized
        pointers = [tensor.untyped_storage().data_ptr() for tensor in arguments]
        assert len(set(pointers)) == 4
        assert pointers[0] == source.untyped_storage().data_ptr()
        arguments[-1].fill_(19)
        assert all(not tensor.count_nonzero() for tensor in arguments[:-1])


def test_priming_accepts_disjoint_views_of_one_pool_chunk():
    from dataclasses import replace

    backing = torch.zeros((2, 4, 10240), dtype=torch.bfloat16)
    source = backing[0]
    scratch = replace(HyperconnectionScratch.allocate(4, torch.device("cpu")), normalized=backing[1])
    assert source.untyped_storage().data_ptr() == scratch.normalized.untyped_storage().data_ptr()
    arguments = scratch.priming_arguments(source, torch.ones(10240))
    for operands in arguments.values():
        operands[-1].fill_(17)
    assert not source.count_nonzero()
    with pytest.raises(ValueError, match="priming source"):
        scratch.priming_arguments(scratch.normalized, torch.ones(10240))


def test_combine_norm_fixture_retains_fp32_gain_and_owns_both_outputs():
    scratch = HyperconnectionScratch.allocate(4, torch.device("cpu"))
    source = torch.zeros_like(scratch.normalized)
    gain = torch.linspace(.5, 1.5, 10240)
    arguments = (source, scratch.block_input, scratch.injection_logits, gain,
                 scratch.gate_workspace, scratch.normalized)
    fixture = HyperconnectionBenchmarkInputs("combine_norm", arguments)
    fixture.produce()
    assert fixture.arguments[3] is gain
    assert len(fixture.inputs) == 3
    assert fixture.output == fixture.arguments[-2:]
    for actual, original in zip(fixture.arguments, arguments):
        assert actual.stride() == original.stride()
        if original is not gain:
            assert actual.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    for output in fixture.output:
        assert not output.count_nonzero()
    assert not source.count_nonzero()


def test_combine_uses_public_prepared_plan_and_residual_output(monkeypatch):
    operator = hyperconnection.PreparedHyperconnection.__new__(hyperconnection.PreparedHyperconnection)
    operator.scratch = HyperconnectionScratch.allocate(4, torch.device("cpu"))
    injection = torch.zeros((4, 4), dtype=torch.bfloat16)
    plan = SimpleNamespace(selection="default")
    operator.plans = {"combine": plan}
    residual = torch.zeros(4, 10240, dtype=torch.bfloat16)
    block = torch.zeros(4, 2560, dtype=torch.bfloat16)
    calls = []

    def combine(state, block_output, logits, *, plan, out):
        calls.append((state, block_output, logits, plan, plan.selection))
        assert out is state
        return out

    monkeypatch.setattr(hyperconnection.hc, "run_combine", combine)
    for selection in ("default", "tuned"):
        plan.selection = selection
        assert operator.combine_runtime(residual, block, injection) is residual
        state, actual_block, logits, selected_plan, selected_config = calls[-1]
        assert state is residual and actual_block is block
        assert logits is injection
        assert selected_plan is plan and selected_config == selection
        assert not residual.count_nonzero() and not block.count_nonzero()
    assert len(calls) == 2


def test_projection_gate_trials_produce_private_inputs_and_retain_weights():
    source = torch.zeros(6, 320, dtype=torch.bfloat16)
    normalized = torch.zeros(6, 10240, dtype=torch.bfloat16)
    output = torch.zeros(6, 2560, dtype=torch.bfloat16)
    workspace = torch.zeros_like(normalized)
    weight = torch.ones(10240, 320, dtype=torch.bfloat16)
    inputs = HyperconnectionGateInputs(source, normalized, output, workspace)
    calls = []

    def run(x, w, values, out, scratch):
        calls.append((x, w, values, out, scratch))
        out.fill_(7)

    call = inputs.call(SimpleNamespace(run=run), weight)
    call.produce()
    expected = inputs.source.clone(), inputs.normalized.clone()
    call.run()
    assert calls[-1] == (inputs.source, weight, inputs.normalized, inputs.output, inputs.workspace)
    assert call.output is inputs.output and torch.all(call.output == 7)
    inputs.source.fill_(float("nan"))
    inputs.normalized.fill_(float("nan"))
    call.produce()
    assert torch.equal(inputs.source, expected[0]) and torch.equal(inputs.normalized, expected[1])
    assert not source.count_nonzero() and not normalized.count_nonzero() and not output.count_nonzero()
    assert inputs.nbytes == sum(t.nbytes for t in (
        inputs.source, inputs.normalized, inputs.output, inputs.workspace, *inputs.seeds))


@pytest.mark.parametrize("world", [2, 4])
def test_sharded_weights_reconstruct_lowrank_axis_and_scratch_preserves_injection(world):
    from ignition.components.parallel import TensorParallelShard

    down = torch.arange(320 * 16).reshape(320, 16)
    up = torch.arange(16 * 320).reshape(16, 320)
    shards = [hyperconnection.hyperconnection_extents(TensorParallelShard(rank, world), True)
              for rank in range(world)]
    assert torch.equal(torch.cat([down[e["input_mix_weight_down.weight"]] for e in shards]), down)
    assert torch.equal(torch.cat([up[e["input_mix_weight_up.weight"]] for e in shards], dim=1), up)
    assert hyperconnection.hyperconnection_extents(TensorParallelShard(0, world), False) == {}
    lowrank = 320 // world
    scratch = HyperconnectionScratch.allocate(64, torch.device("cpu"), lowrank=lowrank)
    assert scratch.gate_logits.shape == scratch.gate_workspace.shape == (64, 10240)
    assert scratch.gate_logits.data_ptr() != scratch.gate_workspace.data_ptr()
    for tokens in (1, 6, 64):
        rows = scratch.rows(tokens)
        rows.down_and_injection[:, :lowrank].fill_(2)
        rows.down_and_injection[:, lowrank:lowrank + 4].fill_(3)
        assert torch.all(rows.projected_down == 2) and torch.all(rows.injection_logits == 3)
        assert rows.projected_down.stride() == ((lowrank + 19) // 16 * 16, 1)
        assert rows.gate_logits.shape == (tokens, 10240)
        rows.priming_arguments(torch.zeros_like(rows.normalized), torch.ones(10240))
    with pytest.raises(ValueError, match="tensor parallelism"):
        hyperconnection.hyperconnection_extents(TensorParallelShard(0, 1), True)
