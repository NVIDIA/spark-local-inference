"""Prepared composition identities follow executable choices, not process bindings."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from b12x.preparation import FrozenMapping, PreparedCall

from ignition.preparation.composition_tuning import PreparedComposition, _CONTRACT, plan_identity
from ignition.preparation.inductor_tuning import _execution_identity


def test_opaque_triton_identity_ignores_table_indices_but_preserves_launch_constants(monkeypatch):
    import torch
    from torch._higher_order_ops.triton_kernel_wrap import KernelSideTable
    from torch.utils._pytree import tree_flatten
    from ignition.components.owner_topk import _owner_candidates
    from ignition.components.row_ownership import _decode_owned_rows
    from ignition.preparation import aot
    from ignition.preparation.compilation import _Invocation

    def record(dummy, rows):
        table = KernelSideTable()
        table.reset_table()
        monkeypatch.setattr(aot, "kernel_side_table", table)
        if dummy:
            table.add_kernel(_decode_owned_rows)
            table.add_constant_args({"unused": True})
        kwargs = dict(kernel_idx=table.add_kernel(_owner_candidates),
                      constant_args_idx=table.add_constant_args({"ROWS": rows}),
                      kwargs={"Packed": None}, grid=(1, 32, 1))
        leaves, tree = tree_flatten(((), kwargs))
        call = _Invocation(torch.ops.higher_order.triton_kernel_wrapper_mutation, leaves, tree, [])
        return aot._serialized_invocations((call,))

    assert record(False, 8) == record(True, 8)
    assert record(False, 8) != record(False, 16)


def child():
    contract = SimpleNamespace(query_schema_version=2, config_schema_version=3,
        semantic_version=4, candidate_contract_version=5, config_payload=FrozenMapping)
    return SimpleNamespace(component_id="fixture.transport", contract=contract,
        invocation=FrozenMapping({"rows": 8}), selection=SimpleNamespace(
            query=FrozenMapping({"rank": 0, "world_size": 16}),
            config={"threads": 512}, source="tuned"),
        prepared=SimpleNamespace(programs=(SimpleNamespace(dialect="cute", key="program-a", name="label"),)),
        handle=41)


def test_child_identity_tracks_selected_programs_and_not_handles():
    plan = child()
    identity = plan_identity(plan)
    plan.handle = 8192
    plan.selection.source = "cached"
    plan.prepared.programs[0].name = "another diagnostic label"
    assert plan_identity(plan) == identity
    plan.prepared.programs[0].key = "program-b"
    assert plan_identity(plan) != identity
    plan.prepared.programs[0].key = "program-a"
    plan.selection.config = {"threads": 128}
    assert plan_identity(plan) != identity
    plan.selection = None
    with pytest.raises(ValueError, match="prepared child"):
        plan_identity(plan)


def test_compiled_identity_normalizes_devices_and_tracks_cubins_and_launch_config():
    kernel = ("kernel", {"kwargs": {"BLOCK": 256}, "num_warps": 4}, b"cubin-a")
    source = "torch.cuda.set_device(7)\ns = get_raw_stream(7)\n"
    expected = _execution_identity("source-a", source, 7, (kernel,))
    assert expected == _execution_identity("source-a", source.replace("7", "0"), 0, (kernel,))
    assert expected != _execution_identity("source-b", source, 7, (kernel,))
    changed = deepcopy(kernel)
    changed[1]["num_warps"] = 8
    assert expected != _execution_identity("source-a", source, 7, (changed,))
    assert expected != _execution_identity("source-a", source, 7, ((*kernel[:2], b"cubin-b"),))
    assert _execution_identity("source-a", source, 7, (kernel,), (("op", 16),)) != (
        _execution_identity("source-a", source, 7, (kernel,), (("op", 32),)))


def test_composition_identity_binds_the_measured_forward():
    forward = SimpleNamespace(execution_identity="generated-code")
    call = PreparedCall(run=forward, benchmark_producers=(lambda: None,), validate=lambda: None)
    candidate = PreparedComposition(forward, (("exchange", child()),), call)
    assert candidate.identity()["forward"] == "generated-code"
    with pytest.raises(ValueError, match="run callable"):
        PreparedComposition(forward, candidate.plans,
                            PreparedCall(run=lambda: None)).identity()
    from dataclasses import replace
    with pytest.raises(ValueError, match="ownership"):
        PreparedComposition(forward, candidate.plans, replace(call, close=lambda: None)).identity()


def test_compiled_trial_restores_its_prepared_dispatch_context():
    import torch

    def forward():
        return torch.is_inference_mode_enabled(), torch.is_grad_enabled()
    forward.execution_identity = "compiled-under-no-grad"
    with torch.no_grad():
        candidate = PreparedComposition(forward, (), PreparedCall(run=forward))
    with torch.inference_mode():
        assert candidate.invoke() == (False, False)
        assert torch.is_inference_mode_enabled()


def test_collective_signature_agrees_semantics_but_preserves_rank_local_cache_inputs():
    payload = dict(operation="candidate-selection", geometry={"rows": 8}, numerics="exact",
        workload={"version": 1}, default="replicated",
        catalog={"owner": {"program": "rank0"}, "replicated": {"program": "rank0"}})
    first = FrozenMapping(payload)
    second = FrozenMapping({**payload, "catalog": {
        "owner": {"program": "rank1"}, "replicated": {"program": "rank1"}}})
    assert _CONTRACT.collective_signature(first) == _CONTRACT.collective_signature(second)
    assert _CONTRACT.encode_query(first) != _CONTRACT.encode_query(second)
    choices = _CONTRACT.configure(first, device=None)
    assert {config["implementation"] for _, config in _CONTRACT.iterate(choices)} == {"owner", "replicated"}
    assert choices.default["implementation"] == "replicated"
    changed = FrozenMapping({**payload, "geometry": {"rows": 1}})
    assert _CONTRACT.collective_signature(first) != _CONTRACT.collective_signature(changed)
