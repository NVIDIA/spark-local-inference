# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention tuning fixtures remain live through trials and retire before capture."""

from types import SimpleNamespace
import weakref

import pytest
import torch
from b12x._lib.scratch import ScratchBufferSpec

from tests.preparation.test_tuning import backend, declaration


@pytest.mark.parametrize("fail_finalizer", [False, True])
def test_extracted_attention_fixture_lives_through_trials_and_releases_before_capture(
    backend, monkeypatch, fail_finalizer
):
    from b12x.preparation import PreparationSession
    from ignition.preparation.fixtures.varlen_attention import benchmark_call, input_factory
    from ignition.preparation.tuning import RecordedPreparationSession

    session = RecordedPreparationSession(device=0, checkpoint="fixture-lifetime",
        workload_schema="attention-fixture-v1", numerical_policy="default")
    arguments = {name: torch.zeros(4, 2, 8) for name in ("q", "k", "v")}
    arguments["cu_seqlens_q"] = torch.zeros(3, dtype=torch.int32)
    arguments.update(max_seqlen_q=4, max_seqlen_k=4)
    acquire = input_factory(arguments)
    calls, references, events = [], [], []
    spec = ScratchBufferSpec(name="work", shape=(256,), dtype=torch.uint8,
                             device=torch.device("cpu"))
    state = SimpleNamespace(
        scratch_plan=SimpleNamespace(scratch_specs=lambda: (spec,)),
        bind=lambda **kwargs: SimpleNamespace(**kwargs),
        run=lambda binding: binding.q + binding.k + binding.v,
    )

    def trial(_):
        call = benchmark_call(state, arguments=arguments, inputs=acquire, workspace=None)
        calls.append(call)
        inputs = call.owners[0]
        references.append(weakref.ref(inputs))
        call.produce()
        expected = inputs.q.clone()
        inputs.q.fill_(float("nan"))
        call.produce()
        assert torch.equal(inputs.q, expected)
        assert inputs.cu_seqlens_q.tolist() == [0, 4, 4]
        assert torch.isfinite(call.invoke()).all()
        assert all(not tensor.count_nonzero() for tensor in arguments.values() if isinstance(tensor, torch.Tensor))

    original_prepare = PreparationSession.prepare

    def prepare(self, requests, **kwargs):
        result = original_prepare(self, requests, **kwargs)
        if kwargs["autotune"]:
            close = result.close

            def close_trials():
                assert references and all(reference() is not None for reference in references)
                assert len(calls) == 2
                events.append("trial_close")
                close()
                calls.clear()

            result.close = close_trials
        return result

    monkeypatch.setattr(PreparationSession, "prepare", prepare)
    session.declare((declaration("first", benchmark=trial), declaration("second", benchmark=trial)))

    def finalize():
        assert references[0]() is references[1]()
        events.append("finalize")
        if fail_finalizer:
            raise RuntimeError("fixture binding finalization failed")

    session.register_finalizer(finalize)
    if fail_finalizer:
        with pytest.raises(RuntimeError, match="fixture binding finalization failed"):
            session.tune()
        with pytest.raises(RuntimeError, match="completed tuning"):
            with session.capture():
                pytest.fail("failed finalization must prevent capture")
    else:
        session.tune()
        with session.capture():
            assert all(reference() is None for reference in references)
    assert events == ["finalize", "trial_close"]
    assert not calls and all(reference() is None for reference in references)
    assert not session._benchmark_resources and not session._recorded_requests
