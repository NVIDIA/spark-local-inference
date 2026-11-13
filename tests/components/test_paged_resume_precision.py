# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Cached-prefix vocabulary projection preserves its requested logit precision."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import collectives, paged_resume, vocabulary
from ignition.preparation.scratch import ScratchArena


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("quantized", [False, True])
def test_resume_projection_matches_gather_byte_geometry(monkeypatch, dtype, quantized):
    from b12x.comm import pcie

    observed = {}
    arena = ScratchArena("cpu")

    class State:
        def __init__(self, *args, **kwargs):
            self.hidden = torch.zeros(1, 64, dtype=torch.bfloat16)
            self.validation_input = torch.zeros(1, 8, dtype=torch.bfloat16)
            self.validation_sum = torch.zeros_like(self.validation_input)
            self.source = torch.zeros_like(self.hidden)

        def functional(self):
            self.hidden = None
            return self

        def validate(self):
            return self.validation_input, torch.ones(1, dtype=torch.bool)

        def gather(self, totals, *, present):
            return self.source.clone()

    class Projection:
        def __init__(self, source, weight, output, **kwargs):
            assert source.dtype == torch.bfloat16
            if quantized:
                observed["workspace"] = kwargs["workspace"]
            else:
                assert weight.dtype == torch.bfloat16
            self.output = output
            self.shape, self.dtype = output.shape, output.dtype

        def functional(self, *, session):
            self.output = None
            return self

        def __call__(self, source):
            offsets = torch.arange(self.shape[1], dtype=torch.float32, device=source.device)
            return (source.float().sum(-1, keepdim=True) + offsets).to(self.dtype)

    class Sampling:
        def __init__(self, logits, *args, **kwargs):
            self.logits = logits

        @staticmethod
        def scratch_layout(rows, vocab):
            return {"scores": ((rows, vocab), torch.float32)}

        def prepare(self):
            pass

        def prepare_broadcast(self, **kwargs):
            pass

        def functional(self, *, session):
            self.logits = None
            return self

        def run(self, logits):
            observed["sampled_logits"] = logits.clone()

    heads = 4 if dtype == torch.float32 else 2
    runtime = SimpleNamespace(
        world_size=4,
        device=torch.device("cpu"),
        max_batch_size=1,
        heads_per_rank=heads,
        query_head_dim=64,
        total_heads=heads * 4,
        exchange_group=object(),
    )

    def query(owner, *, surface, call):
        observed.update(call)
        return object()

    plan = SimpleNamespace(request=lambda **kwargs: object())
    monkeypatch.setattr(paged_resume, "PreparedResumeState", State)
    monkeypatch.setattr(paged_resume, "PreparedSampling", Sampling)
    monkeypatch.setattr(vocabulary, "UnquantizedProjection", Projection)
    monkeypatch.setattr(vocabulary, "QuantizedProjection", Projection)
    monkeypatch.setattr(collectives, "_all_gather", lambda source, output, handle: output.copy_(source.repeat(1, 4)))
    monkeypatch.setattr(collectives, "_roce", lambda runtime: False)
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(pcie, "query_from_runtime", query)
    monkeypatch.setattr(pcie, "plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: [0, 1, 2, 3]
    )
    inputs = SimpleNamespace(
        row_slot_ids=torch.zeros(1, dtype=torch.int32),
        row_generations=torch.ones(1, dtype=torch.int64),
        live_rows=torch.ones(1, dtype=torch.int32),
        request_state_rows=torch.zeros(4, 8, dtype=torch.int64),
    )
    weights = SimpleNamespace(
        head=SimpleNamespace(out_features=128) if quantized else torch.zeros(128, 64, dtype=torch.bfloat16),
        shard=SimpleNamespace(world_size=4),
    )
    pool = SimpleNamespace(
        backing=torch.empty(1, dtype=torch.uint8),
        last_hidden=torch.zeros(5, 64, dtype=torch.bfloat16),
    )
    options = {} if dtype == torch.bfloat16 else {"logits_dtype": torch.float32}
    operation = paged_resume.PreparedResume(
        weights,
        SimpleNamespace(vocab_size=512),
        pool,
        inputs,
        None,
        None,
        None,
        rows=1,
        runtime=None,
        vocabulary_runtime=runtime,
        coordinator=None,
        session=None,
        name="resume",
        workspace=arena,
        **options,
    )
    assert operation.local_logits is None and operation.logits is None
    assert operation.state.hidden is None
    assert operation.projection.projection.dtype == dtype
    assert operation.completion.logits is None
    if quantized:
        assert observed["workspace"] is arena
    assert observed["local_input"].shape == (1, heads, 64)
    assert observed["out"].shape == (1, heads * 4, 64)
    assert observed["local_input"].dtype == observed["out"].dtype == torch.bfloat16
    assert observed["local_input"].numel() * 2 == 128 * dtype.itemsize
    for value in (0.25, 1.5):
        operation.state.source.fill_(value)
        operation()
        expected = (torch.arange(128, dtype=torch.float32) + value * 64).to(dtype).repeat(4).view(1, -1)
        assert observed["sampled_logits"].dtype == dtype
        torch.testing.assert_close(observed["sampled_logits"], expected, rtol=0, atol=0)
