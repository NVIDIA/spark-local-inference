"""Quantized vocabulary variants bind one finalized operator arena."""

from dataclasses import dataclass
from types import SimpleNamespace
import weakref

import torch
from b12x.gemm import blockscaled

from ignition.components import linear
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.scratch import ScratchArena


def test_quantized_heads_rebind_shared_workspace_and_keep_arena_priming_owner(monkeypatch):
    @dataclass(frozen=True)
    class Query:
        workspace_form: str = "provided"
        workspace_nbytes: int | None = None

    declarations = []
    required = [1024]
    plan = SimpleNamespace(memory_requirements=lambda: SimpleNamespace(scratch_nbytes=required[0]),
        request=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(blockscaled, "query_from_call", lambda *args, **kwargs: Query())
    monkeypatch.setattr(blockscaled, "plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(linear, "declare", lambda session, requests: declarations.extend(requests))
    weight = blockscaled.NVFP4LinearWeight(torch.zeros((128, 128), dtype=torch.uint8),
        torch.zeros(1024, dtype=torch.float8_e4m3fn), torch.ones(1), "multiplier", 256, 128)
    arena = ScratchArena("cpu")
    priming = arena.reserve(4096)
    priming_address, priming_reference = priming.data_ptr(), weakref.ref(priming)
    heads = []
    for rows in (1, 8):
        source = torch.zeros((rows, 256), dtype=torch.bfloat16)
        output = torch.zeros((rows, 128), dtype=torch.bfloat16)
        heads.append(PreparedVocabularyProjection(source, weight, output, output,
            runtime=None, coordinator=None, session=None, name=f"head{rows}", workspace=arena))
    for declaration in declarations:
        call = declaration.prepare_call(SimpleNamespace(required_workspace=1024))
        assert call.owners == (arena,)
        del call
    required[0] = 1536
    arena.trim(required[0])
    arena.finalize()
    arena.freeze()
    assert arena.nbytes == 1536
    for head in heads:
        workspace = head.projection.workspace
        assert workspace.data_ptr() == arena.tensor.data_ptr() != priming_address
        assert workspace.nbytes == required[0]
    del priming
    assert priming_reference() is None
    address = arena.tensor.data_ptr()
    for head in heads:
        head.projection.workspace.fill_(17)
        assert arena.tensor.data_ptr() == address
