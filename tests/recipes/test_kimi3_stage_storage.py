# SPDX-License-Identifier: Apache-2.0

"""Kimi declaration operands share examples while rank constants retain their own values."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3.decoder import DecoderStorage, Scratch


def test_decoder_rank_constants_follow_hidden_and_latent_partitions():
    config = SimpleNamespace(hidden=64, text={"routed_expert_hidden_size": 128})
    storage = DecoderStorage.allocate(config, TensorParallelShard(3, 8), "cpu")
    torch.testing.assert_close(storage.hidden_columns, torch.arange(24, 32))
    torch.testing.assert_close(storage.latent_begin, torch.tensor([48], dtype=torch.int32))


def test_declaration_shapes_reuse_examples_without_aliasing_live_roles():
    config = SimpleNamespace(hidden=64, vocabulary=256)
    examples = {}
    first = Scratch.allocate(8, 8, config, 8, "cpu", examples=examples)
    second = Scratch.allocate(8, 8, config, 8, "cpu", examples=examples)
    assert first.activations is second.activations
    assert first.hidden is second.hidden
    first.activations.fill_(3)
    assert torch.count_nonzero(first.hidden) == 0
    assert first.local_logits.shape == (8, 32)
    assert first.logits.shape == (8, 256)


def test_hidden_declaration_destinations_share_storage_without_operand_aliasing(monkeypatch):
    from ignition.preparation.examples import declaration_tensor
    from ignition.recipes.kimi3 import attention, decoder, kda
    from ignition.recipes.kimi31 import dspark

    operands = {}

    class Projection:
        def __init__(self, source, weight, output, *, name, **kwargs):
            assert source.untyped_storage()._cdata != output.untyped_storage()._cdata, name
            operands[name] = source, output
            self.output = output

        def functional(self, **kwargs):
            return self

    class EndDeclaration(Exception):
        pass

    def stop(*args, **kwargs):
        raise EndDeclaration

    for module in (decoder, attention, kda):
        monkeypatch.setattr(module, "prepare_projection", Projection)
    monkeypatch.setattr(dspark, "projection", Projection)
    monkeypatch.setattr(dspark, "UnquantizedProjection", Projection)
    monkeypatch.setattr(dspark, "PreparedChunkedAllGather",
        lambda source, output, **kwargs: Projection(source, None, output, **kwargs))
    monkeypatch.setattr(dspark, "Rotary", lambda *args: None)
    monkeypatch.setattr(decoder, "PreparedChunkedAllReduce", stop)
    monkeypatch.setattr(kda.short_conv, "Caps", lambda **kwargs: None)
    monkeypatch.setattr(kda.short_conv, "invocation_from_tensors", lambda **kwargs: None)
    monkeypatch.setattr(kda.short_conv, "plan", stop)

    hidden, intermediate = 64, 32
    config = SimpleNamespace(hidden=hidden, vocabulary=256, text={
        "routed_expert_hidden_size": 32, "rms_norm_eps": 1e-5},
        draft=SimpleNamespace(text={"rms_norm_eps": 1e-5}))
    shard = TensorParallelShard(0, 2)
    storage = DecoderStorage.allocate(config, shard, "cpu")
    examples = {}
    outputs = []
    for rows in (8, 2):
        scratch = Scratch.allocate(rows, rows, config, shard.world_size, "cpu", examples=examples)
        source = scratch.activations
        decoder.PreparedMLP({"gate_up": torch.empty(2 * intermediate, hidden),
            "down": torch.empty(hidden, intermediate)}, source, session=None, name="mlp", examples=examples)
        with pytest.raises(EndDeclaration):
            decoder.PreparedLatentMoE({"latent_up": torch.empty(hidden, 16), "latent_norm": torch.ones(32)},
                source, config, shard, runtime=None, router_runtime=None, coordinator=None,
                session=None, name="moe", workspace=None, storage=storage, examples=examples)
        metadata = SimpleNamespace(query_start_loc=torch.arange(2), convolution_slots=torch.zeros(1),
            fresh=torch.zeros(1), accepted_tokens=torch.zeros(1), num_seqs=torch.ones(1), num_tokens=torch.ones(1))
        with pytest.raises(EndDeclaration):
            kda.PreparedKDA({"input": torch.empty(648, hidden), "forget": torch.empty(128, 128),
                "output": torch.empty(hidden, 128), "convolution": torch.empty(384, 4)}, source,
                SimpleNamespace(recurrent=torch.empty(1, 1, 128, 128), convolution=torch.empty(1, 384, 4)),
                metadata, mode="decode", eps=1e-5, session=None, name="kda", workspace=None, buffers=examples)
        attention.PreparedMLAProjection({"key": torch.empty(1, 128, 512), "query_norm": torch.empty(1536),
            "latent_gate": torch.empty(1536 + 576 + 128, hidden), "q_b": torch.empty(192, 1536),
            "output": torch.empty(hidden, 128)}, source, None, None,
            eps=1e-5, mode="decode", session=None, name="mla", workspace=None, buffers=examples)
        weights = SimpleNamespace(config=config, shard=shard, draft=SimpleNamespace(
            projection=torch.empty(hidden // 2, 5 * hidden), context_kv=torch.empty(576, hidden),
            hidden_norm=torch.ones(hidden), layers=[]))
        dspark.PreparedContext(weights, torch.empty(rows, 5 * hidden, dtype=torch.bfloat16), None, None,
            session=None, name="context", workspace=None, runtime=None, coordinator=None,
            context=SimpleNamespace(shards=None), examples=examples)
        reduction = declaration_tensor(examples, "kimi.hidden_output", source.shape, source.dtype, source.device)
        for name in ("mlp.down", "moe.latent_up", "kda.output", "mla.output", "context.gather"):
            output = operands[name][1]
            assert output.shape == source.shape
            assert output.data_ptr() == reduction.data_ptr()
            outputs.append(output)
        assert operands["context.kv"][0].data_ptr() == source.data_ptr()
        assert reduction.untyped_storage()._cdata != source.untyped_storage()._cdata
    assert len({value.untyped_storage()._cdata for value in outputs}) == 1
