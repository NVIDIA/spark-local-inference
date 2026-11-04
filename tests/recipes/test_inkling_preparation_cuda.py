# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-rank preparation and replay of the complete Inkling graph catalog."""

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_single_rank_mixed_checkpoint_prepares_and_replays_catalog(tmp_path):
    from safetensors.torch import save_file
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.sampling import AdmissionBuffers, DeviceRequestState, RequestAdmission
    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes.inkling.bootstrap import PreparationConfig, prepare_rank
    from ignition.recipes.inkling.config import ModelConfig
    from ignition.recipes.inkling.pool import ContinuationPool, continuation_layout
    from .inkling_helpers import checkpoint

    snapshot = Path(os.environ.get("FLARE_INKLING_SNAPSHOT", ""))
    if not (snapshot / "tokenizer.json").is_file():
        pytest.skip("requires pinned Inkling frontend assets")
    _, tensors = checkpoint(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text())
    # Keep the publisher's complete frontend and special-token IDs while reducing
    # layer geometry; this exercises frontend/contract preparation without mocks.
    publisher = ModelConfig.load(snapshot)
    raw["text_config"].update(vocab_size=publisher.text["vocab_size"],
                              unpadded_vocab_size=publisher.vocabulary)
    for name in ("embed", "unembed"):
        tensors[f"model.llm.{name}.weight"] = (
            torch.randn(publisher.text["vocab_size"], 256) * .1).bfloat16()
    (tmp_path / "config.json").write_text(json.dumps(raw))
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(dict(
        metadata={"total_size": sum(t.nbytes for t in tensors.values())},
        weight_map={name: "model.safetensors" for name in tensors})))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (tmp_path / name).symlink_to(snapshot / name)
    del tensors
    config = ModelConfig.load(tmp_path)
    preparation = PreparationConfig(devices=(0,), context=256, prefill_chunk_size=128, max_active=1,
        snapshot_budget_bytes=continuation_layout(config, TensorParallelShard(0, 1)).row_bytes,
        compile_workers=1, autotune_effort="off", headroom_bytes=1 << 30, prompt_logprobs=True)
    with PackageStorage(device=0) as storage:
        with prepare_rank(tmp_path, preparation, rank=0, group=None, package_storage=storage) as model:
            assert {signature.kind for signature in model.signatures.values()} == {
                "init", "restore", "snapshot", "prefill", "scored_prefill", "decode", "resume"}
            assert model.memory_report["world_size"] == 1
            assert model.memory_report["compiled_decoder"]
            assert model.server_config["frontend"]["eos_token_ids"] == [200006]
            pool = next(resource for resource in model.resources if isinstance(resource, ContinuationPool))
            requests = next(resource for resource in model.resources if isinstance(resource, DeviceRequestState))
            buffers = model.buffers
            buffers["live_rows"].zero_()
            buffers["admission_count"].zero_()
            buffers["state_num_ops"].zero_()
            before = pool.backing.clone()
            physical_pages = int(pool.memory_capacity[1].item())
            pages_before = {name: page[:physical_pages].clone() for name, page in pool.pages.items()}
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            for graph in model.graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
            assert torch.equal(pool.backing, before)
            assert all(torch.equal(page[:physical_pages], pages_before[name]) for name, page in pool.pages.items())
            buffers["request_state_rows"][0, 0] = 1
            buffers["row_slot_ids"][0] = 0
            buffers["live_rows"].fill_(1)
            buffers["page_tables"][0].copy_(torch.tensor([0, 1], device="cuda"))
            prompt = list(range(127))

            def replay(name):
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                model.graphs[name].replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                assert requests.error_code[0].item() == 0

            def state_command(*, source, destination, operation, consumed):
                for name, value in dict(request_slots=0, source_slots=source, destination_slots=destination,
                        operation=operation, expected_consumed=consumed, num_ops=1).items():
                    buffers["state_" + name].fill_(value)

            def admit(generation, *, restore=False):
                admission = AdmissionBuffers.allocate("cpu", vocab_size=config.vocabulary, capacity=1)
                admission.stage(0, 0, RequestAdmission(generation, 13, max_new_tokens=8, top_k=1), prompt)
                admission.admission_count.fill_(1)
                for name in admission.__dataclass_fields__:
                    buffers[name].copy_(getattr(admission, name))
                buffers["row_generations"][0] = generation
                state_command(source=2 if restore else -1, destination=1,
                              operation=0 if restore else 1, consumed=127 if restore else 0)
                replay("restore1" if restore else "init1")
                buffers["state_num_ops"].zero_()
                buffers["admission_count"].zero_()

            def prefill(*, scored=False):
                buffers["query_lengths"][0] = len(prompt)
                buffers["token_ids"][:len(prompt)].copy_(torch.tensor(prompt, device="cuda"))
                buffers["sample_enabled"][0] = 0
                replay("scored_prefill128" if scored else "prefill128")
                assert requests.consumed_tokens[0].item() == 127

            def continuation():
                replay("resume1")
                tokens = [requests.pending_token[0].item()]
                buffers["query_lengths"][0] = buffers["sample_enabled"][0] = 1
                for _ in range(2):
                    replay("decode1")
                    tokens.append(requests.pending_token[0].item())
                assert requests.consumed_tokens[0].item() == 129
                # Live histories exclude the embedded partial-page snapshot copies;
                # compare their active external KV pages instead.
                state = {region.name: pool.region(region.name)[1].clone()
                         for region in pool.layout.regions
                         if region.name not in pool.pages and region.name != "consumed_tokens"}
                state.update({f"pages.{layer}.{kind}": pool.cache(layer, kind).flatten(0, 1)[:129].clone()
                              for layer in range(config.layers) if config.attention(layer).window < 0
                              for kind in ("key", "value")})
                return tokens, state

            def assert_state(actual, expected):
                assert actual.keys() == expected.keys()
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0, msg=name)

            admit(7)
            prefill()
            hidden = pool.last_hidden[1].clone()
            state_command(source=1, destination=2, operation=0, consumed=127)
            replay("snapshot1")
            buffers["state_num_ops"].zero_()
            snapshot_row = pool.backing[2].clone()
            expected_tokens, expected_state = continuation()

            admit(8, restore=True)
            assert requests.consumed_tokens[0].item() == 127
            assert torch.equal(pool.last_hidden[1], hidden)
            tokens, state = continuation()
            assert tokens == expected_tokens
            assert_state(state, expected_state)
            assert torch.equal(pool.backing[2], snapshot_row)

            admit(9)
            buffers["prompt_target_tokens"].fill_(-1)
            buffers["prompt_target_tokens"][:126].copy_(torch.tensor(prompt[1:], device="cuda"))
            prefill(scored=True)
            assert torch.equal(pool.last_hidden[1], hidden)
            scores = buffers["prompt_logprobs"][int(buffers["frame_index"].item())].cpu()
            assert scores[:126, 0].tolist() == prompt[1:]
            assert torch.isfinite(scores[:126, 1].contiguous().view(torch.float32)).all()
            assert (scores[:126, 1].contiguous().view(torch.float32) <= 0).all()
            assert ((scores[:126, 2] >= 1) & (scores[:126, 2] <= config.vocabulary)).all()
            assert (scores[126:, 0] == -1).all()
            assert torch.isnan(scores[126:, 1].contiguous().view(torch.float32)).all()
            tokens, state = continuation()
            assert tokens == expected_tokens
            assert_state(state, expected_state)
            assert torch.equal(pool.backing[2], snapshot_row)
