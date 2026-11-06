# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduced-shape mixed checkpoint loading and complete graph-catalog preparation."""

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
    reason='requires an explicitly assigned CUDA device')


@torch.no_grad()
def test_reduced_mixed_checkpoint_prepares_and_replays_catalog(tmp_path):
    from safetensors.torch import save_file
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextModel
    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes.muse_glimmer.bootstrap import PreparationConfig, prepare
    from ignition.recipes.muse_glimmer.config import ModelConfig
    from ignition.recipes.muse_glimmer.pool import ContinuationPool, continuation_layout
    from ignition.components.sampling import AdmissionBuffers, DeviceRequestState, RequestAdmission

    snapshot = Path(os.environ.get('FLARE_MUSE_GLIMMER_SNAPSHOT', ''))
    if not (snapshot / 'tokenizer.json').is_file():
        pytest.skip('requires pinned Muse Glimmer frontend assets')
    torch.manual_seed(573)
    c = MuseGlimmerTextConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128, vocab_size=202048,
        layer_types=['sliding_attention', 'full_attention'], layer_rope_theta=[500000, 0],
        sliding_window=128, max_position_embeddings=256, eos_token_id=200001)
    canonical = MuseGlimmerTextModel(c).eval().bfloat16()
    tensors = {'model.language_model.' + name: value.contiguous() for name, value in canonical.state_dict().items()}
    declarations = {}
    def nvfp4(prefix, rows, columns):
        tensors[prefix + '.weight'] = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8)
        tensors[prefix + '.weight_scale'] = torch.full((rows, columns // 16), .0625, dtype=torch.float8_e4m3fn)
        tensors[prefix + '.weight_scale_2'] = torch.tensor(.125)
        declarations[prefix] = {'quant_algo': 'W4A16_NVFP4', 'group_size': 16}
    nvfp4('lm_head', c.vocab_size, c.hidden_size)
    nvfp4('model.language_model.layers.0.self_attn.v_proj', 128, c.hidden_size)
    nvfp4('model.language_model.layers.1.mlp.down_proj', c.hidden_size, c.intermediate_size)
    for suffix, weight_scale, input_scale in [
            ('0.self_attn.q_proj', 1/32, .031),
            ('0.self_attn.k_proj', 1/16, .017),
            ('1.mlp.up_proj', 1/64, .045)]:
        prefix = 'model.language_model.layers.' + suffix
        tensors[prefix + '.weight'] = (tensors[prefix + '.weight'].float() / weight_scale).to(torch.float8_e4m3fn)
        tensors[prefix + '.weight_scale'], tensors[prefix + '.input_scale'] = torch.tensor(weight_scale), torch.tensor(input_scale)
        declarations[prefix] = {'quant_algo': 'FP8'}
    # Vision tensors are deliberately present: text preparation must not load them.
    tensors['model.vision_model.unused.weight'] = torch.ones(256, dtype=torch.bfloat16)
    raw = dict(model_type='muse_glimmer', architectures=['MuseGlimmerForConditionalGeneration'],
        text_config=c.to_dict(), quantization_config=dict(quant_method='modelopt', quant_algo='MIXED_PRECISION',
                                                         quantized_layers=declarations))
    (tmp_path / 'config.json').write_text(json.dumps(raw))
    save_file(tensors, str(tmp_path / 'weights.safetensors'))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(dict(
        metadata={'total_size': sum(t.nbytes for t in tensors.values())},
        weight_map={name: 'weights.safetensors' for name in tensors})))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        (tmp_path / name).symlink_to(snapshot / name)
    del canonical, tensors
    config = PreparationConfig(context=256, prefill_chunk_size=128, max_active=1,
        snapshot_budget_bytes=continuation_layout(ModelConfig.from_dict(raw)).row_bytes,
        compile_workers=1, headroom_bytes=1 << 30, prompt_logprobs=True)
    with PackageStorage(device=0) as storage:
        with prepare(tmp_path, config, package_storage=storage) as model:
            buffers = model.buffers
            assert {s.kind for s in model.signatures.values()} == {
                'init', 'restore', 'snapshot', 'prefill', 'scored_prefill', 'decode', 'resume'}
            assert model.memory_report['kv_dtype'] == 'bfloat16'
            assert model.server_config['frontend']['eos_token_ids'] == [200001, 200008]
            buffers['live_rows'].zero_()
            buffers['admission_count'].zero_()
            buffers['state_num_ops'].zero_()
            before = buffers['state_pool'].clone()
            allocations = torch.cuda.memory_stats()['allocation.all.allocated']
            for graph in model.graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
            assert torch.equal(buffers['state_pool'], before)
            pool = next(resource for resource in model.resources if isinstance(resource, ContinuationPool))
            requests = next(resource for resource in model.resources if isinstance(resource, DeviceRequestState))
            buffers['request_state_rows'][0, 0] = 1
            buffers['row_slot_ids'][0] = 0
            buffers['live_rows'].fill_(1)
            buffers['page_tables'][0].copy_(torch.tensor([0, 1], device='cuda'))
            prompt = list(range(127))

            def replay(name):
                allocations = torch.cuda.memory_stats()['allocation.all.allocated']
                model.graphs[name].replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated'] == allocations
                assert requests.error_code[0].item() == 0

            def command(graph, source, destination, operation, expected):
                for name, value in dict(request_slots=0, source_slots=source, destination_slots=destination,
                        operation=operation, expected_consumed=expected, num_ops=1).items():
                    buffers['state_' + name].fill_(value)
                replay(graph)
                buffers['state_num_ops'].zero_()

            def admit(generation, *, restore=False):
                admission = AdmissionBuffers.allocate('cpu', vocab_size=c.vocab_size, capacity=1)
                admission.stage(0, 0, RequestAdmission(generation, 13, max_new_tokens=8, top_k=1), prompt)
                admission.admission_count.fill_(1)
                for name in admission.__dataclass_fields__:
                    buffers[name].copy_(getattr(admission, name))
                buffers['row_generations'][0] = generation
                command('restore1' if restore else 'init1', 2 if restore else -1, 1, 0 if restore else 1,
                        127 if restore else 0)
                buffers['admission_count'].zero_()

            def prefill(tokens, *, scored=False):
                buffers['query_lengths'][0] = len(tokens)
                buffers['token_ids'][:len(tokens)].copy_(torch.tensor(tokens, device='cuda'))
                buffers['sample_enabled'][0] = 0
                replay('scored_prefill128' if scored else 'prefill128')

            def continuation():
                replay('resume1')
                tokens = [requests.pending_token[0].item()]
                buffers['query_lengths'][0] = buffers['sample_enabled'][0] = 1
                for _ in range(2):
                    replay('decode1')
                    tokens.append(requests.pending_token[0].item())
                assert requests.consumed_tokens[0].item() == 129
                return tokens, pool.last_hidden[1].clone()

            admit(7)
            prefill(prompt)
            assert requests.consumed_tokens[0].item() == 127
            hidden = pool.last_hidden[1].clone()
            cache = {(layer, kind): pool.cache(layer, kind).flatten(0, 1)[:127].clone()
                     for layer in range(2) for kind in ('key', 'value')}

            def assert_prompt_state():
                assert requests.consumed_tokens[0].item() == 127
                torch.testing.assert_close(pool.last_hidden[1], hidden, atol=0, rtol=0)
                for (layer, kind), reference in cache.items():
                    torch.testing.assert_close(pool.cache(layer, kind).flatten(0, 1)[:127], reference,
                                               atol=0, rtol=0)

            command('snapshot1', 1, 2, 0, 127)
            snapshot_row = pool.backing[2].clone()
            expected_tokens, expected_hidden = continuation()

            admit(8)
            prefill(prompt[:64])
            prefill(prompt[64:])
            assert_prompt_state()
            tokens, final_hidden = continuation()
            assert tokens == expected_tokens
            torch.testing.assert_close(final_hidden, expected_hidden, atol=0, rtol=0)
            assert torch.equal(pool.backing[2], snapshot_row)

            admit(9, restore=True)
            assert_prompt_state()
            tokens, final_hidden = continuation()
            assert tokens == expected_tokens
            torch.testing.assert_close(final_hidden, expected_hidden, atol=0, rtol=0)
            assert torch.equal(pool.backing[2], snapshot_row)

            admit(10)
            buffers['prompt_target_tokens'].fill_(-1)
            buffers['prompt_target_tokens'][:126].copy_(torch.tensor(prompt[1:], device='cuda'))
            prefill(prompt, scored=True)
            assert_prompt_state()
            scores = buffers['prompt_logprobs'][int(buffers['frame_index'].item())].cpu()
            assert scores[:126, 0].tolist() == prompt[1:]
            assert torch.isfinite(scores[:126, 1].contiguous().view(torch.float32)).all()
            assert (scores[:126, 1].contiguous().view(torch.float32) <= 0).all()
            assert ((scores[:126, 2] >= 1) & (scores[:126, 2] <= c.vocab_size)).all()
            assert (scores[126:, 0] == -1).all()
            assert torch.isnan(scores[126:, 1].contiguous().view(torch.float32)).all()
            assert torch.equal(pool.backing[2], snapshot_row)
