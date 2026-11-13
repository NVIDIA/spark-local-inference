"""Fused convolution/preprocessing preserves the independent convolution and GDN contracts."""

import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("width", [3, 8, 16])
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_fused_convolution_records_follow_mutable_captured_metadata(width, world_size):
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.sequence import short_conv
    from ignition.components.gated_norm import PreparedGatedRMSNorm
    from ignition.components.gdn_recovery import GDNLayerRecords, PreparedGDNCommit, PreparedRecoveredRecurrence
    from ignition.components.recurrent import DeltaMetadata, _commit_convolution
    from ignition.components.state import StateLayout, StateStorage
    from ignition.preparation.triton import prepare_triton
    from tests.tensor_examples import TensorExamples

    device = torch.device("cuda:0")
    torch.manual_seed(241)
    key_heads, value_heads = 16 // world_size, 48 // world_size
    channels, tokens = (2 * key_heads + value_heads) * 128, 2 * width
    value_channels = value_heads * 128
    gate_offset = channels + value_channels
    projected = torch.randn((tokens, gate_offset + 2 * value_heads), device=device).bfloat16()
    weight = (torch.randn((channels, 4), device=device) * .1).bfloat16()
    a_log = torch.rand(value_heads, device=device) * .5
    bias = torch.randn(value_heads, device=device) * .1
    norm_weight = torch.ones(128, device=device, dtype=torch.bfloat16)
    metadata = DeltaMetadata.allocate(2, device, verification_columns=width)
    metadata.query_start_loc.copy_(torch.tensor([0, width, tokens], device=device, dtype=torch.int32))
    metadata.state_indices[:, 0].copy_(torch.tensor([1, 2], device=device))
    metadata.convolution_slots.copy_(torch.tensor([1, 2], device=device))
    metadata.num_seqs.fill_(2)
    metadata.num_tokens.fill_(tokens)
    counts = torch.ones(2, device=device, dtype=torch.int32)
    layout = StateLayout.from_specs((
        ("convolution", (channels, 18), torch.bfloat16),
        ("recurrent", (value_heads, 128, 128), torch.float32),
    ))
    stores = [StateStorage.allocate(layout, 3, device) for _ in range(2)]
    stores[0].region("convolution").normal_(std=.5)
    stores[0].region("recurrent").normal_(std=.05)
    initial = stores[0].backing.clone()
    conv_offset = torch.tensor([layout.region("convolution").offset // 2], device=device, dtype=torch.int64)
    paths = []
    recovery_buffers = TensorExamples(device)
    for storage in stores:
        mixed = torch.empty((tokens, channels), device=device, dtype=torch.bfloat16)
        output = torch.empty((tokens, value_channels), device=device, dtype=torch.bfloat16)
        gated = torch.empty_like(output)
        records = GDNLayerRecords.allocate((0,), 2, device, key_heads=key_heads, value_heads=value_heads, width=16)
        operands = dict(x=projected[:, :channels], conv_weight=weight,
            query_start_loc=metadata.query_start_loc, state_slot_ids=metadata.convolution_slots,
            state_is_fresh=metadata.fresh, num_accepted_tokens=metadata.accepted_tokens,
            num_seqs=metadata.num_seqs, num_tokens=metadata.num_tokens,
            conv_state=storage.region("convolution"), out=mixed)
        verify = PreparedRecoveredRecurrence(mixed_qkv=mixed, a=projected[:, gate_offset + value_heads:], b=projected[:, gate_offset:gate_offset + value_heads],
            A_log=a_log, dt_bias=bias, state=storage.region("recurrent"), metadata=metadata, output=output,
            records=records[0], key_heads=key_heads, value_heads=value_heads,
            convolution=None if not paths else operands, buffers=recovery_buffers)
        norm = PreparedGatedRMSNorm(output.view(tokens, value_heads, 128), projected[:, channels:gate_offset].view(tokens, value_heads, 128),
            norm_weight, gated.view(tokens, value_heads, 128), metadata.num_tokens, eps=1e-6, round_normalized=True)
        commit = PreparedGDNCommit(storage.backing, {0: storage.region("recurrent")}, records, metadata, counts,
                                   key_heads=key_heads, value_heads=value_heads)
        commit_conv = prepare_triton(_commit_convolution,
            (storage.backing.view(torch.bfloat16), conv_offset, metadata.convolution_slots, counts),
            (1, 2, channels // 128), STRIDE=layout.row_bytes // 2, CHANNELS=channels, COLUMNS=18, BLOCK=128)
        paths.append(dict(mixed=mixed, output=output, gated=gated, records=records, operands=operands,
                          verify=verify, norm=norm, commit=commit, commit_conv=commit_conv))
    for index in (8, 11):
        assert paths[0]["verify"].prepare_args[index].data_ptr() == paths[1]["verify"].prepare_args[index].data_ptr()
    caps = short_conv.Caps(device=device, mode="decode", channels=channels, kernel_size=4,
        max_tokens=tokens, max_seqs=2, max_state_slots=3, max_speculative_tokens=15)
    plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**paths[0]["operands"]))
    bindings = []

    def prepare(state):
        scratch = {spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                   for spec in plan.scratch_specs()}
        binding = state.bind(plan=plan, scratch=scratch, **paths[0]["operands"])
        bindings.append(binding)
        return PreparedCall(run=lambda: state.run(binding), output=binding.out)

    def run(path):
        path["verify"]()
        path["norm"]()
        path["commit"]()
        path["commit_conv"]()

    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        session.prepare((plan.request(name="convolution", prepare_call=prepare),))
        session.freeze()
        from torch._higher_order_ops.triton_kernel_wrap import get_mutated_tensors, kernel_side_table

        from ignition.components.gdn_recovery import _prepare_convolved_inputs

        producer = paths[1]["verify"]
        arguments = dict(zip(_prepare_convolved_inputs.fn.arg_names, producer.prepare_args))
        constants = {**producer.prepare_options,
                     **{name: value for name, value in arguments.items() if not isinstance(value, torch.Tensor)}}
        tensors = {name: value for name, value in arguments.items() if isinstance(value, torch.Tensor)}
        mutated = get_mutated_tensors(kernel_side_table.add_kernel(_prepare_convolved_inputs),
            kernel_side_table.add_constant_args(constants), tensors, {})
        assert {"Mixed", "ConvState", "Queries", "Keys", "Decay", "Beta"} <= set(mutated)
        stores[1].backing.copy_(initial)
        run(paths[1])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run(paths[1])
        scenarios = (
            ((width, width - 3), (1, 2), (1, 5), (False, True), (3, width - 3)),
            ((width, 0), (2, -1), (16, 0), (False, False), (width, 0)),
            ((1,), (1,), (2,), (False,), (1,)),
            ((), (), (), (), ()),
        )
        for lengths, slots, accepted, fresh, committed in scenarios:
            live, requests = sum(lengths), len(lengths)
            starts = [0, *torch.tensor(lengths).cumsum(0).tolist()]
            metadata.query_start_loc.copy_(torch.tensor(starts + [live] * (3 - len(starts)), device=device, dtype=torch.int32))
            metadata.num_seqs.fill_(requests)
            metadata.num_tokens.fill_(live)
            metadata.convolution_slots.fill_(-1)
            metadata.state_indices.zero_()
            metadata.accepted_tokens.zero_()
            metadata.fresh.fill_(False)
            counts.zero_()
            for row in range(requests):
                metadata.convolution_slots[row] = slots[row]
                metadata.state_indices[row, 0] = max(slots[row], 0)
                metadata.accepted_tokens[row] = accepted[row]
                metadata.fresh[row] = fresh[row]
                counts[row] = committed[row]
            projected.normal_(std=.5)
            original_input = projected.clone()
            for storage, path in zip(stores, paths, strict=True):
                storage.backing.copy_(initial)
                for tensor in (path["mixed"], path["output"], path["gated"], path["records"].delta,
                               path["records"].keys, path["records"].decay,
                               path["verify"].prepare_args[8], path["verify"].prepare_args[11]):
                    tensor.fill_(float("nan"))
            expected_mixed = torch.zeros_like(paths[0]["mixed"])
            expected_history = stores[0].region("convolution").clone()
            for row, length in enumerate(lengths):
                slot = slots[row]
                if slot < 0 or not length:
                    continue
                history = expected_history[slot, :, accepted[row] - 1:accepted[row] + 2].clone()
                if fresh[row]:
                    history.zero_()
                sequence = torch.cat((history, projected[starts[row]:starts[row + 1], :channels].T), dim=1)
                convolved = F.conv1d(sequence[None], weight[:, None, :], groups=channels).bfloat16()
                expected_mixed[starts[row]:starts[row + 1]] = F.silu(convolved).squeeze(0).T
                expected_history[slot, :, :3] = sequence[:, 1:4]
                expected_history[slot, :, 3:] = 0
                expected_history[slot, :, 3:3 + length - 1] = projected[starts[row] + 1:starts[row + 1], :channels].T
                if committed[row] > 1:
                    expected_history[slot, :, :3] = expected_history[slot, :, committed[row] - 1:committed[row] + 2].clone()
            short_conv.run(bindings[0])
            run(paths[0])
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[name] == after[name]
            torch.testing.assert_close(paths[1]["mixed"], expected_mixed, rtol=.008, atol=.002)
            torch.testing.assert_close(stores[1].region("convolution"), expected_history, rtol=0, atol=0)
            torch.testing.assert_close(stores[1].region("recurrent"), stores[0].region("recurrent"), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(paths[1]["gated"], paths[0]["gated"], rtol=2**-6, atol=.02, equal_nan=True)
            if live:
                assert (paths[1]["gated"][:live] == paths[0]["gated"][:live]).float().mean() > .99
            for name in ("delta", "keys", "decay"):
                torch.testing.assert_close(getattr(paths[1]["records"], name), getattr(paths[0]["records"], name),
                                           rtol=1e-5, atol=1e-6, equal_nan=True)
            torch.testing.assert_close(paths[1]["verify"].prepare_args[11], paths[0]["verify"].prepare_args[11],
                                       rtol=0, atol=0, equal_nan=True)
            assert torch.equal(projected, original_input)
        graph.reset()
