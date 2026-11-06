# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export relative attention and residual convolution for native replay."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class OperatorForward:
    def __init__(self, attention, convolution, history, result):
        self.attention, self.convolution, self.history, self.result = attention, convolution, history, result

    def __call__(self):
        attention = self.attention()
        convolution = self.convolution()
        end = attention.numel() + convolution.numel()
        self.result[:attention.numel()].copy_(attention.flatten())
        self.result[attention.numel():end].copy_(convolution.flatten())
        self.result[end:].copy_(self.history.flatten())


def prepare(directory):
    import torch
    from ignition.components.paged_batch import Metadata
    from ignition.components.relative_paged_attention import PreparedRelativePagedAttention
    from ignition.components.residual_short_conv import PreparedResidualShortConvolution
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.package_file import combine
    from ignition.preparation.runtime_package import export_model
    from ignition.preparation.tuning import RecordedPreparationSession
    from fixed_package_fixture import allocation_reference
    from fixture_frontend import server_config
    from signing import publisher

    torch.cuda.set_device(0)
    torch.manual_seed(838)
    device = torch.device("cuda", 0)
    with ModuleImages() as images, PackageStorage() as storage:
        with RecordedPreparationSession(device=device, checkpoint="inkling-operator-export",
                workload_schema="relative-attention-residual-convolution", numerical_policy="bf16-fp32-history",
                compile_workers=1, benchmark_storage=storage) as session:
            query = torch.randn((17, 2, 128), device=device, dtype=torch.bfloat16)
            key = torch.randn((2, 128, 1, 128), device=device, dtype=torch.bfloat16)
            value = torch.randn_like(key)
            pages = torch.tensor([[1, 0]], device=device, dtype=torch.int32)
            lengths = torch.tensor([145], device=device, dtype=torch.int32)
            cu = torch.tensor([0, 13], device=device, dtype=torch.int32)
            bias = (torch.randn((17, 2, 1024), device=device) * .2).bfloat16()
            attention_output = torch.empty_like(query)
            attention = PreparedRelativePagedAttention(query, key, value, pages, lengths, cu, bias, attention_output,
                softmax_scale=1 / 128, session=session, name="relative-attention")
            source = torch.randn((17, 67), device=device, dtype=torch.bfloat16)
            weight = torch.randn((67, 4), device=device) * .1
            history = torch.zeros((3, 67, 3), device=device)
            metadata = Metadata.allocate(17, 1, 2, device)
            metadata.state_rows.fill_(1)
            metadata.starts.zero_()
            metadata.lengths.fill_(13)
            metadata.query_start_loc.copy_(cu)
            convolution_output = torch.empty_like(source)
            convolution = PreparedResidualShortConvolution(source, weight, history, metadata, convolution_output,
                session=session, name="residual-convolution")
            result = torch.empty(query.numel() + source.numel() + history.numel(), device=device)
            run = OperatorForward(attention, convolution, history, result)
            session.tune(show_progress=False)
            signatures = {"decode": GraphSignature("decode", 1, 1)}
            stream = torch.cuda.Stream(device=device)
            graphs, compiled = capture_forwards({"decode": run}, signatures, session=session, device=device,
                stream=stream, pool=storage.graph_pool.id, retain_graphs=True)
            session.freeze()
            graphs["decode"].replay()
            torch.cuda.synchronize(device)
            expected = result.view(torch.uint8).cpu().numpy().tobytes()
            assert torch.count_nonzero(attention_output[13:]) == 0
            assert torch.count_nonzero(convolution_output[13:]) == 0
            torch.testing.assert_close(history[1], source[10:13].float().T, atol=0, rtol=0)
            result.zero_()
            buffers = {
                "frame_index": torch.zeros(1, device=device, dtype=torch.int32),
                "submission_sequence": torch.zeros(1, device=device, dtype=torch.int64),
                "completion_records": torch.zeros((2, 1, 24), device=device, dtype=torch.int64),
                "result": result,
            }
            records = buffers["completion_records"]
            records[..., 4] = -1
            records[..., 12:14] = -1
            records[..., 15] = 1
            records[..., 16:] = -1
            prepared = PreparedExecution(device=0, graphs=graphs, buffers=buffers, signatures=signatures,
                max_active=1, resources=(storage, attention, convolution, result, stream, session, compiled),
                export_zero_ranges=((result.data_ptr(), result.nbytes),))
            prepared.server_config = server_config(directory)
            prepared.package_data = None
            package = export_model(prepared, images, storage, directory / "rank.json", options={}, sources=True)
            combine(directory / "package", [directory / "rank.json"], [], publisher=publisher())
            reference = allocation_reference(package.allocations, result)
            (directory / "expected.json").write_text(json.dumps({
                "reference": reference, "bytes": list(expected),
                "addresses": {allocation.name: allocation.address for allocation in package.allocations},
                "zero_reference": reference, "zero_bytes": result.nbytes,
            }))
            assert package.export_counts["modules"] >= 3
            for graph in graphs.values():
                graph.reset()
    print("exported prepared relative attention and residual convolution", flush=True)


if __name__ == "__main__":
    # Compiler workers import the fixture type by its stable module name.
    from inkling_operator_package_fixture import prepare

    destination = Path(sys.argv[1])
    destination.mkdir(parents=True, exist_ok=True)
    prepare(destination)
