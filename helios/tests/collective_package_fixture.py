# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export small TP4 collective graphs for replay after all preparation ranks exit."""

from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fixture_frontend import server_config
from signing import publisher


def _process():
    return dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]))


def _configuration(folder, signatures, tuned_bindings=None):
    from ignition.preparation import contract as c, runtime_package_pb2 as p

    view = c.state_view("state_pool", 32)
    cache = [dict(kind=p.CONTINUATION_STATE, storage=view, tokens_per_record=1, window_tokens=0),
             dict(kind=p.PAGED_KEY_VALUE, storage=c.state_view("pages", 16),
                  tokens_per_record=64, window_tokens=0)]
    contract = c.model_contract(vocabulary=2, input_vocabulary=2, page_tokens=64,
        page_tables=[], regions=cache,
        graphs=[dict(graph=name, kind=p.GraphOperation.Value(signature.kind.upper()),
            inputs=[], state=[dict(view=view, access=p.READ_WRITE)], state_rows_per_request=1,
            embedding_tables=[], tuned_bindings=[dict(buffer=b.buffer, tuning_graph=b.tuning_graph)
                for b in (tuned_bindings or {}).get(name, ())]) for name, signature in signatures.items()],
        vision=dict(feature_slots=0, features_per_request=0, max_images=0,
                    patches_per_feature=0, patch_capacities=[]),
        frontend=c.frontend(tools=p.QWEN_CODER, assistant_marker="assistant",
            message_end_markers=[], prefix_boundary=c.prefix_boundary([])))
    return {**server_config(folder), "model_id": "collective-fixture",
        "identity": "42" * 32, "model_contract": contract, "max_sequence_tokens": 128,
        "prefill_chunk_size": 128, "pages": 16, "state_rows": 3, "snapshots": 2,
        "page_tokens": 64, "reserve_bytes": 0}


def _rank(rank, directory, startup_tuning=False):
    from ignition.components.collectives import PreparedAllReduce
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.execution import GraphSignature, PreparedExecution, TunedBinding, TuningGraph
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model
    from ignition.preparation.tuning import RecordedPreparationSession
    from tests.components.test_collective_transport_tp4_cuda import _expected, _operands, _selected_query

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=(directory / "gloo").as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(minutes=5))
    folder = directory / f"rank-{rank}"
    folder.mkdir()
    runtime = None
    captured = []
    try:
        with ModuleImages() as images, PackageStorage(rank=rank, world=4, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=33,
                hidden_size=5120, ipc=storage.peer_storage)
            with RecordedPreparationSession(device=device, checkpoint="collective-package",
                    workload_schema="tp4-bf16-peer-storage", numerical_policy="fp32-single-rounding",
                    coordinator=coordinator) as session:
                sources = {width: torch.empty((33, width), dtype=torch.bfloat16, device=device)
                           for width in (2560, 4096, 5120)}
                shapes = {"init": (2560, 12), "decode": (2560, 13),
                          "prefill64": (4096, 33), "prefill128": (5120, 33)}
                outputs = {name: torch.empty((rows, width), dtype=torch.bfloat16, device=device)
                           for name, (width, rows) in shapes.items()}
                operations = {name: PreparedAllReduce(sources[width][:rows], outputs[name],
                    runtime=runtime, session=session, coordinator=coordinator, name=name)
                    for name, (width, rows) in shapes.items()}
                session.tune(show_progress=False)
                queries = {name: _selected_query(operation) for name, operation in operations.items()}
                if rank == 0:
                    for name, operation in operations.items():
                        query = queries[name]
                        print(f"{name}: {operation.plan.selection.config}; surface={query['surface']}; "
                              f"transport={query['call'].get('transport')}", flush=True)
                stream = torch.cuda.Stream(device=device)
                startup_buffers, tuning_graphs, tuned_bindings = {}, {}, {}
                startup_consumer = None
                if startup_tuning:
                    startup_buffers = {
                        name: torch.zeros(1, dtype=torch.int32, device=device)
                        for name in ("startup-selector", "startup-bias", "startup-count", "startup-status",
                                     "startup-failure", "startup-observed", "unused-count", "unused-status")
                    }

                    def produce(selector, bias, count, status, failure):
                        count.add_(1)
                        selector.fill_(3)
                        bias.fill_(7)
                        status.copy_(failure)

                    def consume(selector, bias, observed):
                        observed.copy_(selector * 11 + bias)

                    producer = torch.compile(produce, fullgraph=True)
                    startup_consumer = torch.compile(consume, fullgraph=True)
                    producer(*(startup_buffers[name] for name in (
                        "startup-selector", "startup-bias", "startup-count", "startup-status", "startup-failure")))
                    startup_consumer(*(startup_buffers[name] for name in (
                        "startup-selector", "startup-bias", "startup-observed")))
                    sources[2560].fill_(rank + 1)
                    operations["init"](sources[2560][:12])
                    startup_buffers["unused-count"].add_(1)
                    startup_buffers["unused-status"].zero_()
                    torch.cuda.synchronize(device)
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    dist.barrier()
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                            graph, stream=stream, pool=storage.graph_pool.id):
                        operations["init"](sources[2560][:12])
                        producer(*(startup_buffers[name] for name in (
                            "startup-selector", "startup-bias", "startup-count", "startup-status", "startup-failure")))
                    graph.instantiate()
                    captured.append(graph)
                    tuning_graphs["choose-profile"] = TuningGraph(graph=graph,
                        outputs=("startup-selector", "startup-bias", "startup-count"), status_buffer="startup-status")
                    bindings = tuple(TunedBinding(buffer=name, tuning_graph="choose-profile")
                                     for name in ("startup-selector", "startup-bias"))
                    tuned_bindings = dict.fromkeys(("init", "decode"), bindings)
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, stream=stream, pool=storage.graph_pool.id):
                        startup_buffers["unused-count"].add_(1)
                        startup_buffers["unused-status"].zero_()
                    graph.instantiate()
                    captured.append(graph)
                    tuning_graphs["unused-profile"] = TuningGraph(graph=graph,
                        outputs=("unused-count",), status_buffer="unused-status")
                graphs, signatures = {}, {}
                for name, operation in operations.items():
                    width, rows = shapes[name]
                    sources[width].fill_(rank + 1)
                    operation(sources[width][:rows])
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                            graph, stream=stream, pool=storage.graph_pool.id):
                        operation(sources[width][:rows])
                        if name in tuned_bindings:
                            startup_consumer(*(startup_buffers[field] for field in (
                                "startup-selector", "startup-bias", "startup-observed")))
                    graph.instantiate()
                    captured.append(graph)
                    graphs[name] = graph
                    signatures[name] = GraphSignature("prefill" if name.startswith("prefill") else name,
                        1, int(name.removeprefix("prefill")) if name.startswith("prefill") else 1)
                # The loader requires a complete capacity catalog with distinct
                # executables. No HTTP/model semantics are tested by these graphs.
                for name in ("restore", "snapshot", "resume"):
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(
                            graph, stream=stream, pool=storage.graph_pool.id):
                        operations["init"](sources[2560][:12])
                    graph.instantiate()
                    captured.append(graph)
                    graphs[name] = graph
                    signatures[name] = GraphSignature(name, 1, 1)
                session.freeze()
                state = storage.device_allocate("continuations", (4, 32), torch.uint8,
                    group="STATE_ROWS", maximum=4, units=4)
                pages = storage.device_allocate("pages", (16, 16), torch.uint8,
                    group="KV_PAGES", maximum=16, units=16)
                buffers = {"frame_index": torch.zeros(1, dtype=torch.int32, device=device),
                    "submission_sequence": torch.zeros(1, dtype=torch.int64, device=device),
                    "completion_records": torch.zeros((2, 1, 32), dtype=torch.int64, device=device),
                    "state_pool": state, "memory_capacity": torch.tensor([4, 16], device=device),
                    "pages": pages}
                records = buffers["completion_records"]
                records[..., 4] = -1
                records[..., 12:14] = -1
                records[..., 15] = 1
                records[..., 16:] = -1
                buffers.update({f"source-{width}": value for width, value in sources.items()})
                buffers.update({f"output-{name}": value for name, value in outputs.items()})
                buffers.update(startup_buffers)
                prepared = PreparedExecution(device=rank, graphs=graphs, buffers=buffers,
                    signatures=signatures, max_active=1,
                    tuning_graphs=tuning_graphs, tuned_bindings=tuned_bindings,
                    resources=(runtime, session, sources, outputs, storage, stream, startup_buffers, startup_consumer))
                prepared.server_config = _configuration(folder, signatures, tuned_bindings)
                prepared.package_data = None
                cases = []
                for step in range(3):
                    for width in sources:
                        values = _operands(33, width, step)
                        (folder / f"source-{width}-{step}.bin").write_bytes(
                            values[rank].view(torch.uint8).numpy().tobytes())
                    for name, (width, rows) in shapes.items():
                        expected = _expected([value[:rows] for value in _operands(33, width, step)],
                            rank, queries[name])
                        path = f"expected-{name}-{step}.bin"
                        (folder / path).write_bytes(expected.view(torch.uint8).numpy().tobytes())
                        cases.append(dict(graph=name, source=f"source-{width}", output=f"output-{name}",
                            input=f"rank-{rank}/source-{width}-{step}.bin",
                            expected=f"rank-{rank}/{path}", step=step))
                torch.cuda.synchronize(device)
                dist.barrier()
                storage.peer_storage.reset()
                for value in startup_buffers.values():
                    value.zero_()
                torch.cuda.synchronize(device)
                dist.barrier()
                export_model(prepared, images, storage, directory / f"rank-{rank}.json",
                    options={}, bundle_only=rank != 0, sources=True)
                (folder / "metadata.json").write_text(json.dumps(dict(cases=cases, process=_process(),
                    owned=list(storage.peer_storage.owned.items()))))
                for graph in captured:
                    graph.reset()
                captured.clear()
                runtime.close()
                runtime = None
    finally:
        for graph in captured:
            graph.reset()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def prepare(directory, *, startup_tuning=False):
    from ignition.preparation.package_file import combine

    directory.mkdir(exist_ok=True)
    mp.spawn(_rank, args=(directory, startup_tuning), nprocs=4, join=True)
    ranks = [json.loads((directory / f"rank-{rank}" / "metadata.json").read_text()) for rank in range(4)]
    peers = [dict(owner=rank, address=address, nbytes=size, readers=list(range(4)))
             for rank, metadata in enumerate(ranks) for address, size in metadata["owned"]]
    combine(directory / "package", [directory / f"rank-{rank}.json" for rank in range(4)],
            peers, publisher=publisher())
    (directory / "collective-expected.json").write_text(json.dumps(dict(ranks=ranks, startup_tuning=startup_tuning,
        preparation_processes=[_process(), *(rank["process"] for rank in ranks)])))
    total = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    assert total < 128 << 20, f"collective fixture exceeded 128 MiB: {total}"
    print(f"exported TP4 collective package; all four preparation ranks exited; {total} fixture bytes", flush=True)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]).resolve(), startup_tuning="--startup-tuning" in sys.argv[2:])
