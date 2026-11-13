# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native rank fanout, b12x agreement, and coordinated completion without Python peer steps."""

import os
import socket
import time
from datetime import timedelta

import pytest
import torch

from tests.components.completion_support import completion_outputs
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires four assigned GPUs and FLARE_RUN_TP_CUDA_TESTS=1")


class RankForward:
    def __init__(self, batch, agreement, completion, rank):
        self.batch, self.agreement, self.completion = batch, agreement, completion
        self.compilation_variant = "sample" if rank == 0 else "consume"

    def __call__(self):
        self.batch.begin(self.agreement(self.batch.validate()))
        self.batch.finish()
        self.completion()


def _rank(rank, world, rendezvous, sockets, fields, compilation):
    from ignition.components.collectives import PreparedAllReduce
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.sampling import (
        AdmissionBuffers,
        DeviceRequestState,
        PreparedAdmission,
        PreparedCompletion,
        PreparedSampling,
        RequestAdmission,
    )
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_reduce_runtime,
    )
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.batch import BatchInputs, PreparedBatch
    from ignition.recipes.glm53.state import DecoderMetadata

    peers = []
    for index, (root, peer) in enumerate(sockets, start=1):
        if rank == 0:
            peers.append(root)
            peer.close()
        else:
            root.close()
            if rank == index:
                connection = peer
            else:
                peer.close()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=2))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    runtime = create_all_reduce_runtime(group, device, max_tokens=1, hidden_size=8)
    worker, graphs = None, {}
    try:
        requests = DeviceRequestState.allocate(device, vocab_size=154880)
        inputs = BatchInputs.allocate(128, 32, device)
        admission = AdmissionBuffers.allocate(device, vocab_size=requests.vocab_size)
        records = torch.zeros((2, 4, fields), dtype=torch.int64, device=device)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        init = PreparedAdmission(requests, admission, completion_outputs(records), frame, sequence, capacity=1)
        init.prepare()
        operations, owners = {"init": init}, [init]
        signatures = {"init": GraphSignature("init", 1, 1, tuple(admission.__dataclass_fields__))}
        with RecordedPreparationSession(device=device, checkpoint="native-rank-coordination",
                workload_schema="rank-request-tp-v1", numerical_policy="default", coordinator=coordinator) as session:
            for mode, capacity in (("prefill", 128), ("decode", 1)):
                metadata = DecoderMetadata.allocate(capacity, 1, 32, device)
                batch = PreparedBatch(inputs, requests, metadata, mode=mode,
                    shard=TensorParallelShard(rank, world), pages=64)
                agreement = PreparedAllReduce(batch.validation_input, batch.validation_sum,
                    runtime=runtime, coordinator=coordinator, session=session, name=f"{mode}.validation")
                arguments = (requests, inputs.row_slot_ids[:1], inputs.row_generations[:1],
                             inputs.live_rows, completion_outputs(records), frame, sequence)
                if rank == 0:
                    logits = torch.full((1, requests.vocab_size), -torch.inf, device=device, dtype=torch.bfloat16)
                    logits[0, 11:14] = 0
                    completion = PreparedSampling(logits, *arguments, sample_enabled=inputs.sample_enabled[:1])
                else:
                    completion = PreparedCompletion(*arguments)
                completion.prepare()
                forward = RankForward(batch, agreement, completion, rank)
                operations[mode] = forward
                owners.extend((batch, metadata, agreement, completion))
                signatures[mode] = GraphSignature(mode, 1, capacity, tuple(inputs.buffers()))
            session.tune()
            stream = torch.cuda.Stream(device=device)
            if compilation == "parallel":
                from ignition.preparation.capture import capture_forwards

                import ignition.preparation.aot as aot

                def unexpected(*args, **kwargs):
                    raise AssertionError("serving ranks must not trace or compile")

                aot.aot_forward = aot.make_fx = torch.compile = unexpected
                captured, compiled = capture_forwards(operations, signatures, session=session,
                    device=device, stream=stream, pool=torch.cuda.graph_pool_handle(), collective=runtime,
                    coordinator=coordinator)
                graphs.update(captured)
                operations.update(compiled)
            else:
                for name, operation in operations.items():
                    if compilation == "aot":
                        import io
                        from functools import partial
                        from ignition.preparation.aot import aot_forward

                        artifact = aot_forward(operation)
                        artifact(operation)
                        restored = torch.compiler.load_compiled_function(
                            io.BytesIO(type(artifact).serialize(artifact).serialized_data))
                        call = partial(restored, operation)
                    else:
                        call = compile_forward(operation)
                    operations[name] = call
                    call()
                    torch.cuda.synchronize(device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    graph = torch.cuda.CUDAGraph()
                    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                        call()
                    graphs[name] = graph
            session.freeze()
            buffers = inputs.buffers()
            buffers.update({name: getattr(admission, name) for name in admission.__dataclass_fields__})
            buffers.update(completion_records=records, frame_index=frame, submission_sequence=sequence)
            prepared = PreparedExecution(device=rank, graphs=graphs, buffers=buffers, signatures=signatures,
                resources=(requests, inputs, admission, operations, owners, session, runtime, stream))
            torch.cuda.synchronize(device)
            dist.barrier(group=group)
            before = torch.cuda.memory_stats(device)
            if rank:
                prepared.serve_rank(connection, rank=rank, world=world)
            else:
                worker = prepared.start_executor(peers=peers)
                with pytest.raises(RuntimeError, match="upload is not bound"):
                    worker.submit(prepared.graph_indices["prefill"], [("unknown", 0, b"12345678")])

                def replay(name, fields):
                    uploads = [(name, 0, value.numpy().tobytes()) for name, value in fields.items()]
                    ticket = worker.submit(prepared.graph_indices[name], uploads)
                    assert ticket is not None
                    assert worker.submit(prepared.graph_indices[name], uploads) is None
                    deadline = time.monotonic() + 30
                    result = None
                    while result is None and time.monotonic() < deadline:
                        result = worker.poll()
                        if result is None:
                            time.sleep(.001)
                    assert result is not None and result.fence == ticket
                    return result.records[2]

                host_admission = AdmissionBuffers.allocate("cpu", vocab_size=requests.vocab_size)
                host_admission.stage(0, 2, RequestAdmission(7, 13, max_new_tokens=4), list(range(65)))
                host_admission.admission_count.fill_(1)
                record = replay("init", {name: getattr(host_admission, name) for name in host_admission.__dataclass_fields__})
                assert record[:8] == [7, 1, 0, 0, -1, 1, 0, 0]
                host = BatchInputs.allocate(128, 32, "cpu")
                host.row_slot_ids[0], host.row_generations[0] = 2, 7
                host.live_rows.fill_(1)
                host.query_lengths[0] = 65
                host.token_ids[:65].copy_(torch.arange(65))
                host.page_tables[2].copy_(torch.arange(32) + 7)
                host.sample_enabled[0] = 1
                for step in range(4):
                    record = replay("prefill" if step == 0 else "decode", host.buffers())
                    assert record[:4] == [7, step + 2, 65 + step, step + 1]
                    assert record[4] in (11, 12, 13)
                    assert record[5:8] == [2 if step == 3 else 1, 0, 0]
                    host.query_lengths[0] = 1
                    host.token_ids[0] = record[4]
                worker.close()
                assert worker.drained
                worker = None
            after = torch.cuda.memory_stats(device)
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key], key
            assert requests.consumed_tokens[2] == 68
            assert requests.generated_tokens[2] == (4 if rank == 0 else 0)
            assert requests.rng_counter[2] == (4 if rank == 0 else 0)
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()
        runtime.close()
        dist.destroy_process_group()
        for pair in sockets:
            for connection in pair:
                connection.close()


@pytest.mark.parametrize("world,fields", [(2, 32), (4, 32)])
@pytest.mark.parametrize("compilation", ["lowered", "aot", "parallel"])
def test_native_rank_coordination(world, fields, compilation, tmp_path, monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 4
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    sockets = [socket.socketpair() for _ in range(world - 1)]
    processes = mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), sockets, fields, compilation),
                         nprocs=world, join=False)
    for pair in sockets:
        for connection in pair:
            connection.close()
    while not processes.join(timeout=1):
        pass


def _startup_rank(rank, world, rendezvous, sockets, fail_peer):
    from ignition.components.collectives import PreparedAllReduce
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.execution import GraphSignature, PreparedExecution, TunedBinding, TuningGraph
    from ignition.preparation.tuning import RecordedPreparationSession

    peers = []
    for index, (root, peer) in enumerate(sockets, start=1):
        if rank == 0:
            peers.append(root)
            peer.close()
        else:
            root.close()
            if rank == index:
                connection = peer
            else:
                peer.close()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=2))
    coordinator = CollectivePreparation(dist.group.WORLD)
    runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=1, hidden_size=8)
    worker, captured = None, []
    try:
        source = torch.full((1, 8), rank + 1, dtype=torch.bfloat16, device=device)
        reduced = torch.empty_like(source)
        buffers = {name: torch.zeros(1, dtype=torch.int32, device=device)
                   for name in ("selector", "startup_count", "startup_status", "startup_failure",
                                "request_value", "first_output", "second_output", "replay_count", "output_sum")}
        if fail_peer and rank == 1:
            buffers["startup_failure"].fill_(17)
        buffers.update(frame_index=torch.zeros(1, dtype=torch.int32, device=device),
                       submission_sequence=torch.zeros(1, dtype=torch.int64, device=device),
                       completion_records=torch.zeros((2, 1, 32), dtype=torch.int64, device=device))
        with RecordedPreparationSession(device=device, checkpoint="process-startup-tuning",
                workload_schema="shared-selector", numerical_policy="exact-integer-sum",
                coordinator=coordinator) as session:
            reduction = PreparedAllReduce(source, reduced, runtime=runtime, session=session,
                coordinator=coordinator, name="startup-sum")
            session.tune(show_progress=False)
            source.fill_(rank + 1)

            def produce():
                reduction(source)
                buffers["selector"].copy_(reduced.flatten()[:1].to(torch.int32))
                buffers["startup_count"].add_(1)
                buffers["startup_status"].copy_(buffers["startup_failure"])

            def consume(output):
                output.copy_(buffers["selector"] * 11 + buffers["request_value"])
                buffers["output_sum"].add_(output)
                buffers["replay_count"].add_(1)

            producer = torch.compile(produce, fullgraph=True)
            consumer = torch.compile(consume, fullgraph=True)
            producer()
            for name in ("first", "second"):
                consumer(buffers[f"{name}_output"])
            torch.cuda.synchronize(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            pool = torch.cuda.graph_pool_handle()
            dist.barrier()
            startup = torch.cuda.CUDAGraph()
            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(startup, stream=stream, pool=pool):
                producer()
            captured.append(startup)
            graphs = {}
            for name in ("first", "second"):
                graph = torch.cuda.CUDAGraph()
                with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream, pool=pool):
                    consumer(buffers[f"{name}_output"])
                captured.append(graph)
                graphs[name] = graph
            session.freeze()
            prepared = PreparedExecution(device=rank, graphs=graphs, buffers=buffers, max_active=1,
                signatures={name: GraphSignature("decode", 1, 1, ("request_value",)) for name in graphs},
                tuning_graphs={"choose-selector": TuningGraph(graph=startup,
                    outputs=("selector", "startup_count"), status_buffer="startup_status")},
                tuned_bindings={name: (TunedBinding(buffer="selector", tuning_graph="choose-selector"),)
                                for name in graphs},
                resources=(runtime, session, reduction, source, reduced, producer, consumer, stream))
            for name in ("selector", "startup_count", "startup_status", "first_output", "second_output",
                         "replay_count", "output_sum"):
                buffers[name].zero_()
            torch.cuda.synchronize(device)
            dist.barrier()
            before = torch.cuda.memory_stats(device)
            expected_selector = world * (world + 1) // 2
            if rank:
                prepared.serve_rank(connection, rank=rank, world=world)
            elif fail_peer:
                with pytest.raises(RuntimeError, match="status 17"):
                    prepared.start_executor(peers=peers)
            else:
                worker = prepared.start_executor(peers=peers)
                assert buffers["selector"].item() == expected_selector
                assert buffers["startup_count"].item() == 1
                for name, value in (("first", 2), ("second", 3), ("first", 5), ("second", 9)):
                    payload = torch.tensor([value], dtype=torch.int32).numpy().tobytes()
                    ticket = worker.submit(prepared.graph_indices[name], [("request_value", 0, payload)])
                    assert ticket is not None
                    deadline = time.monotonic() + 30
                    completion = None
                    while completion is None and time.monotonic() < deadline:
                        completion = worker.poll()
                        if completion is None:
                            time.sleep(.001)
                    assert completion is not None and completion.fence == ticket
                    assert buffers[f"{name}_output"].item() == expected_selector * 11 + value
                    assert buffers["startup_count"].item() == 1
                worker.close()
                assert worker.drained
                worker = None
            after = torch.cuda.memory_stats(device)
            for key in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key], key
            assert buffers["selector"].item() == expected_selector
            assert buffers["startup_count"].item() == 1
            assert buffers["startup_status"].item() == (17 if fail_peer and rank == 1 else 0)
            assert buffers["replay_count"].item() == (0 if fail_peer else 4)
            assert buffers["first_output"].item() == (0 if fail_peer else expected_selector * 11 + 5)
            assert buffers["second_output"].item() == (0 if fail_peer else expected_selector * 11 + 9)
            assert buffers["output_sum"].item() == (0 if fail_peer else expected_selector * 44 + 19)
    finally:
        if worker is not None:
            worker.close()
        for graph in captured:
            graph.reset()
        runtime.close()
        dist.destroy_process_group()
        for pair in sockets:
            for connection in pair:
                connection.close()


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("fail_peer", (False, True), ids=("shared-producer", "peer-status-failure"))
def test_native_process_startup_tuning(world, fail_peer, tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 4
    sockets = [socket.socketpair() for _ in range(world - 1)]
    processes = mp.spawn(_startup_rank,
        args=(world, (tmp_path / "rendezvous").as_uri(), sockets, fail_peer), nprocs=world, join=False)
    for pair in sockets:
        for connection in pair:
            connection.close()
    while not processes.join(timeout=1):
        pass
