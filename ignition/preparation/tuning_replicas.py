# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local tuning replicas with independent serving and candidate-sharding groups."""

from dataclasses import dataclass, fields, replace
from datetime import timedelta
import importlib
import multiprocessing
import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class TuningReplica:
    rendezvous: str
    index: int
    replicas: int
    tensor_parallel: int

    @property
    def processes(self):
        return self.replicas * self.tensor_parallel


class TuningComplete(BaseException):
    """A preparation-only replica finished tuning and may release its model."""


def authorization(states, *, replica, rank, tensor_parallel, ranks):
    """Release replica-local collectives and corresponding-rank candidate exchanges."""
    peers = states[rank::tensor_parallel]
    if all(item[3] is not None for item in peers):
        return {"cache": tuple(item[3] for item in peers)}
    if all(item[4] for item in peers):
        from .collectives import _tuning_results

        keys = [sorted(value.key for value in item[4]) for item in peers]
        if any(value != keys[0] for value in keys[1:]):
            raise RuntimeError("corresponding tuning ranks declared different candidate races")
        results = _tuning_results(value for item in peers for value in item[4])
        return {"tuning": results[0] if len(results) == 1 else results}
    if any(item[0] for item in peers) and any(item[3] is not None or item[4] for item in peers):
        raise RuntimeError("a tuning replica finished before its corresponding rank")
    local = states[replica * tensor_parallel:(replica + 1) * tensor_parallel]
    if all(item[5] is not None for item in local):
        if any(item[5].ranks != tuple(ranks) for item in local):
            raise ValueError("collective tuning must span its serving TP group")
        return {"collective_tuning": tuple(item[5] for item in local)}
    ready = [item[1] for item in local]
    if all(ready):
        if any(item != ready[0] for item in ready[1:]):
            raise RuntimeError("collective declarations differ within a tuning replica")
        key, members = ready[0][0]
        if tuple(members) != tuple(ranks):
            raise ValueError("a tuning replica collective must span its serving TP group")
        return {"collective_key": key}
    if any(item[0] for item in local) and (any(ready) or any(item[5] is not None for item in local)):
        raise RuntimeError("a tuning rank finished before its replica's collective")
    return {"collective_key": None}


class ReplicaPreparation:
    def __init__(self, role, rank, *, ranks):
        import torch.distributed as dist

        self.role, self.rank, self.ranks = role, rank, tuple(ranks)
        self.store = dist.FileStore(role.rendezvous, role.processes)
        # This group is independent of torch.distributed's default TP group.
        self.group = dist.ProcessGroupGloo(self.store, role.index * role.tensor_parallel + rank,
                                          role.processes, timedelta(minutes=5))

    def gather(self, value):
        import torch

        payload = pickle.dumps(value, protocol=5)
        size = torch.tensor([len(payload)], dtype=torch.int64)
        lengths = [torch.empty_like(size) for _ in range(self.role.processes)]
        self.group.allgather([lengths], [size]).wait()
        buffer = torch.zeros(max(item.item() for item in lengths), dtype=torch.uint8)
        buffer[:len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        buffers = [torch.empty_like(buffer) for _ in lengths]
        self.group.allgather([buffers], [buffer]).wait()
        return [pickle.loads(bytes(data[:length.item()].tolist()))
                for data, length in zip(buffers, lengths, strict=True)]

    def prepare(self, session, requests, *, progress, autotune):
        job = state = None
        error = None
        allowed = {"collective_key": None}
        try:
            while True:
                try:
                    if job is None:
                        job = session.begin(requests, autotune=autotune)
                    if state is None or not state.done:
                        state = job.advance(**allowed)
                        if progress is not None:
                            progress(state)
                except Exception as failure:
                    error = failure
                pending = () if state is None else tuple((item.key, item.ranks) for item in state.ready_collectives)
                states = self.gather((state is not None and state.done, pending,
                    None if error is None else f"{type(error).__name__}: {error}",
                    None if state is None else state.ready_cache,
                    () if state is None else tuple(state.ready_tuning),
                    None if state is None else state.ready_collective_tuning))
                if failures := [(index, item[2]) for index, item in enumerate(states) if item[2] is not None]:
                    raise RuntimeError(f"local tuning preparation failed: {failures}") from error
                if all(item[0] for item in states):
                    return job.result()
                allowed = authorization(states, replica=self.role.index, rank=self.rank,
                                        tensor_parallel=self.role.tensor_parallel, ranks=self.ranks)
        except BaseException as failure:
            if job is not None:
                try:
                    if state is not None and state.done:
                        job.result().close()
                    else:
                        job.close()
                except BaseException as cleanup:
                    failure.add_note(f"tuning cleanup failed: {cleanup!r}")
            raise


def _replica(parent, prepare, snapshot, values, gpu_uuids, options):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_uuids)
    from ignition import runtime_patches

    runtime_patches.apply()
    import flare_heliopause

    flare_heliopause.bind_parent_lifetime(parent)
    import torch
    from ignition.planning import Plan
    from .core_storage import CoreStorage

    recipe = importlib.import_module(prepare.__module__).RECIPE
    plan = Plan(recipe=recipe, **values)
    torch.cuda.set_device(0)
    with CoreStorage(world=plan.world_size) as storage:
        try:
            prepare(snapshot, plan, compile_forwards=False, retain_graphs=False, progress=None,
                    core_storage=storage, **options)
        except TuningComplete:
            torch.cuda.synchronize()
        else:
            raise RuntimeError("a tuning replica continued beyond kernel preparation")


def prepare_recipe(prepare, snapshot, plan, **options):
    """Prepare serving ranks while optional replicas contribute kernel candidates."""
    if not plan.tuning_gpus:
        return prepare(snapshot, plan, **options)
    if options.get("ranks") is not None:
        raise ValueError("local tuning replicas cannot replace multi-host preparation")
    replicas = len(plan.tuning_gpus) // plan.world_size
    processes = []
    context = multiprocessing.get_context("spawn")
    with TemporaryDirectory(prefix="flare-tuning-") as directory:
        role = TuningReplica(str(Path(directory) / "gloo"), 0, replicas, plan.world_size)
        root = replace(plan, tuning_replica=role)
        values = {field.name: getattr(root, field.name) for field in fields(root) if field.name != "recipe"}
        values["checkpoint"] = replace(plan.checkpoint,
            scale_compressed_device_bytes=dict(plan.checkpoint.scale_compressed_device_bytes))
        try:
            for replica in range(1, replicas):
                gpus = plan.tuning_gpus[replica * plan.world_size:(replica + 1) * plan.world_size]
                peer = dict(values, gpus=gpus, tuning_replica=replace(role, index=replica))
                worker = context.Process(target=_replica,
                    args=(os.getpid(), prepare, snapshot, peer, tuple(gpu.uuid for gpu in gpus),
                          {"kernel_cache": options.get("kernel_cache")}), name=f"flare-tuning-{replica}")
                worker.start()
                processes.append(worker)
            prepared = prepare(snapshot, root, **options)
            for worker in processes:
                worker.join(timeout=60)
                if worker.exitcode != 0:
                    prepared.close()
                    raise RuntimeError(f"tuning replica {worker.name} exited with {worker.exitcode}")
            return prepared
        finally:
            for worker in processes:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
