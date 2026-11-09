# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured collective consumers and their prepared startup selector programs."""

from contextlib import contextmanager
from contextvars import ContextVar

from .execution import TunedBinding, TuningGraph

_CAPTURE = ContextVar("startup_tuning_capture", default=None)


def replay_startup_tuning(model, stream):
    """Initialize diagnostic graph bindings before requests on every rank."""
    import torch
    import os
    import time

    manifest = model.manifest()
    referenced = {binding["tuning_graph"] for graph in manifest["graphs"]
                  for binding in graph.get("tuned_bindings", ())}
    names = tuple(graph["name"] for graph in manifest.get("tuning_graphs", ())
                  if graph["name"] in referenced)
    buffers = {buffer["name"]: buffer for buffer in manifest["buffers"]}
    storage = {field for graph in manifest.get("tuning_graphs", ()) if graph["name"] in referenced
               for field in (*graph["outputs"], graph["status_buffer"])}
    signature = (
        tuple((graph["name"], tuple(graph["outputs"]), graph["status_buffer"])
              for graph in manifest.get("tuning_graphs", ()) if graph["name"] in referenced),
        tuple((graph["name"], tuple((binding["buffer"], binding["tuning_graph"])
                                   for binding in graph.get("tuned_bindings", ())))
              for graph in manifest["graphs"]),
        tuple((name, buffers[name]["dtype"], buffers[name]["nbytes"],
               tuple(buffers[name]["shape"]), tuple(buffers[name]["strides"])) for name in sorted(storage)),
    )
    coordinator = getattr(model.session, "_coordinator", None)
    if coordinator is not None and not coordinator.same_on_every_rank(signature):
        raise RuntimeError("diagnostic startup tuning graph order or binding schema differs across ranks")
    for name in names:
        tuner = model.tuning_graphs[name]
        status = model.buffers[tuner.status_buffer]
        error = None
        try:
            with torch.cuda.stream(stream):
                status.fill_(-1)
                tuner.graph.replay()
        except BaseException as failure:
            error = failure
        deadline = time.monotonic() + 120
        try:
            while not stream.query():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"diagnostic startup tuning graph {name!r} did not drain")
                time.sleep(.001)
        except BaseException:
            # A rank may still access its peers' storage; process teardown must reclaim it.
            os.abort()
        if error is not None:
            # A partial launch can leave a peer waiting inside its graph.
            os.abort()
        value = int(status.item())
        agreed = coordinator is None or coordinator.same_on_every_rank(value)
        if value != 0 or not agreed:
            raise RuntimeError(f"diagnostic startup tuning graph {name!r} failed: local status {value}")


def record_consumer(owner):
    capture = _CAPTURE.get()
    if capture is not None:
        capture.add(id(owner))


@contextmanager
def capture_consumers(session, graph):
    consumers = set()
    token = _CAPTURE.set(consumers)
    try:
        yield
    finally:
        _CAPTURE.reset(token)
    registry = getattr(session, "_startup_collectives", None)
    if registry is not None:
        registry.captures[id(graph)] = consumers


def resident_numeric_storage(numeric, *, device, minimum_bytes):
    """The largest live checkpoint storage on the device that can displace L2."""
    import torch

    best = None
    for reference, address, nbytes, _ in numeric.entries.values():
        if nbytes < minimum_bytes:
            continue
        storage = torch.UntypedStorage._new_with_weak_ptr(reference.cdata)
        if (storage is None or storage.device.type != "cuda" or storage.device != device
                or (storage.data_ptr(), storage.nbytes()) != (address, nbytes)):
            continue
        if best is None or nbytes > best.nbytes():
            best = storage
    if best is None:
        return None
    return torch.empty(0, dtype=torch.uint8, device=best.device).set_(best, 0, (best.nbytes(),))


class CollectiveStartupTuning:
    def __init__(self, session):
        self.session = session
        self.owners = []
        self.captures = {}
        self.groups = {}
        self.owner_groups = {}
        self.group_selectors = {}
        self.startup_storage = None

    def prepare(self):
        from b12x.comm.pcie import StartupTuningStorage, joint_startup_spec, prepare_joint_startup_tuner

        groups, owner_groups = {}, {}
        for owner in self.owners:
            spec = joint_startup_spec(owner.plan, profiles=owner.runtime_profiles)
            key = (spec.key, id(spec.state.runtime))
            groups.setdefault(key, []).append(spec)
            owner_groups[id(owner)] = key
        selectors = {key: tuple(spec.selector.data_ptr() for spec in specs) for key, specs in groups.items()}
        policies = {owner_groups[id(owner)]: owner.runtime_cache_policy for owner in self.owners}
        if groups and self.startup_storage is None:
            self.startup_storage = StartupTuningStorage(next(iter(groups.values()))[0].selector.device)
        self.groups = {key: self.groups[key] if self.group_selectors.get(key) == selectors[key]
                       else prepare_joint_startup_tuner(specs, cache_policy=policies[key],
                                                       storage=self.startup_storage)
                       for key, specs in groups.items()}
        self.group_selectors = selectors
        self.owner_groups = owner_groups

    def memory_report(self):
        from b12x.preparation import MemoryRequirements

        memory = MemoryRequirements.sequential(producer.memory_requirements() for producer in self.groups.values())
        return {"startup_tuning_bytes": sum(entry.resident_nbytes for entry in memory.persistent),
                "startup_tuning_producers": len(self.groups)}

    def catalog(self, graphs):
        import torch

        consumers = {name: self.captures.get(id(graph), set()) for name, graph in graphs.items()}
        needed = {self.owner_groups[owner] for owners in consumers.values() for owner in owners
                  if owner in self.owner_groups}
        storage = self.session.benchmark_storage
        if needed and storage is None:
            raise RuntimeError("collective startup capture requires core storage")
        buffers, tuners, bindings, resources = {}, {}, {}, []
        if any(self.groups[key].cache_policy == "cold" for key in needed):
            # Cold races only read their L2 eviction input; resident weights serve.
            source = resident_numeric_storage(storage.numeric_storage,
                device=self.startup_storage.device, minimum_bytes=self.startup_storage.eviction_bytes)
            if source is not None:
                self.startup_storage.use_eviction(source)
        for index, (key, producer) in enumerate(self.groups.items()):
            if key not in needed:
                continue
            name = f"collective-tuning-{index}"
            outputs, addresses = [], {}
            for spec in producer.specs:
                pointer = spec.selector.data_ptr()
                if pointer not in addresses:
                    field = f"{name}-selector-{len(outputs)}"
                    addresses[pointer] = field
                    outputs.append(field)
                    buffers[field] = spec.selector.view(torch.int32)
            status = f"{name}-status"
            buffers[status] = producer.status
            device = self.session.device.ordinal
            coordinator = getattr(self.session, "_coordinator", None)
            if coordinator is not None:
                coordinator.barrier(f"capture {name}", coordinator.ranks)
            with torch.cuda.device(device):
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with self.session.capture(), torch.cuda.graph(graph, pool=storage.graph_pool.id, stream=stream):
                    producer.run()
                graph.instantiate()
                stream.synchronize()
            tuners[name] = TuningGraph(graph, tuple(outputs), status)
            resources.append(producer)
            for serving, owners in consumers.items():
                selected = {id(owner): owner for owner in self.owners if id(owner) in owners
                            and self.owner_groups[id(owner)] == key}
                fields = {addresses[owner.plan.prepared.state.selector.data_ptr()] for owner in selected.values()}
                bindings.setdefault(serving, []).extend(TunedBinding(field, name) for field in sorted(fields))
        torch.cuda.synchronize(self.session.device.ordinal)
        return buffers, tuners, bindings, tuple(resources)


def register_collective(owner, session):
    from .tuning import RecordedPreparationSession, register_finalizer

    registry = getattr(session, "_startup_collectives", None)
    if registry is None:
        registry = session._startup_collectives = CollectiveStartupTuning(session)
        registry.owners.append(owner)
        register_finalizer(session, registry.prepare)
    else:
        registry.owners.append(owner)
        if not isinstance(session, RecordedPreparationSession):
            registry.prepare()
