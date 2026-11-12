# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap coordination of b12x collectives over an established Gloo group."""

import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def host_identity():
    """This boot of this host; ranks sharing it can map each other's memory through CUDA IPC."""
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def create_roce_runtime(group, device, *, max_tokens, pipe_tokens, hidden_size, gather_bytes, core_storage=None,
                        varlen_bytes=0, ring=False, ring_open=False):
    """Allocate RoCEnante for ranks on separate DGX Sparks: BF16 hidden reductions and logit gathers.

    One runtime serves every reduction and gather in rank-serial order. Reductions of up
    to `max_tokens` rows, and at least the two-shot threshold, run one-shot: every rank
    sends its whole payload to every peer, which is fastest for decode rows. Longer
    reductions, up to `pipe_tokens` rows, run as a pipelined two-shot reduce-scatter and
    all-gather, which moves a ring's bytes for prefill chunks.

    With `core_storage`, the pinned region is core host storage at a recorded address and
    the runtime is registered for export, so a native rank can rebuild the transport over it.
    `varlen_bytes` sizes each rank's contribution to variable-length gathers; zero prepares none.
    """
    from b12x.comm import roce

    if any(type(value) is not int or value <= 0 for value in (max_tokens, pipe_tokens, hidden_size, gather_bytes)):
        raise ValueError("RoCE capacities must be positive integers")
    # Below this size the runtime always reduces one-shot (b12x's two-shot threshold is 1 or 2 MiB).
    nbytes = max(max_tokens * hidden_size * 2, 2 << 20)
    pipe_bytes = pipe_tokens * hidden_size * 2
    if nbytes % 16 or pipe_bytes % 16 or gather_bytes % 16:
        raise ValueError("RoCE capacities must occupy complete 16-byte packs")
    import torch

    region = None if core_storage is None else (
        lambda size: core_storage.host_allocate("roce", (size,), torch.uint8).host_view)
    if type(varlen_bytes) is not int or varlen_bytes < 0 or varlen_bytes % 16:
        raise ValueError("variable-length gather capacity must be a nonnegative multiple of 16 bytes")
    runtime = roce.AllReduce(exchange_group=group, device=device, max_size=nbytes, max_gather_bytes=gather_bytes,
                             pipe_bytes=pipe_bytes, region_allocator=region,
                             ring=ring, **({"ring_open": True} if ring_open else {}),
                             **({"max_varlen_bytes": varlen_bytes} if varlen_bytes else {}))
    if core_storage is not None:
        core_storage.transports.append(runtime)
    return runtime

def create_all_reduce_runtime(group, device, *, max_tokens, hidden_size, ipc=None, element_bytes=2,
                              collective_tuning="static"):
    """Allocate one b12x IPC channel for the largest hidden activation.

    Every rank supplies identical capacities before preparing graph variants.
    The bootstrap retains the runtime until every native rank worker drains.
    """
    from b12x.comm.pcie import AllReduce

    if any(type(value) is not int or value <= 0 for value in (max_tokens, hidden_size)):
        raise ValueError("all-reduce capacities must be positive integers")
    if element_bytes not in (2, 4):
        raise ValueError("all-reduce storage must use two- or four-byte elements")
    if collective_tuning not in ("static", "runtime"):
        raise ValueError("collective_tuning must be static or runtime")
    nbytes = max_tokens * hidden_size * element_bytes
    if nbytes % 16:
        raise ValueError("all-reduce capacity must occupy complete 16-byte packs")
    return AllReduce.from_exchange_group(exchange_group=group, device=device, single_channel=True,
                                         max_size=nbytes, eager_buffer_bytes=nbytes, ipc=ipc,
                                         joint=collective_tuning == "runtime")


def create_all_gather_runtime(group, device, *, max_rows, columns, ipc=None):
    """Allocate b12x staging for serial BF16 column gathers in prepared graphs."""
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    if (world_size < 2 or type(max_rows) is not int or max_rows <= 0
            or type(columns) is not int or columns <= 0 or columns % (world_size * 64)):
        raise ValueError("column gathers require positive rows and 64-column-aligned rank partitions")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device,
        max_batch_size=max_rows, total_heads=columns // 64, head_dim=64, stream_affine=False, ipc=ipc)


def create_result_broadcast_runtime(group, device, *, layouts, ipc=None):
    """Allocate one local PCIe channel covering every declared result payload.

    Layouts describe captured operation fields, independently of request-slot storage.
    RoCE callers retain their existing variable-payload transport.
    """
    from ignition.components.collectives import packed_broadcast_layout

    sizes = [packed_broadcast_layout(layout.values())[1] for layout in layouts]
    if not sizes or min(sizes) <= 0:
        raise ValueError("result broadcasts require nonempty payload layouts")
    # Gather warps copy 512-byte segments. Use one segment per rank per row
    # so short results transfer fewer rows while large results keep every
    # copying lane active. All payload sizes share the same channel storage.
    max_rows = (max(sizes) + 511) // 512
    return create_all_gather_runtime(group, device, max_rows=max_rows,
        columns=256 * dist.get_world_size(group), ipc=ipc)


def create_dma_all_reduce_runtime(group, device, *, max_tokens, hidden_size, ipc=None, element_bytes=2):
    """Allocate a b12x copy-engine ring for prefill reductions in caller-owned shared storage."""
    from b12x.comm.pcie import DmaAllReduce

    if any(type(value) is not int or value <= 0 for value in (max_tokens, hidden_size)):
        raise ValueError("all-reduce capacities must be positive integers")
    if element_bytes not in (2, 4):
        raise ValueError("all-reduce storage must use two- or four-byte elements")
    return DmaAllReduce(exchange_group=group, device=device, max_bytes=max_tokens * hidden_size * element_bytes, ipc=ipc)


def create_head_exchange(group, device, *, max_rows, heads, head_dim, query_head_dim=None, ipc=None):
    """Allocate b12x staging for all-gathering head shards and LSE-combining partial attention.

    Queries may be wider than the combined values: latent attention gathers absorbed queries of
    `query_head_dim` and combines `head_dim` outputs.
    """
    from b12x.comm.pcie import DcpAllToAll

    world_size = dist.get_world_size(group)
    query_head_dim = head_dim if query_head_dim is None else query_head_dim
    if world_size < 2 or max_rows <= 0 or heads % world_size or head_dim % 8 or query_head_dim % 8:
        raise ValueError("head exchanges require multiple equal head shards and positive capacity")
    return DcpAllToAll.from_exchange_group(exchange_group=group, device=device, max_batch_size=max_rows,
        total_heads=heads, head_dim=head_dim, query_head_dim=query_head_dim, stream_affine=False, ipc=ipc)


def _tuning_results(contributions):
    """Forward finalist shards in rank order and select ordinary race winners."""
    races = {}
    for item in contributions:
        races.setdefault(item.key, []).append(item)
    results = []
    for key in sorted(races):
        members = races[key]
        policy = getattr(members[0], "finalist_policy", None)
        if any(getattr(item, "finalist_policy", None) != policy for item in members):
            raise ValueError("sharded finalist policies differ across ranks")
        if policy is not None:
            results.extend(members)
        else:
            completed = [item for item in members if item.assignment is not None]
            if not completed:
                raise RuntimeError(f"no rank completed a candidate of tuning race {key!r}")
            results.append(min(completed, key=lambda item: (item.latency_us, item.candidate_index)))
    return tuple(results)


def _selected_plan_catalog(requests):
    from b12x.preparation.session import _declaration_key
    from b12x.preparation.types import _CompositePlan

    def declaration(plan):
        if isinstance(plan, _CompositePlan):
            return (plan.component_id, plan.composite_semantic_version, plan.capacity_metadata,
                    tuple((count, declaration(child)) for count, child in plan.variants.items()))
        return _declaration_key(plan)[:-1]

    by_name = {request.name: request for request in requests}
    identities = {}

    def identity(name):
        if name not in identities:
            request = by_name[name]
            identities[name] = (declaration(request.plan),
                                tuple(identity(dependency) for dependency in request.dependencies))
        return identities[name]

    entries = []

    def selected(plan, key):
        if isinstance(plan, _CompositePlan):
            for count, child in plan.variants.items():
                selected(child, (key, count))
        else:
            config = plan.selection.config
            entries.append((plan, key, plan.contract.config_payload(config), config))

    for request in requests:
        if request.collective is None:
            selected(request.plan, identity(request.name))
    return entries


class CollectivePreparation:
    """Authorize matching collective declarations and synchronize kernel entry.

    Local compilation may complete at different times on each rank. This
    coordinator releases a collective only after all group members report the
    same declaration. When the session shards tuning, it also relays b12x's
    two exchanges: every rank's cached selections, which b12x reconciles in
    rank order, and candidate results. Ordinary races use the fastest result;
    finalist races retain every shard for b12x to consolidate. It is unused
    after graph handoff.
    """

    def __init__(self, group, *, timeout=timedelta(minutes=2)):
        if dist.get_backend(group) != "gloo":
            raise ValueError("collective preparation requires a Gloo control group")
        self.group = group
        self.ranks = tuple(dist.get_process_group_ranks(group))
        self.control_groups = {self.ranks: group}
        self.operation_ranks = {}
        self.timeout = timeout
        hosts = [None] * len(self.ranks)
        dist.all_gather_object(hosts, host_identity(), group=group)
        # Each rank's host, in group order; ranks on one host share its CPUs and CUDA IPC.
        self.hosts = tuple(hosts)
        self.local_ranks = self.hosts.count(host_identity())

    def register_group(self, ranks, group):
        """Retain a control subgroup used by hybrid TP/DCP collective barriers."""
        ranks = tuple(ranks)
        if not ranks or any(rank not in self.ranks for rank in ranks) or ranks in self.control_groups:
            raise ValueError("collective subgroup must be a new nonempty subset of the control ranks")
        self.control_groups[ranks] = group

    def _global_rank(self):
        try:
            return dist.get_rank()
        except (RuntimeError, ValueError):
            return self.ranks[0]

    def _local_index(self):
        return self.ranks.index(self._global_rank())

    def _remember_operation_ranks(self, states):
        """Record each operation's control group for final per-group selection checks."""
        for state in states:
            if type(state[0]) is bool:
                for key, ranks in state[1]:
                    self.operation_ranks[key] = tuple(ranks)
                if state[5] is not None and getattr(state[5], "key", None) is not None:
                    self.operation_ranks[state[5].key] = tuple(state[5].ranks)
                for contribution in state[4]:
                    if getattr(contribution, "key", None) is not None and getattr(contribution, "ranks", None) is not None:
                        self.operation_ranks[contribution.key] = tuple(contribution.ranks)
            else:
                for key, ranks in state[0]:
                    self.operation_ranks[key] = tuple(ranks)
                if state[1] is not None and getattr(state[1], "key", None) is not None:
                    self.operation_ranks[state[1].key] = tuple(state[1].ranks)

    def _members(self, states, ranks):
        """Select only the local operation's control group from a TP-wide progress exchange."""
        ranks = tuple(ranks)
        if not ranks or self._global_rank() not in ranks or any(rank not in self.ranks for rank in ranks):
            raise ValueError("collective operation requires a subgroup containing the local rank")
        return [state for state, rank in zip(states, self.ranks, strict=True) if rank in ranks]

    @property
    def spans_hosts(self):
        return len(set(self.hosts)) > 1

    def same_on_every_rank(self, value):
        reference = [value if dist.get_rank() == self.ranks[0] else None]
        dist.broadcast_object_list(reference, src=self.ranks[0], group=self.group)
        matching = torch.tensor(int(value == reference[0]), dtype=torch.int32, device="cpu")
        dist.all_reduce(matching, op=dist.ReduceOp.MIN, group=self.group)
        return bool(matching.item())

    def share_selections(self, cache):
        """Adopt rank zero's cached kernel selections on every rank.

        Ranks on other hosts keep their own tuning caches, and independent races choose
        different kernels for about a quarter of GLM's requests. Compiled forwards bind rank
        zero's scratch layouts, so every rank must run the same selections.
        """
        records = [cache.records if dist.get_rank() == self.ranks[0] else None]
        dist.broadcast_object_list(records, src=self.ranks[0], group=self.group)
        if dist.get_rank() != self.ranks[0]:
            # The session's agreed records, which b12x's own cross-rank reconciliation sets, take
            # precedence over the rank's file. Saving a newly tuned key reloads that file, so
            # replacing only the loaded records would restore the rank's own choices.
            cache._agreed_records = dict(records[0])

    def agree_selections(self, selections):
        """Fail before compilation when ranks selected different kernels for the same request."""
        local = {item["name"]: (item["component"], json.dumps(item["config"], sort_keys=True))
                 for item in selections}
        gathered = [None] * len(self.ranks)
        dist.all_gather_object(gathered, local, group=self.group)
        operation_ranks = getattr(self, "operation_ranks", {})
        differing = set()
        for name in local:
            ranks = operation_ranks.get(name, self.ranks)
            positions = [index for index, rank in enumerate(self.ranks) if rank in ranks]
            if not positions:
                continue
            root = gathered[positions[0]]
            for index in positions[1:]:
                other = gathered[index]
                if name in root and name in other and root[name] != other[name]:
                    differing.add(name)
        differing = sorted(differing)
        if differing:
            raise RuntimeError(f"{len(differing)} kernel selections differ from rank zero's, such as "
                               f"{differing[:3]}; preparing again adopts rank zero's tuning decisions")

    def reconcile_selections(self, requests):
        """Choose one actual configuration for matching declarations across ranks.

        Unequal catalogs cannot shard candidate races. Their shared declarations
        still need matching scratch layouts before compiled forwards are shared.
        """
        entries = _selected_plan_catalog(requests)
        local = {}
        for _, key, payload, config in entries:
            local.setdefault(key, {}).setdefault(payload, config)
        gathered = [None] * len(self.ranks)
        dist.all_gather_object(gathered, local, group=self.group)
        chosen, participants = {}, {}
        for catalog in gathered:
            for key, values in catalog.items():
                chosen.setdefault(key, next(iter(values.items())))
                participants[key] = participants.get(key, 0) + 1
        common = {key for key, count in participants.items() if count > 1}
        changed = any(payload != chosen[key][0]
                      for catalog in gathered for key, values in catalog.items() if key in common
                      for payload in values)
        configs = {}
        for plan, key, payload, _ in entries:
            if key in common and payload != chosen[key][0]:
                config = chosen[key][1]
                if plan in configs and plan.contract.config_payload(configs[plan]) != chosen[key][0]:
                    raise ValueError("one plan has conflicting rank-shared configurations")
                configs[plan] = config
        return configs, changed

    def __call__(self, progress):
        if progress.ready_cache is not None or progress.ready_tuning:
            raise ValueError("collective preparation does not shard tuning candidates")
        pending = tuple((item.key, item.ranks) for item in progress.ready_collectives)
        local = (pending, progress.ready_collective_tuning)
        states = [None] * len(self.ranks)
        dist.all_gather_object(states, local, group=self.group)
        self._remember_operation_ranks(states)
        if not all(item[0] or item[1] is not None for item in states):
            return None
        if all(item[1] is not None for item in states):
            local = states[self._local_index()][1]
            members = self._members(states, local.ranks)
            values = tuple(item[1] for item in members)
            if any(getattr(item, "key", None) != getattr(local, "key", None) or item.ranks != local.ranks
                   for item in values):
                return None
            return values
        local = states[self._local_index()][0]
        ranks = local[0][1]
        states = [item[0] for item in self._members(states, ranks)]
        if any(state != local for state in states):
            raise RuntimeError(f"collective preparation declarations differ across ranks: {states}")
        key, _ = local[0]
        return key

    def prepare(self, session, requests, *, progress=None, autotune=None, selected_configs=None):
        """Drive a mixed local/collective job until every rank has finished.

        A finished rank must still exchange readiness while its peers compile
        or finish local work. Returning on local completion can strand peers in
        their next control-group exchange.
        """
        job = None
        state = None
        error = None
        authorization = {"collective_key": None}
        try:
            while True:
                try:
                    if job is None:
                        options = {} if selected_configs is None else {"selected_configs": selected_configs}
                        job = session.begin(requests, autotune=autotune, **options)
                    if state is None or not state.done:
                        state = job.advance(**authorization)
                        if progress is not None:
                            progress(state)
                except Exception as exc:  # noqa: BLE001 - Every rank must receive preparation failures.
                    error = exc
                pending = () if state is None else tuple((item.key, item.ranks) for item in state.ready_collectives)
                local = (state is not None and state.done, pending,
                         None if error is None else f"{type(error).__name__}: {error}",
                         None if state is None else state.ready_cache, () if state is None else tuple(state.ready_tuning),
                         None if state is None else state.ready_collective_tuning)
                states = [None] * len(self.ranks)
                dist.all_gather_object(states, local, group=self.group)
                self._remember_operation_ranks(states)
                failures = [(rank, item[2]) for rank, item in zip(self.ranks, states, strict=True) if item[2] is not None]
                if failures:
                    raise RuntimeError(f"rank preparation failed: {failures}") from error
                if all(item[0] for item in states):
                    return job.result()
                authorization = {"collective_key": None}
                local = states[self._local_index()]
                operation, local_ranks = None, None
                for index in (5, 3, 4, 1):
                    value = local[index]
                    if value:
                        operation = index
                        local_ranks = (getattr(value, "ranks", self.ranks) if index in (5, 3) else
                                       getattr(value[0], "ranks", self.ranks) if index == 4 else value[0][1])
                        break
                if operation is None:
                    if local[0]:
                        for item in states:
                            for index in (5, 3, 4, 1):
                                value = item[index]
                                if value:
                                    ranks = (getattr(value, "ranks", self.ranks) if index in (5, 3) else
                                             getattr(value[0], "ranks", self.ranks) if index == 4 else value[0][1])
                                    if self._global_rank() in ranks:
                                        raise RuntimeError("a rank finished before its peer's collective declaration")
                    continue
                members = self._members(states, local_ranks)
                def busy(item):
                    return item[1] or item[3] is not None or item[4] or item[5] is not None
                if any(item[0] for item in members) and any(busy(item) for item in members):
                    raise RuntimeError("a rank finished before its peer's collective declaration")
                if not all(item[operation] for item in members):
                    continue
                if operation == 5:
                    if any(item[5] is None or getattr(item[5], "key", None) != getattr(local[5], "key", None)
                           or item[5].ranks != local_ranks for item in members):
                        continue
                    authorization = {"collective_tuning": tuple(item[5] for item in members)}
                elif operation == 3:
                    if any(item[3] is None or getattr(item[3], "ranks", self.ranks) != local_ranks for item in members):
                        continue
                    # Each operation reconciles its own group's cached selections, first rank first.
                    authorization = {"cache": tuple(item[3] for item in members)}
                elif operation == 4:
                    contributions = [item[4] for item in members]
                    if any({getattr(contribution, "ranks", self.ranks) for contribution in group} != {local_ranks}
                           for group in contributions):
                        continue
                    races = [sorted(contribution.key for contribution in group) for group in contributions]
                    if any(keys != races[0] for keys in races[1:]):
                        reference = set(races[0])
                        counts = dict(zip(local_ranks, map(len, races), strict=True))
                        differences = {}
                        for rank, keys in zip(local_ranks[1:], races[1:], strict=True):
                            missing, extra = reference - set(keys), set(keys) - reference
                            if missing or extra:
                                differences[rank] = {"missing": len(missing), "extra": len(extra),
                                                     "examples": sorted(missing | extra)[:2]}
                        raise RuntimeError(f"tuning races differ across ranks: counts={counts}, differences={differences}")
                    results = _tuning_results(contribution for group in contributions for contribution in group)
                    authorization = {"tuning": results[0] if len(results) == 1 else results}
                else:
                    if any(tuple(ranks) != local_ranks for _, ranks in local[1]):
                        raise ValueError("collective declaration mixes incompatible control groups")
                    if any(item[1] != local[1] for item in members):
                        continue
                    authorization = {"collective_key": local[1][0][0]}
        except BaseException as failure:
            try:
                if job is not None:
                    if state is not None and state.done:
                        job.result().close()
                    else:
                        job.close()
            except BaseException as cleanup:  # noqa: BLE001 - Preserve the distributed failure.
                failure.add_note(f"preparation cleanup failed: {cleanup!r}")
            raise

    def barrier(self, key, ranks):
        ranks = tuple(ranks)
        if ranks not in self.control_groups:
            raise ValueError(f"collective {key!r} does not span its control group")
        dist.monitored_barrier(group=self.control_groups[ranks], timeout=self.timeout, wait_all_ranks=True)
