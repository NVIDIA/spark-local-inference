# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared b12x reductions and gathers embedded in compiled CUDA graphs.

Ranks on one host communicate through b12x PCIe runtimes; ranks on separate
DGX Sparks through RoCEnante (b12x.comm.roce), whose one-shot all-reduce and
all-gather are captured in the same graphs. A component given no runtime serves
one rank: the reduction or gather is the rank's own values, copied into the
output its consumers read.
"""

import itertools
from math import prod
import weakref
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x.preparation import CollectiveRequirement, PreparedCall

from ignition.preparation.tuning import declare, register_finalizer

_OWNERS = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


def packed_broadcast_layout(fields):
    """Return byte offsets and extent for eight-byte-aligned typed fields."""
    offsets, size = [], 0
    for shape, dtype in fields:
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("broadcast field dimensions must be nonnegative integers")
        size = (size + 7) // 8 * 8
        offsets.append(size)
        size += prod(shape) * dtype.itemsize
    return tuple(offsets), size


def _roce(runtime):
    """True for a RoCEnante runtime joining ranks on separate hosts."""
    from b12x.comm import roce

    return isinstance(runtime, roce.AllReduce)


def _roce_plan(runtime, coordinator, surface):
    """One fixed RoCEnante plan; every operation's geometry is validated by the runtime at launch."""
    from b12x.comm import roce

    query = roce.query_from_runtime(runtime, surface=surface, call={"dtypes": ("bfloat16",)},
                                    topology="roce_rdma", peer_hosts=coordinator.hosts)
    return roce.plan(query, runtime=runtime)


@torch.library.custom_op("flare::pcie_all_reduce", mutates_args=("output",))
def _all_reduce(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    from ignition.preparation.startup_tuning import record_consumer
    record_consumer(owner)
    owner.runtime.all_reduce(source, out=output, plan=owner.plan)


@_all_reduce.register_fake
def _fake_all_reduce(source, output, handle):
    pass


class PreparedAllReduce:
    """One fixed-shape reduction through an established b12x PCIe runtime.

    The bootstrap owns a single-channel runtime, exchange group, and coordinator.
    Every participating rank must declare the same operation name and enter
    preparation together. Graph ownership must retain this component and its
    runtime until every rank has drained native replay.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        import torch.distributed as dist
        from b12x.comm import pcie

        self.runtime, self.output = runtime, output
        startup_joint = False
        # Traced forwards test this flag, never the runtime: compile workers see a runtime that
        # owns RDMA or IPC resources only as a placeholder.
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank reduction copies its source into a matching output")
            return
        if _roce(runtime):
            if (source.shape != output.shape or source.dtype != torch.bfloat16 or output.dtype != source.dtype
                    or not source.is_contiguous() or not output.is_contiguous()
                    or not 0 < source.numel() * 2 <= max(runtime.max_size, runtime.pipe_bytes)
                    or source.numel() * 2 % 16):
                raise ValueError("RoCE reductions require contiguous BF16 16-byte packs within the runtime's capacity")
            self.plan = _roce_plan(runtime, coordinator, "AllReduce.all_reduce")
            group = runtime._group

            def prepare(state):
                return PreparedCall(run=lambda: state.all_reduce(source, out=output), output=output,
                                    owners=(runtime,))
        elif isinstance(runtime, pcie.DmaAllReduce):
            # Copy-engine ring for prefill-sized activations.
            from b12x.comm.pcie._dma_preparation import query_from_metadata

            self.plan = pcie.plan(query_from_metadata(runtime, shape=tuple(source.shape), dtype=source.dtype),
                                  runtime=runtime)
            group = runtime.group

            def prepare(state):
                return PreparedCall(run=lambda: state.run(source, out=output), output=output,
                                    owners=(runtime,))
        elif runtime.algorithm == "hierarchical":
            from b12x.comm.pcie._owner_preparation import prepared_call

            self.plan = runtime.plan(source, out=output)
            group = runtime.for_stream().group

            def prepare(state):
                return prepared_call(state, inp=source, out=output)
        else:
            if (runtime.world_size < 2 or runtime.algorithm != "oneshot"
                    or not runtime.single_channel):
                raise ValueError("prepared PCIe reductions require single-channel b12x direct all-reduce")
            from b12x.comm.pcie import CollectiveConfig

            channel = runtime.for_stream()
            startup_joint = (getattr(session, "collective_tuning", "static") == "runtime"
                             and source.dtype == torch.bfloat16
                             and getattr(channel, "_joint_workspace", None) is not None)
            plan_options = {}
            if startup_joint:
                from ignition.preparation.collective_ladder import select

                ladder = session.collective_ladder
                if ladder is None:
                    raise ValueError("runtime collective tuning requires a collective ladder")
                row = select(ladder, source.numel() * source.element_size(), world_size=runtime.world_size,
                             capability=torch.cuda.get_device_capability(source.device),
                             inplace=source.data_ptr() == output.data_ptr())
                self.runtime_profiles = tuple((mode, active) for mode, counts in row["profiles"].items() for active in counts)
                self.runtime_cache_policy = ladder["cache_policy"]
                cluster_size = row.get("cluster_size", 1)
                mode, active = self.runtime_profiles[0]
                # Phased modes need a complete cluster, so use a declared clustered profile.
                initial_active = active if cluster_size > 1 else 1
                plan_options = dict(joint_only=True, override=CollectiveConfig(
                    implementation=row["implementation"], threads=row["threads"], blocks=row["blocks"],
                    submode=mode, active_blocks=initial_active))
                if cluster_size > 1:
                    plan_options["joint_cluster_size"] = cluster_size
            self.plan = runtime.plan(source, out=output, **plan_options)
            group = runtime.exchange_group

            def prepare(state):
                if state.query.surface == "PCIeTwoShotBF16.all_reduce":
                    from b12x.comm.pcie._twoshot_preparation import prepared_call

                    return prepared_call(state, payload=source, out=output)
                state.bind_input(source)
                return PreparedCall(run=lambda: state.run_plain(source, output),
                                    output=output, owners=(runtime,))

        ranks = tuple(dist.get_process_group_ranks(group))
        request = self.plan.request(name=name, prepare_call=prepare,
            collective=CollectiveRequirement(key=name, ranks=ranks))
        declare(session, (request,), coordinator=coordinator)
        if startup_joint:
            from ignition.preparation.startup_tuning import register_collective
            register_collective(self, session)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        """Allocate the reduction destination inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
        return self

    def __call__(self, source, output=None):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        else:
            _all_reduce(source, output, self.handle)
        return output


class PreparedChunkedAllGather:
    """Column gathers of more rows than the channel's batch capacity, gathered in row chunks it accepts.

    Each chunk is a PreparedAllGather over contiguous row views of the source and output, so a channel sized
    for decode can also gather prefill activations; the chunks run in order on one stream.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        rows = source.shape[0]
        step = rows if runtime is None or _roce(runtime) else max(1, min(rows, runtime.max_batch_size))
        self.bounds = [(start, min(start + step, rows)) for start in range(0, rows, step)]
        self.parts = [PreparedAllGather(source[begin:end], output[begin:end], runtime=runtime, session=session,
                                        coordinator=coordinator, name=name if len(self.bounds) == 1 else f"{name}.{index}")
                      for index, (begin, end) in enumerate(self.bounds)]
        self.output = output

    def functional(self, *, session):
        """Allocate one destination shared by the ordered row chunks."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
            for part in self.parts:
                part.functional(session=session)
        return self

    def __call__(self, source):
        output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            part(source[begin:end], output[begin:end])
        return output

    def run_eager(self, source):
        """Gather on the current stream from inside another eager operation."""
        output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            part.run_eager(source[begin:end], output[begin:end])
        return output


class PreparedChunkedAllReduce:
    """A row-major reduction larger than its PCIe runtime's capacity, reduced in row chunks the runtime accepts.

    Each chunk is a PreparedAllReduce over contiguous row views of the source and output, so a channel sized
    for decode can also reduce prefill activations; the chunks run in order on one stream.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        rows = source.shape[0]
        step = rows
        if runtime is not None and _roce(runtime):
            row_bytes = source[0].numel() * source.element_size()
            step = min(rows, max(runtime.max_size, runtime.pipe_bytes) // row_bytes)
            if step < 1:
                raise ValueError("one reduction row exceeds the RoCE channel capacity")
        elif runtime is not None:
            from b12x.comm import pcie

            # One-shot pools answer through their stream channel; hierarchical runtimes directly.
            accepts = (runtime.for_stream().should_allreduce
                       if not isinstance(runtime, pcie.DmaAllReduce) and runtime.algorithm == "oneshot"
                       else runtime.should_allreduce)
            while step > 1 and not accepts(source[:step]):
                step = (step + 1) // 2
        self.bounds = [(start, min(start + step, rows)) for start in range(0, rows, step)]
        self.parts = [PreparedAllReduce(source[begin:end], output[begin:end], runtime=runtime, session=session,
                                        coordinator=coordinator, name=name if len(self.bounds) == 1 else f"{name}.{index}")
                      for index, (begin, end) in enumerate(self.bounds)]
        self.output = output

    def functional(self, *, session):
        """Allocate one destination shared by the ordered row chunks."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
            for part in self.parts:
                part.functional(session=session)
        return self

    def __call__(self, source):
        output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        for (begin, end), part in zip(self.bounds, self.parts, strict=True):
            part(source[begin:end], output[begin:end])
        return output


@torch.library.custom_op("flare::roce_reduce_scatter", mutates_args=("output",))
def _reduce_scatter(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.reduce_scatter(source, out=output, plan=owner.plan)


@_reduce_scatter.register_fake
def _fake_reduce_scatter(source, output, handle):
    pass


@torch.library.custom_op("flare::roce_gather_shards", mutates_args=("output",))
def _gather_shards(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.gather_shards(source, out=output, plan=owner.plan)


@_gather_shards.register_fake
def _fake_gather_shards(source, output, handle):
    pass


class PreparedReduceScatter:
    """Reduce every rank's [rows, columns] partial sums and keep this rank's contiguous row shard.

    Rank r's output holds rows [r * rows / world, (r + 1) * rows / world) of the
    sum with the arithmetic of the runtime's two-shot all-reduce, so a
    PreparedShardGather of every rank's output reproduces that all-reduce's
    bits. Ranks on separate hosts reduce through RoCEnante's pipelined path,
    whose eligibility `fits` reports. A component given no runtime serves one
    rank: the output is the source, copied.
    """

    @staticmethod
    def fits(runtime, source):
        """True when every rank can reduce-scatter `source` through `runtime` (rank-uniform)."""
        return runtime is None or (_roce(runtime) and runtime.should_reduce_scatter(source))

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        import torch.distributed as dist

        self.runtime, self.output = runtime, output
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank reduce-scatter copies its source into a matching output")
            return
        if not _roce(runtime):
            raise ValueError("reduce-scatter runs over RoCEnante")
        if (source.ndim != 2 or source.dtype != torch.bfloat16 or not runtime.should_reduce_scatter(source)
                or output.shape != (source.shape[0] // runtime.world_size, source.shape[1])
                or output.dtype != source.dtype or not output.is_contiguous()):
            raise ValueError("RoCE reduce-scatter requires a contiguous BF16 [rows, columns] source whose rows divide "
                             "by the world size within the pipelined capacity, and a matching row-shard output")
        self.plan = _roce_plan(runtime, coordinator, "AllReduce.reduce_scatter")
        ranks = tuple(dist.get_process_group_ranks(runtime._group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: runtime._run_prepared_reduce_scatter(source, prepared=state, out=output),
                output=output, owners=(runtime, source, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, source):
        if self.single:
            self.output.copy_(source)
            return self.output
        _reduce_scatter(source, self.output, self.handle)
        return self.output


class PreparedShardGather:
    """Gather every rank's [rows / world, columns] row shard into [rows, columns] in rank order.

    The inverse of PreparedReduceScatter over the same pipelined RoCEnante path.
    A component given no runtime serves one rank: the output is the source, copied.
    """

    @staticmethod
    def fits(runtime, source):
        """True when every rank can gather its shard `source` through `runtime` (rank-uniform)."""
        return runtime is None or (_roce(runtime) and runtime.should_gather_shards(source))

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        import torch.distributed as dist

        self.runtime, self.output = runtime, output
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank shard gather copies its source into a matching output")
            return
        if not _roce(runtime):
            raise ValueError("shard gathers run over RoCEnante")
        if (source.ndim != 2 or source.dtype != torch.bfloat16 or not runtime.should_gather_shards(source)
                or output.shape != (source.shape[0] * runtime.world_size, source.shape[1])
                or output.dtype != source.dtype or not output.is_contiguous()):
            raise ValueError("RoCE shard gathers require a contiguous BF16 [rows, columns] shard within the pipelined "
                             "capacity and a matching world-fold output")
        self.plan = _roce_plan(runtime, coordinator, "AllReduce.gather_shards")
        ranks = tuple(dist.get_process_group_ranks(runtime._group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: runtime._run_prepared_gather_shards(source, prepared=state, out=output),
                output=output, owners=(runtime, source, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def __call__(self, source):
        if self.single:
            self.output.copy_(source)
            return self.output
        _gather_shards(source, self.output, self.handle)
        return self.output


@dataclass(frozen=True)
class WeightedGroupSource:
    """Ordered weighted groups with physical [rows, groups, hidden] storage.

    Weights are FP32 [rows, groups]; IDs are int32/int64 with the same shape.
    IDs outside [0, id_limit) supply zero values before multiplication.
    """

    values: torch.Tensor
    weights: torch.Tensor
    ids: torch.Tensor
    id_limit: int

    def arguments(self):
        return dict(group_weights=self.weights, group_ids=self.ids, group_id_limit=self.id_limit)


@torch.library.custom_op("flare::pcie_all_reduce_rmsnorm", mutates_args=("residual", "output"))
def _all_reduce_rmsnorm(source: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                       output: torch.Tensor, handle: int, input_addend: torch.Tensor | None = None,
                       group_weights: torch.Tensor | None = None, group_ids: torch.Tensor | None = None) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce_fused_add_rms_norm(source, residual, weight, owner.eps, plan=owner.plan,
        out=output, residual_out=residual, zero_centered=owner.zero_centered, reduction_dtype=owner.reduction_dtype,
        input_addend=input_addend, group_weights=group_weights, group_ids=group_ids,
        group_id_limit=owner.group_id_limit)


@_all_reduce_rmsnorm.register_fake
def _fake_all_reduce_rmsnorm(source, residual, weight, output, handle, input_addend=None,
                             group_weights=None, group_ids=None):
    pass


class PreparedAllReduceRMSNorm:
    """Reduce an optional rounded source sum, update the residual, and normalize.

    Weighted groups reduce in group order with separate FP32 multiplies and
    adds, then round to source precision. An input addend shares the logical
    [rows, hidden] geometry and dtype; its FP32 addition rounds to source
    precision before tensor-parallel reduction.
    """

    @staticmethod
    def fits(runtime, source):
        from b12x.comm import pcie

        if isinstance(source, WeightedGroupSource):
            from b12x.comm.pcie._oneshot_preparation import _grouped_inputs, _state

            if not isinstance(runtime, pcie.AllReduce) or runtime.algorithm != "oneshot" or not runtime.single_channel:
                return False
            channel = runtime._runtime._prepared_channel_for_stream(None, None)
            if _state(channel).eager_tables is None:
                return False
            source, *_ = _grouped_inputs(source.values, **source.arguments())
        return (isinstance(runtime, pcie.AllReduce) and runtime.algorithm == "oneshot"
                and runtime.single_channel and source.ndim == 2 and source.shape[0] > 0
                and source.is_contiguous() and source.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and source.shape[1] * source.element_size() % 16 == 0
                and 0 < source.numel() * source.element_size() <= runtime.max_size)

    def __init__(self, source, residual, weight, output, *, eps, runtime, session, coordinator, name,
                 zero_centered=False, reduction_dtype=None, input_addend=None):
        import torch.distributed as dist
        from b12x.comm.pcie._oneshot_preparation import _prepare_fused_call

        if not self.fits(runtime, source):
            raise ValueError("fused all-reduce RMSNorm requires a single-channel one-shot runtime within its capacity")
        self.runtime, self.residual, self.weight, self.output = runtime, residual, weight, output
        self.input_addend = input_addend
        grouped = source if isinstance(source, WeightedGroupSource) else None
        group_arguments = {} if grouped is None else grouped.arguments()
        self.group_weights = None if grouped is None else grouped.weights
        self.group_ids = None if grouped is None else grouped.ids
        self.group_id_limit = 0 if grouped is None else grouped.id_limit
        source = source if grouped is None else grouped.values
        self.eps, self.zero_centered, self.reduction_dtype = eps, zero_centered, reduction_dtype
        self.plan = runtime.plan(source, operation="all_reduce_fused_add_rms_norm", residual=residual,
            weight=weight, out=output, residual_out=residual, zero_centered=zero_centered, reduction_dtype=reduction_dtype,
            input_addend=input_addend, **group_arguments)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: _prepare_fused_call(state, inp=source, residual=residual, weight=weight,
                out=output, residual_out=residual, epsilon=eps, input_addend=input_addend, **group_arguments,
                owners=(runtime, source, residual, weight, output, input_addend, self.group_weights, self.group_ids)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype
        self.output = self.residual = self.input_addend = self.group_weights = self.group_ids = None
        return self

    def __call__(self, source, residual, *, input_addend=None, group_weights=None, group_ids=None):
        output = self.output
        if output is None:
            output = torch.empty_strided(self.output_layout[0], self.output_layout[1], dtype=self.output_layout[2], device=source.device)
        else:
            input_addend, group_weights, group_ids = self.input_addend, self.group_weights, self.group_ids
        _all_reduce_rmsnorm(source, residual, self.weight, output, self.handle, input_addend,
                           group_weights, group_ids)
        return output


@torch.library.custom_op("flare::pcie_all_reduce_add", mutates_args=("residual",))
def _all_reduce_add(source: torch.Tensor, residual: torch.Tensor, handle: int,
                    input_addend: torch.Tensor | None = None,
                    group_weights: torch.Tensor | None = None,
                    group_ids: torch.Tensor | None = None) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_reduce_fused_add(source, residual, plan=owner.plan,
        input_addend=input_addend, group_weights=group_weights,
        group_ids=group_ids, group_id_limit=owner.group_id_limit)


@_all_reduce_add.register_fake
def _fake_all_reduce_add(source, residual, handle, input_addend=None, group_weights=None, group_ids=None):
    pass


class PreparedAllReduceAdd:
    """Reduce weighted contributions and update every residual row in place."""

    fits = staticmethod(PreparedAllReduceRMSNorm.fits)

    def __init__(self, source, residual, *, runtime, session, coordinator, name,
                 reduction_dtype=None, input_addend=None):
        import torch.distributed as dist
        from b12x.comm.pcie._oneshot_preparation import _prepare_fused_call

        if not self.fits(runtime, source):
            raise ValueError("residual reduction requires a single-channel one-shot runtime within capacity")
        self.runtime, self.input_addend = runtime, input_addend
        grouped = source if isinstance(source, WeightedGroupSource) else None
        arguments = {} if grouped is None else grouped.arguments()
        self.group_weights = None if grouped is None else grouped.weights
        self.group_ids = None if grouped is None else grouped.ids
        self.group_id_limit = 0 if grouped is None else grouped.id_limit
        source = source if grouped is None else grouped.values
        self.plan = runtime.plan(source, operation="all_reduce_fused_add", residual=residual,
            residual_out=residual, reduction_dtype=reduction_dtype, input_addend=input_addend, **arguments)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: _prepare_fused_call(state, inp=source, residual=residual,
                weight=None, out=None, residual_out=residual, epsilon=0.0,
                input_addend=input_addend, **arguments,
                owners=(runtime, source, residual, input_addend, self.group_weights, self.group_ids)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        self.input_addend = self.group_weights = self.group_ids = None
        return self

    def __call__(self, source, residual, *, input_addend=None, group_weights=None, group_ids=None):
        _all_reduce_add(source, residual, self.handle,
                        self.input_addend if input_addend is None else input_addend,
                        self.group_weights if group_weights is None else group_weights,
                        self.group_ids if group_ids is None else group_ids)
        return residual


def _gather_transport_view(tensor):
    # Two BF16 transport lanes preserve every FP32 bit without conversion.
    return tensor.view(torch.bfloat16) if tensor.dtype == torch.float32 else tensor


def _gather_columns(source, output, handle):
    source, output = _gather_transport_view(source), _gather_transport_view(output)
    owner = _OWNERS[handle]
    from ignition.preparation.startup_tuning import record_consumer
    record_consumer(owner)
    runtime = owner.runtime
    if owner.roce:
        runtime.all_gather(source, dim=-1, out=output, plan=owner.plan)
        return
    runtime.all_gather_heads(
        source.view(source.shape[0], runtime.heads_per_rank, runtime.query_head_dim),
        out=output.view(output.shape[0], runtime.total_heads, runtime.query_head_dim),
        plan=owner.plan)


@torch.library.custom_op("flare::pcie_all_gather", mutates_args=("output",))
def _all_gather(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor], handle: int) -> None:
    _gather_columns(source, output, handle)


@_all_gather.register_fake
def _fake_all_gather(source, output, dependencies, handle):
    pass


@torch.library.custom_op("flare::start_all_gather", mutates_args=("output",))
def _start_all_gather(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                      handle: int, stream: int) -> None:
    auxiliary = torch.cuda.ExternalStream(stream, device=source.device)
    auxiliary.wait_stream(torch.cuda.current_stream(source.device))
    with torch.cuda.stream(auxiliary):
        _gather_columns(source, output, handle)


@_start_all_gather.register_fake
def _fake_start_all_gather(source, output, dependencies, handle, stream):
    pass


@torch.library.custom_op("flare::join_all_gather", mutates_args=("output",))
def _join_all_gather(source: torch.Tensor, output: torch.Tensor, dependencies: list[torch.Tensor],
                     dependency: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(source.device).wait_stream(torch.cuda.ExternalStream(stream, device=source.device))


@_join_all_gather.register_fake
def _fake_join_all_gather(source, output, dependencies, dependency, stream):
    pass


@dataclass(frozen=True)
class PendingAllGather:
    source: torch.Tensor
    output: torch.Tensor
    dependencies: list[torch.Tensor]


class ConcurrentAllGather:
    """Overlap a prepared PCIe gather with local work, joining before channel reuse.

    The join retains the source through its asynchronous read and follows a local
    output dependency. A subsequent collective must read the joined output as a
    dependency, keeping communication serialized through compilation.
    """

    def __init__(self, gather, *, workspace, session, parallel_plans):
        from ignition.preparation.scratch import ScratchArena

        plans = tuple(parallel_plans)
        if gather.single or gather.roce:
            raise ValueError("concurrent column gather requires a prepared PCIe channel")
        if not isinstance(workspace, ScratchArena) or not workspace.overlap or not plans:
            raise ValueError("concurrent column gather requires an overlap arena and local plans")
        self.gather = gather
        self.stream = workspace.auxiliary_stream
        self.stream_handle = self.stream.cuda_stream
        for plan in plans:
            workspace.register_parallel(gather.plan, plan)
        register_finalizer(session, self.validate_scratch)

    def validate_scratch(self):
        # PCIe gather scratch belongs to its persistent channel, outside the shared arena.
        if self.gather.plan.memory_requirements().scratch_nbytes:
            raise ValueError("concurrent column gather requires a plan without caller scratch")

    def functional(self, *, session):
        self.gather.functional(session=session)
        return self

    def start(self, source, *, dependencies=()):
        output = self.gather.output
        if output is None:
            shape, stride, dtype = self.gather.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        dependencies = [*self.gather.dependencies, *dependencies]
        _start_all_gather(source, output, dependencies, self.gather.handle, self.stream_handle)
        return PendingAllGather(source, output, dependencies)

    def finish(self, pending, dependency):
        _join_all_gather(pending.source, pending.output, pending.dependencies, dependency, self.stream_handle)
        return pending.output


@triton.jit
def _broadcast_pack(sources, send, OFFSETS: tl.constexpr, SIZES: tl.constexpr,
                    BOOLS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for binding in tl.static_range(len(SIZES)):
        value = tl.load(sources[binding] + index, index < SIZES[binding], other=0)
        if BOOLS[binding]:
            value = (value != 0).to(tl.uint8)
        tl.store(send + OFFSETS[binding] + index, value, index < SIZES[binding])


@triton.jit
def _broadcast_unpack(received, destinations, OFFSETS: tl.constexpr, SIZES: tl.constexpr,
                      BOOLS: tl.constexpr, ROW_BYTES: tl.constexpr, WORLD: tl.constexpr,
                      BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for binding in tl.static_range(len(SIZES)):
        offset = OFFSETS[binding] + index
        # Every received row places rank zero before the other rank partitions.
        source = offset // ROW_BYTES * (ROW_BYTES * WORLD) + offset % ROW_BYTES
        value = tl.load(received + source, index < SIZES[binding], other=0)
        if BOOLS[binding]:
            value = (value != 0).to(tl.uint8)
        tl.store(destinations[binding] + index, value, index < SIZES[binding])


# Typed operands preserve bool consumers and storage aliases during compilation.
@torch.library.custom_op("flare::broadcast_pack", mutates_args=("send",))
def _broadcast_pack_call(sources: list[torch.Tensor], send: torch.Tensor, handle: int) -> None:
    operation = _OWNERS[handle]._pack
    operation.compiled[operation.grid](tuple(source.view(torch.uint8).view(-1) for source in sources), send)


@_broadcast_pack_call.register_fake
def _fake_broadcast_pack_call(sources, send, handle):
    pass


@torch.library.custom_op("flare::broadcast_unpack", mutates_args=("destinations",))
def _broadcast_unpack_call(received: torch.Tensor, destinations: list[torch.Tensor], handle: int,
                           group: int) -> None:
    operation = _OWNERS[handle]._unpacks[group]
    operation.compiled[operation.grid](received,
        tuple(destination.view(torch.uint8).view(-1) for destination in destinations))


@_broadcast_unpack_call.register_fake
def _fake_broadcast_unpack_call(received, destinations, handle, group):
    pass


class PreparedBroadcast:
    """Copy rank zero's tensor bytes through a serial all-gather channel.

    Integer decisions and floating-point payloads retain their exact bit patterns.
    The BF16 channel views transport bytes without floating-point arithmetic.
    Overlapping destination views are restored in tensor order.
    """

    def __init__(self, tensors, *, runtime, session, coordinator, name, buffers=None):
        self.single = runtime is None
        if self.single:
            return
        tensors = tuple(tensors)
        if not tensors or any(not tensor.is_contiguous() or tensor.device != tensors[0].device
                              for tensor in tensors):
            raise ValueError("broadcast requires contiguous tensors on one device")
        offsets, size = packed_broadcast_layout((tuple(tensor.shape), tensor.dtype) for tensor in tensors)
        columns = ((size + 15) // 16 * 8 if _roce(runtime)
                   else runtime.heads_per_rank * runtime.query_head_dim)
        rows = (size + 2 * columns - 1) // (2 * columns)

        def allocate(shape):
            return (torch.zeros(shape, dtype=torch.uint8, device=tensors[0].device) if buffers is None
                    else buffers.zeros(shape, torch.uint8))

        self.send = allocate((rows, columns * 2))
        self.received = allocate((rows, columns * 2 * runtime.world_size))
        self.gather = PreparedAllGather(self.send.view(torch.bfloat16), self.received.view(torch.bfloat16),
            runtime=runtime, session=session, coordinator=coordinator, name=name)

        self._tensors = tensors
        self._byte_views = tuple(tensor.view(torch.uint8).view(-1) for tensor in tensors)
        self._offsets = tuple(offsets)
        self._sizes = tuple(tensor.numel() * tensor.element_size() for tensor in tensors)
        self._bools = tuple(tensor.dtype == torch.bool for tensor in tensors)
        groups, group, ranges = [], [], []
        for index, tensor in enumerate(tensors):
            begin, size = tensor.data_ptr(), self._sizes[index]
            end = begin + size
            if size and any(begin < other_end and other_begin < end for other_begin, other_end in ranges):
                groups.append(tuple(group))
                group, ranges = [], []
            group.append(index)
            if size:
                ranges.append((begin, end))
        groups.append(tuple(group))
        self._unpack_groups = tuple(groups)
        self._row_bytes = columns * 2
        self._world = runtime.world_size
        register_finalizer(session, self.prepare)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def prepare(self):
        from ignition.preparation.triton import prepare_triton

        if self._byte_views is None:
            return
        options = dict(OFFSETS=self._offsets, SIZES=self._sizes, BOOLS=self._bools, BLOCK=256, num_warps=4)
        grid = (triton.cdiv(max(self._sizes), 256), 1, 1)
        self._pack = prepare_triton(_broadcast_pack, (self._byte_views, self.send), grid, **options)
        self._unpacks = tuple(prepare_triton(_broadcast_unpack,
            (self.received, tuple(self._byte_views[index] for index in group)),
            (triton.cdiv(max(self._sizes[index] for index in group), 256), 1, 1),
            OFFSETS=tuple(self._offsets[index] for index in group),
            SIZES=tuple(self._sizes[index] for index in group),
            BOOLS=tuple(self._bools[index] for index in group),
            ROW_BYTES=self._row_bytes, WORLD=self._world, BLOCK=256, num_warps=4)
            for group in self._unpack_groups)

    def functional(self, *, session):
        """Accept runtime payloads and allocate transport scratch in the forward."""
        if not self.single and not hasattr(self, "send_shape"):
            self.send_shape = tuple(self.send.shape)
            self.gather.functional(session=session)
            register_finalizer(session, self._release_buffers)
        return self

    def _release_buffers(self):
        self.send = self.received = self._tensors = self._byte_views = None
        self._pack.args = ()
        for operation in self._unpacks:
            operation.args = ()

    def __call__(self, tensors=None):
        if self.single:
            return
        tensors = self._tensors if tensors is None else tuple(tensors)
        send = self.send
        if send is None:
            send = torch.zeros(self.send_shape, dtype=torch.uint8, device=tensors[0].device)
        _broadcast_pack_call(list(tensors), send, self.handle)
        received = self.gather(send.view(torch.bfloat16)).view(torch.uint8)
        # Aliased destinations keep their field order across separate stream-ordered launches.
        # Disjoint fields share one launch, including the ordinary sampling payload.
        for index, group in enumerate(self._unpack_groups):
            _broadcast_unpack_call(received, [tensors[field] for field in group], self.handle, index)


class PreparedAllGather:
    """Gather contiguous column partitions through a caller-owned b12x channel.

    Each output row contains rank zero's columns, then rank one's, and so on.
    Operations sharing the channel must execute serially on every rank.
    Dependencies order storage writes before the collective publishes them to peers.
    """

    def __init__(self, source, output, *, runtime, session, coordinator, name, dependencies=()):
        import torch.distributed as dist
        from b12x.comm import pcie

        self.runtime, self.output = runtime, output
        self.dependencies = list(dependencies)
        # Traced forwards test this flag, never the runtime: compile workers see a runtime that
        # owns RDMA or IPC resources only as a placeholder.
        self.single = runtime is None
        if runtime is None:
            if source.shape != output.shape or source.dtype != output.dtype:
                raise ValueError("a single-rank gather copies its source into a matching output")
            return
        if (source.ndim != 2 or source.dtype not in (torch.bfloat16, torch.float32, torch.float8_e4m3fn)
                or output.dtype != source.dtype or not source.is_contiguous() or not output.is_contiguous()):
            raise ValueError("column gathers require contiguous matching BF16, FP32 or E4M3 matrices")
        transport_source = _gather_transport_view(source)
        transport_output = _gather_transport_view(output)
        self.roce = _roce(runtime)
        if self.roce:
            if (transport_source.dtype != torch.bfloat16
                    or transport_output.shape != (source.shape[0], transport_source.shape[1] * runtime.world_size)
                    or transport_source.shape[1] * 2 % 16
                    or not 0 < transport_source.numel() * 2 <= runtime.max_gather_bytes):
                raise ValueError("RoCE gathers require BF16 or FP32 column partitions within the byte capacity")
            self.plan = _roce_plan(runtime, coordinator, "AllReduce.all_gather")
            ranks = tuple(dist.get_process_group_ranks(runtime._group))
            # Priming runs before the plan is published, so it launches the materialized gather directly.
            declare(session, (self.plan.request(name=name,
                prepare_call=lambda state: PreparedCall(
                    run=lambda: runtime._run_prepared_all_gather(transport_source, prepared=state, dim=-1, out=transport_output),
                    output=output, owners=(runtime,)),
                collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
            self.handle = next(_HANDLES)
            _OWNERS[self.handle] = self
            return
        if (runtime.world_size < 2 or source.ndim != 2
                or source.dtype not in (torch.bfloat16, torch.float32, torch.float8_e4m3fn) or output.dtype != source.dtype
                or source.device != runtime.device or output.device != source.device
                or not source.is_contiguous() or not output.is_contiguous()
                or not 0 < source.shape[0] <= runtime.max_batch_size
                or transport_source.shape[1] != runtime.heads_per_rank * runtime.query_head_dim
                or output.shape != (source.shape[0], source.shape[1] * runtime.world_size)):
            raise ValueError("all-gather requires contiguous BF16, FP32 or E4M3 column partitions matching its collective "
                             "channel")
        self.runtime, self.output = runtime, output
        call = {"local_input": transport_source.view(source.shape[0], runtime.heads_per_rank, runtime.query_head_dim),
                "out": transport_output.view(output.shape[0], runtime.total_heads, runtime.query_head_dim)}
        query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.all_gather_heads", call=call)
        startup = False
        if getattr(session, "collective_tuning", "static") == "runtime":
            from dataclasses import replace
            from b12x.preparation import FrozenMapping
            from ignition.preparation.collective_ladder import select_gather

            ladder = session.collective_ladder
            if ladder is None:
                raise ValueError("runtime collective tuning requires a collective ladder")
            if ladder["all_gather"] is not None:
                row = select_gather(ladder, source.numel() * source.element_size(),
                    world_size=runtime.world_size, capability=torch.cuda.get_device_capability(source.device))
                self.runtime_profiles = tuple((mode, active) for mode, counts in row["profiles"].items() for active in counts)
                self.runtime_cache_policy = ladder["cache_policy"]
                query = replace(query, call=FrozenMapping(dict(query.call,
                    gather_transport=row["implementation"].removeprefix("gather_"),
                    threads=row["threads"], block_limit=row["blocks"],
                    submode=self.runtime_profiles[0][0], active_blocks=1)))
                startup = True
        self.plan = pcie.plan(query, runtime=runtime)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(**call), output=output,
                                                    owners=(runtime,)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        if startup:
            from ignition.preparation.startup_tuning import register_collective
            register_collective(self, session)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        """Allocate the gathered destination inside the traced forward."""
        if self.output is not None:
            self.output_layout = (tuple(self.output.shape), self.output.stride(), self.output.dtype)
            self.output = None
        self.dependencies = []
        return self

    def __call__(self, source, output=None, *, dependencies=()):
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        else:
            _all_gather(source, output, [*self.dependencies, *dependencies], self.handle)
        return output

    def run_eager(self, source, output=None):
        """Gather on the current stream from inside another eager operation."""
        if output is None:
            output = self.output
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
        if self.single:
            output.copy_(source)
        else:
            _gather_columns(source, output, self.handle)
        return output


@torch.library.custom_op("flare::roce_all_gather_varlen", mutates_args=("output",))
def _all_gather_varlen(source: torch.Tensor, counts: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.all_gather_varlen(source, counts, out=output, plan=owner.plan)


@_all_gather_varlen.register_fake
def _fake_all_gather_varlen(source, counts, output, handle):
    pass


class PreparedVarlenGather:
    """Gather byte prefixes of every rank's row of `output` [world, capacity] to every rank over RoCEnante.

    Each rank writes its contribution into its own row in place; `counts` is a [world] int64 device
    tensor of each rank's bytes, a multiple of 16, holding the same values on every rank and free to
    change between replays. Row r then holds rank r's first counts[r] bytes on every rank; bytes
    past them keep their values. Operations sharing the runtime must execute serially on every rank.
    """

    def __init__(self, output, counts, *, runtime, session, coordinator, name):
        import torch.distributed as dist

        if not _roce(runtime):
            raise ValueError("variable-length gathers run over RoCEnante")
        ranks = tuple(dist.get_process_group_ranks(runtime._group))
        if (output.shape[0] != len(ranks) or not output.is_contiguous() or counts.shape != (len(ranks),)
                or counts.dtype != torch.int64 or output[0].numel() * output.element_size() % 16):
            raise ValueError("variable-length gathers require one contiguous 16-byte-multiple row per rank")
        self.runtime, self.output, self.counts = runtime, output, counts
        # A rank's own row is its source, so the gather leaves it in place.
        self.local_rank = ranks.index(dist.get_rank())
        self.source = output[self.local_rank]
        self.plan = _roce_plan(runtime, coordinator, "AllReduce.all_gather_varlen")
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: runtime._run_prepared_all_gather_varlen(self.source, counts, prepared=state, out=output),
                output=output, owners=(runtime, self.source, counts, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.source = self.counts = self.output = None

    def __call__(self, output=None, counts=None):
        output = self.output if output is None else output
        counts = self.counts if counts is None else counts
        _all_gather_varlen(output[self.local_rank], counts, output, self.handle)
        return output


@torch.library.custom_op("flare::pcie_lse_reduce_scatter", mutates_args=("output",))
def _lse_reduce_scatter(partial: torch.Tensor, lse: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    owner = _OWNERS[handle]
    owner.runtime.lse_reduce_scatter(partial, lse, out=output, plan=owner.plan, is_lse_base_on_e=True)


@_lse_reduce_scatter.register_fake
def _fake_lse_reduce_scatter(partial, lse, output, handle):
    pass


class PreparedLseReduceScatter:
    """Combine every rank's partial attention by natural-log LSE and keep this rank's heads.

    Partial outputs are [rows, heads, dim] over all heads with FP32 [rows, heads] LSE;
    the output holds this rank's contiguous head shard. Operations sharing the
    channel must execute serially on every rank. Over RoCEnante, ranks all-gather
    the shards each peer keeps through `staging` (send [rows, width] and received
    [rows, W * width] BF16, from head_exchange.pack_width) and merge locally;
    `rank` is this rank as a one-element device tensor.
    """

    def __init__(self, partial, lse, output, *, runtime, session, coordinator, name, staging=None, rank=None,
                 threads=None, block_limit=None):
        import torch.distributed as dist
        from b12x.comm import pcie

        rows = partial.shape[0]
        self.roce = _roce(runtime)
        if self.roce:
            from .head_exchange import pack_width

            heads, dim = partial.shape[1:]
            world = runtime.world_size
            width = pack_width(world, heads, dim)
            send, received = (None, None) if staging is None else staging
            if (rank is None or send is None or heads % world or partial.dtype != torch.bfloat16
                    or lse.shape != (rows, heads) or lse.dtype != torch.float32
                    or output.shape != (rows, heads // world, dim) or output.dtype != torch.bfloat16
                    or send.shape[1:] != (width,) or received.shape[1:] != (world * width,)
                    or min(send.shape[0], received.shape[0]) < rows):
                raise ValueError("RoCE head combination requires BF16 partials, FP32 LSE, staging and the rank")
            self.output, self.rank, self.world = output, rank, world
            self.send, self.received = send[:rows], received[:rows]
            self.gather = PreparedAllGather(self.send, self.received, runtime=runtime, session=session,
                                            coordinator=coordinator, name=f"{name}.gather")
            return
        # Partials and outputs have the channel's value width, which a latent query may exceed.
        if (partial.shape != (rows, runtime.total_heads, runtime.head_dim) or partial.dtype != torch.bfloat16
                or lse.shape != (rows, runtime.total_heads) or lse.dtype != torch.float32
                or output.shape != (rows, runtime.heads_per_rank, runtime.head_dim)
                or output.dtype != torch.bfloat16 or not 0 < rows <= runtime.max_batch_size):
            raise ValueError("LSE reduce-scatter requires BF16 partials, FP32 LSE and a head shard of its channel")
        self.runtime, self.output = runtime, output
        call = {"partial_output": partial, "partial_lse": lse, "out": output, "is_lse_base_on_e": True}
        geometry = {key: value for key, value in (("threads", threads), ("block_limit", block_limit))
                    if value is not None}
        query = pcie.query_from_runtime(runtime, surface="DcpAllToAll.lse_reduce_scatter", call={**call, **geometry})
        self.plan = pcie.plan(query, runtime=runtime)
        ranks = tuple(dist.get_process_group_ranks(runtime.exchange_group))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(**call), output=output,
                                                    owners=(runtime, partial, lse, output)),
            collective=CollectiveRequirement(key=name, ranks=ranks)),), coordinator=coordinator)
        self.handle = next(_HANDLES)
        _OWNERS[self.handle] = self

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype
        self.output = None
        if self.roce:
            self.send_layout = tuple(self.send.shape), self.send.dtype
            self.send = self.received = None
            self.gather.functional(session=session)
        return self

    def __call__(self, partial, lse):
        output = self.output
        if output is None:
            output = torch.empty_strided(self.output_layout[0], self.output_layout[1], dtype=self.output_layout[2], device=partial.device).zero_()
        if self.roce:
            from .head_exchange import merge, pack

            send = self.send
            if send is None:
                send = torch.zeros(self.send_layout[0], dtype=self.send_layout[1], device=partial.device)
            pack(partial, lse, send, self.rank, self.world)
            received = self.gather(send)
            merge(received, partial, lse, output, self.rank, self.world)
            return output
        _lse_reduce_scatter(partial, lse, output, self.handle)
        return output
