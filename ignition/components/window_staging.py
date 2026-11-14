"""Prepared paired circular KV staging into caller-owned FP8 buffers."""

from dataclasses import dataclass

import torch
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.tuning import declare

from .native_operator import native_contract, native_plan


@dataclass(frozen=True)
class Query:
    heads: int
    dim: int
    capacity: int
    window: int
    history_dtype: str
    first_position: int


CONTRACT = native_contract(component_id="flare.window_stage", query_type=Query, tactics=lambda q: (0,))


def _operands(history_dtype):
    import cutlass

    history = cutlass.Float8E4M3FN if history_dtype == "float8_e4m3fn" else cutlass.BFloat16
    return ((4, history), (4, history), (3, cutlass.BFloat16), (3, cutlass.BFloat16),
            (4, cutlass.Uint8), (4, cutlass.Uint8), (1, cutlass.Int64), (1, cutlass.Int64),
            (1, cutlass.Int32), (1, cutlass.Int32), (1, cutlass.Int32))


@program_cache(scope="preparation")
def compile_staging(payload, ordinal):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor
    from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
    from b12x._lib.utils.device import current_cuda_stream
    from b12x.attention._window_stage import WindowStageKernel

    q = Query(**dict(payload))
    operands = tuple(make_fake_tensor(dtype, tuple(cute.sym_int(32) for _ in range(rank)),
        tuple(cute.sym_int(64) for _ in range(rank - 1)) + (1,), assumed_align=4)
        for rank, dtype in _operands(q.history_dtype))
    with torch.cuda.device(ordinal):
        return compile_kernel(WindowStageKernel(heads=q.heads, dim=q.dim, capacity=q.capacity,
            window=q.window, first_position=q.first_position), *operands, cutlass.Int32(1), current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key("attention.window_stage", 2,
                (q.heads, q.dim, q.capacity, q.window, q.history_dtype, q.first_position)))


@dataclass(frozen=True)
class State:
    query: Query
    program: object


def _run(arguments, state):
    import cutlass
    from cutlass.cute.runtime import from_dlpack
    from b12x._lib.compiler import run_compiled
    from b12x._lib.utils.device import current_cuda_stream

    converted = []
    for tensor, (_, dtype) in zip(arguments, _operands(state.query.history_dtype), strict=True):
        storage = tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor
        value = from_dlpack(storage, assumed_align=4)
        value.element_type = dtype
        converted.append(value.mark_layout_dynamic(leading_dim=tensor.ndim - 1))
    run_compiled(state.program, (*converted, cutlass.Int32(arguments[6].numel()), current_cuda_stream()))


@torch.library.custom_op("flare::window_stage", mutates_args=("output_key", "output_value"))
def _execute(history_key: torch.Tensor, history_value: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
             output_key: torch.Tensor, output_value: torch.Tensor, starts: torch.Tensor, states: torch.Tensor,
             offsets: torch.Tensor, lengths: torch.Tensor, counts: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    _run((history_key, history_value, key, value, output_key, output_value, starts, states, offsets, lengths, counts),
         require_prepared(plan, plan.component_id, key.device))


@_execute.register_fake
def _fake(history_key, history_value, key, value, output_key, output_value, starts, states, offsets, lengths, counts, handle):
    pass


def _row_spans(tensor):
    inner = tensor.element_size() * (1 + sum((n - 1) * s for n, s in zip(tensor.shape[1:], tensor.stride()[1:])))
    base, step = tensor.data_ptr(), tensor.stride(0) * tensor.element_size()
    return tuple((base + i * step, base + i * step + inner) for i in range(tensor.shape[0]))


def _overlap(left, right):
    a, b = sorted(_row_spans(left)), sorted(_row_spans(right))
    if a[-1][1] <= b[0][0] or b[-1][1] <= a[0][0]:
        return False
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] < b[j][1] and b[j][0] < a[i][1]:
            return True
        if a[i][1] <= b[j][0]:
            i += 1
        else:
            j += 1
    return False


class PreparedWindowStaging:
    """Gather paired history and candidates with BF16 selection and FP8 saturation."""

    def __init__(self, history_key, history_value, key, value, output_key, output_value, metadata, *,
                 counts=None, first_position=0, session, name):
        from b12x.attention.paged_kv_write._preparation import _check_layout

        m = metadata
        rows = m.starts.numel()
        if (key.device.type != "cuda" or rows <= 0 or key.ndim != 3 or key.shape[0] <= 0 or key.shape != value.shape
                or key.dtype != torch.bfloat16 or value.dtype != key.dtype
                or history_key.ndim != 4 or history_key.shape != history_value.shape
                or history_key.shape[0] <= 0 or history_key.shape[1] <= 0
                or history_key.shape[2:] != key.shape[1:] or min(key.shape[1:]) <= 0 or key.shape[-1] % 4
                or history_key.dtype not in (torch.float8_e4m3fn, torch.bfloat16)
                or history_value.dtype != history_key.dtype or output_key.shape != output_value.shape
                or output_key.ndim != 4 or output_key.shape[2:] != key.shape[1:]
                or output_key.dtype != torch.float8_e4m3fn or output_value.dtype != output_key.dtype
                or output_key.shape[0] * output_key.shape[1] % rows or output_key.numel() == 0
                or type(first_position) is not int):
            raise ValueError("window staging requires matching positive BF16 candidates, BF16/FP8 history and FP8 output geometry")
        capacity = output_key.shape[0] * output_key.shape[1] // rows
        self.page_shapes = tuple(output_key.shape), tuple(output_value.shape)
        outputs = tuple(t.view(rows, capacity, *key.shape[1:]) for t in (output_key, output_value))
        counts = m.lengths if counts is None else counts
        self.arguments = (history_key, history_value, key, value, *outputs,
                          m.starts, m.state_rows, m.query_start_loc, m.lengths, counts)
        for index, tensor in enumerate(self.arguments):
            rank, _ = _operands(str(history_key.dtype).removeprefix("torch."))[index]
            _check_layout(f"window staging operand {index}", tensor, device=key.device, rank=rank, align=4)
        for tensor, dtype, size in ((m.starts, torch.int64, rows), (m.state_rows, torch.int64, rows),
                (m.query_start_loc, torch.int32, rows + 1), (m.lengths, torch.int32, rows), (counts, torch.int32, rows)):
            if tensor.dtype != dtype or tensor.numel() != size:
                raise ValueError("window staging metadata has incompatible dtype or row count")
        for output in outputs:
            if any(_overlap(output, source) for source in (*self.arguments[:4], *self.arguments[6:])):
                raise ValueError("window staging output must not overlap inputs")
        if _overlap(*outputs):
            raise ValueError("window staging outputs must not overlap")
        q = Query(key.shape[1], key.shape[2], capacity, history_key.shape[1],
                  str(history_key.dtype).removeprefix("torch."), first_position)
        payload = CONTRACT.encode_query(q)
        arguments = self.arguments
        device_index = key.device.index

        def materialize(tactic):
            program = compile_staging(payload, device_index)
            load_programs({"stage": program})
            return attach_programs(State(q, program), program)

        self.plan = native_plan(contract=CONTRACT, query=q, device=key.device, scratch_bytes=lambda t: 0,
            materialize=materialize, compile_jobs=lambda t, d: (CompileJob.create(
                "ignition.components.window_staging:compile_staging", payload, d.ordinal),))
        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: _run(arguments, state))),))
        self._functional_layout = None

    def functional(self, *, session):
        """Allocate staged cache windows inside the compiled forward."""
        del session
        if self._functional_layout is None:
            a = self.arguments
            output_key, output_value = a[4], a[5]
            self._functional_layout = (
                tuple(a[0].shape), tuple(a[1].shape), tuple(a[2].shape), tuple(a[3].shape),
                tuple(output_key.shape), output_key.dtype, tuple(output_value.shape), output_value.dtype,
                tuple(a[6].shape), tuple(a[7].shape), tuple(a[8].shape), tuple(a[9].shape),
            )
            self.history_key, self.history_value = a[:2]
            self.key_shape, self.key_dtype = tuple(a[2].shape), a[2].dtype
            self.value_shape, self.value_dtype = tuple(a[3].shape), a[3].dtype
            self.metadata_shapes = tuple(tuple(t.shape) for t in a[6:10])
            self.rows = a[6].numel()
            self.metadata = None
            self.counts = None
            self.plan_handle = self.plan.handle
            self.arguments = None
        return self

    def __call__(self, key=None, value=None, counts=None, metadata=None):
        if self._functional_layout is None:
            _execute(*self.arguments, self.plan.handle)
            return self.arguments[4], self.arguments[5]
        if key is None or value is None or tuple(key.shape) != self.key_shape or tuple(value.shape) != self.value_shape:
            raise ValueError("functional window staging requires candidate tensors with prepared geometry")
        if metadata is None:
            metadata = self.metadata
        if metadata is None:
            raise ValueError("functional window staging requires runtime metadata")
        starts, states, offsets, lengths = (metadata.starts, metadata.state_rows,
                                            metadata.query_start_loc, metadata.lengths)
        if tuple(tuple(t.shape) for t in (starts, states, offsets, lengths)) != self.metadata_shapes:
            raise ValueError("functional window staging metadata differs from prepared geometry")
        for tensor, dtype in ((starts, torch.int64), (states, torch.int64),
                              (offsets, torch.int32), (lengths, torch.int32)):
            if tensor.device != key.device or tensor.dtype != dtype:
                raise ValueError("functional window staging metadata has incompatible device or dtype")
        (_, _, _, _, out_key_shape, out_dtype, out_value_shape, out_value_dtype,
         _, _, _, _) = self._functional_layout
        rows = self.rows
        capacity = out_key_shape[1]
        output_key = torch.empty(self.page_shapes[0], dtype=out_dtype, device=key.device)
        output_value = torch.empty(self.page_shapes[1], dtype=out_value_dtype, device=value.device)
        selected_counts = lengths if counts is None else counts
        if (selected_counts.device != key.device or selected_counts.dtype != torch.int32
                or selected_counts.numel() != rows):
            raise ValueError("functional window staging counts differ from prepared geometry")
        _execute(self.history_key, self.history_value, key, value,
                 output_key.view(rows, capacity, *key.shape[1:]),
                 output_value.view(rows, capacity, *value.shape[1:]),
                 starts, states, offsets, lengths, selected_counts, self.plan_handle)
        return output_key, output_value
