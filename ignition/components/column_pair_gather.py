# SPDX-License-Identifier: Apache-2.0

"""Gather BF16 column partitions together with bit-preserved FP32 columns."""

import torch

from ignition.components.collectives import PreparedAllGather


def _validate_storage(first, second, **destinations):
    # Storage identity and byte offsets also describe aliases during metadata-only tracing.
    fields = (("first", first), ("second", second), *destinations.items())
    ranges = [(name, value.untyped_storage()._cdata, value.storage_offset() * value.element_size(),
               (value.storage_offset() + value.numel()) * value.element_size()) for name, value in fields]
    for index, (name, storage, begin, end) in enumerate(ranges):
        for other, other_storage, other_begin, other_end in ranges[max(index + 1, 2):]:
            if storage == other_storage and max(begin, other_begin) < min(end, other_end):
                raise ValueError(f"paired-column live storage overlaps: {name} and {other}")


def validate_columns(first, second, output_first, output_second, *, world, columns, max_rows):
    if type(world) is not int or world < 2 or type(max_rows) is not int or max_rows <= 0:
        raise ValueError("paired column gathering requires multiple ranks and positive row capacity")
    if any(value.ndim != 2 for value in (first, second, output_first, output_second)):
        raise ValueError("paired column gathering requires matrices")
    rows, first_width = first.shape
    second_width = second.shape[1]
    if (min(rows, first_width, second_width) <= 0 or second.shape[0] != rows
            or output_first.shape != (rows, world * first_width)
            or output_second.shape != (rows, world * second_width)
            or columns != first_width + 2 * second_width or columns % 8):
        raise ValueError("paired column dimensions differ from their exchange channel")
    if first.dtype != torch.bfloat16 or output_first.dtype != torch.bfloat16:
        raise TypeError("first paired columns require BF16 storage")
    if second.dtype != torch.float32 or output_second.dtype != torch.float32:
        raise TypeError("second paired columns require FP32 storage")
    if any(value.device != first.device or not value.is_contiguous()
           for value in (first, second, output_first, output_second)):
        raise ValueError("paired columns require contiguous storage on one device")
    _validate_storage(first, second, output_first=output_first, output_second=output_second)


class PreparedColumnPairGather:
    """Pack two column shards into bounded caller-owned send/receive buffers.

    FP32 columns use two BF16 transport lanes without numerical conversion.
    Each chunk is consumed before another overwrites the shared transport buffers.
    Outputs and transport buffers must be disjoint from each other and both inputs.
    """

    def __init__(self, first, second, output_first, output_second, *, runtime,
                 session, coordinator, name, buffers, allocator=None):
        world = runtime.world_size
        columns = runtime.heads_per_rank * runtime.query_head_dim
        validate_columns(first, second, output_first, output_second, world=world,
                         columns=columns, max_rows=runtime.max_batch_size)
        self.first, self.second = first, second
        self.output_first, self.output_second = output_first, output_second
        self.world, self.columns = world, columns
        self.first_width, self.second_width = first.shape[1], second.shape[1]
        chunk_rows = min(first.shape[0], runtime.max_batch_size)

        def allocate(field, width):
            value = buffers.get(field)
            if value is not None and (value.ndim != 2 or value.shape[1] != width
                    or value.dtype != torch.bfloat16 or value.device != first.device or not value.is_contiguous()):
                raise ValueError("shared paired-column buffers have incompatible geometry")
            if value is None or value.shape[0] < chunk_rows:
                shape = (chunk_rows, width)
                value = (torch.zeros(shape, dtype=torch.bfloat16, device=first.device) if allocator is None
                         else allocator.zeros(shape, torch.bfloat16))
                buffers[field] = value
            if (value.shape[1:] != (width,) or value.dtype != torch.bfloat16
                    or value.device != first.device or not value.is_contiguous()):
                raise ValueError("shared paired-column buffers have incompatible geometry")
            return value[:chunk_rows]

        self.send = allocate("send", columns)
        self.received = allocate("received", world * columns)
        _validate_storage(first, second, output_first=output_first, output_second=output_second,
                          send=self.send, received=self.received)
        self.bounds = tuple((begin, min(begin + chunk_rows, first.shape[0]))
                            for begin in range(0, first.shape[0], chunk_rows))
        self.parts = tuple(PreparedAllGather(self.send[:end - begin], self.received[:end - begin],
            runtime=runtime, session=session, coordinator=coordinator, name=f"{name}.{index}")
            for index, (begin, end) in enumerate(self.bounds))

    def functional(self, *, session):
        self.rows = self.first.shape[0]
        self.chunk_rows = self.send.shape[0]
        for part in self.parts:
            part.functional(session=session)
        self.first = self.second = self.output_first = self.output_second = self.send = self.received = None
        return self

    def __call__(self, first=None, second=None):
        local = self.first is None
        if local:
            if first is None or second is None:
                raise ValueError("functional paired gather requires both column shards")
            output_first = torch.empty((self.rows, self.world * self.first_width), dtype=first.dtype, device=first.device)
            output_second = torch.empty((self.rows, self.world * self.second_width), dtype=second.dtype, device=second.device)
        else:
            first, second = self.first, self.second
            output_first, output_second = self.output_first, self.output_second
        for (begin, end), gather in zip(self.bounds, self.parts, strict=True):
            count = end - begin
            send = (torch.empty((count, self.columns), dtype=torch.bfloat16, device=first.device)
                    if local else self.send[:count])
            send[:, :self.first_width].copy_(first[begin:end])
            send[:, self.first_width:].copy_(second[begin:end].view(torch.bfloat16))
            received = gather(send).view(count, self.world, self.columns)
            output_first[begin:end].view(count, self.world, self.first_width).copy_(received[:, :, :self.first_width])
            output_second[begin:end].view(count, self.world, self.second_width).view(torch.bfloat16).copy_(
                received[:, :, self.first_width:])
        return output_first, output_second
