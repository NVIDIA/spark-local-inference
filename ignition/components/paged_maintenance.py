# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collective admission and complete continuation snapshots."""

from ignition.components.collectives import PreparedAllReduce
from ignition.components.maintenance import PreparedMaintenance as StateMaintenance
from ignition.components.state_transfer import PreparedStateTransfer

PAGE_TOKENS = 128


class PreparedMaintenance:
    def __init__(
        self,
        mode,
        requests,
        pool,
        commands,
        inputs,
        records,
        frame,
        sequence,
        *,
        runtime,
        coordinator,
        session,
        name,
        capacity,
        admission_buffers,
        draft_state=True,
        page_tokens=PAGE_TOKENS,
        buffers=None,
        trailing_pages=(),
        trailing_tables=None,
        local_metadata=False,
    ):
        self.pool, self.commands, self.inputs, self.requests = pool, commands, inputs, requests
        self.page_tokens = page_tokens
        self.trailing_pages, self.trailing_tables = trailing_pages, trailing_tables
        self.excluded_regions = (
            () if draft_state else tuple(r.name for r in pool.layout.regions if r.name.startswith("draft"))
        )
        if mode == "snapshot":
            slots, epochs, count = inputs.row_slot_ids, inputs.row_generations, inputs.live_rows
        else:
            slots, epochs, count = (
                admission_buffers.admission_slots,
                admission_buffers.admission_generations,
                admission_buffers.admission_count,
            )
        self.state = StateMaintenance(
            mode,
            requests,
            self._transfer(),
            slots,
            epochs,
            count,
            records,
            frame,
            sequence,
            world_size=1 if runtime is None else runtime.world_size,
            capacity=capacity,
            admission_buffers=admission_buffers,
            buffers=buffers,
        )
        self.state.prepare()
        self.validation = PreparedAllReduce(
            self.state.validation_input,
            self.state.validation_sum,
            runtime=runtime,
            coordinator=coordinator,
            session=session,
            name=name + ".validation",
        )
        self.local_metadata = local_metadata
        if local_metadata:
            self.validation.functional(session=session)
            self.state.functional()

    def _transfer(self):
        return PreparedStateTransfer(
            self.pool,
            self.commands,
            self.inputs.page_tables,
            pages=self.pool.partial_pages(),
            page_tokens=self.page_tokens,
            state_rows=self.inputs.request_state_rows,
            requests=self.requests,
            excluded_regions=self.excluded_regions,
            trailing_pages=self.trailing_pages,
            trailing_tables=self.trailing_tables,
        )

    def __call__(self):
        if self.local_metadata:
            validation, admission_errors = self.state.validate()
            totals = self.validation(validation)
            self.state.apply(totals, admission_errors=admission_errors)
        else:
            self.validation(self.state.validate())
            self.state.apply()
