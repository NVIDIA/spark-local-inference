# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM continuation transitions with b12x rank agreement before mutation."""

from ignition.components.collectives import PreparedAllReduce
from ignition.components.maintenance import PreparedMaintenance as StateMaintenance
from ignition.components.state_transfer import PreparedStateTransfer


class PreparedMaintenance:
    def __init__(self, mode, requests, pool, commands, inputs, records, frame, sequence, *,
                 runtime, coordinator, session, name, capacity=4, admission_buffers=None, draft_tables=None,
                 page_tokens=256):
        self.pool, self.commands, self.tables, self.draft_tables = pool, commands, inputs.page_tables, draft_tables
        self.requests, self.state_rows = requests, inputs.request_state_rows
        self.page_tokens = page_tokens
        transfer = self._transfer()
        if mode == "snapshot":
            slots, epochs, count = inputs.row_slot_ids, inputs.row_generations, inputs.live_rows
        elif admission_buffers is not None:
            slots, epochs, count = (admission_buffers.admission_slots, admission_buffers.admission_generations,
                                   admission_buffers.admission_count)
        else:
            raise ValueError("GLM init and restore require request admission buffers")
        self.state = StateMaintenance(mode, requests, transfer, slots, epochs, count, records, frame, sequence,
            world_size=1 if runtime is None else runtime.world_size, capacity=capacity,
            admission_buffers=admission_buffers)
        self.state.prepare()
        self.validation = PreparedAllReduce(self.state.validation_input, self.state.validation_sum,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.validation")
        self.state.functional()
        self.validation.functional(session=session)

    def _transfer(self):
        return PreparedStateTransfer(self.pool, self.commands, self.tables, pages=self.pool.partial_pages(),
            page_tokens=self.page_tokens, state_rows=self.state_rows, requests=self.requests,
            trailing_pages=() if self.draft_tables is None else self.pool.draft_pages(), trailing_tables=self.draft_tables,
            excluded_regions=tuple(r.name for r in self.pool.layout.regions
                                   if self.draft_tables is None and r.name.startswith("draft.")))

    def rebind_pages(self, pages):
        """Copy partial pages of the continuation pool's current page pools."""
        self.state.transfer = self._transfer()
        self.state.transfer.prepare()

    def __call__(self):
        validation, admission_errors = self.state.validate()
        self.state.apply(self.validation(validation), admission_errors=admission_errors)
