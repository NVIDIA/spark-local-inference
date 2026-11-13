# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export independent CUDA ranks for the native threaded executor regression."""

from pathlib import Path
import argparse

import torch

from fixture_frontend import server_config
from signing import publisher
from ignition.preparation.execution import GraphSignature, PreparedExecution
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.core_storage import CoreStorage
from ignition.preparation.runtime_core import export_model
from tests.components.test_packed_speculation_cuda import _fixture


@torch.no_grad()
def prepare(root, rank):
    root.mkdir(parents=True)
    torch.cuda.set_device(rank)
    with ModuleImages() as images, CoreStorage(rank=rank, world=4, device=rank) as storage:
        fixture = _fixture(3)
        fixture.logits.fill_(-torch.inf)
        for position in range(8):
            fixture.logits[:, position, 5 + position] = 8.
        reset_state = torch.ones(1, dtype=torch.int32, device="cuda")
        confidence = torch.full((4, 3), .75, device="cuda")
        initial_state = {name: value.clone() for name, value in vars(fixture.state).items()
                         if isinstance(value, torch.Tensor)}
        initial_rng = {name: value.clone() for name, value in vars(fixture.rng).items()
                       if isinstance(value, torch.Tensor)}
        initial_offset = fixture.owner.offset.clone()
        initial_generation = fixture.owner.generation.clone()
        terminal = torch.zeros(1, dtype=torch.int32, device="cuda")

        def forward():
            reset = reset_state[0] != 0
            for name, value in initial_state.items():
                target = getattr(fixture.state, name)
                target.copy_(torch.where(reset, value, target))
            for name, value in initial_rng.items():
                target = getattr(fixture.rng, name)
                target.copy_(torch.where(reset, value, target))
            fixture.owner.offset.copy_(torch.where(reset, initial_offset, fixture.owner.offset))
            fixture.owner.generation.copy_(torch.where(reset, initial_generation, fixture.owner.generation))
            fixture.state.max_new_tokens.copy_(torch.where(terminal[0] != 0,
                fixture.state.generated_tokens + 1, fixture.state.max_new_tokens))
            fixture.confidence.values.copy_(confidence)
            fixture.operation()

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, pool=storage.graph_pool.id):
            forward()
        graph.instantiate()
        try:
            model = PreparedExecution(device=rank, max_active=4, graphs={"round": graph},
                signatures={"round": GraphSignature("speculative", 4, 8,
                    ("speculative_prefix_limits", "test_confidence", "live_rows", "reset_state",
                     "terminal_after_round", "row_slot_ids", "row_generations"),
                    speculation_enabled=True, packed_token_capacity=12, draft_tokens=3,
                    confidence_width=3)},
                buffers=dict(frame_index=fixture.owner.frame, submission_sequence=fixture.owner.sequence,
                    completion_records=fixture.owner.completion.records,
                    completion_speculative_survival=fixture.owner.completion.speculative_survival,
                    proposal_confidence=fixture.confidence_output,
                    speculative_prefix_limits=fixture.prefix_limits,
                    test_confidence=confidence, live_rows=fixture.inputs.live_rows,
                    reset_state=reset_state, terminal_after_round=terminal,
                    row_slot_ids=fixture.inputs.row_slot_ids, row_generations=fixture.inputs.row_generations,
                    pending_inputs=fixture.target.candidates),
                resources=(fixture, forward, initial_state, initial_rng, storage))
            model.server_config = server_config(root)
            model.core_data = None
            export_model(model, images, storage, root / "core", options={}, publisher=publisher())
        finally:
            graph.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--rank", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    prepare(args.directory, args.rank)
