# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 Small text serving at TP4 across four hosts, and its per-rank memory costs.

The recipe prepares text without speculation or images. The checkpoint's
drafter and vision encoder are not prepared, and plans that request them are
rejected rather than partially prepared.
"""

from types import MappingProxyType

from ignition.recipes.declaration import RankCosts, RankUsage, Recipe

from .checkpoint import CHECKPOINT
from .frontend import SAMPLING_DEFAULTS

# One rank's continuation row: K/V rings of the 40 sliding-window layers
# (1,152 FP8 slots of two 128-wide heads), four short-convolution histories of
# every layer, the last hidden state and the consumed-token count, padded to the
# page stride. A 128-token page holds the FP8 K and V of two heads for each of
# the 8 global layers.
_STATE_ROW_BYTES = 26_935_296
_PAGE_BYTES = 8 * 2 * 128 * 2 * 128
# Rough estimate of what a rank holds beyond weights, state rows and pages:
# operator scratch, graphs, RoCEnante buffers and the CUDA context.
_GRAPH_BYTES = 6 << 30


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    """Prepare the plan's text catalog with one rank on GPU 0 of each of four hosts."""
    if plan.speculation is not None:
        raise ValueError("Inkling 2 preparation has no drafter; speculative decoding is unsupported")
    if plan.prompt_logprobs:
        raise ValueError("Inkling 2 preparation captures no scored prefill graphs")
    if plan.roce_chain:
        raise ValueError("Inkling 2 preparation supports direct and ring RoCE topologies, not chains")
    if ranks is None:
        raise ValueError("Inkling 2 preparation runs one rank on each of four hosts; use --ranks 4")
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        compile_workers=plan.compile_workers, compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        devices=(0,) * plan.world_size, hosts=ranks.hosts, roce_ring=plan.roce_ring, context=plan.context,
        prefill_chunk_size=plan.prefill_chunk, max_active=plan.concurrency,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, decoder_layers=plan.decoder_layers)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="inkling2", served_model_name="Inkling-2-Small", prepare=prepare, usage=usage,
    sampling_defaults=SAMPLING_DEFAULTS, checkpoints=(CHECKPOINT,),
    context=131072, page_tokens=128, concurrency=16, max_concurrency=32, max_draft_tokens=0, prefill_chunk=4096,
    layer_prefix=True,
    ranks=MappingProxyType({4: RankCosts(_STATE_ROW_BYTES, _PAGE_BYTES, _GRAPH_BYTES)}),
)
