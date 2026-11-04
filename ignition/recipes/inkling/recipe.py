# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inkling-Small BF16 text preparation with learned relative attention."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None:
        raise ValueError("Inkling-Small preparation requires eight GPUs on one host")
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs, compile_workers=plan.compile_workers,
        autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)), context=plan.context, max_active=plan.concurrency,
        prefill_chunk_size=plan.prefill_chunk, snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"])
                 for rank in prepared.memory_report.get("ranks", (prepared.memory_report,)))


RECIPE = Recipe(name="inkling", served_model_name="Inkling-Small",
    # Publisher guidance: thinking-machines-lab/tinker-cookbook, skills/inkling/SKILL.md.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1), prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(alias="inkling-small", repository="thinkingmachines/Inkling-Small",
        revision="8cc5877b44d343f88b92086aa1fb72897950f06a", architecture="InklingForConditionalGeneration",
        stored_bytes=531_912_898_740, device_bytes=527_948_358_944, expansion_bytes=1 << 30),),
    context=262144, page_tokens=128, concurrency=4, max_concurrency=4, max_draft_tokens=0, prefill_chunk=4096,
    ranks=MappingProxyType({8: RankCosts(14_024_704, 458_752, 8 << 30)}))
