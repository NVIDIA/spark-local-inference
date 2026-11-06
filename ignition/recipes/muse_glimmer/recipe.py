# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU Muse Glimmer mixed NVFP4/FP8 text preparation."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("Muse Glimmer preparation supports one local GPU")
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs, compile_workers=plan.compile_workers,
        autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        context=plan.context, max_active=plan.concurrency, prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache)
    return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


RECIPE = Recipe(name="muse_glimmer", served_model_name="Muse-Glimmer-30B", prepare=prepare, usage=usage,
    # https://huggingface.co/nvidia/Muse-Glimmer-30B-NVFP4#usage
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=64),
    checkpoints=(Checkpoint(alias="muse-glimmer-30b-nvfp4", repository="nvidia/Muse-Glimmer-30B-NVFP4",
        revision="47818374517751c48c55cde2621594926b1888b6", architecture="MuseGlimmerForConditionalGeneration",
        stored_bytes=24_669_346_036, device_bytes=24_669_346_036, expansion_bytes=1 << 28),),
    context=131072, page_tokens=128, concurrency=4, max_concurrency=16, max_draft_tokens=0, prefill_chunk=4096,
    ranks=MappingProxyType({1: RankCosts(6_829_312, 6_815_744, 4 << 30)}))
