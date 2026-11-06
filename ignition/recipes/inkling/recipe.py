# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inkling-Small NVFP4 text preparation on one SM103 GPU."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if plan.world_size != 1 or ranks is not None:
        raise ValueError("Inkling preparation requires one local GPU")
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs, compile_workers=plan.compile_workers,
        autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        context=plan.context, max_active=plan.concurrency,
        prefill_chunk_size=plan.prefill_chunk, snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, stream_overlap=plan.stream_overlap)
    return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)


def usage(prepared):
    return (RankUsage(prepared.memory_report["pages"], prepared.memory_report["free_device_bytes"]),)


RECIPE = Recipe(name="inkling", served_model_name="Inkling-Small-NVFP4",
    # Recommended sampling: https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/inkling/SKILL.md.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1), prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(alias="inkling-small-nvfp4", repository="thinkingmachines/Inkling-Small-NVFP4",
        revision="b6a99534467840620d411e4cd4ad5819b2610d9c", architecture="InklingForConditionalGeneration",
        # Upper bound: includes the unused multimodal and MTP tensors.
        stored_bytes=170_733_074_592, device_bytes=170_733_074_592, expansion_bytes=1 << 30),),
    context=262144, page_tokens=128, concurrency=4, max_concurrency=4, max_draft_tokens=0, prefill_chunk=4096,
    ranks=MappingProxyType({1: RankCosts(82_837_504, 3_670_016, 8 << 30)}))
