# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer 30B single-GPU NVFP4 text preparation with the z-lab DFlash2 drafter."""

from pathlib import Path
from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from .bootstrap import PreparationConfig, prepare as prepare_model

    if ranks is not None or plan.world_size != 1:
        raise ValueError("Muse Glimmer preparation runs on exactly one GPU")
    drafting = {}
    if plan.speculation is not None:
        from huggingface_hub import snapshot_download
        from .draft_config import REPOSITORY, REVISION

        draft = snapshot_download(REPOSITORY, revision=REVISION, cache_dir=plan.hub_cache,
                                  allow_patterns=("config.json", "model.safetensors"))
        drafting = dict(draft_snapshot=Path(draft),
            speculative_widths=plan.speculation.widths,
            speculative_fixed_width=plan.speculation.fixed_width,
            speculative_token_capacity=plan.speculative_token_capacity,
            speculative_graph_budget=plan.speculation.graph_budget or 64)
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs, compile_workers=plan.compile_workers,
        autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        device=0, context=plan.context, max_active=plan.concurrency,
        prefill_chunk_size=plan.prefill_chunk, snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache, language_model_only=plan.language_model_only,
        image_concurrency=plan.image_concurrency, dflash=plan.speculation is not None, **drafting)
    return prepare_model(snapshot, config, progress=progress, package_storage=package_storage)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


# Export VRAM inventory (gb202-tp1-202611061529): graph scratch pools 1,901.14 MiB + capture pool 1,126.00 MiB +
# disposable vision arena 837.56 MiB + other owned tensors 459.43 MiB + graph I/O 79.45 MiB.
_GRAPH_BYTES = 4_617_488_302

RECIPE = Recipe(name="muse_glimmer_nvfp4", served_model_name="Muse-Glimmer-30B",
    # Sampling defaults carried from Muse Spark; the model card gives none.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1),
    prepare=prepare, usage=usage,
    checkpoints=(Checkpoint(alias="muse-glimmer-30b-nvfp4", repository="original-el8/Muse-Glimmer-30B-NVFP4",
        revision="b47e1e9f77d9a86e18cda5e318c68c632a7fbe59", architecture="MuseGlimmerForConditionalGeneration",
        # `stored_bytes` is the index `total_size`, which covers the separate input-scale shard.
        # `device_bytes` holds every stored text and vision tensor (23,380,399,360 payload bytes) plus the drafter;
        # the export inventory's 28,885.60 MiB of resident weights exceeds that by the expanded block scales.
        stored_bytes=23_380_728_424, device_bytes=28_492_369_664, expansion_bytes=1_796_377_242,
        draft_bytes=5_544_328_424),),
    context=131072, page_tokens=128, concurrency=4, max_concurrency=4, max_draft_tokens=15, prefill_chunk=4096,
    # The packed round catalog is bounded by `graph_budget` (64 rounds: 29 + 21 + 14 across batches 4, 2, 1); the
    # rebuild manifest requires it declared.
    speculation=Speculation(draft_tokens=15, token_capacity=64, packed_verification=True, graph_budget=64),
    max_images=4, image_bytes=32 << 20, language_model_only=True,
    verification_rows=0,
    # 39 FP8 sliding rings of 2048 × 2 × 128 per key/value, 13 paged full layers at 128-token pages, and with
    # drafting five FP8 draft rings of 2048 × 8 × 128: `continuation_layout` / `page_bytes` at TP1.
    ranks=MappingProxyType({1: RankCosts(62_914_560, 851_968, _GRAPH_BYTES)}),
    plain_ranks=MappingProxyType({1: RankCosts(41_943_040, 851_968, _GRAPH_BYTES)}))
