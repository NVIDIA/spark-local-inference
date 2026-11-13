# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""MiMo V2.6 Pro MOPD and Flash RL text, DFlash and image preparation."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False, core_storage=None,
            ranks=None):
    """Prepare the plan's catalog on process-local GPUs 0 through N-1.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    """
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_token_capacity": plan.speculative_token_capacity, "speculative_graph_budget": speculation.graph_budget or 64, "speculative_fixed_width": speculation.fixed_width}
    config = PreparationConfig(mtp_only=plan.speculation is not None and plan.speculation.mtp_only,
        collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder, prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, roce_ring=plan.roce_ring, roce_chain=plan.roce_chain,
        compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, max_active=plan.concurrency, image_concurrency=plan.image_concurrency, context=plan.context,
        prefill_chunk_size=plan.prefill_chunk,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache,
        dflash=speculation is not None, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="mimo_v2",
    served_model_name="MiMo-V2.6-Pro-MOPD",
    # Recommended sampling, identical for both checkpoints:
    # https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD#4-deployment and
    # https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL#5-deployment.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1),
    prepare=prepare,
    usage=usage,
    # Device bytes count the loaded main-index tensors (decoder layers, embedding, final norm,
    # vocabulary head, and image encoder; not the MTP, audio encoder, or speech embedding tensors)
    # plus the DFlash draft checkpoint.
    checkpoints=(Checkpoint(alias="mimo-v2.6-pro-mopd", repository="XiaomiMiMo/MiMo-V2.6-Pro-MOPD",
        revision="adea8e2c5373181e5a973fa1ecb343cb31af214b", architecture="MiMoV2ForCausalLM",
        stored_bytes=566_027_990_784, device_bytes=568_511_624_448, draft_bytes=5_536_632_576,
        expansion_bytes=14 << 30),
        # Flash RL stores four fused-QKV checkpoint shards, so it prepares two or four ranks. Its
        # expansion covers the package fixed mappings two-rank GB10 exports measured beyond the
        # loaded tensors: 10.71 GiB at eight concurrent requests with 4096-token chunks (88,522 MiB
        # per rank) and 15.66 GiB at sixteen with 8192-token chunks (91,082 MiB per rank).
        Checkpoint(alias="mimo-v2.6-flash-rl", repository="XiaomiMiMo/MiMo-V2.6-Flash-RL",
        revision="f657ba2f7d24c9adc5730675baaa2d34de878a49", architecture="MiMoV2ForCausalLM",
        served_model_name="MiMo-V2.6-Flash-RL",
        stored_bytes=172_923_364_096, device_bytes=174_147_823_616, draft_bytes=2_936_114_304,
        expansion_bytes=16 << 30,
        # Plain dicts: multi-host preparation pickles the checkpoint for peer ranks.
        ranks={2: RankCosts(12_386_304, 737_280, 8 << 30), 4: RankCosts(6_291_456, 368_640, 8 << 30)},
        plain_ranks={2: RankCosts(7_274_496, 737_280, 8 << 30), 4: RankCosts(3_637_248, 368_640, 8 << 30)}),),
    context=1048576, page_tokens=128, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=128, packed_verification=True, graph_budget=64), max_draft_tokens=7, max_images=4, image_bytes=32 << 20,
    verification_rows=0,
    ranks=MappingProxyType({world: RankCosts(33_816_576 // world, 3_276_800 // world, 8 << 30)
                            for world in (2, 4, 8)}),
    plain_ranks=MappingProxyType({world: RankCosts(23_199_744 // world, 3_276_800 // world, 8 << 30)
                                  for world in (2, 4, 8)}),
)
