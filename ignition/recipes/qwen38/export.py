# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Disposable Qwen storage cleared before publishing a runtime package."""

from ignition.preparation.disposable import storage_ranges


def disposable_ranges(*, cache, scratch, workspaces, vision_scratch, decoders, owners, speculation):
    from .mixed_state import MixedScratch

    values = [scratch, workspaces.tensor, vision_scratch]
    for state in cache.values():
        values.extend((state.main_k_cache, state.main_v_cache, state.compressed_k_cache, state.raw_k_ring))
    for owner in (*decoders, *owners):
        if isinstance(owner, MixedScratch):
            values.append(owner)
        for name in ("sampler", "prompt_sampler", "decode_sampler"):
            sampler = getattr(owner, name, None)
            if sampler is not None:
                values.append(sampler.scratch_tensors)
    from ignition.components.dflash2_preparation import DFlash2Preparation

    if isinstance(speculation, DFlash2Preparation):
        values.append(speculation.scratch_tensors())
    elif speculation is not None:
        from .speculative_model import PreparedPrefillAbsorption, PreparedSpeculativeRound

        values.append(speculation.mtp_scratch)
        values.extend((speculation.sampling_buffers.chunks, speculation.feedback_hidden, speculation.prefill_residual,
            speculation.verification_residual, speculation.raw_backup))
        for operation in speculation.owners:
            if isinstance(operation, PreparedPrefillAbsorption):
                values.extend((operation.target_residual, operation.previous, operation.previous_rope, operation.before))
            elif isinstance(operation, PreparedSpeculativeRound):
                values.extend(getattr(operation, name) for name in (
                    "target_residual", "rope_delta", "before", "pending", "previous", "previous_rope", "active", "drafts",
                    "draft_probabilities", "draft_valid", "copy_mask", "raw_backup", "seed", "offset",
                    "base_offset", "sampled"))
                target, proposals, verify, _, _ = operation.operations
                for distribution in (target, *proposals):
                    values.extend(getattr(distribution, name) for name in (
                        "scores", "probabilities", "maximum", "valid",
                        "packed_top_k", "packed_top_p"))
                for proposal in proposals:
                    values.extend((proposal.proposal.candidates, proposal.proposal.selected))
                values.extend(getattr(verify, name) for name in (
                    "safe_draft_probs", "safe_target_probs", "safe_ids", "seed_bits", "offset",
                    "independent_accepted", "emitted_drafts", "output"))
    return storage_ranges(values)
