# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Trace FlashInfer sampling operators with explicit persistent workspace inputs."""

import torch
from flashinfer import sampling
from flashinfer.utils import device_support_pdl


def probability_filter_workspace_size(rows, vocabulary):
    return sampling._top_p_renorm_workspace_size(rows, vocabulary, True)


def probability_filter_workspace(rows, vocabulary, device):
    """Reserve FlashInfer's deterministic top-p workspace before graph capture."""
    return torch.empty(probability_filter_workspace_size(rows, vocabulary),
                       dtype=torch.uint8, device=device)


@torch.compiler.allow_in_graph
def filter_probabilities_out(logits, top_k, top_p, row_states, softmax_workspace,
                             output, top_p_workspace):
    """Consume prepared scores and write filtered probabilities into caller storage."""
    module = sampling.get_sampling_module()
    torch.ops.flashinfer.top_k_mask_logits_out(logits, logits, top_k, 0, row_states)
    probabilities = module.softmax(softmax_workspace, logits, None, 1.,
                                   device_support_pdl(logits.device))
    torch.ops.flashinfer.top_p_renorm_probs_out(
        probabilities, output, top_p, 0., True, top_p_workspace)


@torch.compiler.allow_in_graph
def sample_probabilities(probabilities, seed, offset, return_valid=False):
    return sampling.sampling_from_probs(probabilities, seed=seed, offset=offset,
                                       deterministic=True, check_nan=False, return_valid=return_valid)


@torch.compiler.allow_in_graph
def verify_chain(draft_probs, draft_ids, target_probs, accepted, emitted, seed, offset):
    # Each row draws from its own seed and offset only, so its tokens do not depend on its batch slot.
    return sampling.chain_speculative_sampling(draft_probs, draft_ids, target_probs,
        accepted, emitted, deterministic=True, seed=seed, offset=offset, row_independent_rng=True)
