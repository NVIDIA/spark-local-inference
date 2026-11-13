# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unfused target distribution implementation used as the compiled score oracle."""

import torch
from ignition.components.speculation import _geometry, _metadata, _tensor, _live, _sliced
from flashinfer import sampling
from flashinfer.utils import device_support_pdl
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.scratch import bind_scratch


@torch.compiler.allow_in_graph
def filter_probabilities(logits, top_k, top_p, row_states, softmax_workspace):
    """Reference filtering with separate input and output tensors."""
    module = sampling.get_sampling_module()
    masked = module.top_k_mask_logits(logits, top_k, 0, row_states)
    probabilities = module.softmax(softmax_workspace, masked, None, 1.,
                                   device_support_pdl(logits.device))
    return sampling.top_p_renorm_probs(probabilities, top_p, is_deterministic=True)

class UnfusedSpeculativeDistributions:
    """Conditional request filtering using the ordinary FlashInfer policy.

    Row j includes candidate IDs before j+prefix_offset in its penalty counts.
    The committed histogram already contains the incoming pending token.
    Optional physical-slot grammar masks describe each conditional prefix.
    """

    @staticmethod
    def scratch_layout(batch, width, vocab):
        return dict(scores=((batch, width, vocab), torch.float32),
            probabilities=((batch, width, vocab), torch.float32),
            counts=((batch, width, vocab), torch.int32),
            maximum=((batch, width), torch.float32), valid=((batch, width), torch.bool),
            packed_top_k=((batch * width,), torch.int32), packed_top_p=((batch * width,), torch.float32))

    def __init__(self, logits, requests, slots, generations, live_rows, candidate_prefix_ids,
                 *, prefix_offset=0, grammar_masks=None, grammar_mask_count=None, workspace=None):
        batch, width, vocab = logits.shape
        _geometry(batch, width)
        device = logits.device
        if logits.dtype not in (torch.float32, torch.bfloat16) or not logits.is_contiguous() or vocab != requests.vocab_size:
            raise ValueError("conditional logits require contiguous FP32/BF16 [B,W,vocabulary]")
        _metadata(requests, slots, generations, live_rows, batch, device)
        if candidate_prefix_ids.ndim != 2 or candidate_prefix_ids.shape[0] != batch:
            raise ValueError("candidate_prefix_ids requires [B,K]")
        _tensor(candidate_prefix_ids, candidate_prefix_ids.shape, torch.int64, device, "candidate_prefix_ids")
        if type(prefix_offset) is not int or prefix_offset < 0 or width - 1 + prefix_offset > candidate_prefix_ids.shape[1]:
            raise ValueError("prefix_offset must keep every conditional prefix within candidate IDs")
        self.logits, self.state, self.slots, self.generations, self.live_rows = logits, requests, slots, generations, live_rows
        self.candidates, self.prefix_offset = candidate_prefix_ids, prefix_offset
        if (grammar_masks is None) != (grammar_mask_count is None):
            raise ValueError("conditional grammar masks and counts must be provided together")
        if grammar_masks is not None:
            _tensor(grammar_masks, (requests.capacity, 8, (vocab + 31) // 32), torch.int32, device, "grammar_masks")
            _tensor(grammar_mask_count, (requests.capacity,), torch.int32, device, "grammar_mask_count")
            if width + prefix_offset > 8:
                raise ValueError("conditional grammar prefix exceeds eight mask rows")
        self.grammar_masks, self.grammar_mask_count = grammar_masks, grammar_mask_count
        self.rows = torch.arange(batch, device=device)
        self.columns = torch.arange(width, device=device)
        self.vocab_ids = torch.arange(vocab, device=device)
        for tensor in (self.rows, self.columns, self.vocab_ids):
            record_numeric_tensor(tensor)
        for name, tensor in bind_scratch(self.scratch_layout(batch, width, vocab),
                                        device=device, workspace=workspace).items():
            setattr(self, name, tensor)
        from ignition.components.probability_normalization import PreparedProbabilityNormalization
        self.normalization = PreparedProbabilityNormalization(self.probabilities, self.valid)
        self._prepared = False

    def prepare_scores(self):
        state, columns = self.state, self.columns
        safe, live = _live(state, self.slots, self.generations, self.live_rows, self.rows)
        self.counts.copy_(state.histogram[safe, None, :].expand_as(self.counts))
        prefix = columns + self.prefix_offset
        prefix_valid = torch.ones_like(self.valid)
        for index in range(columns.numel() - 1 + self.prefix_offset):
            ids = self.candidates[:, index]
            used = index < prefix
            valid = (ids >= 0) & (ids < state.vocab_size)
            prefix_valid &= valid[:, None] | ~used[None, :]
            indices = ids.clamp(0, state.vocab_size - 1)[:, None, None].expand(-1, columns.numel(), 1)
            additions = (valid[:, None] & used[None, :]).to(torch.int32).unsqueeze(-1)
            self.counts.scatter_add_(-1, indices, additions)
        params = state.parameters[safe]
        scores = self.logits.float()
        penalty = params[:, None, 3:4]
        adjusted = torch.where(scores > 0, scores / penalty, scores * penalty)
        scores = torch.where(self.counts > 0, adjusted, scores)
        scores = scores - (self.counts > 0) * params[:, None, 4:5] - self.counts * params[:, None, 5:6]
        temperature = params[:, None, :1]
        scores = scores / torch.where(temperature > 0, temperature, 1)
        if self.grammar_masks is None:
            words = state.grammar_mask[safe][:, None, self.vocab_ids // 32]
        else:
            words = self.grammar_masks[safe][:, prefix][:, :, self.vocab_ids // 32]
        allowed = ((words >> (self.vocab_ids % 32)) & 1).bool() | ~state.grammar_enabled[safe, None, None].bool()
        self.scores.copy_(torch.where(allowed, scores, -torch.inf))
        self.maximum.copy_(_sliced(self.scores).amax(-1).amax(-1))
        self.valid.copy_(live[:, None] & prefix_valid & torch.isfinite(self.maximum))
        if self.grammar_masks is not None:
            count = self.grammar_mask_count[safe, None]
            self.valid.logical_and_(~state.grammar_enabled[safe, None].bool()
                                    | ((prefix[None, :] < count) & (count <= 8)))
        elif columns.numel() > 1:
            self.valid.logical_and_(~state.grammar_enabled[safe, None].bool() | (prefix[None, :] == 0))
        self.packed_top_k.copy_(torch.where(state.top_k[safe] > 0, state.top_k[safe], state.vocab_size)[:, None].expand_as(self.valid).reshape(-1))
        self.packed_top_p.copy_(params[:, 1:2].expand_as(self.valid).reshape(-1))
        return safe, params

    def prepare(self):
        if self.logits.device.type != "cuda":
            raise ValueError("FlashInfer probability filtering requires CUDA")
        from flashinfer import sampling
        self._flashinfer = sampling
        self.library_scratch = (
            sampling._get_cache_buf(f"top_k_mask_logits_row_states_{self.logits.device}", 1024 * 1024, self.logits.device, zero_init=True),
            sampling._get_cache_buf("softmax_workspace", 1024 * 1024, self.logits.device),
        )
        self._prepared = True
        self()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("probability filtering requires prepare() before capture")
        _, params = self.prepare_scores()
        dummy = torch.where(self.vocab_ids == 0, 0., -torch.inf)
        self.scores.copy_(torch.where(self.valid[:, :, None], self.scores, dummy))
        flat = self.scores.view(-1, self.state.vocab_size)
        probabilities = filter_probabilities(flat, self.packed_top_k, self.packed_top_p,
            *self.library_scratch).view_as(self.probabilities)
        return self.normalization(probabilities, params)
