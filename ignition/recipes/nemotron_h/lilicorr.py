# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-lattice correlations and predecessor-conditioned draft logits."""

import torch
import torch.nn.functional as F

from ignition.components.embedding import PreparedEmbedding
from ignition.preparation.examples import declaration_tensor
from ignition.components.linear import QuantizedProjection, UnquantizedProjection


def _projection(source, weights, output, *, session, name, workspace):
    weight, bias = weights
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name, bias=bias)
    return QuantizedProjection(source, weight, output, session=session, name=name,
        workspace=workspace, activation_mode="auto", bias=bias)


def _output_width(weight):
    return weight.shape[0] if isinstance(weight, torch.Tensor) else weight.out_features


class _LatticeLayer:
    def __init__(self, weights, source, *, batch, heads, eps, session, name, workspace, examples=None):
        self.weights, self.batch, self.heads, self.eps = weights, batch, heads, eps
        rows, hidden = source.shape
        self.normalized = declaration_tensor(examples, "nemotron.lattice.layer.normalized", source.shape, torch.bfloat16, source.device)
        self.projected = declaration_tensor(examples, "nemotron.lattice.layer.qkv", (rows, 3 * hidden), torch.bfloat16, source.device)
        self.attended = declaration_tensor(examples, "nemotron.lattice.layer.attended", source.shape, torch.bfloat16, source.device)
        self.qkv = _projection(self.normalized, weights["qkv"], self.projected,
            session=session, name=name + ".qkv", workspace=workspace)
        self.output = _projection(self.attended, weights["output"], declaration_tensor(examples, "nemotron.lattice.layer.output", source.shape, torch.bfloat16, source.device),
            session=session, name=name + ".output", workspace=workspace)
        up, down = weights["mlp"]
        self.mlp_up = _projection(self.normalized, (up[0], up[1]),
            declaration_tensor(examples, "nemotron.lattice.layer.mlp_up", (rows, _output_width(up[0])), torch.bfloat16, source.device),
            session=session, name=name + ".mlp.up", workspace=workspace)
        self.mlp_down = _projection(declaration_tensor(examples, "nemotron.lattice.layer.mlp_intermediate",
            (rows, _output_width(up[0])), torch.bfloat16, source.device), (down[0], down[1]),
            declaration_tensor(examples, "nemotron.lattice.layer.mlp_down", source.shape, torch.bfloat16, source.device),
            session=session, name=name + ".mlp.down", workspace=workspace)

    def functional(self, *, session):
        for operation in (self.qkv, self.output, self.mlp_up, self.mlp_down):
            operation.functional(session=session)
        self.normalized = self.projected = self.attended = None
        return self

    def __call__(self, source, bias):
        b, h, hidden = self.batch, self.heads, source.shape[-1]
        normalized = F.rms_norm(source, (hidden,), self.weights["attn_norm"], self.eps)
        qkv = self.qkv(normalized).view(b, -1, 3, h, hidden // h)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape_as(source)
        residual = source + self.output(attended)
        normalized = F.rms_norm(residual, (hidden,), self.weights["mlp_norm"], self.eps)
        return residual + self.mlp_down(F.silu(self.mlp_up(normalized)))


class PreparedLattice:
    """Compute learned first-slot and adjacent-slot scores from raw head log probabilities."""

    def __init__(self, weights, config, token_embeddings, log_probabilities, pass_hidden, anchor_hidden, *,
                 session, name, workspace, examples=None):
        c = config.text["dflash_config"]
        self.weights, self.eps = weights, config.text["rms_norm_eps"]
        self.vector_eps, self.logit_scale = c["lilicorr_vector_eps"], c["lilicorr_logit_scale"]
        b, slots, k = log_probabilities.shape
        hidden = c["lilicorr_hidden_size"]
        if not 1 <= slots < config.block_size or k != c["lilicorr_candidate_topk"]:
            raise ValueError("LiLiCorr lattice dimensions exceed the trained block or candidate count")
        self.batch, self.slots, self.top_k, self.hidden = b, slots, k, hidden
        rows = b * slots * k
        def matrix(role, n, width):
            return declaration_tensor(examples, "nemotron.lattice." + role,
                                      (n, width), torch.bfloat16, log_probabilities.device)
        self.states, self.features, self.anchor = matrix("states", rows, hidden), matrix("features", rows, 5), matrix("anchor", b, hidden)
        self.token = _projection(token_embeddings.view(rows, -1), weights["token_proj"], matrix("token", rows, hidden),
            session=session, name=name + ".token", workspace=workspace)
        self.passed = _projection(pass_hidden.reshape(b * slots, -1), weights["pass_hidden_proj"], matrix("passed", b * slots, hidden),
            session=session, name=name + ".pass_hidden", workspace=workspace)
        self.context = _projection(anchor_hidden, weights["context_proj"], matrix("context", b, hidden),
            session=session, name=name + ".context", workspace=workspace)
        up, down = weights["feature_mlp"]
        self.feature_up = _projection(self.features, (up[0], up[1]), matrix("feature_up", rows, hidden),
            session=session, name=name + ".features.up", workspace=workspace)
        self.feature_down = _projection(matrix("feature_intermediate", rows, _output_width(up[0])),
            (down[0], down[1]), matrix("feature_down", rows, hidden),
            session=session, name=name + ".features.down", workspace=workspace)
        self.layers = tuple(_LatticeLayer(w, self.states, batch=b, heads=c["lilicorr_num_heads"], eps=self.eps,
            session=session, name=f"{name}.layer{i}", workspace=workspace, examples=examples)
            for i, w in enumerate(weights["layers"]))
        factors, bias = weights["factor"]
        self.cross_input = matrix("cross_input", rows, hidden)
        self.factor_self = _projection(self.states, (factors[0], bias), matrix("factor_self", rows, hidden),
            session=session, name=name + ".factor_self", workspace=workspace)
        self.factor_anchor = _projection(self.anchor, (factors[1], None), matrix("factor_anchor", b, hidden),
            session=session, name=name + ".factor_anchor", workspace=workspace)
        self.factor_cross = _projection(self.cross_input, (factors[2], None), matrix("factor_cross", rows, hidden),
            session=session, name=name + ".factor_cross", workspace=workspace)
        self.factor_hidden = matrix("factor_hidden", rows, hidden)
        self.edges = _projection(self.factor_hidden, weights["edges"], matrix("edges", rows, 2 * c["lilicorr_factor_dim"]),
            session=session, name=name + ".edges", workspace=workspace)
        self.anchor_out = _projection(self.anchor, weights["anchor_out_head"], matrix("anchor_out", b, c["lilicorr_factor_dim"]),
            session=session, name=name + ".anchor_out", workspace=workspace)
        self.device, self.block_size = log_probabilities.device, config.block_size

    def functional(self, *, session):
        for operation in (self.token, self.passed, self.context, self.feature_up, self.feature_down,
                          self.factor_self, self.factor_anchor, self.factor_cross, self.edges, self.anchor_out):
            operation.functional(session=session)
        for layer in self.layers:
            layer.functional(session=session)
        self.states = self.features = self.anchor = self.cross_input = self.factor_hidden = None
        return self

    def __call__(self, token_embeddings, log_probabilities, pass_hidden, anchor_hidden, anchor_valid):
        b, s, k, h = self.batch, self.slots, self.top_k, self.hidden
        logp = log_probabilities.float()
        rank = torch.arange(k, dtype=torch.float32, device=logp.device)
        rank_fraction, is_top1 = rank / (k - 1), (rank == 0).float()
        slot_ids = torch.arange(s, device=logp.device).repeat_interleave(k)
        relative = slot_ids[:, None] - slot_ids[None, :]
        relative_bias = self.weights["relative_slot_bias"][:, relative + self.block_size - 1]
        attention_bias = (relative_bias + (relative == 0).to(relative_bias.dtype)[None]
                          * self.weights["same_slot_bias"][:, None, None])[None].contiguous()
        features = torch.stack((logp, logp.exp(), logp - logp.amax(-1, keepdim=True),
            rank_fraction.expand_as(logp), is_top1.expand_as(logp)), dim=-1).bfloat16()
        weight, bias = self.weights["feature_norm"]
        features = F.layer_norm(features, (5,), weight, bias, 1e-5).view(-1, 5)
        token = self.token(token_embeddings.view(b * s * k, -1)).view(b, s, k, h)
        passed = self.passed(pass_hidden.reshape(b * s, -1)).view(b, s, 1, h)
        states = token + passed
        states = states + self.feature_down(F.silu(self.feature_up(features))).view(b, s, k, h)
        states = states + self.weights["slot_embedding"][:, 0, :s]
        states = states + self.weights["rank_embedding"][:, 0]
        states = states.flatten(0, 2)
        for layer in self.layers:
            states = layer(states, attention_bias)
        states = F.rms_norm(states, (h,), self.weights["output_norm"], self.eps)
        anchor = self.context(anchor_hidden) * anchor_valid[:, None].to(anchor_hidden.dtype)
        anchor = F.rms_norm(anchor, (h,), self.weights["anchor_norm"], self.eps)
        x = states.view(b, s, k, h)
        cross_input = (x * anchor[:, None, None]).flatten(0, 2)
        pre = self.factor_self(states).view(b, s, k, h) + self.factor_anchor(anchor)[:, None, None]
        pre = pre + self.factor_cross(cross_input).view(b, s, k, h)
        edges = self.edges(F.silu(pre).flatten(0, 2)).view(b, s, k, 2, -1)
        out_vector, in_vector = F.normalize(edges, dim=-1, eps=self.vector_eps).unbind(-2)
        anchor_out = F.normalize(self.anchor_out(anchor), dim=-1, eps=self.vector_eps)
        first = (anchor_out[:, None] * in_vector[:, 0]).sum(-1).float() * self.logit_scale
        pairs = torch.matmul(out_vector[:, :-1], in_vector[:, 1:].transpose(-1, -2)).float() * self.logit_scale
        return torch.cat((first[:, None, None].expand(-1, 1, k, -1), pairs), dim=1)


class PreparedLatticeSelector:
    """Restrict each proposal to the lattice row selected by its realized predecessor."""

    def __init__(self, weights, config, embedding, hidden, logits, anchor, *, session, name, workspace, examples=None):
        b, block, width = hidden.shape
        self.top_k, self.slots = config.text["dflash_config"]["lilicorr_candidate_topk"], block - 1
        self.ids = declaration_tensor(examples, "nemotron.lattice.selector.ids", (b, self.slots, self.top_k), torch.int64, hidden.device)
        self.logp = declaration_tensor(examples, "nemotron.lattice.selector.logp", self.ids.shape, torch.float32, hidden.device)
        self.pass_hidden = declaration_tensor(examples, "nemotron.lattice.selector.passed", (b, self.slots, width), torch.bfloat16, hidden.device)
        self.embedded = declaration_tensor(examples, "nemotron.lattice.selector.embedded", (b, self.slots, self.top_k, width), torch.bfloat16, hidden.device)
        self.lookup = PreparedEmbedding(embedding, self.ids.flatten(), self.embedded.view(-1, width),
            session=session, name=name + ".embedding")
        self.lattice = PreparedLattice(weights, config, self.embedded, self.logp, self.pass_hidden, anchor,
            session=session, name=name + ".lattice", workspace=workspace, examples=examples)
        self.batch, self.device = b, hidden.device

    def functional(self, *, session):
        self.lookup.functional(session=session, dynamic_inputs=True)
        self.lattice.functional(session=session)
        self.ids = self.logp = self.pass_hidden = self.embedded = None
        return self

    def prepare(self, hidden, logits, anchor, anchor_valid):
        values = logits[:, 1:].float()
        unary, ids = values.topk(self.top_k, dim=-1, sorted=True)
        logp = unary - values.logsumexp(-1, keepdim=True)
        pass_hidden = hidden[:, 1:]
        embedded = self.lookup(ids.flatten()).view(self.batch, self.slots, self.top_k, -1)
        scores = self.lattice(embedded, logp, pass_hidden, anchor, anchor_valid)
        return scores

    def prepare_shortlists(self, logits, *, features=None):
        unary, ids = logits[:, 1:].float().topk(self.top_k, dim=-1, sorted=True)
        prefix_shape = (self.batch, 1, self.top_k)
        unary = torch.cat((torch.full(prefix_shape, -torch.inf, dtype=unary.dtype, device=unary.device), unary), 1)
        ids = torch.cat((torch.zeros(prefix_shape, dtype=ids.dtype, device=ids.device), ids), 1)
        return unary, ids

    def apply_shortlist(self, logits, predecessor, position, selector_features, unary, ids):
        scores = selector_features
        step = position - 1
        previous = (ids[:, position - 1] == predecessor[:, None]).int().argmax(-1) if step else torch.zeros_like(predecessor)
        selected = scores[torch.arange(self.batch, device=logits.device), step, previous]
        return torch.full_like(logits, -torch.inf).scatter(1, ids[:, position], selected)
