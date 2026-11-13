# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heterogeneous latent ReLU-squared experts with full-width shared experts."""

import torch

from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mlp import PreparedUngatedMLP
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.examples import declaration_tensor


class PreparedLatentMoE:
    def __init__(self, weights, source, config, index, *, session, name, workspace,
                 num_valid_tokens=None, examples=None):
        c, tokens = config.text, source.shape[0]
        block = c["block_configs"][index]

        def empty(role, width, dtype=torch.bfloat16):
            return declaration_tensor(examples, "nemotron.target.moe." + role, (tokens, width), dtype, source.device)

        logits = empty("logits", block["n_routed_experts"], torch.float32)
        routes = RouterScratch.allocate(tokens, source.device, experts=block["n_routed_experts"],
            top_k=block["num_experts_per_tok"])
        self.router_projection = UnquantizedProjection(source, weights["router"], logits,
            session=session, name=name + ".router_logits")
        self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
            renormalize=True, routed_scaling_factor=c["routed_scaling_factor"],
            num_valid_tokens=num_valid_tokens, session=session, name=name + ".route")
        latent, expert_output, output = (empty("latent", c["moe_latent_size"]),
            empty("expert_output", c["moe_latent_size"]), empty("routed_output", config.hidden))
        self.down = QuantizedProjection(source, weights["latent_down"], latent,
            session=session, name=name + ".latent_down", activation_mode="auto", workspace=workspace,
            examples=examples, example_role="nemotron.target.moe.latent_down")
        self.experts = PreparedRoutedExperts(latent, weights["experts"], routes.ids, routes.weights, expert_output,
            session=session, name=name + ".experts", workspace=workspace)
        self.up = QuantizedProjection(expert_output, weights["latent_up"], output,
            session=session, name=name + ".latent_up", activation_mode="auto", workspace=workspace,
            examples=examples, example_role="nemotron.target.moe.latent_up")
        shared_width = c["moe_shared_expert_intermediate_size"]
        self.shared = PreparedUngatedMLP(source, weights["shared_up"], weights["shared_down"],
            empty("shared_projected", shared_width), empty("shared_intermediate", shared_width), empty("shared_output", config.hidden), activation="relu2",
            session=session, name=name + ".shared", workspace=workspace)
        self.concurrent_shared = None
        if tokens <= 64 and isinstance(workspace, ScratchArena) and workspace.overlap:
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.router_projection.plan, self.router.plan, self.down.plan,
                                self.experts.plan, self.up.plan),
                session=session, workspace=workspace, defer_launch=1 < tokens <= 4)

    def functional(self, *, session):
        self.router_projection = self.router_projection.functional(session=session)
        self.router.functional(session=session)
        self.down.functional(session=session)
        self.experts.functional(session=session)
        self.up.functional(session=session)
        self.shared.functional(session=session)
        if self.concurrent_shared is not None:
            self.concurrent_shared.functional(session=session)
        return self

    def __call__(self, source, metadata):
        pending = self.concurrent_shared.start(source) if self.concurrent_shared is not None else None
        logits = self.router_projection(source)
        routes = self.router(logits, num_valid_tokens=metadata.live_counts[:1])
        latent = self.down(source)
        routed = self.up(self.experts(latent, routes.ids, routes.weights))
        shared = (self.shared(source) if self.concurrent_shared is None
                  else self.concurrent_shared.finish(pending, routed))
        return routed + shared
