# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Choose prepared attention plans by replaying complete catalog layers."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import torch
from b12x.preparation import ContextCandidate, FinalistPolicy, FrozenMapping
from b12x.gemm import blockscaled
from b12x.quantization import mla_norm_mxfp8

from ignition.preparation.capture import capture_forwards
from ignition.progress import phase

from .contextual_tuning import CatalogLayerFixture


_ROLES = ("qkv_a", "norm", "q_b")
_POLICIES = {
    "qkv_a": FinalistPolicy(family_fields=("split_k_slices", "programmatic_serialization", "evict_first"),
                             per_family=1, max_total=16),
    "norm": FinalistPolicy(family_fields=("programmatic_serialization",), per_family=1, max_total=2),
    "q_b": FinalistPolicy(family_fields=("split_k_slices", "programmatic_serialization", "evict_first"),
                           per_family=1, max_total=16),
}


def contextual_summary(report):
    """Report the measured contexts and selected projection tuple for each shape."""
    groups = []
    for group in report["groups"]:
        winner = group["winner"]
        groups.append({"contexts": group["contexts"], "target_rows": group["target_rows"],
            "configs": group["winner_configs"], "layer_us": group["latencies_us"][winner],
            "sum_us": group["total_us"][winner], "standalone_sum_us": group["standalone_total_us"],
            "cache_policy": group["cache_policy"], "rounds": group["rounds"], "samples": group["samples"],
            "lock_gpu_clocks": group["lock_gpu_clocks"]})
    return "Selected joint plans: " + json.dumps(groups, sort_keys=True, separators=(",", ":"))


def _role_plans(layer):
    attention = layer.attention
    if attention.norm_quant is None:
        raise ValueError("contextual layer requires fused MXFP8 query normalization")
    return (attention.projections["qkv_a"].plan, attention.norm_quant.plan,
            attention.projections["q_b"].plan)


def catalog_contexts(forwards, signatures):
    """Visit every prepared text shape and each shared proposal decoder once."""
    seen_decoders = set()
    for name, signature in signatures.items():
        if signature.kind not in ("prefill", "scored_prefill", "decode", "speculative"):
            continue
        forward = forwards[name]
        while hasattr(forward, "forward"):
            forward = forward.forward
        if hasattr(forward, "target"):
            decoders = ((name + ".target", forward.target.decoder),
                        (forward.source.name + ".draft", forward.source.draft))
        else:
            decoders = ((name, forward.decoder),
                        (name + ".draft", getattr(forward, "draft", None)))
        for context_name, decoder in decoders:
            if decoder is None or id(decoder) in seen_decoders:
                continue
            seen_decoders.add(id(decoder))
            shapes = set()
            found = False
            for index, layer in enumerate(decoder.layers):
                attention = layer.attention
                if (attention.norm_quant is None or layer.engram is not None
                        or layer.boundary is not None or attention.global_kv is not None):
                    continue
                shape = (layer.metadata.positions.numel(), layer.metadata.starts.numel(),
                         layer.metadata.max_query_tokens, attention.mode, attention.ratio,
                         layer.carry, layer.post is not None, bool(attention.selectors),
                         attention.concurrent_query is not None)
                if shape in shapes:
                    continue
                shapes.add(shape)
                found = True
                yield context_name, signature, decoder, index
            if not found:
                raise ValueError(f"joint tuning has no restorable attention context for {context_name}")


class CatalogContextualSelection:
    """Race projection tuples in every prepared text execution shape.

    Matching operator declarations share one choice, scored by the sum of the
    complete-layer contexts in which they execute. Distinct declarations have
    independent races. Inputs and trial graphs are released between contexts.
    """

    def __init__(self, forwards, signatures, *, coordinator, collective,
                 device, graph_pool, cache_root, compile_decoder, warmup_storage,
                 progress=None):
        if coordinator is None or len(coordinator.ranks) < 2:
            raise ValueError("joint tuning requires the model's tensor-parallel coordinator")
        self.fixtures = tuple(CatalogLayerFixture(*context)
                              for context in catalog_contexts(forwards, signatures))
        if not self.fixtures:
            raise ValueError("joint tuning requires prepared text graphs")
        self.layers = tuple(fixture.layer for fixture in self.fixtures)
        groups = []
        for fixture in self.fixtures:
            plans = _role_plans(fixture.layer)
            for group in groups:
                if all(first.same_declaration_except_override(second) for first, second in
                       zip(_role_plans(group[0].layer), plans, strict=True)):
                    group.append(fixture)
                    break
            else:
                groups.append([fixture])
        self.groups = tuple(tuple(group) for group in groups)
        self.coordinator, self.collective = coordinator, collective
        self.device, self.graph_pool = device, graph_pool
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.compile_decoder, self.warmup_storage = compile_decoder, warmup_storage
        self.progress = progress
        self._free_after = {}
        self._race_stage = None

    def _names(self, requests):
        by_plan = {}
        for name, request in requests.items():
            by_plan.setdefault(id(request.plan), []).append(name)
        names = []
        for layer in self.layers:
            roles = _role_plans(layer)
            if any(len(by_plan.get(id(plan), ())) != 1 for plan in roles):
                raise ValueError("contextual layer plan must name one recorded request")
            names.append(tuple(by_plan[id(plan)][0] for plan in roles))
        return tuple(names)

    def finalist_policies(self, requests):
        names = self._names(requests)
        return {name: _POLICIES[role] for group in names
                for name, role in zip(group, _ROLES, strict=True)}

    def _configs(self, names, finalists):
        configs = {}
        for role_index, role in enumerate(_ROLES):
            groups = [tuple(finalists.get(names[index][role_index], ()))
                      for index in range(len(self.layers))]
            if any(not group for group in groups):
                raise RuntimeError(f"contextual {role} has no qualified legal finalists")
            reference = self.layers[0].attention.projections[role].plan if role != "norm" else self.layers[0].attention.norm_quant.plan
            def payload(group):
                return {json.dumps(reference.contract.config_payload(item.selection.config).to_dict(),
                                   sort_keys=True) for item in group}
            if any(payload(group) != payload(groups[0]) for group in groups[1:]):
                raise ValueError(f"representative layers have different {role} finalist catalogs")
            configs[role] = tuple(item.selection.config for item in groups[0])
        return configs

    @staticmethod
    def _override_plan(role, original, config):
        factory = mla_norm_mxfp8.plan if role == "norm" else blockscaled.plan
        alternative = factory(original.query, invocation=original.invocation, override=config)
        if not original.same_declaration_except_override(alternative):
            raise ValueError("contextual override changes the operator declaration")
        return alternative

    def _prepare_alternatives(self, session, requests, names, configs):
        plans, obligations = {}, []
        for group_index, group in enumerate(names):
            for role, name in zip(_ROLES, group, strict=True):
                original = requests[name]
                variants = []
                for index, config in enumerate(configs[role]):
                    alternative = self._override_plan(role, original.plan, config)
                    obligations.append(replace(original,
                        name=f"{name}.context.{group_index}.{index}", plan=alternative,
                        benchmark_call=None, finalist_policy=None))
                    variants.append(alternative)
                plans[(group_index, role)] = tuple(variants)
        result = session.prepare_context_plans(obligations)
        result.close()
        if any(plan.selection is None or plan.selection.source != "override"
               for variants in plans.values() for plan in variants):
            raise RuntimeError("contextual alternatives must use prepared explicit overrides")
        return plans

    def _tuple_name(self, index, configs, selection):
        plans = _role_plans(self.layers[0])
        payload = [{"component": plan.component_id,
                    "query": FrozenMapping(plan.contract.encode_query(plan.query)).to_dict(),
                    "config": plan.contract.config_payload(configs[role][offset]).to_dict()}
                   for plan, role, offset in zip(plans, _ROLES, selection, strict=True)]
        encoded = json.dumps({"operators": payload,
            "target_rows": self.fixtures[0].tokens,
            "contexts": tuple(fixture.label for fixture in self.fixtures)},
            sort_keys=True, separators=(",", ":")).encode()
        return f"context-{index:02d}-{sha256(encoded).hexdigest()[:12]}"

    def _capture(self, session, functions, signatures, *, label, before_run=None,
                 compile_group=None):
        stream = torch.cuda.Stream(device=self.device)
        return capture_forwards(functions, signatures, session=session, device=self.device,
            stream=stream, pool=self.graph_pool, collective=self.collective,
            coordinator=self.coordinator, compile_decoder=self.compile_decoder,
            progress=self.progress, label=label, cache_root=self.cache_root, report_catalog=False,
            warmup_storage=self.warmup_storage, capture_names=tuple(functions),
            before_run=before_run, compile_group=compile_group)

    def _context_exchange(self, step, payload):
        import torch.distributed as dist

        send = dict(payload, step=step)
        peers = [None] * len(self.coordinator.ranks)
        dist.all_gather_object(peers, send, group=self.coordinator.group)
        if any(peer.get("step") != step for peer in peers):
            raise ValueError("tensor-parallel context ranks reached different race steps")
        if step == "launch" and payload.get("repetition") == 0 and self._race_stage is not None:
            self._race_stage.update(advance=len(payload["indices"]))
        return tuple(peers)

    def _race_layer(self, session, fixture, fixture_index, configs, plans, selections):
        functions, signatures, members = {}, {}, {}
        for index, selection in enumerate(selections):
            name = self._tuple_name(index, configs, selection) + f"-context{fixture_index}"
            variant_plans = tuple(plans[(fixture_index, role)][offset]
                                  for role, offset in zip(_ROLES, selection, strict=True))
            layer = prepared_layer_variant(self.layers[fixture_index],
                qkv_plan=variant_plans[0], norm_plan=variant_plans[1], query_plan=variant_plans[2])
            functions[name] = fixture.trial_forward(layer)
            signatures[name] = fixture.trial_signature
            members[name] = (variant_plans, layer)
        graphs, compiled, candidates = {}, {}, ()
        try:
            captured, artifacts = self._capture(session, functions, signatures,
                label=f"Capture joint tuning {fixture.label}",
                before_run=lambda name: fixture.restore_tune(members[name][1]),
                compile_group=f"{fixture.label}-plan-family" if self.compile_decoder else None)
            graphs.update(captured)
            compiled.update(artifacts)
            observations = {name + "-observe": fixture.observation_forward(layer)
                            for name, (_, layer) in members.items()}
            captured, artifacts = self._capture(session, observations,
                {name: fixture.trial_signature for name in observations},
                label=f"Capture joint tuning observations {fixture.label}",
                before_run=lambda name: fixture.restore_tune(members[name.removesuffix("-observe")][1]),
                compile_group=f"{fixture.label}-observation-family" if self.compile_decoder else None)
            graphs.update(captured)
            compiled.update(artifacts)
            after_free = torch.cuda.mem_get_info(self.device)[0]
            if after_free < max(1 << 30, session.headroom_bytes):
                raise MemoryError("contextual graph catalog exceeded the reserved capture headroom")
            candidates = tuple(ContextCandidate(name=name, plans=members[name][0],
                call=fixture.prepared_call(name, graphs[name], compiled.get(name), members[name][1],
                    graphs[name + "-observe"], compiled.get(name + "-observe"))) for name in functions)
            with phase(self.progress, f"Compare complete layer {fixture.label}",
                       total=len(candidates) * session.rounds) as stage:
                self._race_stage = stage
                try:
                    result = session.race_contexts(candidates,
                        collective_exchange=self._context_exchange,
                        collective_ranks=self.coordinator.ranks, max_candidates=len(candidates))
                finally:
                    self._race_stage = None
            return result, after_free
        finally:
            for graph in graphs.values():
                graph.reset()
            candidates = ()
            graphs.clear()
            compiled.clear()
            functions.clear()
            members.clear()

    def _validate_winner(self, session, fixture, fixture_index, plans, selection):
        chosen = tuple(plans[(fixture_index, role)][offset]
                       for role, offset in zip(_ROLES, selection, strict=True))
        layer = prepared_layer_variant(self.layers[fixture_index], qkv_plan=chosen[0],
            norm_plan=chosen[1], query_plan=chosen[2])
        name = f"{fixture.label}-heldout"
        observation = name + "-observe"
        graphs, compiled = self._capture(session,
            {name: fixture.trial_forward(layer), observation: fixture.observation_forward(layer)},
            {name: fixture.trial_signature, observation: fixture.trial_signature},
            label="Validate contextual layer choice",
            before_run=lambda name: fixture.restore_tune(layer))
        try:
            fixture.restore_holdout(layer)
            graphs[name].replay()
            torch.cuda.synchronize(self.device)
            return fixture.validate_trial(layer, graphs[observation])
        finally:
            for graph in graphs.values():
                graph.reset()
            compiled.clear()

    def _select_group(self, session, requests, finalists):
        names = self._names(requests)
        configs = self._configs(names, finalists)
        selections = tuple(product(*(range(len(configs[role])) for role in _ROLES)))
        if not selections:
            raise RuntimeError("contextual tuning has no legal operator tuples")
        plans = self._prepare_alternatives(session, requests, names, configs)
        session.preview_context_bindings()
        measurements, race_results = {}, {}
        for fixture_index, fixture in enumerate(self.fixtures):
            self.collective = fixture.layer.attention_reduce.runtime
            fixture.initialize()
            try:
                result, free_after = self._race_layer(session, fixture, fixture_index, configs,
                    plans, selections)
                race_results[fixture_index] = result
                measurements[fixture_index] = result.latencies_us
                self._free_after[fixture.label] = free_after
            finally:
                fixture.close()
        totals = {self._tuple_name(index, configs, selection): sum(
            measurements[fixture_index][self._tuple_name(index, configs, selection)
                + f"-context{fixture_index}"]
            for fixture_index, fixture in enumerate(self.fixtures))
            for index, selection in enumerate(selections)}
        standalone = tuple(next((index for index, config in enumerate(configs[role])
                                 if config == plan.selection.config), None)
                           for role, plan in zip(_ROLES, _role_plans(self.layers[0]), strict=True))
        if None in standalone or standalone not in selections:
            raise RuntimeError("contextual finalists omit the ordinary standalone plan tuple")
        standalone_index = selections.index(standalone)
        standalone_name = self._tuple_name(standalone_index, configs, standalone)
        winner_index = min(range(len(selections)),
            key=lambda index: (totals[self._tuple_name(index, configs, selections[index])], index))
        winning_selection = selections[winner_index]
        validation = {}
        for fixture_index, fixture in enumerate(self.fixtures):
            self.collective = fixture.layer.attention_reduce.runtime
            fixture.initialize()
            try:
                validation[fixture.label] = self._validate_winner(session, fixture, fixture_index,
                    plans, winning_selection)
            finally:
                fixture.close()
        choices, obligations = {}, []
        shared = {id(requests[group[role_index]].plan): plans[(fixture_index, role)][offset]
                  for fixture_index, group in enumerate(names)
                  for role_index, (role, offset) in enumerate(zip(_ROLES, winning_selection, strict=True))}
        for role, offset in zip(_ROLES, winning_selection, strict=True):
            original = requests[names[0][_ROLES.index(role)]].plan
            config = configs[role][offset]
            for name, request in requests.items():
                if not original.same_declaration_except_override(request.plan):
                    continue
                if name not in session.context_plan_names:
                    raise ValueError(f"matching contextual request {name!r} lacks a binding owner")
                if id(request.plan) in shared:
                    choices[name] = shared[id(request.plan)]
                    continue
                alternative = self._override_plan(role, request.plan, config)
                obligations.append(replace(request, name=f"{name}.context.winner",
                    plan=alternative, benchmark_call=None, finalist_policy=None))
                choices[name] = alternative
                shared[id(request.plan)] = alternative
        if obligations:
            result = session.prepare_context_plans(obligations)
            result.close()
        if any(plan.selection is None or plan.selection.source != "override" for plan in choices.values()):
            raise RuntimeError("contextual winner was not prepared as an explicit legal override")
        selected = set(choices.values())
        unused = {plan for variants in plans.values() for plan in variants if plan not in selected}
        if unused:
            session.release_many(unused)
        report = {"objective": "equal-weight sum of complete layer GPU replay, each timed as TP max-rank",
            "tuples": len(selections), "winner": self._tuple_name(winner_index, configs, winning_selection),
            "winner_configs": {role: plan.contract.config_payload(configs[role][offset]).to_dict()
                               for plan, role, offset in zip(_role_plans(self.layers[0]), _ROLES,
                                                              winning_selection, strict=True)},
            "standalone_tuple": standalone_name,
            "standalone_total_us": totals[standalone_name],
            "target_rows": self.fixtures[0].tokens,
            "contexts": tuple(fixture.label for fixture in self.fixtures),
            "cache_policy": {fixture.label: race_results[index].cache_policy
                             for index, fixture in enumerate(self.fixtures)},
            "rounds": session.rounds, "samples": session.samples,
            "lock_gpu_clocks": session.lock_gpu_clocks,
            "capture_free_after_bytes": dict(self._free_after),
            "latencies_us": {name: {fixture.label: measurements[index][
                                        name + f'-context{index}']
                                    for index, fixture in enumerate(self.fixtures)}
                             for name in totals},
            "total_us": totals, "heldout": validation}
        return choices, report

    def __call__(self, session, requests, finalists):
        fixtures, layers = self.fixtures, self.layers
        choices, reports = {}, []
        try:
            for group in self.groups:
                self.fixtures = group
                self.layers = tuple(fixture.layer for fixture in group)
                selected, report = self._select_group(session, requests, finalists)
                if choices.keys() & selected.keys():
                    raise ValueError("joint tuning groups overlap operator declarations")
                choices.update(selected)
                reports.append(report)
        finally:
            self.fixtures, self.layers = fixtures, layers
            for fixture in fixtures:
                fixture.release_examples()
        return choices, {"groups": reports, "contexts": len(fixtures)}


def prepared_layer_variant(layer, *, qkv_plan, norm_plan, query_plan):
    """Bind prepared attention plans while sharing immutable weights and cache owners."""
    attention = layer.attention
    if attention.norm_quant is None or set(("qkv_a", "q_b")) - attention.projections.keys():
        raise ValueError("contextual layer requires fused query normalization and two MXFP8 projections")
    owners = ((attention.projections["qkv_a"], qkv_plan),
              (attention.norm_quant, norm_plan),
              (attention.projections["q_b"], query_plan))
    for original, alternative in owners:
        if (not original.plan.same_declaration_except_override(alternative)
                or alternative.selection is None or alternative.prepared is None):
            raise ValueError("contextual layer alternative is not a prepared operator declaration")

    candidate = copy(layer)
    candidate.attention = copy(attention)
    candidate.attention.projections = dict(attention.projections)
    query_input = copy(attention.projections["qkv_a"])
    query_input.plan = qkv_plan
    query_input._finalize_functional()
    norm = copy(attention.norm_quant)
    norm.plan = norm_plan
    query_output = copy(attention.projections["q_b"])
    query_output.plan = query_plan
    query_output._finalize_functional()
    candidate.attention.projections["qkv_a"] = query_input
    candidate.attention.projections["q_b"] = query_output
    candidate.attention.norm_quant = norm
    chain = attention.concurrent_query
    if chain is not None:
        if chain.encoded_layouts is None or len(chain.plan_owners) != 1:
            raise ValueError("contextual query overlap requires one functional encoded projection")
        candidate.attention.concurrent_query = copy(chain)
        candidate.attention.concurrent_query.plan_owners = (query_output,)
        candidate.attention.concurrent_query._finalize_functional()
    return candidate
