# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared layer variants retain catalog storage and isolate candidate plans."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from b12x.preparation import Finalist, FrozenMapping, PreparedCall, Selection

from ignition.recipes.deepseek41.contextual_selection import (
    CatalogContextualSelection, catalog_contexts, contextual_summary, prepared_layer_variant,
)


def test_context_discovery_covers_catalog_shapes_and_shared_draft_decoders():
    from ignition.preparation.execution import GraphSignature
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    def decoder(batch, tokens, width, mode):
        def layer(ratio=0, carry=False):
            return SimpleNamespace(metadata=SimpleNamespace(positions=torch.empty(tokens),
                starts=torch.empty(batch), max_query_tokens=width), engram=None, boundary=None,
                carry=carry, post=object(), attention=SimpleNamespace(norm_quant=object(),
                    global_kv=None, mode=mode, ratio=ratio, selectors=(), concurrent_query=None))
        return SimpleNamespace(layers=(layer(), layer(), layer(1), layer(1, True)))

    config = PreparationConfig(speculative_widths=tuple(range(1, 9)))
    forwards, signatures, expected = {}, {}, {}
    drafts = {batch: SimpleNamespace(name=f"proposal{batch}",
        draft=decoder(batch, batch * 7, 7, "draft")) for batch in config.batch_capacities}
    for batch, tokens in config.packed_speculative_shapes:
        name = f"packed{batch}x{tokens}"
        target = decoder(batch, tokens, 8, "verify")
        forwards[name] = SimpleNamespace(target=SimpleNamespace(decoder=target), source=drafts[batch])
        signatures[name] = GraphSignature("speculative", batch, 8, packed_token_capacity=tokens)
        expected[name + ".target"] = target
        expected[f"proposal{batch}.draft"] = drafts[batch].draft
    for kind, shapes in (("prefill", [(1, tokens, tokens) for tokens in config.prefill_capacities]),
                         ("scored_prefill", [(1, 4096, 4096)]),
                         ("decode", [(batch, batch, 1) for batch in config.batch_capacities]),
                         ("speculative", [(batch, batch, 1) for batch in config.batch_capacities])):
        for batch, tokens, width in shapes:
            name = f"{kind}{batch}x{tokens}"
            model = decoder(batch, tokens, width, "prefill" if "prefill" in kind else "decode")
            forward = SimpleNamespace(decoder=model)
            forwards[name] = SimpleNamespace(forward=forward) if kind == "scored_prefill" else forward
            signatures[name] = GraphSignature(kind, batch, width,
                inputs=("prompt_target_tokens",) if kind == "scored_prefill" else ())
            expected[name] = model
    contexts = list(catalog_contexts(forwards, signatures))
    assert {name for name, _, _, _ in contexts} == set(expected)
    assert len(contexts) == len(expected) * 3
    for name, _, model, index in contexts:
        assert model is expected[name]
        assert index in (0, 2, 3)


@pytest.mark.parametrize("kind,batch,width,packed", [("prefill", 1, 4096, 0),
    ("decode", 32, 1, 0), ("speculative", 4, 6, 13)])
def test_joint_tuning_is_explicit_and_accepts_every_text_shape(monkeypatch, kind, batch, width, packed):
    from b12x.quantization import mla_norm_mxfp8
    from ignition.preparation.execution import GraphSignature
    from ignition.recipes.deepseek41.bootstrap import _can_race_contextual_plans

    config = SimpleNamespace(joint_tuning=False, autotune_effort="default")
    signatures = {"graph": GraphSignature(kind, batch, width, packed_token_capacity=packed)}
    device = torch.device("cuda:0")
    monkeypatch.setattr(mla_norm_mxfp8, "is_supported", lambda device: True)
    assert not _can_race_contextual_plans(config, signatures, device)
    config.joint_tuning = True
    assert _can_race_contextual_plans(config, signatures, device)
    config.autotune_effort = "off"
    with pytest.raises(ValueError, match="autotune-effort"):
        _can_race_contextual_plans(config, signatures, device)
    config.autotune_effort = "default"
    monkeypatch.setattr(mla_norm_mxfp8, "is_supported", lambda device: False)
    with pytest.raises(ValueError, match="selected device"):
        _can_race_contextual_plans(config, signatures, device)


class Plan:
    def __init__(self, size=0):
        self.size = size
        self.selection = object()
        self.prepared = object()
        self.handle = id(self)

    def same_declaration_except_override(self, other):
        return isinstance(other, Plan)

    def memory_requirements(self):
        return SimpleNamespace(scratch_nbytes=self.size)


def catalog_layer():
    from ignition.components.linear import QuantizedProjection
    from ignition.components.projection_overlap import ConcurrentProjectionChain

    class Projection(SimpleNamespace):
        _finalize_functional = QuantizedProjection._finalize_functional

    class Chain(SimpleNamespace):
        _finalize_functional = ConcurrentProjectionChain._finalize_functional

    qkv, query = [Projection(plan=Plan(), output=None, weight=object(), quantizer_plan=None) for _ in range(2)]
    norm = SimpleNamespace(plan=Plan())
    chain = Chain(plan_owners=(query,), encoded_layouts=(((8, 16), torch.float8_e4m3fn),),
        encoded_projection=None, projections=None, parallel_plans=None, arena=None, workspace=None,
        outputs=None)
    chain._finalize_functional()
    attention = SimpleNamespace(projections={"qkv_a": qkv, "q_b": query}, norm_quant=norm, concurrent_query=chain)
    return SimpleNamespace(attention=attention, cache=torch.empty(8, 16, dtype=torch.bfloat16))


def test_variant_rebinds_functional_plan_owners_and_preserves_cache_and_weights():
    layer = catalog_layer()
    alternatives = (Plan(), Plan(), Plan(64))
    variant = prepared_layer_variant(layer, qkv_plan=alternatives[0], norm_plan=alternatives[1],
                                     query_plan=alternatives[2])
    assert variant is not layer and variant.attention is not layer.attention
    assert variant.attention.projections is not layer.attention.projections
    assert variant.attention.projections["qkv_a"].plan is alternatives[0]
    assert variant.attention.norm_quant.plan is alternatives[1]
    assert variant.attention.projections["q_b"].plan is alternatives[2]
    chain = variant.attention.concurrent_query
    assert chain.plan_owners == (variant.attention.projections["q_b"],)
    assert chain.dense_handle == alternatives[2].handle and chain.workspace_nbytes == 64
    assert chain.side_plans == (alternatives[2],)
    assert chain.workspace is None
    assert variant.cache is layer.cache
    assert variant.attention.projections["q_b"].weight is layer.attention.projections["q_b"].weight
    assert layer.attention.projections["q_b"].plan is not alternatives[2]
    assert layer.attention.concurrent_query.workspace_nbytes == 0


def test_variant_rejects_unprepared_plan_and_bound_overlap_storage():
    layer = catalog_layer()
    qkv, norm, query = Plan(), Plan(), Plan(64)
    query.prepared = None
    with pytest.raises(ValueError, match="prepared operator"):
        prepared_layer_variant(layer, qkv_plan=qkv, norm_plan=norm, query_plan=query)
    query.prepared = object()
    layer.attention.concurrent_query.encoded_layouts = None
    with pytest.raises(ValueError, match="functional encoded projection"):
        prepared_layer_variant(layer, qkv_plan=qkv, norm_plan=norm, query_plan=query)


def test_tuple_identity_serializes_nested_b12x_query_metadata():
    selector = object.__new__(CatalogContextualSelection)
    contract = SimpleNamespace(
        encode_query=lambda query: {"codegen": FrozenMapping({"target": "sm120"}), "rows": 8},
        config_payload=lambda config: FrozenMapping({"split_k_slices": config}))
    plans = [SimpleNamespace(component_id=name, contract=contract, query=object())
             for name in ("qkv", "norm", "query")]
    attention = SimpleNamespace(projections={"qkv_a": SimpleNamespace(plan=plans[0]),
                                              "q_b": SimpleNamespace(plan=plans[2])},
                                norm_quant=SimpleNamespace(plan=plans[1]))
    selector.layers = (SimpleNamespace(attention=attention, metadata=SimpleNamespace(positions=torch.empty(8))),)
    selector.fixtures = (SimpleNamespace(target_signature=SimpleNamespace(token_capacity=8), layer_index=0, label="packed.layer0", tokens=8),
                         SimpleNamespace(layer_index=37, label="packed.layer37", tokens=8))
    configs = {"qkv_a": (1,), "norm": (2,), "q_b": (3,)}
    identity = selector._tuple_name(0, configs, (0, 0, 0))
    assert identity == selector._tuple_name(0, configs, (0, 0, 0))
    assert identity.startswith("context-00-")


def test_public_b12x_declarations_supply_selector_families_and_override_requests():
    from b12x.gemm import blockscaled
    from b12x.gemm._tuning import DenseGemmConfig, DenseGemmQuery
    from b12x.quantization import mla_norm_mxfp8
    from b12x.quantization.mla_norm_mxfp8._tuning import Config, Query

    qkv_query = DenseGemmQuery(recipe="mxfp8", entry_point="gemm.blockscaled.mm",
        weight_storage="native", output_dtype="bfloat16", batch=1, max_rows=8,
        in_features=5120, out_features=1792, output_mode="provided", alpha_mode="unit",
        workspace_form="provided", rhs_prefetch_ready=True,
        allow_bf16_atomic_reduction=False)
    qb_query = replace(qkv_query, in_features=1280, out_features=8192)
    norm_query = Query(rows=8, query_weight_dtype="bfloat16", kv_weight_dtype="bfloat16",
                       affine_prefetch_ready=True, release_dependents=True)
    dense_config = DenseGemmConfig(backend="cutedsl", tile_m=16, tile_n=64, tile_k=128,
        load_path="tma", swap_ab=False, split_k_slices=1, large_m_unroll=None,
        target_occupancy=1)
    config_by_role = {"qkv_a": dense_config, "norm": Config(), "q_b": dense_config}
    selector = object.__new__(CatalogContextualSelection)
    layers, requests = [], {}
    for index in (0, 37):
        plans = (blockscaled.plan(qkv_query), mla_norm_mxfp8.plan(norm_query),
                 blockscaled.plan(qb_query))
        attention = SimpleNamespace(projections={"qkv_a": SimpleNamespace(plan=plans[0]),
                                                  "q_b": SimpleNamespace(plan=plans[2])},
                                    norm_quant=SimpleNamespace(plan=plans[1]))
        layers.append(SimpleNamespace(attention=attention, metadata=SimpleNamespace(positions=torch.empty(8))))
        for role, plan in zip(("qkv_a", "norm", "q_b"), plans, strict=True):
            name = f"layer{index}.{role}"
            call = lambda state: PreparedCall(run=lambda: None)
            requests[name] = plan.request(name=name, prepare_call=call, benchmark_call=call)
    selector.layers = tuple(layers)
    selector.fixtures = (SimpleNamespace(target_signature=SimpleNamespace(token_capacity=8), layer_index=0, label="packed.layer0", tokens=8),
                         SimpleNamespace(layer_index=37, label="packed.layer37", tokens=8))
    names = selector._names(requests)
    policies = selector.finalist_policies(requests)
    assert set(policies) == set(requests)
    assert all(replace(request, finalist_policy=policies[name]).finalist_policy == policies[name]
               for name, request in requests.items())
    finalists = {}
    for role_index, role in enumerate(("qkv_a", "norm", "q_b")):
        for group in names:
            plan = requests[group[role_index]].plan
            finalists[group[role_index]] = (Finalist(selection=Selection(
                component_id=plan.component_id,
                query=FrozenMapping(plan.contract.encode_query(plan.query)),
                config=config_by_role[role], source="tuned"),
                family=FrozenMapping({"family": role}), candidate_index=0, latency_us=1.0),)
    configs = selector._configs(names, finalists)
    assert {role: len(values) for role, values in configs.items()} == {
        "qkv_a": 1, "norm": 1, "q_b": 1}
    for role_index, role in enumerate(("qkv_a", "norm", "q_b")):
        original_request = requests[names[0][role_index]]
        alternative = selector._override_plan(role, original_request.plan, configs[role][0])
        assert alternative.override == configs[role][0]
        assert original_request.plan.same_declaration_except_override(alternative)
        pinned = replace(original_request, plan=alternative, finalist_policy=None,
                         benchmark_call=None)
        assert pinned.finalist_policy is None and pinned.benchmark_call is None
    identity = selector._tuple_name(0, configs, (0, 0, 0))
    assert json.dumps({"winner": identity, "query": FrozenMapping(
        requests[names[0][0]].plan.contract.encode_query(qkv_query)).to_dict()})


def test_context_progress_counts_each_candidate_once_per_balanced_round(monkeypatch):
    import torch.distributed as dist

    selector = object.__new__(CatalogContextualSelection)
    selector.coordinator = SimpleNamespace(group=object(), ranks=(0, 1))
    advances = []
    selector._race_stage = SimpleNamespace(update=lambda *, advance: advances.append(advance))
    monkeypatch.setattr(dist, "all_gather_object",
        lambda peers, value, *, group: peers.__setitem__(slice(None), [value, value]))
    for repetition in (0, 1):
        peers = selector._context_exchange("launch", {"indices": (4,), "turn": 2,
                                                       "repetition": repetition})
        assert len(peers) == 2 and all(peer["indices"] == (4,) for peer in peers)
    assert advances == [1]


def test_export_summary_identifies_tuple_configs_and_both_measured_layers():
    report = {"objective": "equal-weight sum of complete layer GPU replay, each timed as TP max-rank",
        "winner": "context-selected", "standalone_tuple": "context-standalone",
        "winner_configs": {"qkv_a": {"split_k_slices": 4},
                           "norm": {"programmatic_serialization": False},
                           "q_b": {"split_k_slices": 2}},
        "latencies_us": {"context-selected": {"0": 90.0, "37": 105.0},
                         "context-standalone": {"0": 92.0, "37": 106.0}},
        "total_us": {"context-selected": 195.0}, "standalone_total_us": 198.0,
        "cache_policy": {"0": "cold", "37": "cold"}, "rounds": 3, "samples": 2,
        "lock_gpu_clocks": False, "contexts": ("packed.layer0", "packed.layer37"), "target_rows": 8}
    line = contextual_summary({"groups": [report]})
    assert line.startswith("Selected joint plans: ")
    payload = json.loads(line.split(": ", 1)[1])[0]
    assert payload["configs"] == report["winner_configs"]
    assert payload["layer_us"] == {"0": 90.0, "37": 105.0}
    assert payload["contexts"] == ["packed.layer0", "packed.layer37"]
    assert payload["sum_us"] == 195.0 and payload["standalone_sum_us"] == 198.0


@pytest.mark.parametrize("compile_decoder", (False, True))
def test_context_race_captures_every_tuple_with_optional_compilation(monkeypatch,
                                                                     compile_decoder):
    from ignition.preparation.execution import GraphSignature
    from ignition.recipes.deepseek41 import contextual_selection as module

    selector = object.__new__(CatalogContextualSelection)
    selector.layers = (object(),)
    selector.device = "cuda:0"
    selector.coordinator = SimpleNamespace(ranks=(0, 1))
    selector.progress = None
    selector._race_stage = None
    selector.compile_decoder = compile_decoder
    selector._tuple_name = lambda index, configs, selection: f"tuple-{index}"
    captured, restored, reset = [], [], []

    def capture(session, functions, signatures, *, label, before_run, compile_group):
        captured.append((tuple(functions), compile_group))
        for name in functions:
            before_run(name)
        return ({name: SimpleNamespace(reset=lambda name=name: reset.append(name))
                 for name in functions}, {})

    selector._capture = capture
    monkeypatch.setattr(module, "prepared_layer_variant", lambda layer, **plans: plans)
    monkeypatch.setattr(module, "ContextCandidate", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (2 << 30, 4 << 30))
    fixture = SimpleNamespace(layer_index=0, label="packed.layer0",
        trial_signature=GraphSignature("speculative", 1, 8),
        trial_forward=lambda layer: lambda: None,
        observation_forward=lambda layer: lambda: None,
        restore_tune=lambda layer: restored.append(layer),
        prepared_call=lambda name, graph, compiled, layer, observation_graph, observation_compiled: PreparedCall(
            run=lambda: None, produce=lambda: None))
    calls = []
    session = SimpleNamespace(headroom_bytes=0, rounds=3,
        race_contexts=lambda candidates, **kwargs: calls.append(
            (len(candidates), kwargs["max_candidates"])) or SimpleNamespace(
                winner=candidates[0].name, latencies_us={}))
    plans = {(0, "qkv_a"): (object(), object()), (0, "norm"): (object(),),
             (0, "q_b"): (object(),)}
    selector._race_layer(session, fixture, 0, {}, plans,
                         ((0, 0, 0), (1, 0, 0)))
    assert captured == [(("tuple-0-context0", "tuple-1-context0"),
                         "packed.layer0-plan-family" if compile_decoder else None),
                        (("tuple-0-context0-observe", "tuple-1-context0-observe"),
                         "packed.layer0-observation-family" if compile_decoder else None)]
    assert len(restored) == 4 and len(reset) == 4
    assert calls == [(2, 2)]


def test_attention_release_preserves_contextual_projection_plan_owners():
    from ignition.recipes.deepseek41.attention import PreparedAttention
    from ignition.recipes.deepseek41.contextual_selection import _role_plans

    layer = catalog_layer()
    attention = object.__new__(PreparedAttention)
    attention.__dict__.update(vars(layer.attention))
    layer.attention = attention
    original = _role_plans(layer)
    attention.release_examples()
    assert _role_plans(layer) == original
    assert attention.concurrent_query.plan_owners == (attention.projections["q_b"],)
    alternatives = (Plan(), Plan(), Plan(64))
    variant = prepared_layer_variant(layer, qkv_plan=alternatives[0], norm_plan=alternatives[1],
                                     query_plan=alternatives[2])
    assert _role_plans(variant) == alternatives
    assert _role_plans(layer) == original
