# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare Python and Rust replay of the exact same prepared Qwen graph catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes())


def byte_view(tensor):
    if tensor.device.type != "cpu":
        raise ValueError("hashing requires an explicit fenced CPU readback")
    return memoryview(tensor.contiguous().reshape(-1).view(__import__("torch").uint8).numpy())


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Checks:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.value = {"checks": [], "commands": [], "observations": {}, "mismatches": []}
        self.baselines = {}
        self.bytes = 0

    def check(self, name, passed, **details):
        self.value["checks"].append({"name": name, "passed": bool(passed), **details})
        if not passed:
            print("FAIL", name, flush=True)

    def tensor(self, side, stage, name, cpu, *, parent=None, offset=0):
        raw = byte_view(cpu)
        digest = sha256(raw)
        descriptor = {"dtype": str(cpu.dtype), "shape": list(cpu.shape), "nbytes": len(raw), "sha256": digest}
        observation = self.value["observations"].setdefault(stage, {}).setdefault(name, {})
        observation[side] = descriptor
        if side == "python":
            if parent is None and digest not in self.baselines:
                if self.bytes + len(raw) > 32 << 30:
                    raise MemoryError("deduplicated Python observations exceed the 32 GiB host-memory bound")
                self.baselines[digest] = bytes(raw)
                self.bytes += len(raw)
        else:
            expected = observation["python"]
            descriptor["equal"] = expected == {key: descriptor[key] for key in expected}
            if not descriptor["equal"]:
                import numpy as np
                key = expected["sha256"] if parent is None else self.value["observations"][stage][parent]["python"]["sha256"]
                baseline = np.frombuffer(self.baselines[key], dtype=np.uint8, count=len(raw), offset=offset)
                actual = np.frombuffer(raw, dtype=np.uint8)
                different = np.flatnonzero(actual != baseline)
                first = int(different[0]) if len(different) else None
                number = len(self.value["mismatches"])
                start = max(0, (first or 0) - 256)
                end = min(len(actual), start + (64 << 10))
                path = self.artifacts / f"mismatch-{number:05d}.npz"
                np.savez(path, expected=baseline[start:end], actual=actual[start:end])
                self.value["mismatches"].append({"stage": stage, "tensor": name,
                    "different_bytes": int(len(different)), "first_different_byte": first,
                    "window_byte_offset": start, "window_bytes": end - start,
                    "readback": str(path), "readback_sha256": file_hash(path)})


_PRISTINE_COUNT_CHUNK_BYTES = 64 << 20


class _ScratchArenaAudit:
    """Inspect final tensor bindings without reading or changing device memory."""

    def __init__(self, arena):
        self.arena = arena
        self.storage = arena.tensor.untyped_storage()
        self.views = {}
        self.plans = {}
        self.payloads = set()
        self.families = Counter()
        self.external_count = 0

    def view(self, name, tensor):
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise AssertionError(f"{name} is not a scratch tensor")
        storage = tensor.untyped_storage()
        offset = tensor.storage_offset() * tensor.element_size()
        extent = (1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))) * tensor.element_size() if tensor.numel() else 0
        if (tensor.device != self.arena.device or storage.data_ptr() != self.storage.data_ptr()
                or storage.nbytes() != self.storage.nbytes()
                or any(stride < 0 for stride in tensor.stride())
                or not 0 <= offset <= offset + extent <= self.arena.nbytes):
            raise AssertionError(f"{name} does not fit the finalized scratch arena")
        if name in self.views:
            raise AssertionError(f"duplicate scratch audit name: {name}")
        self.views[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "strides": list(tensor.stride()), "offset_bytes": offset,
            "extent_bytes": extent, "logical_bytes": tensor.nbytes}

    def external(self, name, tensor):
        if tensor is not None:
            if tensor.untyped_storage().data_ptr() == self.storage.data_ptr():
                raise AssertionError(f"{name} must survive scratch arena reuse")
            self.external_count += 1

    def binding(self, name, family, binding, scratch_fields, *, external_fields=()):
        self.families[family] += 1
        for field in scratch_fields:
            self.view(f"{name}.{field}", getattr(binding, field))
        for field in external_fields:
            self.external(f"{name}.{field}", getattr(binding, field))

    def plan_owners(self, name, plan):
        from b12x.preparation import MemoryRequirements
        from b12x.preparation.types import _CompositePlan

        prepared = plan.prepared
        if prepared is None:
            raise AssertionError(f"{name} has no live prepared payload")
        owners = tuple(prepared.owners)
        if any(owner is not self.arena for owner in owners):
            raise AssertionError(f"{name} retains a concrete priming owner instead of the arena")
        scratch_bytes = MemoryRequirements(scratch=plan.scratch_specs()).scratch_nbytes
        composite = isinstance(plan, _CompositePlan)
        if not composite and scratch_bytes and not (len(owners) == 1 and owners[0] is self.arena):
            raise AssertionError(f"{name} prepared scratch has no arena owner")
        # b12x publishes priming owners separately from final bindings and compiled programs.
        # Inspecting those owners detects obsolete storage even when final pointers are correct.
        self.plans[name] = {"component": plan.component_id, "scratch_bytes": scratch_bytes,
            "arena_owner_count": len(owners), "composite": composite}
        self.payloads.add(id(prepared))
        if composite:
            if prepared.variants is None or set(prepared.variants) != set(plan.variants):
                raise AssertionError(f"{name} has incomplete prepared child plans")
            for capacity, child in plan.variants.items():
                if prepared.variants[capacity] is not child:
                    raise AssertionError(f"{name} has a stale prepared child plan")
                self.plan_owners(f"{name}.variant.{capacity}", child)

    def projection(self, name, projection):
        from ignition.components.linear import UnquantizedProjection, QuantizedProjection

        memory = projection.plan.memory_requirements()
        if isinstance(projection, QuantizedProjection):
            if projection.plan.query.workspace_form != "provided":
                raise AssertionError(f"{name} uses privately owned projection scratch")
            self.view(f"{name}.workspace", projection.workspace)
            if memory.scratch_nbytes > projection.workspace.nbytes:
                raise AssertionError(f"{name} workspace is smaller than the selected plan")
            self.plan_owners(name, projection.plan)
            self.families["quantized_projection"] += 1
        elif isinstance(projection, UnquantizedProjection):
            if memory.scratch_nbytes:
                raise AssertionError(f"{name} BF16 projection requires unbound scratch")
            self.families["bf16_projection_without_scratch"] += 1
        else:
            raise AssertionError(f"{name} has an unknown projection type")
        self.external(f"{name}.output", projection.output)

    def moe(self, name, operator):
        import torch

        self.view(f"{name}.scratch", operator.scratch)
        self.plan_owners(name, operator.plan)
        binding = operator.binding
        if binding.implementation not in ("micro", "dynamic", "w4a16"):
            raise AssertionError(f"{name} uses an unaudited expert binding implementation")
        self.families["moe"] += 1
        external = {"a", "topk_ids", "topk_weights", "output", "output_cast_target",
            "activation_amax", "route_expert_map", "output_expert_map"}
        direct_scales = {"input_gs": binding.experts.a1_gscale,
            "down_input_scale": binding.experts.a2_gscale}
        for field in fields(binding):
            tensor = getattr(binding, field.name)
            if not isinstance(tensor, torch.Tensor):
                continue
            if field.name in external:
                self.external(f"{name}.binding.{field.name}", tensor)
            elif field.name in direct_scales and tensor.untyped_storage().data_ptr() == direct_scales[field.name].untyped_storage().data_ptr():
                self.external(f"{name}.binding.{field.name}", tensor)
            else:
                self.view(f"{name}.binding.{field.name}", tensor)
        self.external(f"{name}.output", operator.output)

    def vision_binding(self, name, binding):
        from b12x.attention.varlen import VarlenBinding

        if not isinstance(binding, VarlenBinding):
            raise AssertionError(f"{name} requires the public prepared varlen binding")
        self.plan_owners(name, binding.plan)
        # Varlen output/LSE are work buffers, copied to VisionScratch before projection reuse.
        self.binding(name, "vision_attention", binding.binding,
            ("output", "lse"), external_fields=("q", "k", "v"))
        for field in ("sink_source", "sink_storage", "block_indices", "block_offsets"):
            self.external(f"{name}.{field}", getattr(binding, field))

    def report(self):
        return {"arena_bytes": self.arena.nbytes, "arena_frozen": self.arena.frozen,
            "families": dict(sorted(self.families.items())), "view_count": len(self.views),
            "view_logical_bytes_with_aliases": sum(view["logical_bytes"] for view in self.views.values()),
            "highest_view_end_bytes": max((view["offset_bytes"] + view["extent_bytes"] for view in self.views.values()), default=0),
            "external_tensor_count": self.external_count, "views": self.views,
            "plan_count_with_aliases": len(self.plans), "distinct_prepared_payloads": len(self.payloads),
            "plans": self.plans}


def verify_prepared_scratch_arena(model, decoders, owners, evidence):
    """Check every Qwen operator's final scratch bindings before parity reset."""
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.qwen38.attention import PreparedAttention
    from ignition.recipes.qwen38.gdn import PreparedGDN
    from ignition.recipes.qwen38.vision import PreparedVision

    arenas = [owner for owner in model._resources if isinstance(owner, ScratchArena)]
    if len(arenas) != 1:
        evidence.check("prepared_scratch_arena", False, arena_count=len(arenas))
        raise AssertionError("prepared model must retain exactly one scratch arena")
    arena = arenas[0]
    audit, error = _ScratchArenaAudit(arena), None
    try:
        if not arena.frozen or arena._finalizers:
            raise AssertionError("scratch arena is not frozen with finalized bindings")
        for index, decoder in enumerate(decoders):
            prefix = f"decoder.{index}"
            if decoder.workspaces is not arena or len(decoder.layers) != 48:
                raise AssertionError(f"{prefix} has incomplete layers or another scratch arena")
            for layer, (_, attention, _, moe) in enumerate(decoder.layers):
                name = f"{prefix}.layer.{layer}"
                if isinstance(attention, PreparedGDN):
                    audit.plan_owners(name + ".gdn.conv", attention.conv_plan)
                    audit.plan_owners(name + ".gdn.recurrence", attention.recurrent_plan)
                    audit.binding(name + ".gdn.conv", "gdn_convolution", attention.conv_binding,
                        ("scratch", "gathered_state", "request_ids"), external_fields=("conv_state", "out"))
                    recurrence = attention.recurrent_binding
                    views = ("scratch",) if attention.mode == "decode" else (
                        "scratch", "band_base", "sorted_seq", "rank_of", "pos_seq", "pos_local", "window_table", "ready_flags", "ws")
                    audit.binding(name + ".gdn.recurrence", "gdn_recurrence", recurrence, views,
                        external_fields=("recurrent_state", "output"))
                    audit.view(name + ".gdn.workspace", attention.workspace)
                elif isinstance(attention, PreparedAttention):
                    audit.plan_owners(name + ".qsa", attention.plan)
                    audit.binding(name + ".qsa", "qsa", attention.binding, (
                        "scratch", "prepared_index_query", "scores", "eligible_group_counts", "merge_lengths",
                        "topk_values", "topk_group_ids", "topk_values_b", "topk_group_ids_b",
                        "partial_output", "partial_lse", "output_lse"),
                        external_fields=("main_k_cache", "main_v_cache", "compressed_k_cache", "raw_k_ring", "output", "selected_positions"))
                    audit.view(name + ".qsa.workspace", attention.workspace)
                else:
                    raise AssertionError(f"{name} has an unknown attention type")
                for key, projection in attention.projections.items():
                    audit.projection(f"{name}.attention.projection.{key}", projection)
                audit.moe(name + ".moe.experts", moe.experts)
                for key in ("up_projection", "down_projection", "gate_projection"):
                    audit.projection(f"{name}.moe.shared.{key}", getattr(moe.shared, key))
                audit.projection(name + ".moe.router", moe.router_projection)
            ple = decoder.ple
            audit.plan_owners(prefix + ".ple.embedding", ple.embedding_plan)
            audit.plan_owners(prefix + ".ple.conv", ple.convolution_plan)
            audit.binding(prefix + ".ple.embedding", "ple_embedding", ple.embedding_binding,
                ("scratch", "_ids", "_hash_scratch"), external_fields=("out",))
            audit.binding(prefix + ".ple.hash", "ple_hash", ple.embedding_binding._hash_binding,
                ("scratch", "request_ids", "out"))
            audit.binding(prefix + ".ple.conv", "ple_convolution", ple.convolution_binding,
                ("scratch", "gathered_state", "request_ids"), external_fields=("conv_state", "out"))
            for key in ("key_projection", "value_projection"):
                audit.projection(f"{prefix}.ple.{key}", getattr(ple, key))
            audit.projection(prefix + ".lm_head", decoder.lm_head)
        visions = [owner for owner in owners if isinstance(owner, PreparedVision)]
        if not decoders or not visions:
            raise AssertionError("scratch audit requires decoder and vision graph owners")
        for index, vision in enumerate(visions):
            prefix = f"vision.{index}"
            audit.view(prefix + ".attention_workspace", vision.attention_workspace)
            audit.projection(prefix + ".patch_embed", vision.patch_embed.projection)
            for block_index, block in enumerate(vision.module.blocks):
                name = f"{prefix}.block.{block_index}"
                audit.view(name + ".attention.workspace", block.attn.workspace)
                audit.vision_binding(name + ".attention", block.attn.binding)
                audit.external(name + ".attention.escaped_output", block.attn.scratch.attention)
                for key in ("qkv", "proj"):
                    audit.projection(f"{name}.attention.{key}", getattr(block.attn, key))
                for key in ("linear_fc1", "linear_fc2"):
                    audit.projection(f"{name}.mlp.{key}", getattr(block.mlp, key).projection)
            for key in ("linear_fc1", "linear_fc2"):
                audit.projection(f"{prefix}.merger.{key}", getattr(vision.module.merger, key).projection)
    except BaseException as failure:
        error = str(failure)
        raise
    finally:
        evidence.value["scratch_arena_audit"] = audit.report()
        evidence.check("prepared_scratch_arena", error is None, error=error,
            view_count=len(audit.views), families=dict(audit.families),
            stage="after prepare_model and before MutableReset")


def verify_prepared_hybrid_state(state, evidence):
    """Check benchmark restoration before the parity reset overwrites model state."""
    import torch

    from ignition.recipes.qwen38.state import EOS_TOKEN_ID

    history = state.ple_history.cpu()
    eos = torch.tensor([EOS_TOKEN_ID], dtype=torch.int64)
    expected_bytes = int(torch.count_nonzero(eos.view(torch.uint8))) * history.shape[1] * (history.shape[0] - 1)
    backing = state.backing.view(-1)
    actual_bytes = sum(int(torch.count_nonzero(backing[offset:offset + _PRISTINE_COUNT_CHUNK_BYTES]).item())
        for offset in range(0, backing.numel(), _PRISTINE_COUNT_CHUNK_BYTES))
    history_valid = bool((history[0] == 0).all()) and bool((history[1:] == EOS_TOKEN_ID).all())
    evidence.check("prepared_hybrid_state_pristine", history_valid and actual_bytes == expected_bytes,
        history_valid=history_valid, expected_nonzero_bytes=expected_bytes,
        observed_nonzero_bytes=actual_bytes, state_bytes=state.backing.nbytes,
        count_chunk_bytes=_PRISTINE_COUNT_CHUNK_BYTES,
        stage="after prepare_model and before MutableReset", history=history.tolist())


class MutableReset:
    """Restore explicit retained scratch and reinitialize all model state in place."""

    def __init__(self, model, state, requests, decoders, owners, cache):
        import torch

        from ignition.preparation.scratch import ScratchArena
        from ignition.recipes.qwen38.decoder import DecoderScratch, ModelWeights
        from ignition.recipes.qwen38.maintenance import PreparedMaintenance
        from ignition.recipes.qwen38.vision import PreparedVision, VisionScratch
        from ignition.recipes.qwen38.vision_state import (
            PreparedVisionHandoff,
            VisionFeatureState,
        )

        self.model, self.state, self.requests, self.cache = model, state, requests, cache
        self.features = next(item for item in model._resources if isinstance(item, VisionFeatureState))
        excluded = {state.backing.untyped_storage().data_ptr(), self.features.arena.untyped_storage().data_ptr()}
        excluded.update(getattr(layer, field.name).untyped_storage().data_ptr()
            for layer in cache.values() for field in fields(layer))
        self.storage = {}

        def add(name, value):
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                key = storage.data_ptr()
                if key in excluded or key in self.storage or not storage.nbytes():
                    return
                raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))
                self.storage[key] = (name, raw, raw.cpu())
            elif is_dataclass(value):
                for field in fields(value):
                    add(f"{name}.{field.name}", getattr(value, field.name))
            elif isinstance(value, dict):
                for key, tensor in value.items():
                    add(f"{name}.{key}", tensor)
            elif isinstance(value, (tuple, list)):
                for index, tensor in enumerate(value):
                    add(f"{name}.{index}", tensor)

        for name, value in model.buffers.items():
            add(f"input.{name}", value)
        add("request", requests)
        add("features", self.features)
        for value in model._resources:
            if isinstance(value, (DecoderScratch, VisionScratch)):
                add(type(value).__name__, value)
            elif isinstance(value, ScratchArena):
                add("workspace.arena", value.tensor)
            elif isinstance(value, ModelWeights):
                add("expert_scale_scratch", value.expert_scale_scratch)
                for index, layer in enumerate(value.layers):
                    add(f"expert.{index}.active", layer.moe.experts._impl.nvfp4_csf.active)

        def sampler(name, value):
            for field, tensor in value.scratch_tensors.items():
                add(f"{name}.{field}", tensor)

        for index, decoder in enumerate(decoders):
            prefix = f"decoder.{index}"
            add(prefix + ".gdn", decoder.state.gdn)
            add(prefix + ".qsa", decoder.state.qsa)
            for name in ("pending_tokens", "positions", "ple_history", "active"):
                add(prefix + ".token_state." + name, getattr(decoder.state.token_state, name))
            if decoder.sampler is not None:
                sampler(prefix + ".sampler", decoder.sampler)
            if decoder.embedding_injection is not None:
                add(prefix + ".injection_errors", decoder.embedding_injection.errors)
            for layer, (_, attention, _, _) in enumerate(decoder.layers):
                if hasattr(attention, "inactive_rows"):
                    add(f"{prefix}.gdn.{layer}.inactive_rows", attention.inactive_rows)
        for index, owner in enumerate(owners):
            prefix = f"owner.{index}"
            if isinstance(owner, PreparedMaintenance):
                for name in ("stage_errors", "admission_errors", "filtered"):
                    add(prefix + "." + name, getattr(owner, name))
            elif isinstance(owner, PreparedVisionHandoff):
                for name in ("pixels", "errors", "metadata"):
                    add(prefix + "." + name, getattr(owner, name))
            elif isinstance(owner, PreparedVision):
                add(prefix + ".attention_workspace", owner.attention_workspace)
            elif hasattr(owner, "sampler"):
                sampler(prefix + ".sampler", owner.sampler)
        self.pristine_request = {field.name: getattr(requests, field.name).cpu().clone() for field in fields(requests)}

    def _require_idle_executor(self, executor):
        if (executor is None or executor is not self.model.runtime or executor.outstanding != 0):
            raise RuntimeError("qualification reset requires the model's idle native executor")

    def retain_startup_bindings(self, executor):
        """Preserve finalized tuning bytes without changing pristine sibling views."""
        self._require_idle_executor(executor)
        for tuner in self.model.tuning_graphs.values():
            for field in (*tuner.outputs, tuner.status_buffer):
                tensor = self.model.buffers[field]
                if not tensor.is_contiguous():
                    raise ValueError("qualification tuning buffers must be contiguous")
                storage = tensor.untyped_storage()
                _, raw, pristine = self.storage[storage.data_ptr()]
                offset = tensor.data_ptr() - storage.data_ptr()
                pristine[offset:offset + tensor.nbytes].copy_(raw[offset:offset + tensor.nbytes].cpu())

    def reset(self, *, executor=None):
        if self.model.consumed and executor is None:
            raise RuntimeError("Python reset is forbidden after native ownership handoff")
        if executor is not None:
            self._require_idle_executor(executor)
        self.state.backing.fill_(0x3D)
        self.state.backing[0].zero_()
        for _, device, pristine in self.storage.values():
            device.copy_(pristine)
        for name, pristine in self.pristine_request.items():
            getattr(self.requests, name).copy_(pristine)
        for layer in self.cache.values():
            for name in ("main_k_cache", "main_v_cache", "compressed_k_cache", "raw_k_ring"):
                getattr(layer, name).fill_(0.125)
            for name in ("raw_logical_positions", "raw_rope_positions", "raw_interval_start_positions"):
                getattr(layer, name).fill_(-1)
        self.features.arena.fill_(-0.125)
        __import__("torch").cuda.synchronize(self.model.device)

    def hashes(self):
        result = {name: {"nbytes": cpu.numel(), "sha256": sha256(byte_view(device.cpu()))}
            for name, device, cpu in self.storage.values()}
        for row in range(self.state.backing.shape[0]):
            result[f"hybrid.{row}"] = {"nbytes": self.state.layout.row_bytes,
                "sha256": sha256(byte_view(self.state.backing[row].cpu()))}
        for index, layer in self.cache.items():
            for field in fields(layer):
                tensor = getattr(layer, field.name)
                result[f"qsa.{index}.{field.name}"] = {"nbytes": tensor.nbytes,
                    "sha256": sha256(byte_view(tensor.cpu()))}
        result["feature_arena"] = {"nbytes": self.features.arena.nbytes,
            "sha256": sha256(byte_view(self.features.arena.cpu()))}
        return result


@dataclass
class Request:
    prompt: list[int]
    pages: list[int]
    generation: int
    seed: int
    max_new: int = 32
    consumed: int = 0
    generated: int = 0
    temperature: float = 0.7


class Scenario:
    def __init__(self, model, tokenizer, image, artifacts):

        from ignition.components.sampling import AdmissionBuffers

        self.model, self.tokenizer, self.image, self.artifacts = model, tokenizer, image, artifacts
        self.host = {name: tensor.cpu().clone() for name, tensor in model.buffers.items()}
        self.admission = AdmissionBuffers(**{field.name: self.host[field.name] for field in fields(AdmissionBuffers)})
        self.requests, self.snapshots, self.groups, self.payloads = {}, {}, [], {}
        self.prefill_capacities = sorted({signature.token_capacity for signature in model.signatures.values()
            if signature.kind == "prefill"})
        self.extended_cases = []
        self.next_page = 0
        self.sequence = 0

    def tokens(self, length, label):
        ids = self.tokenizer.encode(f"Replay qualification {label}. " + "The capital of France is Paris. " * (length + 1), add_special_tokens=False)
        return ids[:length]

    def pages(self, prefix=(), *, tokens=0):
        count = max(16, (tokens + 63) // 64) - len(prefix)
        if count < 0:
            raise ValueError("snapshot pages exceed the request reservation")
        pages = [*prefix, *range(self.next_page, self.next_page + count)]
        self.next_page += count
        return pages

    def emit(self, name, slots, *, sample=False, length=0, note="", pair=False):
        signature = self.model.signatures[name]
        temperatures = [self.requests[slot].temperature for slot in slots]
        records = {}
        for slot in slots:
            request = self.requests[slot]
            token = -1 if signature.kind == "decode" and request.generated >= request.max_new else None
            if signature.kind == "prefill":
                request.consumed += length
                if sample:
                    request.generated += 1
            elif signature.kind == "decode" and request.generated < request.max_new:
                request.consumed += 1
                request.generated += 1
            elif signature.kind == "resume":
                request.generated += 1
            if signature.kind in ("init", "restore", "snapshot", "vision") or (signature.kind == "prefill" and not sample):
                token = -1
            records[str(slot)] = {"generation": request.generation, "consumed": request.consumed,
                "generated": request.generated, "status": 2 if request.generated == request.max_new else 1,
                "token": token}
        uploads = []
        for name_field in signature.inputs:
            raw = byte_view(self.host[name_field])
            digest = sha256(raw)
            if digest not in self.payloads:
                self.payloads[digest] = bytes(raw)
                (self.artifacts / "uploads" / digest).write_bytes(raw)
            uploads.append((name_field, digest))
        self.sequence += 1
        command = {"graph": name, "sequence": self.sequence, "frame": 1 if pair else 0,
            "slots": list(slots), "expected_records": records, "uploads": uploads, "note": note,
            "temperatures": temperatures}
        if signature.kind in ("init", "restore"):
            command["admissions"] = {str(slot): {"prompt": self.requests[slot].prompt, "seed": self.requests[slot].seed}
                for slot in slots}
        if signature.kind in ("init", "restore", "snapshot"):
            command["transfers"] = [{"source": int(self.host["state_source_slots"][row]),
                "destination": int(self.host["state_destination_slots"][row]),
                "operation": int(self.host["state_operation"][row]), "slot": slot,
                "consumed": int(self.host["state_expected_consumed"][row]),
                "pages": list(self.requests[slot].pages)}
                for row, slot in enumerate(slots)]
        if pair:
            if len(self.groups[-1]) != 1:
                raise ValueError("two-frame pairs require exactly one preceding command")
            self.groups[-1].append(command)
        else:
            self.groups.append([command])

    def admit(self, kind, capacity, specifications, *, temperatures=None):
        from ignition.recipes.qwen38.sampling import RequestAdmission

        h = self.host
        h["admission_count"].fill_(len(specifications))
        h["admission_slots"].fill_(-1)
        h["state_num_ops"].fill_(len(specifications))
        h["state_source_slots"].zero_()
        h["state_destination_slots"].zero_()
        h["state_qsa_slots"].fill_(-1)
        h["state_operation"].zero_()
        h["state_accepted_tokens"].zero_()
        h["state_expected_consumed"].fill_(-1)
        for row, (slot, prompt, generation, seed, source, max_new) in enumerate(specifications):
            previous = self.snapshots[source] if source is not None else None
            if previous is not None and any(token is None for token in previous[0]):
                raise ValueError("restore prompt requires explicit known snapshot token IDs")
            consumed = len(previous[0]) if previous else 0
            if previous is not None and prompt[:consumed] != previous[0]:
                raise ValueError("restore prompt must extend the exact checkpoint token prefix")
            temperature = 0.7 if temperatures is None else temperatures[slot]
            request = Request(prompt, self.pages(previous[1] if previous else (), tokens=len(prompt) + max_new), generation, seed,
                max_new, consumed, temperature=temperature)
            self.requests[slot] = request
            self.admission.stage(row, slot, RequestAdmission(generation, seed, max_new_tokens=max_new,
                temperature=temperature, top_k=20, top_p=0.95, repetition_penalty=1.05,
                presence_penalty=0.05, frequency_penalty=0.02, eos_token_ids=()), prompt)
            for name, value in (("source_slots", 5 + source if source is not None else slot + 1),
                    ("destination_slots", slot + 1), ("operation", 0 if source is not None else 1),
                    ("qsa_slots", slot + 1), ("accepted_tokens", 1),
                    ("expected_consumed", consumed)):
                h["state_" + name][row] = value
        slots = [spec[0] for spec in specifications]
        self.rows(slots)
        self.emit(f"{kind}{capacity}", slots, note="exact reset/restore and fresh seeded admission")

    def rows(self, slots):
        import torch

        h = self.host
        h["row_slot_ids"].fill_(-1)
        h["row_generations"].zero_()
        h["live_rows"].fill_(len(slots))
        h["query_lengths"].zero_()
        h["sample_enabled"].fill_(1)
        h["vision_token_indices"].fill_(-1)
        h["vision_feature_slot"].fill_(-1)
        for row, slot in enumerate(slots):
            request = self.requests[slot]
            h["row_slot_ids"][row] = slot
            h["row_generations"][row] = request.generation
            h["page_tables"][slot].fill_(-1)
            h["page_tables"][slot, :len(request.pages)] = torch.tensor(request.pages, dtype=torch.int32)

    def prefill(self, slot, end, capacity, *, sample=True, vision=False):
        import torch

        self.rows([slot])
        request, h = self.requests[slot], self.host
        begin = request.consumed
        length = end - begin
        if not 0 < length <= capacity or end > len(request.prompt):
            raise ValueError("prefill sequence must consume a nonempty in-prompt span within capacity")
        h["query_lengths"][0] = length
        h["sample_enabled"].fill_(int(sample))
        h["token_ids"].fill_(248044)
        h["token_ids"][:length] = torch.tensor(request.prompt[begin:end])
        h["prefill_positions"].zero_()
        h["prefill_positions"][:length] = (self.image["mrope"][begin:end] if vision
            else torch.arange(begin, end)[:, None].expand(-1, 3))
        if vision:
            h["vision_token_indices"][:length] = self.image["token_indices"][begin:end]
            h["vision_feature_slot"].fill_(0)
        name = f"prefill{capacity}"
        self.emit(name, [slot], sample=sample, length=length,
            note=f"consume [{begin},{end}) with {'image' if vision else 'text'} positions")

    def forward(self, kind, capacity, slots, *, pair=False):
        self.rows(slots)
        name = f"{kind}{capacity}"
        self.emit(name, slots, note="packed row permutation and explicit RNG advancement", pair=pair)

    def snapshot(self, capacity, slots, snapshot_ids):
        self.rows(slots)
        h = self.host
        h["state_num_ops"].fill_(len(slots))
        h["state_source_slots"].zero_()
        h["state_destination_slots"].zero_()
        h["state_qsa_slots"].fill_(-1)
        h["state_operation"].zero_()
        h["state_accepted_tokens"].zero_()
        h["state_expected_consumed"].fill_(-1)
        for row, (slot, identifier) in enumerate(zip(slots, snapshot_ids, strict=True)):
            request = self.requests[slot]
            prefix = request.prompt[:request.consumed] + [None] * max(0, request.consumed - len(request.prompt))
            self.snapshots[identifier] = (prefix, request.pages[:request.consumed // 64])
            h["state_expected_consumed"][row] = request.consumed
            h["state_source_slots"][row] = slot + 1
            h["state_destination_slots"][row] = 5 + identifier
        self.emit(f"snapshot{capacity}", slots, note="copy exact live state and private cache tails into an immutable snapshot")

    def build(self):
        if not {128, 256, 512} <= set(self.prefill_capacities):
            raise ValueError("graph execution scenarios require prefill capacities 128, 256 and 512")
        prompts = {3: self.tokens(64, "aligned"), 0: self.tokens(513, "chunked"),
            2: self.tokens(67, "cold-tail"), 1: self.tokens(77, "split-tail")}
        self.admit("init", 4, [(slot, prompts[slot], 11, 0xF0123456789ABCDE + slot, None, 32) for slot in (3, 0, 2, 1)])
        self.prefill(3, 64, 128, sample=False)
        self.prefill(0, 512, 512, sample=False)
        self.prefill(0, 513, 128)
        self.prefill(2, 67, 128)
        self.prefill(1, 64, 128, sample=False)
        self.prefill(1, 77, 128)
        self.snapshot(4, [1, 3, 0, 2], [1, 3, 0, 2])
        self.forward("resume", 1, [3])
        self.forward("decode", 4, [0, 3, 1, 2])
        self.forward("decode", 4, [2, 1, 3, 0], pair=True)
        self.forward("decode", 2, [2, 0])
        self.forward("decode", 4, [3, 2, 1])
        self.forward("decode", 1, [0])
        branch = prompts[2][:64] + self.tokens(13, "branch")
        # The branch changes tokens after 64; produce its exact prefix independently.
        self.admit("init", 1, [(0, prompts[2], 12, 3456, None, 32)])
        self.prefill(0, 64, 128, sample=False)
        self.snapshot(1, [0], [4])
        self.admit("restore", 2, [(1, prompts[3], 12, 2345, 3, 32), (0, branch, 13, 3456, 4, 32)])
        self.forward("resume", 2, [1])
        self.prefill(0, 77, 128)
        self.forward("decode", 2, [1, 0])
        self.forward("decode", 2, [0, 1], pair=True)
        self.snapshot(2, [2, 3], [4, 5])
        fifth, sixth = self.tokens(255, "wide-tail"), self.tokens(128, "full-hit")
        self.admit("init", 2, [(2, fifth, 12, 4567, None, 32), (3, sixth, 12, 5678, None, 32)])
        self.prefill(2, 255, 256)
        self.prefill(3, 128, 128, sample=False)
        self.snapshot(2, [2, 3], [6, 7])
        self.admit("restore", 1, [(0, sixth, 14, 6789, 7, 32)])
        self.forward("resume", 4, [0])
        self.admit("init", 1, [(1, self.tokens(63, "page-boundary"), 13, 7890, None, 2)])
        self.prefill(1, 63, 128)
        self.forward("decode", 1, [1])
        self.forward("decode", 1, [1], pair=True)
        self.snapshot(1, [1], [8])
        self.admit("restore", 4, [(slot, list(self.snapshots[source][0]), 15 if slot == 0 else 14, 8901 + slot, source, 32)
            for slot, source in ((3, 3), (2, 1), (1, 2), (0, 0))])
        self.forward("resume", 4, [2, 0, 3, 1])
        self.forward("decode", 4, [3, 1, 0, 2])
        self.forward("decode", 4, [0, 2, 1, 3], pair=True)
        for index, capacity in enumerate((256, 1024, 4096, 16384)):
            epoch = 101 + index
            self.admit("init", 1, [(3, self.image["prompt"], epoch, 9000 + index, None, 32)])
            for name, value in self.image["vision_inputs"].items():
                self.host[name].zero_()
                self.host[name].reshape(-1)[:value.numel()].copy_(value.reshape(-1))
            self.host["vision_slot"].fill_(3)
            self.host["vision_feature_slot"].fill_(0)
            self.host["vision_generation"].fill_(epoch)
            self.host["vision_feature_offset"].zero_()
            self.emit(f"vision{capacity}", [3], note="same pinned 256-patch fixture at this graph capacity")
            self.prefill(3, len(self.image["prompt"]), 128, vision=True)
            self.forward("decode", 1, [3])
        prompts = {3: self.tokens(128, "greedy-aligned"), 0: self.tokens(513, "greedy-chunked"),
            2: self.tokens(255, "greedy-wide-tail"), 1: self.tokens(77, "greedy-split-tail")}
        self.admit("init", 4, [(slot, prompts[slot], 201, 12000 + slot, None, 32) for slot in (3, 0, 2, 1)],
            temperatures={slot: 1. for slot in range(4)})
        self.prefill(3, 64, 128, sample=False)
        self.prefill(3, 128, 128, sample=False)
        self.prefill(0, 512, 512, sample=False)
        self.prefill(0, 513, 128)
        self.prefill(2, 255, 256)
        self.prefill(1, 64, 128, sample=False)
        self.prefill(1, 77, 128)
        self.snapshot(4, [1, 3, 0, 2], [1, 3, 0, 2])
        self.forward("resume", 1, [3])
        self.forward("decode", 4, [0, 3, 1, 2])
        self.forward("decode", 4, [2, 1, 3, 0], pair=True)
        self.forward("decode", 4, [3, 2, 1])
        self.forward("decode", 4, [1, 2, 3], pair=True)
        self.forward("decode", 2, [2, 0])
        self.forward("decode", 1, [0])
        self.admit("restore", 2, [(slot, list(self.snapshots[source][0]), 202, 13000 + slot, source, 32)
            for slot, source in ((1, 3), (0, 2))], temperatures={1: 1., 0: 0.7})
        self.forward("resume", 2, [1, 0])
        self.forward("decode", 2, [0, 1])
        self.forward("decode", 1, [1], pair=True)
        self.admit("restore", 1, [(2, list(self.snapshots[0][0]), 202, 14002, 0, 32)], temperatures={2: 1.})
        self.forward("resume", 2, [2])
        self.admit("restore", 4, [(slot, list(self.snapshots[source][0]), 203, 15000 + slot, source, 32)
            for slot, source in ((3, 3), (2, 1), (1, 2), (0, 0))], temperatures={slot: 1. for slot in range(4)})
        self.forward("resume", 4, [2, 0, 3, 1])
        self.forward("decode", 4, [3, 1, 0, 2])
        self.extended_prefills()
        if self.next_page + 1 >= self.model.memory_report["pages"]:
            raise ValueError("prepared KV pool cannot cover scripted private pages and canaries")
        if set(command["graph"] for group in self.groups for command in group) != set(self.model.serving_graphs):
            raise ValueError("scenario must execute every prepared graph variant")
        if not any(len(group) == 2 for group in self.groups):
            raise ValueError("scenario requires two-frame execution")
        if not any(len(set(command["temperatures"])) > 1
                and all(value > 0 for value in command["temperatures"])
                for group in self.groups for command in group if self.model.signatures[command["graph"]].kind == "decode"):
            raise ValueError("scenario requires distinct positive temperatures in one decode batch")
        return self

    def extended_prefills(self):
        for capacity in self.prefill_capacities:
            if capacity <= 512:
                continue
            for temperature, fixture_label in ((0.7, "general"), (1., "greedy")):
                index = len(self.extended_cases)
                slot, restored_slot = index % 4, (index + 1) % 4
                generation, seed = 300 + 2 * index, 17000 + index
                prompt = self.tokens(capacity + 65, f"large-{capacity}-{fixture_label}")
                first_sequence = self.sequence + 1
                self.admit("init", 1, [(slot, prompt, generation, seed, None, 32)],
                    temperatures={slot: temperature})
                self.prefill(slot, capacity, capacity, sample=False)
                self.snapshot(1, [slot], [8])
                self.prefill(slot, len(prompt), capacity)
                self.forward("decode", 1, [slot])
                self.admit("restore", 1, [(restored_slot, prompt, generation + 1, seed, 8, 32)],
                    temperatures={restored_slot: temperature})
                self.prefill(restored_slot, len(prompt), capacity)
                self.forward("decode", 1, [restored_slot])
                self.extended_cases.append({"capacity": capacity, "temperature": temperature,
                    "full_tokens": capacity, "continuation_tokens": 65,
                    "source_slot": slot, "restored_slot": restored_slot, "snapshot_id": 8,
                    "commands": list(range(first_sequence, self.sequence + 1))})


def run_comparison(model, state, requests, scratch, cache, reset, scenario, evidence, progress):
    import torch

    graph_indices = {graph["name"]: index for index, graph in enumerate(model.manifest()["graphs"])}
    tracked_rows = [*range(14), state.backing.shape[0] - 1]
    tracked_pages = scenario.next_page + 1
    if len(state.layout.regions) < 77 or state.backing.shape[0] <= 14:
        raise ValueError("qualification requires every hybrid region and untouched snapshot canaries")
    readback_bytes = len(tracked_rows) * state.layout.row_bytes
    for layer in cache.values():
        for field in fields(layer):
            tensor = getattr(layer, field.name)
            readback_bytes += tensor[0].nbytes * (tracked_pages + 2) if field.name.endswith("cache") else tensor.nbytes
    readback_bytes += sum(getattr(requests, field.name).nbytes for field in fields(requests))
    readback_bytes += sum(tensor.nbytes for tensor in model.buffers.values()) + scratch.logits.nbytes
    readback_bytes += reset.features.generation.nbytes + 2 * reset.features.count.nbytes
    feature_rows = max(1, int(scenario.image["vision_inputs"]["vision_num_patches"].item()) // 4)
    readback_bytes += 4 * (feature_rows + 1) * reset.features.arena[0, 0].nbytes
    evidence.value["readback_contract"] = {"hybrid_rows": tracked_rows, "regions_per_row": len(state.layout.regions),
        "hybrid_padding": "Every tracked row is compared bytewise, including alignment padding.",
        "qsa_pages": [0, tracked_pages], "qsa_far_canary_page": model.memory_report["pages"] - 1,
        "qsa_raw": "All seven QSA fields are covered; full selected pages and all raw slots include invalid payload bytes.",
        "request_fields": [field.name for field in fields(requests)],
        "mismatches": "Byte differences retain a64KiB window beginning256bytes before the first difference, plus total differing-byte count and full tensor hashes.",
        "two_frame_groups": [[command["sequence"] for command in group] for group in scenario.groups if len(group) == 2],
        "expected_stage_readback_bytes_upper_bound": readback_bytes,
        "expected_total_stage_readback_bytes_upper_bound": 2 * (len(scenario.groups) + 1) * readback_bytes,
        "readback_bound_scope": "Expected valid-state reads after each group and initial state on both executors; excludes reset hashes and pristine-scratch snapshots.",
        "deduplicated_baseline_limit_bytes": 32 << 30}
    cpu_uploads = {digest: torch.frombuffer(bytearray(payload), dtype=torch.uint8) for digest, payload in scenario.payloads.items()}
    frame_sources = [torch.tensor([frame], dtype=torch.int32) for frame in range(2)]
    sequence_sources = {command["sequence"]: torch.tensor([command["sequence"]], dtype=torch.int64)
        for group in scenario.groups for command in group}
    python_stream = torch.cuda.Stream(device=model.device)

    def allocated():
        statistics = torch.cuda.memory_stats(model.device)
        names = ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated",
            "segment.all.freed", "num_device_alloc", "num_device_free")
        return {name: statistics[name] for name in names}

    def readback(side, stage):
        for row in tracked_rows:
            cpu = state.backing[row].cpu()
            name = f"hybrid.row{row}"
            evidence.tensor(side, stage, name, cpu)
            for region in state.layout.regions:
                value = cpu[region.offset:region.offset + region.nbytes].view(region.dtype).reshape(region.shape)
                evidence.tensor(side, stage, f"{name}.{region.name}", value, parent=name, offset=region.offset)
        for index, layer in cache.items():
            for field in fields(layer):
                tensor = getattr(layer, field.name)
                name = f"qsa.layer{index}.{field.name}"
                if field.name.endswith("cache"):
                    for begin in range(0, tracked_pages, 16):
                        end = min(begin + 16, tracked_pages)
                        evidence.tensor(side, stage, f"{name}.pages{begin}-{end}", tensor[begin:end].cpu())
                    evidence.tensor(side, stage, name + ".far_canary", tensor[-1].cpu())
                    evidence.tensor(side, stage, name + ".unused_page", tensor[scenario.next_page].cpu())
                else:
                    cpu = tensor.cpu()
                    evidence.tensor(side, stage, name, cpu)
                    for slot in range(cpu.shape[0]):
                        evidence.tensor(side, stage, f"{name}.raw_slot{slot}", cpu[slot], parent=name,
                            offset=slot * cpu[slot].nbytes)
        request_cpu = {field.name: getattr(requests, field.name).cpu() for field in fields(requests)}
        for name, cpu in request_cpu.items():
            evidence.tensor(side, stage, "request." + name, cpu)
            for slot in range(4):
                evidence.tensor(side, stage, f"request.{name}.slot{slot}", cpu[slot:slot + 1],
                    parent="request." + name, offset=slot * cpu[slot:slot + 1].nbytes)
        for name, tensor in model.buffers.items():
            evidence.tensor(side, stage, "buffer." + name, tensor.cpu())
        logits = scratch.logits.cpu()
        evidence.tensor(side, stage, "logits", logits)
        for name in ("generation", "count"):
            evidence.tensor(side, stage, "vision." + name, getattr(reset.features, name).cpu())
        feature_count = reset.features.count.cpu().tolist()
        for slot in range(4):
            count = max(1, feature_count[slot])
            evidence.tensor(side, stage, f"vision.slot{slot}.features", reset.features.arena[slot, :count].cpu())
            evidence.tensor(side, stage, f"vision.slot{slot}.far_canary", reset.features.arena[slot, -1].cpu())
        return request_cpu, logits

    def verify_unrelated_state(side, previous, stage, group):
        active = {slot for command in group for slot in command["slots"]}
        destinations = {transfer["destination"] for command in group for transfer in command.get("transfers", [])}
        snapshots_only = all(model.signatures[command["graph"]].kind == "snapshot" for command in group)
        mutated_slots = {slot for command in group if model.signatures[command["graph"]].kind != "snapshot"
                         for slot in command["slots"]}
        allowed_rows = destinations | {slot + 1 for slot in mutated_slots}
        observations = evidence.value["observations"]
        unchanged = []
        for name, values in observations[stage].items():
            if name.startswith("hybrid.row") and "." not in name[len("hybrid."):] and int(name.removeprefix("hybrid.row")) not in allowed_rows:
                unchanged.append(name)
            elif name.startswith("request.") and ".slot" in name and int(name.rsplit("slot", 1)[1]) not in active:
                unchanged.append(name)
            elif name.startswith("qsa.") and (snapshots_only or name.endswith((".far_canary", ".unused_page")) or
                    ".raw_slot" in name and int(name.rsplit("raw_slot", 1)[1]) not in {slot + 1 for slot in active}):
                unchanged.append(name)
            elif name.startswith("vision.slot") and (name.endswith(".far_canary") or int(name.split(".")[1][4:]) not in active):
                unchanged.append(name)
        changed = [name for name in unchanged if observations[stage][name][side]["sha256"] != observations[previous][name][side]["sha256"]]
        evidence.check(f"{side}.{stage}.unrelated_state_unchanged", not changed,
            observed_tensors=len(unchanged), changed=changed)

    def verify_records(side, command, records, reference):
        sequence = command["sequence"]
        command[side + "_records"] = records
        if side == "native":
            evidence.check(f"completion.{sequence}.same_bytes", records == command["python_records"])
        for slot in range(4):
            record = records[slot]
            expected = command["expected_records"].get(str(slot))
            if expected is None:
                evidence.check(f"{side}.{sequence}.inactive-slot{slot}", record == [0, 0, 0, 0, -1, 0, 0, 0], record=record)
                continue
            valid = (record[:4] == [expected["generation"], sequence, expected["consumed"], expected["generated"]]
                and record[5:] == [expected["status"], 0, 0]
                and (record[4] == expected["token"] if expected["token"] is not None else 0 <= record[4] < 248320))
            evidence.check(f"{side}.{sequence}.slot{slot}", valid, expected=expected, record=record)
            if not valid:
                raise RuntimeError(f"{side} {command['graph']} returned an invalid active completion")
            admission = command.get("admissions", {}).get(str(slot))
            if admission is not None:
                reference[slot] = {"histogram": Counter(admission["prompt"]), "generated": 0, "seed": admission["seed"]}
            current = reference[slot]
            if record[3] > current["generated"]:
                if record[3] != current["generated"] + 1:
                    raise RuntimeError("completion generated count jumped by more than one")
                current["histogram"][record[4]] += 1
                current["generated"] = record[3]

    executor = None
    try:
        executor = model.start_executor()
        model.runtime = executor
        reset.retain_startup_bindings(executor)
        reset.reset(executor=executor)
        reset_hashes = reset.hashes()
        evidence.value["reset"]["python"] = reset_hashes
        for side in ("python", "native"):
            if side == "native":
                python_stream.synchronize()
                reset.reset(executor=executor)
                native_reset = reset.hashes()
                evidence.value["reset"]["native"] = native_reset
                evidence.check("identical_initial_storage_bytes", native_reset == reset_hashes)
                if native_reset != reset_hashes:
                    raise RuntimeError("model reset did not reproduce the Python baseline's initial bytes")
                evidence.check("same_graph_handles_and_buffer_addresses", model.manifest() == evidence.value["manifest"])
                if model.manifest() != evidence.value["manifest"]:
                    raise RuntimeError("graph or buffer identity changed between comparison sides")
            referenced = {binding["tuning_graph"] for graph in model.manifest()["graphs"]
                          for binding in graph.get("tuned_bindings", ())}
            for name in referenced:
                tuner = model.tuning_graphs[name]
                status = int(model.buffers[tuner.status_buffer].item())
                evidence.check(f"{side}.startup.{name}.status", status == 0, status=status)
                if status != 0:
                    raise RuntimeError(f"{side} startup tuning graph {name!r} failed: {status}")
            reference = {}
            readback(side, "initial")
            previous_stage = "initial"
            for group_index, group in enumerate(scenario.groups):
                stage = f"group{group_index:02d}." + "+".join(command["graph"] for command in group)
                before = allocated()
                if side == "python":
                    with torch.cuda.stream(python_stream):
                        for command in group:
                            for field, digest in command["uploads"]:
                                model.buffers[field].view(torch.uint8).reshape(-1).copy_(cpu_uploads[digest], non_blocking=True)
                            model.buffers["frame_index"].copy_(frame_sources[command["frame"]], non_blocking=True)
                            model.buffers["submission_sequence"].copy_(sequence_sources[command["sequence"]], non_blocking=True)
                            model.graphs[command["graph"]].replay()
                    python_stream.synchronize()
                    after = allocated()
                    records = model.buffers["completion_records"].cpu().tolist()
                    completed = [records[command["frame"]] for command in group]
                else:
                    fences = []
                    for command in group:
                        fence = executor.submit(graph_indices[command["graph"]],
                            [(field, 0, scenario.payloads[digest]) for field, digest in command["uploads"]])
                        if fence != command["sequence"]:
                            raise RuntimeError(f"native submission returned {fence}, expected {command['sequence']}")
                        fences.append(fence)
                    completed = []
                    deadline = time.monotonic() + 120
                    while len(completed) < len(fences):
                        completion = executor.poll()
                        if completion is None:
                            if time.monotonic() > deadline:
                                raise TimeoutError(f"{stage} native completion timed out")
                            time.sleep(.001)
                            continue
                        index = len(completed)
                        if completion.fence != fences[index] or completion.graph_index != graph_indices[group[index]["graph"]]:
                            raise RuntimeError("native completion fence/graph did not match submission order")
                        completed.append([list(record) for record in completion.records])
                    after = allocated()
                evidence.check(f"{side}.{stage}.allocation_stability", before == after, before=before, after=after)
                for command, records in zip(group, completed, strict=True):
                    verify_records(side, command, records, reference)
                request_cpu, logits = readback(side, stage)
                verify_unrelated_state(side, previous_stage, stage, group)
                previous_stage = stage
                for slot, current in reference.items():
                    actual = request_cpu["histogram"][slot]
                    nonzero = actual.nonzero().flatten().tolist()
                    histogram = {token: int(actual[token]) for token in nonzero}
                    evidence.check(f"{side}.{stage}.slot{slot}-histogram", histogram == dict(current["histogram"]))
                    seed = request_cpu["seed"][slot].item()
                    counter = request_cpu["rng_counter"][slot].item()
                    evidence.check(f"{side}.{stage}.slot{slot}-rng", seed == current["seed"]
                        and counter == current["generated"], seed=seed, counter=counter, generated=current["generated"])
                last = group[-1]
                if model.signatures[last["graph"]].kind in ("prefill", "decode", "resume"):
                    evidence.check(f"{side}.{stage}.finite_active_logits", bool(torch.isfinite(logits[:len(last["slots"])]).all()))
                for transfer in last.get("transfers", []):
                    if transfer["operation"] == 0:
                        from qualify.checkpoint_state import checkpoint_transfer_checks

                        snapshot_row = (transfer["destination"] if model.signatures[last["graph"]].kind == "snapshot"
                                        else transfer["source"])
                        checks = checkpoint_transfer_checks(state, cache, slot=transfer["slot"],
                            snapshot=snapshot_row - 5, pages=transfer["pages"], consumed=transfer["consumed"])
                        if model.signatures[last["graph"]].kind == "restore":
                            checks.update({f"qsa.{layer}.restored_anchor":
                                int(value.raw_interval_start_positions[transfer["slot"] + 1].cpu().item())
                                == transfer["consumed"] - 1 for layer, value in cache.items()})
                        evidence.check(f"{side}.{stage}.state-copy-{transfer['source']}-{transfer['destination']}",
                            bool(checks) and all(checks.values()), checks=checks)
                progress(f"{side} {group_index + 1}/{len(scenario.groups)} {stage}; baseline bytes {evidence.bytes / (1 << 30):.2f} GiB")
        evidence.check("all_observations_bitwise_equal", not evidence.value["mismatches"], mismatches=len(evidence.value["mismatches"]))
    finally:
        if executor is not None:
            executor.close()
            evidence.check("native_streams_drained", executor.drained)
            if not executor.drained:
                raise RuntimeError("native executor did not confirm drain; model resources remain retained")


def image_fixture(snapshot):
    from types import MethodType, SimpleNamespace

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
    from transformers.vision_utils import (
        get_vision_cu_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    fixture = json.loads((ROOT / "tests/frontend/qwen38_images.json").read_text())
    case = next(case for case in fixture["cases"] if case["name"] == "identity")
    descriptor = case["inputs"][0]
    state = descriptor["seed"]
    data = bytearray(descriptor["width"] * descriptor["height"] * 3)
    for index in range(len(data)):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        data[index] = state & 255
    image = Image.fromarray(np.frombuffer(data, np.uint8).reshape(descriptor["height"], descriptor["width"], 3))
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    result = processor(text=case["rendered"], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
    grids = result["image_grid_thw"]
    indices, weights = get_vision_interpolation_indices_and_weights(grids, 48, mode="bilinear", align_corners=True, spatial_merge_size=2)
    positions = get_vision_position_ids(grids, 2)
    positional = SimpleNamespace(config=AutoConfig.from_pretrained(snapshot, local_files_only=True))
    positional.get_vision_position_ids = MethodType(Qwen4ExpModel.get_vision_position_ids, positional)
    rope, _ = Qwen4ExpModel.get_rope_index(positional, result["input_ids"], result["mm_token_type_ids"], image_grid_thw=grids)
    tensors = {"pixel_values": result["pixel_values"], "position_ids": positions,
        "interpolation_indices": indices, "interpolation_weights": weights, "mrope": rope[:, 0, :]}
    if {name: sha256(byte_view(tensor)) for name, tensor in tensors.items()} != case["hashes"]:
        raise ValueError("CPU image preparation differs from the pinned fast-processor fixture")
    if result["input_ids"][0].tolist() != case["token_ids"] or result["pixel_values"].shape[0] != 256:
        raise ValueError("image fixture token or patch geometry differs")
    token_indices = torch.full_like(result["input_ids"][0], -1, dtype=torch.int32)
    is_image = result["mm_token_type_ids"][0] == 1
    token_indices[is_image] = torch.arange(int(is_image.sum()), dtype=torch.int32)
    return {"prompt": case["token_ids"], "mrope": rope[:, 0, :].T.contiguous(), "token_indices": token_indices,
        "fixture": case, "vision_inputs": {"vision_" + name: tensor for name, tensor in tensors.items() if name != "mrope"}
        | {"vision_cu_seqlens": get_vision_cu_seqlens(grids).to(torch.int32),
            "vision_num_patches": torch.tensor([256], dtype=torch.int32)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    if args.prefill_chunk_size < 512 or args.prefill_chunk_size % 64:
        parser.error("qualification requires a 64-aligned prefill chunk size of at least 512 to retain every original case")
    if args.prefill_chunk_size + 65 + 32 > 262144:
        parser.error("qualification requires context space for a full chunk, 65-token continuation and 32 output tokens")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("assign CUDA_VISIBLE_DEVICES explicitly")
    if args.artifacts.exists() and any(args.artifacts.iterdir()):
        parser.error("diagnostic tensor directory must be empty")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "uploads").mkdir()

    import torch
    from transformers import AutoTokenizer

    from ignition.preparation.qwen38 import MODEL_ID, REVISION
    from ignition.recipes.qwen38.bootstrap import PreparationConfig, prepare_model
    from ignition.recipes.qwen38.decoder import DecoderScratch, PreparedDecoder
    from ignition.recipes.qwen38.maintenance import PreparedMaintenance
    from ignition.recipes.qwen38.sampling import DeviceRequestState
    from ignition.recipes.qwen38.state import HybridStatePool

    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    snapshot = args.snapshot or Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / REVISION
    evidence = Checks(args.artifacts)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    image = image_fixture(snapshot)
    started = time.monotonic()
    progress = lambda text: print(f"{time.monotonic() - started:.1f}s {text}", flush=True)
    try:
        with prepare_model(snapshot, PreparationConfig(prefill_chunk_size=args.prefill_chunk_size), progress=progress) as model:
            state = next(item for item in model._resources if isinstance(item, HybridStatePool))
            requests = next(item for item in model._resources if isinstance(item, DeviceRequestState))
            scratch = next(item for item in model._resources if isinstance(item, DecoderScratch))
            decoders = next(item for item in model._resources if isinstance(item, list) and item and isinstance(item[0], PreparedDecoder))
            owners = next(item for item in model._resources if isinstance(item, list)
                          and any(isinstance(owner, PreparedMaintenance) for owner in item))
            cache = decoders[0].qsa_states
            verify_prepared_scratch_arena(model, decoders, owners, evidence)
            verify_prepared_hybrid_state(state, evidence)
            reset = MutableReset(model, state, requests, decoders, owners, cache)
            reset.reset()
            scenario = Scenario(model, tokenizer, image, args.artifacts).build()
            evidence.value["commands"] = [command for group in scenario.groups for command in group]
            evidence.value["extended_prefill_cases"] = scenario.extended_cases
            evidence.value["memory_report"] = model.memory_report
            evidence.value["image_fixture"] = image["fixture"]
            evidence.value["reset"] = {"hybrid_byte_poison": 0x3D, "kv_and_raw_payload_poison": 0.125,
                "feature_poison": -0.125, "invalid_raw_tag": -1,
                "retained_storage_count": len(reset.storage), "retained_storage_bytes": sum(raw.numel() for _, raw, _ in reset.storage.values())}
            original_manifest = model.manifest()
            evidence.value["manifest"] = original_manifest
            run_comparison(model, state, requests, scratch, cache, reset, scenario, evidence, progress)
    except BaseException as error:
        import traceback
        evidence.value["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        evidence.value["elapsed_seconds"] = time.monotonic() - started
        evidence.value["deduplicated_baseline_bytes"] = evidence.bytes
        evidence.value["passed"] = not evidence.value.get("error") and not evidence.value["mismatches"] and all(check["passed"] for check in evidence.value["checks"])
    print(json.dumps({"passed": evidence.value["passed"],
        "checks": len(evidence.value["checks"]), "mismatches": len(evidence.value["mismatches"])}), flush=True)
    return 0 if evidence.value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
