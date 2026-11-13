# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer-native operators declared through b12x tuning and scratch planning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import torch
from b12x._lib.scratch import scratch_buffer_spec
from b12x.preparation import Knob, MemoryRequirements, Plan, TuningContract


def _tactic(value):
    """Restore tuple tactics after a JSON round trip through the selection cache."""
    if isinstance(value, (list, tuple)):
        return tuple(_tactic(item) for item in value)
    if value is not None and type(value) not in (int, bool, str):
        raise TypeError("native tactics are nested tuples of integers, booleans, and strings")
    return value


@dataclass(frozen=True)
class NativeConfig:
    """One legal launch tactic of a FlashInfer-native kernel."""

    tactic: object

    def __post_init__(self):
        object.__setattr__(self, "tactic", _tactic(self.tactic))

    def to_dict(self):
        return {"tactic": self.tactic}

    @classmethod
    def from_config(cls, payload):
        if set(payload) != {"tactic"}:
            raise ValueError("native kernel configs contain exactly one tactic")
        return cls(payload["tactic"])


def native_contract(*, component_id: str, query_type: type,
                    tactics: Callable[[object], tuple], semantic_version: int = 1) -> TuningContract:
    """Tune one native kernel over ``tactics(query)``, whose first entry is the default.

    ``query_type`` is a frozen dataclass of JSON-compatible fields that fully
    determines the legal tactics; ``tactics`` should be cached by query.
    """
    names = frozenset(field.name for field in fields(query_type))

    def legal(query):
        values = tuple(_tactic(value) for value in tactics(query))
        if not values:
            raise ValueError(f"{component_id} has no legal tactic for {query}")
        return values

    def validate_query(query, device):
        if not isinstance(query, query_type):
            raise TypeError(f"{component_id} queries must be {query_type.__name__}")

    def validate_config(query, config, device):
        if not isinstance(config, NativeConfig):
            raise TypeError(f"{component_id} configs must be NativeConfig")
        if config.tactic not in legal(query):
            raise ValueError(f"{config.tactic!r} is not a legal {component_id} tactic")

    return TuningContract(
        component_id=component_id,
        query_schema_version=1,
        config_schema_version=1,
        query_fields=names,
        config_fields=frozenset({"tactic"}),
        encode_query=lambda query: {name: getattr(query, name) for name in names},
        encode_config=NativeConfig.to_dict,
        decode_config=NativeConfig.from_config,
        validate_query=validate_query,
        validate_config=validate_config,
        default_config=lambda query, device: NativeConfig(legal(query)[0]),
        knobs=(Knob(name="tactic", values=None),),
        parameters=lambda query, device: {"tactic": legal(query)},
        semantic_version=semantic_version,
    )


def native_plan(*, contract: TuningContract, query, device, scratch_bytes: Callable[[object], int],
                materialize: Callable[[object], object],
                compile_jobs: Callable[[object, object], tuple] = lambda tactic, detected: ()) -> Plan:
    """Declare one native operator over a single caller-owned byte buffer.

    ``scratch_bytes(tactic)`` sizes the buffer; ``materialize(tactic)``
    returns the prepared state. FlashInfer kernels build or load when their
    state is materialized, never at declaration. Any Triton programs the
    state launches must come from ``compile_jobs(tactic, detected_device)``,
    since b12x forbids Triton compilation after its compile phase.
    """
    device = torch.device(device)

    def memory(config, detected):
        return MemoryRequirements(scratch=(scratch_buffer_spec(
            contract.component_id, nbytes=scratch_bytes(config.tactic),
            device=torch.device("cuda", detected.ordinal)),))

    return Plan(contract=contract, query=query,
                _compile_jobs=lambda config, detected: tuple(compile_jobs(config.tactic, detected)),
                _memory_requirements=memory,
                _materialize=lambda selection, detected: materialize(selection.config.tactic),
                _device=device)
