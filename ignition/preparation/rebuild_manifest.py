# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Describe preparation inputs and named views of stored weights without copying payloads.

The catalog describes prepared layouts, not checkpoint tensors. Reading it does
not instantiate Python objects or establish that a recipe can rebuild from it.

Recipes declare semantic weight groups explicitly with `declare_weights`. The
declaration states which container fields hold layers, which groups are tuning
selected, and which views are aliases. Object traversal only enumerates the
resulting views; it never decides immutability, grouping, or alias identity from
object paths or pointer values. Tensors under types without a declared
classification are reported as unavailable instead of being treated as weights.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
import math
import re

from . import runtime_package_pb2 as proto

STABLE = 'stable'
TUNED = 'tuned'

# Field allowlists for external weight container types, keyed by exact type.
# Entries name the fields that carry stored weight bytes; a listed type is never
# traversed through any other field, so runtime buffers of that type stay
# unclassified and end up in the initial-state payload.
CONTAINER_FIELDS: dict[str, tuple[str, ...]] = {}
CONTAINER_TUNED: dict[str, frozenset[str]] = {}
# Per-module-type fields that hold stored bytes. Module parameters are always
# included; buffers are only included for types declared here, so rotary tables,
# masks, and other derived constants never become weight owners.
MODULE_BUFFER_FIELDS: dict[str, frozenset[str]] = {}


def declare_container_fields(type_key, fields, *, tuned=()):
    """Declare the weight-bearing fields of an external container type."""
    if not isinstance(type_key, str) or not type_key:
        raise ValueError('declared container types need a module-qualified name')
    CONTAINER_FIELDS[type_key] = tuple(fields)
    CONTAINER_TUNED[type_key] = frozenset(tuned)


def declare_module_buffers(type_key, fields):
    """Declare which module buffers of one type hold stored bytes."""
    if not isinstance(type_key, str) or not type_key:
        raise ValueError('declared module types need a module-qualified name')
    MODULE_BUFFER_FIELDS[type_key] = frozenset(fields)


@dataclass(frozen=True)
class WeightDeclaration:
    """Explicit semantic weight declaration supplied by a recipe."""

    roots: Mapping[str, object]
    # `(path, group)` pairs; a path may name a root or a dotted field/attribute path.
    components: tuple[tuple[str, str], ...]
    layer_fields: tuple[str, ...] = ('layers',)
    tuned_groups: tuple[str, ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()


def declare_weights(roots, *, components, layers=('layers',), tuned=(), aliases=()):
    """Declare the semantic grouping of a recipe's prepared weight objects.

    `components` names the root fields that hold stored weight bytes; other
    fields of the same objects stay unclassified. A name equal to a root keeps
    that whole root in one component group.
    """
    if roots is None:
        raise ValueError('weight declarations require a weight object')
    if not isinstance(roots, Mapping):
        roots = {'weights': roots}
    if not roots:
        raise ValueError('weight declarations require at least one root')
    if not components:
        raise ValueError('weight declarations require declared components')
    if isinstance(components, Mapping):
        entries = tuple((str(path), str(group)) for path, group in components.items())
    else:
        entries = tuple((str(name), str(name)) for name in components)
    return WeightDeclaration(roots=MappingProxyType(dict(roots)), components=entries,
        layer_fields=tuple(layers), tuned_groups=tuple(tuned), aliases=tuple(aliases))


def _type_key(value):
    kind = type(value)
    return f'{kind.__module__}.{kind.__qualname__}'


def _storage(group, declaration):
    return TUNED if group in declaration.tuned_groups else STABLE


def declared_views(declaration):
    """Enumerate declared weight views into layer or misc groups, retaining storage class."""
    import torch

    disabled = (-1,)

    def walk(value, name, owner, group, storage, ancestors):
        if isinstance(value, torch.Tensor):
            payload_group = group if re.fullmatch(r"layer-[0-9]+", group) else "misc"
            yield name, owner, value, payload_group, storage
            return
        if value is None or id(value) in ancestors:
            return
        ancestry = ancestors | {id(value)}
        key = _type_key(value)
        child_storage = {}
        if key in CONTAINER_FIELDS:
            owner = key
            tuned = CONTAINER_TUNED.get(key, frozenset())
            children = [(f'.{field}', getattr(value, field)) for field in CONTAINER_FIELDS[key]]
            child_storage = {f'.{field}': TUNED if field in tuned else storage
                             for field in CONTAINER_FIELDS[key]}
        elif isinstance(value, Mapping):
            children = [(f'[{k!r}]', child) for k, child in value.items()
                        if isinstance(k, (str, int))]
        elif isinstance(value, (tuple, list)):
            children = [(f'[{index}]', child) for index, child in enumerate(value)]
        elif isinstance(value, torch.nn.Module):
            owner = key
            buffers = MODULE_BUFFER_FIELDS.get(key, frozenset())
            children = [(f'.{k}', child) for k, child in value._parameters.items()]
            children += [(f'.{k}', value._buffers[k]) for k in sorted(buffers) if k in value._buffers]
            children += [(f'.{k}', child) for k, child in value._modules.items()]
        elif is_dataclass(value) and not isinstance(value, type):
            owner = key
            children = [(f'.{field.name}', getattr(value, field.name)) for field in fields(value)]
        else:
            return
        for suffix, child in children:
            yield from walk(child, name + suffix, owner, group,
                            child_storage.get(suffix, storage), ancestry)

    def resolve(value, path):
        for segment in path.split('.'):
            if isinstance(value, Mapping):
                if segment not in value:
                    raise KeyError(path)
                value = value[segment]
            elif isinstance(value, (torch.nn.Module,)) or is_dataclass(value):
                value = getattr(value, segment)
            else:
                raise AttributeError(path)
        return value

    components = dict(declaration.components)
    for root, value in sorted(declaration.roots.items()):
        if root in components:
            yield from walk(value, root, '', components[root],
                            _storage(components[root], declaration), frozenset())
            continue
        members = _members(value)
        prefix = root + '.'
        for suffix, child in members:
            path = root + suffix
            relative = path[len(prefix):] if path.startswith(prefix) else path
            field = suffix.rsplit('.', 1)[-1].partition('[')[0]
            group = components.get(path)
            if group is None:
                group = components.get(relative, components.get(field))
            if group is not None:
                yield from walk(child, path, '', group,
                                _storage(group, declaration), frozenset())
                continue
            for declared, declared_group in components.items():
                if not declared.startswith(relative + '.'):
                    continue
                try:
                    selected = resolve(child, declared[len(relative) + 1:])
                except (KeyError, AttributeError) as error:
                    raise ValueError(
                        f'declared weight component {declared!r} does not resolve') from error
                yield from walk(selected, f'{root}.{declared}', '', declared_group,
                                _storage(declared_group, declaration), frozenset())

    for root, value in sorted(declaration.roots.items()):
        if root in components:
            continue
        for declared in declaration.layer_fields:
            try:
                layers = resolve(value, declared)
            except (KeyError, AttributeError):
                continue
            if not isinstance(layers, (tuple, list, torch.nn.ModuleList)):
                continue
            for index, layer in enumerate(layers):
                group = f'layer-{index:03d}'
                yield from walk(layer, f'{root}.{declared}[{index}]', '', group,
                                _storage(group, declaration), frozenset())


def _members(value):
    import torch

    if isinstance(value, Mapping):
        return [(f'[{key!r}]', child) for key, child in value.items() if isinstance(key, (str, int))]
    if isinstance(value, (tuple, list)):
        return [(f'[{index}]', child) for index, child in enumerate(value)]
    if isinstance(value, torch.nn.Module):
        return [(f'.{key}', child) for group in (value._parameters, value._buffers, value._modules)
                for key, child in group.items()]
    return [(f'.{field.name}', getattr(value, field.name)) for field in fields(value)]


def from_plan(model, plan, *, compiled_forwards):
    from .source_identity import source_identity

    result = dict(version=1, recipe=plan.recipe.name, source_identity=source_identity(),
        checkpoint_repository=plan.checkpoint.repository,
        checkpoint_revision=plan.checkpoint.revision if model.directory is None else '',
        local_checkpoint=model.directory is not None, tensor_parallel=plan.world_size,
        max_concurrency=plan.concurrency, prefill_chunk=plan.prefill_chunk,
        image_concurrency=plan.image_concurrency, context_parallel=plan.context_parallel,
        stream_overlap=plan.stream_overlap, mixed_prefill=plan.mixed_prefill,
        mla_compute_mode=plan.mla_compute_mode, roce_ring=plan.roce_ring, roce_chain=plan.roce_chain,
        online_mxfp8=plan.online_mxfp8, online_mxfp8_scope=plan.online_mxfp8_scope, language_model_only=plan.language_model_only,
        compiled_forwards=compiled_forwards, reserve_bytes=plan.reserve_bytes,
        collective_tuning=plan.collective_tuning)
    if plan.collective_ladder is not None:
        import json

        result["collective_ladder_json"] = json.dumps(plan.collective_ladder, separators=(",", ":"), sort_keys=True)
    if plan.expert_scale_compression is not None:
        result['expert_scale_compression'] = plan.expert_scale_compression
    if plan.force_moe_decode_a16:
        result['force_moe_decode_a16'] = True
    if plan.shard_hyperconnections:
        result['shard_hyperconnections'] = True
    if plan.joint_tuning:
        result['joint_tuning'] = True
    if (s := plan.speculation) is not None:
        result['speculation'] = dict(draft_tokens=s.draft_tokens,
            mtp_only=s.mtp_only,
            adaptive=s.adaptive,
            token_capacity=plan.speculative_token_capacity, packed_verification=s.packed_verification,
            draft_prefix_selection=s.draft_prefix_selection,
            vocabulary_parallel_sampling=s.vocabulary_parallel_sampling)
        if s.method is not None:
            result['speculation']['method'] = s.method
        if s.packed_verification and s.draft_tokens > 0 and s.graph_budget is None:
            raise ValueError('packed speculative rebuild metadata requires an explicit graph budget')
        if s.graph_budget is not None:
            result['speculation']['graph_budget'] = s.graph_budget
        for field in ('draft_head_quantization', 'markov_head_quantization'):
            if (value := getattr(s, field)) is not None:
                result['speculation'][field] = value
    return result


def weight_views(roots):
    """Walk named weight owners without evaluating properties, closures, or module code.

    Paths identify fields within the recorded source identity. Aliased tensors keep
    each field name; cycles in container ownership do not recurse indefinitely.
    """
    import torch

    def walk(value, name, owner, ancestors):
        if isinstance(value, torch.Tensor):
            yield name, owner, value
            return
        if value is None or id(value) in ancestors:
            return
        ancestry = ancestors | {id(value)}
        if isinstance(value, Mapping):
            children = ((f'[{key!r}]', child) for key, child in value.items()
                        if isinstance(key, (str, int)))
        elif isinstance(value, (tuple, list)):
            children = ((f'[{index}]', child) for index, child in enumerate(value))
        elif isinstance(value, torch.nn.Module):
            owner = f'{type(value).__module__}.{type(value).__qualname__}'
            children = ((f'.{key}', child) for group in (value._parameters, value._buffers, value._modules)
                        for key, child in group.items())
        elif is_dataclass(value) and not isinstance(value, type):
            owner = f'{type(value).__module__}.{type(value).__qualname__}'
            children = ((f'.{field.name}', getattr(value, field.name)) for field in fields(value))
        elif type(value).__module__.startswith(('b12x.', 'ignition.')) and hasattr(value, '__dict__'):
            owner = f'{type(value).__module__}.{type(value).__qualname__}'
            children = ((f'.{key}', child) for key, child in vars(value).items())
        else:
            return
        for suffix, child in children:
            yield from walk(child, name + suffix, owner, ancestry)

    for name, value in sorted(roots.items()):
        yield from walk(value, name, '', set())


def usable_weight_tensor(tensor):
    """True for a view that has materialized, non-empty, non-negative strides."""
    import torch

    return (tensor.device.type != 'meta' and tensor.layout == torch.strided
            and tensor.numel() and not any(stride < 0 for stride in tensor.stride()))


def weight_payload_declaration(declaration):
    """Return the thin declaration Heliopause resolves against retained storage."""
    views = []
    for name, _owner, tensor, group, storage in declared_views(declaration):
        if not usable_weight_tensor(tensor):
            continue
        entry = dict(name=name, group=group, storage=storage,
                     element_bytes=tensor.element_size(), shape=list(tensor.shape),
                     stride=list(tensor.stride()), address=tensor.data_ptr())
        from . import provenance

        recorded = provenance.lookup(tensor)
        if recorded is not None:
            entry['sources'] = recorded['sources']
            entry['packing'] = recorded['packing']
            if recorded.get('packing_parameters') is not None:
                entry['packing_parameters'] = recorded['packing_parameters']
            entry['statistics_domain'] = recorded['statistics_domain']
        views.append(entry)
    by_name = {entry['name']: entry for entry in views}
    for alias, canonical in declaration.aliases:
        if alias not in by_name:
            raise ValueError(f'declared weight alias {alias!r} is not a declared view')
        if canonical not in by_name:
            raise ValueError(f'declared weight alias target {canonical!r} is not a declared view')
        source, target = by_name[alias], by_name[canonical]
        if (source['address'], source['element_bytes'], source['shape'], source['stride']) != (
                target['address'], target['element_bytes'], target['shape'], target['stride']):
            raise ValueError(f'declared weight alias {alias!r} does not share {canonical!r} storage')
        source['alias_of'] = canonical
    return dict(version=1, views=views)


def weight_payload_request(roots):
    """Return the thin declaration Heliopause resolves, or None for legacy roots."""
    if not isinstance(roots, WeightDeclaration):
        return None
    return weight_payload_declaration(roots)


def _declared_metadata(declaration):
    import torch

    tensors, unavailable = [], []
    for name, owner, tensor, _group, _storage in declared_views(declaration):
        if not usable_weight_tensor(tensor):
            unavailable.append(name)
            continue
        extent = tensor.element_size() * (1 + sum((size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)))
        tensors.append(dict(name=name, dtype=str(tensor.dtype).removeprefix('torch.'),
            element_bytes=tensor.element_size(), shape=list(tensor.shape), stride=list(tensor.stride()),
            layout_type=owner, address=tensor.data_ptr(), extent=extent))
    return dict(version=1, tensors=tensors, unavailable=unavailable)


def rebuild_weights_metadata(roots):
    """Describe tensor views for native resolution against retained storage."""
    import torch

    if isinstance(roots, WeightDeclaration):
        return _declared_metadata(roots)

    tensors, unavailable = [], []
    for name, owner, tensor in weight_views(roots):
        if (tensor.device.type == 'meta' or tensor.layout != torch.strided or not tensor.numel()
                or any(stride < 0 for stride in tensor.stride())):
            unavailable.append(name)
            continue
        extent = tensor.element_size() * (1 + sum((size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)))
        tensors.append(dict(name=name, dtype=str(tensor.dtype).removeprefix('torch.'),
            element_bytes=tensor.element_size(), shape=list(tensor.shape), stride=list(tensor.stride()),
            layout_type=owner, address=tensor.data_ptr(), extent=extent))
    return dict(version=1, tensors=tensors, unavailable=unavailable)


def validate_weights(bundle):
    """Check catalog references before a preparation tool uses them; serving ignores this metadata."""
    if not bundle.HasField('rebuild_weights'):
        return
    catalog = bundle.rebuild_weights
    if catalog.version != 1:
        raise ValueError(f'unsupported prepared-weight catalog version {catalog.version}')
    allocations = {a.name: a for a in bundle.allocations}
    names = set()
    widths = {'bool': 1, 'uint8': 1, 'int8': 1, 'int16': 2, 'uint16': 2, 'float16': 2,
              'bfloat16': 2, 'int32': 4, 'uint32': 4, 'float32': 4, 'int64': 8, 'uint64': 8,
              'float64': 8, 'complex32': 4, 'complex64': 8, 'complex128': 16,
              'float8_e4m3fn': 1, 'float8_e5m2': 1, 'float8_e4m3fnuz': 1,
              'float8_e5m2fnuz': 1, 'float8_e8m0fnu': 1}
    for tensor in catalog.tensors:
        if (not tensor.name or tensor.name in names or not tensor.HasField('storage')
                or tensor.element_bytes != widths.get(tensor.dtype)
                or len(tensor.shape) != len(tensor.stride) or not math.prod(tensor.shape)):
            raise ValueError('invalid prepared-weight tensor description')
        names.add(tensor.name)
        extent = tensor.element_bytes * (1 + sum((size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride, strict=True)))
        allocation = allocations.get(tensor.storage.allocation)
        if (allocation is None or allocation.HasField('sizing') or allocation.memory == proto.PEER
                or tensor.storage.offset + extent > allocation.nbytes):
            raise ValueError(f'prepared-weight tensor {tensor.name!r} exceeds fixed stored allocation')
    for name in catalog.unavailable:
        if not name or name in names:
            raise ValueError('duplicate or empty unavailable prepared-weight name')
        names.add(name)
