# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained-owner provenance for native NVFP4/MXFP4 experts."""

import json
import math
import os

import pytest
import torch
from safetensors.torch import save_file
from b12x._lib.quant.csf_encode import encode_csf_source

from ignition.components.weights import DirectWeightLoader, record_final_fp4_expert_provenance
from ignition.preparation import provenance
from ignition.preparation.checkpoint import inspect_indexed_checkpoint

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def checkpoint(path, tensors):
    save_file(tensors, str(path / "weights.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {name: "weights.safetensors" for name in tensors}}))
    (path / "config.json").write_text(json.dumps({"quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": {}}}))
    return inspect_indexed_checkpoint(path)


def packed(rows, columns, fmt, generator):
    group = 32 if fmt == "mxfp4" else 16
    weight = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator)
    if fmt == "mxfp4":
        scale = torch.randint(121, 125, (rows, columns // group), dtype=torch.uint8, generator=generator)
    else:
        scale = (torch.rand((rows, columns // group), generator=generator) * .02 + .01).to(torch.float8_e4m3fn)
    tensors = {"weight": weight, "weight_scale": scale}
    if fmt != "mxfp4":
        tensors["weight_scale_2"] = torch.tensor(.5)
    return tensors


def assert_weight_pieces(owner, entry, sources, encoding):
    assert entry is not None
    assert len(entry["sources"]) == sources, entry["sources"]
    for source in entry["sources"]:
        assert source.get("pieces"), source["tensor"]
        assert source.get("piece_encoding") == encoding, source["tensor"]
    covered = sum(math.prod(piece["extents"]) * piece["contiguous_bytes"]
                  for source in entry["sources"] for piece in source["pieces"])
    assert covered == owner.numel() * owner.element_size(), (covered, owner.numel() * owner.element_size())



@pytest.mark.parametrize(("fmt", "mode"), [("nvfp4", "a16"), ("mxfp4", "a8")])
@torch.no_grad()
def test_transformed_fp4_owner_provenance_declares_layout_identity(tmp_path, fmt, mode):
    gen = torch.Generator().manual_seed(811)
    hidden, intermediate = 256, 256
    rows = {}
    for expert in range(2):
        for projection in ("up_proj", "gate_proj"):
            rows[(expert, projection)] = packed(intermediate, hidden, fmt, gen)
        rows[(expert, "down_proj")] = packed(hidden, intermediate, fmt, gen)
    checkpoint_tensors = {
        f"experts.{expert}.{projection}.{name}": value
        for (expert, projection), tensors in rows.items() for name, value in tensors.items()}
    manifest = checkpoint(tmp_path, checkpoint_tensors)
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        experts = loader.load_fp4_experts(
            ("experts.0", "experts.1"), hidden_size=hidden, intermediate_size=intermediate,
            format=fmt, activation_mode=mode)
        retained = (experts._impl.w1_fp4, experts._impl.w1_blockscale,
                    experts._impl.w2_fp4, experts._impl.w2_blockscale)
        recorded = [provenance.lookup(owner) for owner in retained]
        assert all(recorded)
        assert all(entry["packing"] in ("w4a16_mma_packed", "w4a8_qmma")
                   for entry in recorded)
        assert all(entry["statistics_domain"] == "shard"
                   for entry in recorded[1::2] if fmt == "nvfp4")
        if fmt == "nvfp4":
            assert_weight_pieces(retained[0], recorded[0], 4, "w4a16_mma_k16_n64_v1")
            assert_weight_pieces(retained[2], recorded[2], 2, "w4a16_mma_k16_n64_v1")
        else:
            assert_weight_pieces(retained[0], recorded[0], 4, "w4a8_qmma_n256_k128_v1")
            assert_weight_pieces(retained[2], recorded[2], 2, "w4a8_qmma_n256_k128_v1")
        assert not any(source.get("pieces") for source in recorded[1]["sources"])
        assert not any(source.get("pieces") for source in recorded[3]["sources"])


@torch.no_grad()
def test_w31_normalization_registry_selects_retained_half_order(tmp_path):
    from b12x.moe.fused_moe import _impl as b12x_impl

    gen = torch.Generator().manual_seed(812)
    hidden, intermediate, local = 256, 256, 128
    rows = {}
    for expert in range(1):
        rows[(expert, "gate_proj")] = packed(intermediate, hidden, "nvfp4", gen)
        rows[(expert, "up_proj")] = packed(intermediate, hidden, "nvfp4", gen)
        rows[(expert, "down_proj")] = packed(hidden, intermediate, "nvfp4", gen)
    checkpoint_tensors = {
        f"experts.{expert}.{projection}.{name}": value
        for (expert, projection), tensors in rows.items() for name, value in tensors.items()}
    manifest = checkpoint(tmp_path, checkpoint_tensors)
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        experts = loader.load_fp4_experts(
            ("experts.0",), hidden_size=hidden, intermediate_size=intermediate,
            intermediate_range=(128, 256), format="nvfp4", activation_mode="a16",
            projection_names=("gate_proj", "up_proj", "down_proj"), w13_layout="w31")
        w13, s13 = experts._impl.w1_fp4, experts._impl.w1_blockscale
        key = (w13.data_ptr(), s13.data_ptr())
        # W4A16 folds the w31 half order into preparation; no runtime registry
        # mutation remains to invalidate a later declaration.
        assert key not in b12x_impl._W13_NORMALIZED_STORAGES

    # The loader has cleared its source dictionary; deferred recording must use
    # the immutable snapshot and not claim transformed bytes as copies.
    record_final_fp4_expert_provenance(experts)
    recorded = provenance.lookup(w13)
    assert recorded["packing"] == "w4a16_mma_packed"
    assert recorded["sources"][0]["tensor"].endswith("gate_proj.weight")
    assert recorded["sources"][1]["tensor"].endswith("up_proj.weight")
    assert all(source.get("piece_encoding") == "w4a16_mma_k16_n64_v1"
               for source in recorded["sources"])
    assert all(source.get("pieces") for source in recorded["sources"])
    assert not any(source.get("pieces") for source in provenance.lookup(s13)["sources"])


@torch.no_grad()
def test_compressed_scale_owner_does_not_suppress_weight_provenance(tmp_path):
    gen = torch.Generator().manual_seed(813)
    hidden, intermediate = 256, 256
    rows = {}
    for expert in range(2):
        for projection in ("up_proj", "gate_proj"):
            rows[(expert, projection)] = packed(intermediate, hidden, "nvfp4", gen)
        rows[(expert, "down_proj")] = packed(hidden, intermediate, "nvfp4", gen)
    checkpoint_tensors = {
        f"experts.{expert}.{projection}.{name}": value
        for (expert, projection), tensors in rows.items() for name, value in tensors.items()}
    manifest = checkpoint(tmp_path, checkpoint_tensors)
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        scratch = (torch.empty((2, 512, 16), dtype=torch.float8_e4m3fn, device="cuda"),
                   torch.empty((2, 256, 16), dtype=torch.float8_e4m3fn, device="cuda"))
        experts = loader.load_fp4_experts(
            ("experts.0", "experts.1"), hidden_size=hidden, intermediate_size=intermediate,
            format="nvfp4", activation_mode="a16", scale_compression="csf",
            scale_scratch=scratch)
        weight13 = provenance.lookup(experts._impl.w1_fp4)
        weight2 = provenance.lookup(experts._impl.w2_fp4)
        scale13 = provenance.lookup(experts._impl.w1_blockscale)
        scale2 = provenance.lookup(experts._impl.w2_blockscale)
        assert all((weight13, weight2))
        assert scale13 is None and scale2 is None
        assert all(source.get("pieces") for source in weight13["sources"])
        assert all(source.get("pieces") for source in weight2["sources"])


@torch.no_grad()
def test_w4a16_tp2_encoded_tiles_copy_from_tp1_tiles():
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_w4a16_packed_weights

    torch.manual_seed(814)
    experts, hidden, intermediate = 2, 256, 256
    device = "cuda"
    generator = torch.Generator().manual_seed(815)
    w13 = torch.randint(0, 256, (experts, 2 * intermediate, hidden // 2),
                        dtype=torch.uint8, generator=generator).to(device)
    w2 = torch.randint(0, 256, (experts, hidden, intermediate // 2),
                       dtype=torch.uint8, generator=generator).to(device)
    s13 = torch.randint(0, 256, (experts, 2 * intermediate, hidden // 16),
                        dtype=torch.uint8, generator=generator).to(device).view(torch.float8_e4m3fn)
    s2 = torch.randint(0, 256, (experts, hidden, intermediate // 16),
                       dtype=torch.uint8, generator=generator).to(device).view(torch.float8_e4m3fn)
    gains = torch.full((experts,), .5, device=device)

    def prepare(w13, w2, s13, s2):
        return prepare_w4a16_packed_weights(
            w13, s13, gains, w2, s2, gains, activation="silu",
            source_format="modelopt_nvfp4", w13_layout="w13", reuse_input_storage=False)

    def tiles(owner):
        return owner.reshape(experts, owner.shape[1], -1, 128)

    canonical = prepare(w13, w2, s13, s2)
    canonical13, canonical2 = tiles(canonical.w13), tiles(canonical.w2)
    for rank in (0, 1):
        start = rank * 128
        shard13 = torch.cat((w13[:, start:start + 128],
                             w13[:, intermediate + start:intermediate + start + 128]), dim=1)
        shard2 = w2[:, :, rank * 64:(rank + 1) * 64]
        shard_s13 = torch.cat((s13[:, start:start + 128],
                               s13[:, intermediate + start:intermediate + start + 128]), dim=1)
        shard_s2 = s2[:, :, rank * 8:(rank + 1) * 8]
        shard = prepare(shard13, shard2, shard_s13, shard_s2)
        shard13_tiles, shard2_tiles = tiles(shard.w13), tiles(shard.w2)
        for expert in range(experts):
            for kt in range(shard13_tiles.shape[1]):
                for nt in range(shard13_tiles.shape[2]):
                    target = (rank * 2 + nt) if nt < 2 else (4 + rank * 2 + nt - 2)
                    assert torch.equal(shard13_tiles[expert, kt, nt],
                                       canonical13[expert, kt, target])
            for nt in range(shard2_tiles.shape[2]):
                for kt in range(shard2_tiles.shape[1]):
                    assert torch.equal(shard2_tiles[expert, kt, nt],
                                       canonical2[expert, rank * 8 + kt, nt])


@torch.no_grad()
def test_w4a16_tp2_piece_metadata_reconstructs_tp1_tiles(tmp_path):
    gen = torch.Generator().manual_seed(816)
    hidden, intermediate = 256, 256
    rows = {}
    for expert in range(2):
        for projection in ("up_proj", "gate_proj"):
            rows[(expert, projection)] = packed(intermediate, hidden, "nvfp4", gen)
        rows[(expert, "down_proj")] = packed(hidden, intermediate, "nvfp4", gen)
    checkpoint_tensors = {
        f"experts.{expert}.{projection}.{name}": value
        for (expert, projection), tensors in rows.items() for name, value in tensors.items()}
    manifest = checkpoint(tmp_path, checkpoint_tensors)

    def load(intermediate_range):
        with DirectWeightLoader(manifest, "cuda:0") as loader:
            experts = loader.load_fp4_experts(
                ("experts.0", "experts.1"), hidden_size=hidden,
                intermediate_size=intermediate, intermediate_range=intermediate_range,
                format="nvfp4", activation_mode="a16")
        return experts

    def owner_bytes(owner):
        return owner.contiguous().view(torch.uint8).reshape(-1)

    canonical = load(None)
    for rank in (0, 1):
        shard = load((rank * 128, (rank + 1) * 128))
        for shard_owner, canonical_owner in (
                (shard._impl.w1_fp4, canonical._impl.w1_fp4),
                (shard._impl.w2_fp4, canonical._impl.w2_fp4)):
            shard_entry = provenance.lookup(shard_owner)
            canonical_entry = provenance.lookup(canonical_owner)
            expected_sources = 4 if shard_owner is shard._impl.w1_fp4 else 2
            assert_weight_pieces(shard_owner, shard_entry, expected_sources, "w4a16_mma_k16_n64_v1")
            assert_weight_pieces(canonical_owner, canonical_entry, expected_sources, "w4a16_mma_k16_n64_v1")
            shard_bytes, canonical_bytes = owner_bytes(shard_owner), owner_bytes(canonical_owner)
            for source in shard_entry["sources"]:
                canonical_source = next(
                    item for item in canonical_entry["sources"]
                    if item["tensor"] == source["tensor"]
                    and item.get("piece_encoding") == source.get("piece_encoding"))
                full_ntiles = checkpoint_tensors[source["tensor"]].shape[0] // 64
                canonical_piece = canonical_source["pieces"][0]
                for kt in range(source["pieces"][0]["extents"][0]):
                    for nt in range(source["pieces"][0]["extents"][1]):
                        piece = source["pieces"][0]
                        absolute = ((piece["source_offset"]
                                     + kt * piece["source_strides"][0]
                                     + nt * piece["source_strides"][1]) // 512)
                        canonical_base = canonical_piece["source_offset"] // 512
                        ckt, cnt = divmod(absolute - canonical_base, full_ntiles)
                        shard_offset = (piece["destination_offset"]
                                        + kt * piece["destination_strides"][0]
                                        + nt * piece["destination_strides"][1])
                        canonical_offset = (canonical_piece["destination_offset"]
                                            + ckt * canonical_piece["destination_strides"][0]
                                            + cnt * canonical_piece["destination_strides"][1])
                        assert torch.equal(
                            shard_bytes[shard_offset:shard_offset + 512],
                            canonical_bytes[canonical_offset:canonical_offset + 512]), (
                                rank, source["tensor"], kt, nt, shard_offset, canonical_offset)


@torch.no_grad()
def test_qmma_piece_metadata_matches_tp1_and_excludes_unaligned_w13(tmp_path):
    gen = torch.Generator().manual_seed(817)
    hidden, intermediate = 256, 512
    rows = {}
    for expert in range(2):
        for projection in ("up_proj", "gate_proj"):
            rows[(expert, projection)] = packed(intermediate, hidden, "mxfp4", gen)
        rows[(expert, "down_proj")] = packed(hidden, intermediate, "mxfp4", gen)
    checkpoint_tensors = {
        f"experts.{expert}.{projection}.{name}": value
        for (expert, projection), tensors in rows.items() for name, value in tensors.items()}
    manifest = checkpoint(tmp_path, checkpoint_tensors)

    def load(intermediate_range):
        with DirectWeightLoader(manifest, "cuda:0") as loader:
            return loader.load_fp4_experts(
                ("experts.0", "experts.1"), hidden_size=hidden, intermediate_size=intermediate,
                intermediate_range=intermediate_range, format="mxfp4", activation_mode="a8")

    def owner_bytes(owner):
        return owner.contiguous().view(torch.uint8).reshape(-1)

    canonical = load(None)
    for rank in (0, 1):
        shard = load((rank * 256, (rank + 1) * 256))
        for shard_owner, canonical_owner in ((shard._impl.w1_fp4, canonical._impl.w1_fp4),
                                             (shard._impl.w2_fp4, canonical._impl.w2_fp4)):
            shard_entry = provenance.lookup(shard_owner)
            canonical_entry = provenance.lookup(canonical_owner)
            expected_sources = 4 if shard_owner is shard._impl.w1_fp4 else 2
            assert_weight_pieces(shard_owner, shard_entry, expected_sources, "w4a8_qmma_n256_k128_v1")
            assert_weight_pieces(canonical_owner, canonical_entry, expected_sources, "w4a8_qmma_n256_k128_v1")
            shard_bytes, canonical_bytes = owner_bytes(shard_owner), owner_bytes(canonical_owner)
            for source in shard_entry["sources"]:
                canonical_source = next(item for item in canonical_entry["sources"]
                    if item["tensor"] == source["tensor"]
                    and item.get("piece_encoding") == source.get("piece_encoding"))
                piece = source["pieces"][0]
                canonical_piece = canonical_source["pieces"][0]
                ktiles = piece["source_strides"][1] // 16384
                for kt in range(piece["extents"][0]):
                    for nt in range(piece["extents"][1]):
                        absolute = ((piece["source_offset"] + kt * piece["source_strides"][0]
                                     + nt * piece["source_strides"][1]) // 16384)
                        base = canonical_piece["source_offset"] // 16384
                        ckt = (absolute - base) % ktiles
                        cnt = (absolute - base) // ktiles
                        shard_offset = (piece["destination_offset"]
                                        + kt * piece["destination_strides"][0]
                                        + nt * piece["destination_strides"][1])
                        canonical_offset = (canonical_piece["destination_offset"]
                                            + ckt * canonical_piece["destination_strides"][0]
                                            + cnt * canonical_piece["destination_strides"][1])
                        assert torch.equal(
                            shard_bytes[shard_offset:shard_offset + 16384],
                            canonical_bytes[canonical_offset:canonical_offset + 16384]), (
                                rank, source["tensor"], kt, nt, shard_offset, canonical_offset)


@torch.no_grad()
def test_qmma_unaligned_w13_has_no_piece_metadata(tmp_path):
    gen = torch.Generator().manual_seed(818)
    hidden, intermediate = 256, 512
    rows = {}
    for expert in range(2):
        for projection in ("up_proj", "gate_proj"):
            rows[(expert, projection)] = packed(intermediate, hidden, "mxfp4", gen)
        rows[(expert, "down_proj")] = packed(hidden, intermediate, "mxfp4", gen)
    checkpoint_tensors = {
        f"experts.{expert}.{projection}.{name}": value
        for (expert, projection), tensors in rows.items() for name, value in tensors.items()}
    manifest = checkpoint(tmp_path, checkpoint_tensors)
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        experts = loader.load_fp4_experts(
            ("experts.0", "experts.1"), hidden_size=hidden, intermediate_size=intermediate,
            intermediate_range=(0, 128), format="mxfp4", activation_mode="a8")
    entry = provenance.lookup(experts._impl.w1_fp4)
    assert entry is not None
    assert not any(source.get("pieces") for source in entry["sources"])
    assert not any(source.get("piece_encoding") for source in entry["sources"])


@torch.no_grad()
@pytest.mark.parametrize("hidden", (256, 512))
def test_iq2_descriptor_and_metadata_planes_match_tp1(tmp_path, hidden):
    from b12x.moe._shared.kernels.w4a16.iq2_xs import prepare_iq2_xs_moe_weights
    from b12x._lib.quant.block_codec import block_codec
    from ignition.components.weights import _record_iq2_expert_provenance

    def blocks(n, kb, spec, generator):
        value = torch.randint(0, 256, (2, n, kb, spec.block_bytes),
                              dtype=torch.uint8, generator=generator)
        value[..., 0] = 0
        value[..., 1] = 0x3c
        return value.cuda()

    def prepare(w13, w2, codec):
        return prepare_iq2_xs_moe_weights(
            w13, w2, hidden_size=hidden, intermediate_size=w13.shape[1], num_experts=2,
            activation="relu2", w13_layout="w31", codec=codec)

    def record(prepared, w13, w2, codec, first, last):
        sources = {}
        for expert in range(2):
            sources[f'experts.{expert}.up_proj.weight'] = w13[expert]
            sources[f'experts.{expert}.down_proj.weight'] = w2[expert]
        _record_iq2_expert_provenance(
            prepared, prefixes=('experts.0', 'experts.1'), sources=sources,
            projection_names=('up_proj', 'down_proj'), weight_name='weight',
            first=first, last=last, hidden_size=hidden, local=last - first, codec=codec)

    def covered(entry):
        return sum(math.prod(piece['extents']) * piece['contiguous_bytes']
                   for source in entry['sources'] for piece in source['pieces'])

    def assert_piece_bytes(shard_owner, shard_entry, canonical_owner, canonical_entry):
        shard_bytes = shard_owner.contiguous().view(torch.uint8).reshape(-1)
        canonical_bytes = canonical_owner.contiguous().view(torch.uint8).reshape(-1)
        for source in shard_entry['sources']:
            canonical_source = next(
                item for item in canonical_entry['sources']
                if item['tensor'] == source['tensor']
                and item['piece_encoding'] == source['piece_encoding'])
            for piece in source['pieces']:
                canonical_piece = canonical_source['pieces'][0]
                indices = __import__('itertools').product(
                    *(range(extent) for extent in piece['extents']))
                for index in indices:
                    absolute = piece['source_offset'] + sum(
                        part * stride for part, stride in zip(index, piece['source_strides']))
                    delta = absolute - canonical_piece['source_offset']
                    extents = canonical_piece['extents']
                    if not extents:
                        canonical_offset = canonical_piece['destination_offset'] + delta
                    else:
                        canonical_offset = None
                        for coordinate in __import__('itertools').product(
                                *(range(extent) for extent in extents)):
                            source_offset = canonical_piece['source_offset'] + sum(
                                part * stride for part, stride
                                in zip(coordinate, canonical_piece['source_strides']))
                            if source_offset == absolute:
                                canonical_offset = canonical_piece['destination_offset'] + sum(
                                    part * stride for part, stride
                                    in zip(coordinate, canonical_piece['destination_strides']))
                                break
                        assert canonical_offset is not None, (
                            source['tensor'], source.get('piece_encoding'), absolute)
                    shard_offset = piece['destination_offset'] + sum(
                        part * stride for part, stride in zip(index, piece['destination_strides']))
                    count = piece['contiguous_bytes']
                    assert torch.equal(shard_bytes[shard_offset:shard_offset + count],
                                       canonical_bytes[canonical_offset:canonical_offset + count]), (
                                           source['tensor'], source.get('piece_encoding'), index,
                                           shard_offset, canonical_offset)

    for codec in ('iq2_xs', 'iq2_xxs'):
        spec = block_codec(codec)
        generator = torch.Generator().manual_seed(910 if codec == 'iq2_xs' else 911)
        w13 = blocks(512, hidden // 256, spec, generator)
        w2 = blocks(hidden, 2, spec, generator)
        canonical = prepare(w13, w2, codec)
        record(canonical, w13, w2, codec, 0, 512)
        c13 = provenance.lookup(canonical.w13)
        c2 = provenance.lookup(canonical.w2)
        c13m = provenance.lookup(canonical.w13_scale)
        c2m = provenance.lookup(canonical.w2_scale)
        assert covered(c13) == canonical.w13.numel() * canonical.w13.element_size()
        assert covered(c2) == canonical.w2.numel() * canonical.w2.element_size()
        assert covered(c13m) == canonical.w13_scale.numel() * canonical.w13_scale.element_size()
        assert covered(c2m) == canonical.w2_scale.numel() * canonical.w2_scale.element_size()
        for rank in (0, 1):
            start = rank * 256
            s13 = w13[:, start:start + 256]
            s2 = w2[:, :, rank:rank + 1]
            shard = prepare(s13, s2, codec)
            record(shard, w13, w2, codec, start, start + 256)
            s13e = provenance.lookup(shard.w13)
            s2e = provenance.lookup(shard.w2)
            s13m = provenance.lookup(shard.w13_scale)
            s2m = provenance.lookup(shard.w2_scale)
            assert len(s13e['sources']) == 2 and len(s2e['sources']) == 2
            assert len(s13m['sources']) == (2 if codec == 'iq2_xxs' else 4)
            assert len(s2m['sources']) == (2 if codec == 'iq2_xxs' else 4)
            assert covered(s13e) == shard.w13.numel() * shard.w13.element_size()
            assert covered(s2e) == shard.w2.numel() * shard.w2.element_size()
            assert covered(s13m) == shard.w13_scale.numel() * shard.w13_scale.element_size()
            assert covered(s2m) == shard.w2_scale.numel() * shard.w2_scale.element_size()
            assert_piece_bytes(shard.w13, s13e, canonical.w13, c13)
            assert_piece_bytes(shard.w2, s2e, canonical.w2, c2)
            assert_piece_bytes(shard.w13_scale, s13m, canonical.w13_scale, c13m)
            assert_piece_bytes(shard.w2_scale, s2m, canonical.w2_scale, c2m)


@torch.no_grad()
def test_block_quant_loader_records_actual_wrapper_owners(tmp_path):
    from b12x._lib.quant.block_codec import block_codec
    from ignition.preparation.quantization import quantized_layers

    def ggml_checkpoint(path, tensors, entries):
        save_file(tensors, str(path / "weights.safetensors"))
        (path / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": {name: "weights.safetensors" for name in tensors}}))
        (path / "config.json").write_text(json.dumps({"quantization_config": {
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": entries}}))
        return inspect_indexed_checkpoint(path)

    for codec in ("iq2_xxs", "q8_0"):
        spec = block_codec(codec)
        generator = torch.Generator().manual_seed(920 if codec == "iq2_xxs" else 921)
        tensors, entries = {}, {}
        for expert in range(2):
            for projection, rows, columns in (("up_proj", 256, 256), ("down_proj", 256, 256)):
                key = f"experts.{expert}.{projection}"
                raw = torch.randint(0, 256, (rows, columns // spec.block_weights,
                                             spec.block_bytes), dtype=torch.uint8,
                                    generator=generator)
                raw[..., 0] = 0
                raw[..., 1] = 0x3c
                tensors[f"{key}.weight"] = raw
                entries[key] = dict(quant_algo=codec.upper(), group_size=spec.block_weights,
                                    block_payload_bytes=spec.block_bytes, packing="ggml")
        manifest = ggml_checkpoint(tmp_path, tensors, entries)
        with DirectWeightLoader(manifest, "cuda:0") as loader:
            experts = loader.load_block_quant_experts(
                ("experts.0", "experts.1"), hidden_size=256, intermediate_size=256)
        impl = experts._impl
        owners = (impl.w1_fp4, impl.w1_blockscale, impl.w2_fp4, impl.w2_blockscale)
        if codec == "q8_0":
            assert all(provenance.lookup(owner) is None for owner in owners)
        else:
            for owner in (impl.w1_fp4, impl.w2_fp4):
                entry = provenance.lookup(owner)
                assert entry is not None
                assert len(entry["sources"]) == 2
                assert all(source.get("pieces") for source in entry["sources"])
                covered = sum(math.prod(piece["extents"]) * piece["contiguous_bytes"]
                              for source in entry["sources"] for piece in source["pieces"])
                assert covered == owner.numel() * owner.element_size()
            assert all(provenance.lookup(owner) is not None
                       for owner in (impl.w1_blockscale, impl.w2_blockscale))


@torch.no_grad()
def test_mxfp4_csf_wrapper_records_actual_direct_loader_owners(tmp_path):
    from b12x._lib.quant.csf_encode import encode_csf_source

    gen = torch.Generator().manual_seed(932)
    count, hidden, intermediate = 1, 256, 256
    tensors = {}
    for projection, rows, columns in (
            ('up_proj', intermediate, hidden), ('gate_proj', intermediate, hidden),
            ('down_proj', hidden, intermediate)):
        tensors[f'experts.0.{projection}.weight'] = torch.randint(
            0, 256, (rows, columns // 2), dtype=torch.uint8, generator=gen)
        scale = torch.randint(0, 256, (rows, columns // 32), dtype=torch.uint8,
                              generator=gen).cuda()
        encoded = encode_csf_source(scale.unsqueeze(0), quantization='mxfp4')
        fixed, exceptions = encoded.components
        tensors[f'experts.0.{projection}.weight_scale.mxfp4_csf_fixed'] = fixed.cpu().reshape(-1)
        tensors[f'experts.0.{projection}.weight_scale.mxfp4_csf_exceptions'] = exceptions.cpu().reshape(-1)
    manifest = checkpoint(tmp_path, tensors)
    scratch = (torch.empty((1, hidden // 32, 2 * intermediate), dtype=torch.uint8, device='cuda'),
               torch.empty((1, intermediate // 32, hidden), dtype=torch.uint8, device='cuda'))
    with DirectWeightLoader(manifest, 'cuda:0') as loader:
        experts = loader.load_mxfp4_csf_experts(
            ('experts.0',), hidden_size=hidden, intermediate_size=intermediate,
            scale_scratch=scratch, activation_mode='a8')
    impl = experts._impl
    for owner in (impl.w1_fp4, impl.w2_fp4):
        entry = provenance.lookup(owner)
        assert entry is not None and entry['sources'], (owner.shape, entry)
    assert provenance.lookup(impl.w1_blockscale) is None



@torch.no_grad()
def test_nvfp4_csf_wrapper_records_actual_direct_loader_owners(tmp_path):
    from b12x._lib.quant.csf_encode import encode_csf_source

    gen = torch.Generator().manual_seed(933)
    count, hidden, intermediate = 1, 256, 256
    tensors = {}
    for projection, rows, columns in (
            ('up_proj', intermediate, hidden), ('gate_proj', intermediate, hidden),
            ('down_proj', hidden, intermediate)):
        tensors[f'experts.0.{projection}.weight'] = torch.randint(
            0, 256, (rows, columns // 2), dtype=torch.uint8, generator=gen)
        scale = torch.randint(0, 256, (rows, columns // 16), dtype=torch.uint8,
                              generator=gen).cuda()
        encoded = encode_csf_source(scale.unsqueeze(0), quantization='nvfp4')
        fixed, exceptions = encoded.components
        tensors[f'experts.0.{projection}.weight_scale.nvfp4_csf_fixed'] = fixed[0].cpu()
        tensors[f'experts.0.{projection}.weight_scale.nvfp4_csf_exceptions'] = exceptions.cpu().reshape(-1)
    manifest = checkpoint(tmp_path, tensors)
    scratch = (torch.empty((1, 2 * intermediate, hidden // 16), dtype=torch.float8_e4m3fn, device='cuda'),
               torch.empty((1, hidden, intermediate // 16), dtype=torch.float8_e4m3fn, device='cuda'))
    vectors = tuple(torch.full((1,), value, dtype=torch.float32) for value in (0.5, 0.5, 1.0, 1.0))
    with DirectWeightLoader(manifest, 'cuda:0') as loader:
        experts = loader.load_nvfp4_experts(
            ('experts.0',), hidden_size=hidden, intermediate_size=intermediate,
            scale_vectors=vectors, scale_scratch=scratch, activation_mode='a4')
    impl = experts._impl
    for owner in (impl.w1_fp4, impl.w2_fp4):
        entry = provenance.lookup(owner)
        assert entry is not None and entry['sources'], (owner.shape, entry)
    assert provenance.lookup(impl.w1_blockscale) is None

def _csf_checkpoint(tmp_path, codec, *, hidden=256, intermediate=512, experts=2):
    """Real checkpoint with CSF-encoded scales and packed expert weights."""
    generator = torch.Generator().manual_seed(940 if codec == "mxfp4" else 941)
    group = 32 if codec == "mxfp4" else 16
    tensors = {}
    for expert in range(experts):
        for projection, rows, columns in (
                ("up_proj", intermediate, hidden),
                ("gate_proj", intermediate, hidden),
                ("down_proj", hidden, intermediate)):
            tensors[f"experts.{expert}.{projection}.weight"] = torch.randint(
                0, 256, (rows, columns // 2), dtype=torch.uint8, generator=generator)
            native = torch.randint(
                0, 256, (rows, columns // group), dtype=torch.uint8, generator=generator).cuda()
            encoded = encode_csf_source(native.unsqueeze(0), quantization=codec)
            fixed, exceptions = encoded.components
            suffix = "mxfp4_csf" if codec == "mxfp4" else "nvfp4_csf"
            base = f"experts.{expert}.{projection}.weight_scale"
            tensors[f"{base}.{suffix}_fixed"] = (
                fixed.cpu().reshape(-1) if codec == "mxfp4" else fixed[0].cpu())
            tensors[f"{base}.{suffix}_exceptions"] = exceptions.cpu().reshape(-1)
    return checkpoint(tmp_path, tensors)


def _load_csf(manifest, codec, intermediate_range, *, expanded=False):
    with DirectWeightLoader(manifest, "cuda:0") as loader:
        if codec == "mxfp4":
            local = 512 if intermediate_range is None else intermediate_range[1] - intermediate_range[0]
            scratch = (
                torch.empty((2, 256 // 32, 2 * local), dtype=torch.uint8, device="cuda"),
                torch.empty((2, local // 32, 256), dtype=torch.uint8, device="cuda"),
            )
            return loader.load_mxfp4_csf_experts(
                ("experts.0", "experts.1"), hidden_size=256, intermediate_size=512,
                intermediate_range=intermediate_range, scale_scratch=scratch,
                activation_mode="a8")
        local = 512 if intermediate_range is None else intermediate_range[1] - intermediate_range[0]
        scratch = None
        if not expanded:
            scratch = (
                torch.empty((2, (2 * local + 127) // 128 * 128, 16),
                            dtype=torch.float8_e4m3fn, device="cuda"),
                torch.empty((2, 256, (local // 16 + 3) // 4 * 4),
                            dtype=torch.float8_e4m3fn, device="cuda"),
            )
        vectors = tuple(torch.full((2,), value, dtype=torch.float32) for value in (.5, .5, 1., 1.))
        return loader.load_nvfp4_experts(
            ("experts.0", "experts.1"), hidden_size=256, intermediate_size=512,
            intermediate_range=intermediate_range, scale_vectors=vectors,
            scale_scratch=scratch, activation_mode="a4")


def _piece_runs(piece):
    import itertools
    for index in itertools.product(*(range(extent) for extent in piece["extents"])):
        source = piece["source_offset"] + sum(
            part * stride for part, stride in zip(index, piece["source_strides"]))
        destination = piece["destination_offset"] + sum(
            part * stride for part, stride in zip(index, piece["destination_strides"]))
        yield source, destination, piece["contiguous_bytes"]


def _assert_piece_coverage(entry, owner):
    total = 0
    for source in entry["sources"]:
        for piece in source.get("pieces", ()):
            total += sum(length for _, _, length in _piece_runs(piece))
    assert total == owner.numel() * owner.element_size(), (total, owner.numel() * owner.element_size())


def _assert_contained_piece_bytes(shard_owner, shard_entry, canonical_owner, canonical_entry):
    shard_bytes = shard_owner.detach().contiguous().view(torch.uint8).reshape(-1).cpu()
    canonical_bytes = canonical_owner.detach().contiguous().view(torch.uint8).reshape(-1).cpu()
    for source in shard_entry["sources"]:
        canonical_source = next(
            item for item in canonical_entry["sources"]
            if item["tensor"] == source["tensor"]
            and item.get("piece_encoding") == source.get("piece_encoding"))
        canonical_runs = [run for piece in canonical_source["pieces"] for run in _piece_runs(piece)]
        for piece in source["pieces"]:
            for source_offset, destination_offset, length in _piece_runs(piece):
                containing = next(
                    (run for run in canonical_runs
                     if run[0] <= source_offset
                     and source_offset + length <= run[0] + run[2]),
                    None)
                assert containing is not None, (
                    source["tensor"], source.get("piece_encoding"), source_offset, length)
                canonical_destination = containing[1] + (source_offset - containing[0])
                assert torch.equal(
                    shard_bytes[destination_offset:destination_offset + length],
                    canonical_bytes[canonical_destination:canonical_destination + length]), (
                        source["tensor"], source.get("piece_encoding"), source_offset)


@pytest.mark.parametrize("expanded", [False, True])
@torch.no_grad()
def test_nvfp4_csf_tp2_pieces_reconstruct_tp1_containment(tmp_path, expanded):
    manifest = _csf_checkpoint(tmp_path, "nvfp4")
    canonical = _load_csf(manifest, "nvfp4", None, expanded=expanded)
    for shard_range in ((0, 256), (256, 512)):
        shard = _load_csf(manifest, "nvfp4", shard_range, expanded=expanded)
        for shard_owner, canonical_owner in (
                (shard._impl.w1_fp4, canonical._impl.w1_fp4),
                (shard._impl.w2_fp4, canonical._impl.w2_fp4)):
            shard_entry = provenance.lookup(shard_owner)
            canonical_entry = provenance.lookup(canonical_owner)
            assert shard_entry is not None and canonical_entry is not None
            assert any(source.get("pieces") for source in shard_entry["sources"])
            assert all(source.get("pieces") for source in shard_entry["sources"])
            _assert_piece_coverage(shard_entry, shard_owner)
            _assert_piece_coverage(canonical_entry, canonical_owner)
            _assert_contained_piece_bytes(
                shard_owner, shard_entry, canonical_owner, canonical_entry)


@torch.no_grad()
def test_mxfp4_csf_tp2_pieces_reconstruct_tp1_containment(tmp_path):
    manifest = _csf_checkpoint(tmp_path, "mxfp4")
    canonical = _load_csf(manifest, "mxfp4", None)
    for shard_range in ((0, 256), (256, 512)):
        shard = _load_csf(manifest, "mxfp4", shard_range)
        for shard_owner, canonical_owner in (
                (shard._impl.w1_fp4, canonical._impl.w1_fp4),
                (shard._impl.w2_fp4, canonical._impl.w2_fp4)):
            shard_entry = provenance.lookup(shard_owner)
            canonical_entry = provenance.lookup(canonical_owner)
            assert shard_entry is not None and canonical_entry is not None
            assert shard_entry["sources"] and canonical_entry["sources"]
            assert all(source.get("pieces") for source in shard_entry["sources"])
            assert all(source.get("pieces") for source in canonical_entry["sources"])
            _assert_piece_coverage(shard_entry, shard_owner)
            _assert_piece_coverage(canonical_entry, canonical_owner)
            _assert_contained_piece_bytes(
                shard_owner, shard_entry, canonical_owner, canonical_entry)
