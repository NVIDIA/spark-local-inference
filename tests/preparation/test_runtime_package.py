# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native export validation and cross-process Helios CUDA graph replay."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ignition.preparation.runtime_package import FileSource


@pytest.mark.parametrize("flags", [0, 1, 8, 9])
def test_template_export_declares_flags_without_an_executable(flags):
    from types import SimpleNamespace
    from ignition.preparation.runtime_package import _graphs

    def no_executable():
        pytest.fail("template export must not request an executable")

    graph = SimpleNamespace(raw_cuda_graph=lambda: 123, raw_cuda_graph_exec=no_executable,
                            _flare_instantiation_flags=flags)
    assert _graphs({"decode": graph}) == [dict(name="decode", graph=123, instantiate_flags=flags)]


@pytest.mark.parametrize("flags", [-1, True, 2, 4, 16, 1.0])
def test_template_export_rejects_unsupported_flags(flags):
    from types import SimpleNamespace
    from ignition.preparation.runtime_package import _graphs

    with pytest.raises(ValueError, match="instantiation flags"):
        _graphs({"decode": SimpleNamespace(_flare_instantiation_flags=flags)})


@pytest.mark.parametrize("graph", [dict(name="decode"), dict(name="decode", recorded=1, graph=1),
    dict(name="decode", recorded=1, executable=1), dict(name="decode", recorded=1, instantiate_flags=1),
    dict(name="decode", graph=1, executable=1, instantiate_flags=1),
    dict(name="decode", graph=1, instantiate_flags=2), dict(name="decode", instantiate_flags=1)])
def test_export_rejects_inconsistent_graph_resources_and_retains_undrained_owners(tmp_path, graph):
    import gc
    import weakref
    import flare_heliopause

    class Owner:
        pass

    owner = Owner()
    reference = weakref.ref(owner)
    with pytest.raises(RuntimeError, match="named graphs require.*preparation owners retained"):
        flare_heliopause.export_capture(json.dumps(dict(path=str(tmp_path / "rank.json"),
            context=0, graphs=[graph], allocations=[])), (owner,), None)
    del owner
    gc.collect()
    assert reference() is not None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("graph", [dict(name="decode", graph=1, executable=1),
    dict(name="decode", recorded=1), dict(name="decode", graph=1, instantiate_flags=0)])
def test_invalid_serving_metadata_fails_before_cuda_or_weight_copy(tmp_path, graph):
    import flare_heliopause

    request = dict(path=str(tmp_path / "invalid.safetensors"), context=0,
        graphs=[graph], allocations=[],
        execution=None, package_data=None, rebuild_weights=None, shard_size=64 << 30,
        server=dict(assets={"tokenizer": {}, "chat_template": {}}, frontend={},
            sampling_defaults=dict(seed=0, temperature=1., top_p=.95, top_k=-1, min_p=0.,
                repetition_penalty=1., presence_penalty=0., frequency_penalty=0.),
            scheduler_policy="throughput"))
    with pytest.raises(ValueError, match="scheduler_policy"):
        flare_heliopause.export_capture(json.dumps(request), (), None)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_native_graph_export_preserves_construction_order_across_branches(tmp_path):
    import ctypes as C
    from types import SimpleNamespace

    import torch
    from helios.tests.cuda_graph_fixture import MemsetParams
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_file import read_sources
    from ignition.preparation import runtime_package_pb2 as proto
    from ignition.preparation.runtime_package import Allocation, RuntimePackage, _Cuda

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    data = torch.zeros(256, dtype=torch.uint8, device="cuda:0")
    cuda = _Cuda()
    with ModuleImages() as images:
        data.add_(1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            data.add_(1)
        nodes = []
        context = C.c_void_p()
        cuda.call("cuCtxGetCurrent", C.byref(context))
        for value in (17, 29, 41):
            params = MemsetParams(dst=data.data_ptr() + len(nodes) * 64, pitch=0,
                                  value=value, element_size=1, width=64, height=1)
            node = C.c_void_p()
            dependencies = (C.c_void_p * 1)(nodes[0].value) if nodes else None
            cuda.call("cuGraphAddMemsetNode", C.byref(node), C.c_void_p(graph.raw_cuda_graph()),
                      dependencies, C.c_size_t(int(bool(nodes))), C.byref(params), context)
            nodes.append(node)
        executable = C.c_void_p()
        cuda.call("cuGraphInstantiateWithFlags", C.byref(executable),
                  C.c_void_p(graph.raw_cuda_graph()), C.c_uint64(0))
        paths = []
        try:
            for recorded in (False, True):
                if recorded:
                    images.native.record_graph(int(graph.raw_cuda_graph()), executable.value)
                    graph.reset()
                owner = RuntimePackage(allocations=[Allocation("weight", data.data_ptr(), data.nbytes)],
                                       modules={}, kernels=[], recorder=images.native, owners=(data, graph))
                path = tmp_path / ("recorded.json" if recorded else "live.json")
                owner.write(path, {"branches": SimpleNamespace(raw_cuda_graph=(lambda: 0) if recorded else graph.raw_cuda_graph,
                    raw_cuda_graph_exec=lambda: executable.value)}, sources=True)
                paths.append(path)
        finally:
            images.native.release_graph(executable.value)
            cuda.call("cuGraphExecDestroy", executable)
    import struct
    exports = []
    for path in paths:
        index, locate = read_sources(path)
        description, = index.ranks[0].graphs
        contents = bytearray()
        for source, offset, nbytes in locate(index.ranks[0].resources[description.resource]):
            with source.open("rb") as stream:
                stream.seek(offset)
                contents.extend(stream.read(nbytes))
        decoded, cursor = [], 0
        for record in [proto.Node] * description.node_count + [proto.Edge] * description.edge_count:
            size, = struct.unpack_from("<I", contents, cursor)
            cursor += 4
            decoded.append(record.FromString(contents[cursor:cursor + size]))
            cursor += size
        assert cursor == len(contents)
        exports.append(decoded)
    assert exports[0] == exports[1]
    memset = [(i, node.memset.value) for i, node in enumerate(decoded[:description.node_count])
              if node.WhichOneof("operation") == "memset"]
    assert [value for _, value in memset] == [17, 29, 41]
    first, second, third = [i for i, _ in memset]
    edges = {(edge.source, edge.destination) for edge in decoded[description.node_count:]}
    assert {(first, second), (first, third)} <= edges


def test_checkpoint_source_requires_a_complete_nonempty_file_range(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"header" + bytes(128))
    assert FileSource(checkpoint, 6, 128).nbytes == 128
    for offset, size in ((-1, 1), (6, 0), (6, 129)):
        with pytest.raises(ValueError, match="within its checkpoint"):
            FileSource(checkpoint, offset, size)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
@pytest.mark.parametrize("fixture,binary,expected,recorded", [
    ("runtime_package_fixture.py", "test_runtime_package", "192 graph launches", False),
    ("fixed_package_fixture.py", "test_fixed_runtime_package", "native fixed-address replay passed", False),
    ("runtime_package_fixture.py", "test_runtime_package", "192 graph launches", True),
    ("runtime_package_fixture.py", "test_runtime_package", "192 graph launches", "template"),
])
def test_payload_forms_preserve_signatures_and_replay(tmp_path, fixture, binary, expected, recorded):
    import shutil

    import flare_heliopause
    from safetensors import safe_open

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        "cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools",
        "--bin", binary,
    ], check=True, cwd=root, timeout=180)
    original = tmp_path / "original"
    result = subprocess.run([
        sys.executable, str(root / "helios/tests" / fixture), str(original),
        *(["--recorded-graphs", recorded] if isinstance(recorded, str) else ["--recorded-graphs"] if recorded else []),
    ], cwd=root, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    if fixture == "runtime_package_fixture.py":
        assert (original / "package/weights/misc.safetensors").is_file()

    packed = tmp_path / "packed"
    restored = tmp_path / "restored"
    packed.mkdir()
    restored.mkdir()
    flare_heliopause.export_payload_convert(str(original / "package"), str(packed / "package"), "packed")
    flare_heliopause.export_payload_convert(str(packed / "package"), str(restored / "package"), "directory")

    primaries = [original / "package/model.safetensors", packed / "package", restored / "package/model.safetensors"]
    manifests = []
    for primary in primaries:
        with safe_open(primary, framework="numpy") as tensors:
            assert tensors.metadata()["format"] == "flare-payload"
            manifests.append(tensors.get_tensor("flare.manifest").tobytes())
    assert manifests[0] == manifests[1] == manifests[2]
    if fixture == "fixed_package_fixture.py":
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
        descriptor = tmp_path / 'package-schema.pb'
        subprocess.run(['protoc', '-I', str(root / 'helios/proto'), '--include_imports',
                        f'--descriptor_set_out={descriptor}', 'package.proto'], check=True)
        pool = descriptor_pool.DescriptorPool()
        for file in descriptor_pb2.FileDescriptorSet.FromString(descriptor.read_bytes()).file:
            pool.Add(file)
        envelope_type = message_factory.GetMessageClass(pool.FindMessageTypeByName('flare.package.ProtectedPackage'))
        rank_type = message_factory.GetMessageClass(pool.FindMessageTypeByName('flare.package.RankBody'))
        envelope = envelope_type.FromString(manifests[0])
        rank = rank_type.FromString(envelope.ranks[0].body)
        assert 'transform_' + 'template_parameter_' * 32 in {function.symbol for function in rank.functions}
        assert 'bias_' + 'template_parameter_' * 32 in {symbol.symbol for module in rank.modules for symbol in module.globals}
        assert 'assertFile_0' in {symbol.symbol for module in rank.modules for symbol in module.globals}
    companions = sorted(path for path in (original / "package").rglob("*.safetensors")
                        if path.name != "model.safetensors")
    assert companions
    for companion in companions:
        relative = companion.relative_to(original / "package")
        with safe_open(companion, framework="numpy") as left, safe_open(restored / "package" / relative, framework="numpy") as right:
            assert left.keys() == right.keys() == ["flare.payload"]
            assert left.get_tensor("flare.payload").tobytes() == right.get_tensor("flare.payload").tobytes()

    for directory in (original, packed, restored):
        if directory != original:
            for fixture_file in original.iterdir():
                if fixture_file.is_file():
                    shutil.copyfile(fixture_file, directory / fixture_file.name)
        result = subprocess.run([
            str(root / "helios/target/debug" / binary), str(directory),
        ], cwd=root, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert expected in result.stdout



@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_captured_weight_payloads_survive_relocation_and_capacity_changes(tmp_path):
    from safetensors import safe_open

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        "cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools",
        "--bin", "test_runtime_package",
    ], check=True, cwd=root, timeout=180)
    payloads, addresses = [], []
    variants = [
        ("baseline", {}),
        ("relocated", dict(capacity=129, allocation_padding=64 << 20)),
        ("changed", dict(capacity=129, allocation_padding=64 << 20, weight_delta=4.)),
    ]
    for name, options in variants:
        directory = tmp_path / name
        directory.mkdir()
        result = subprocess.run([
            sys.executable, "-c",
            "import json, sys; from pathlib import Path; "
            "sys.path.insert(0, 'helios/tests'); "
            "from runtime_package_fixture import prepare; "
            "prepare(Path(sys.argv[1]), layer_groups=True, **json.loads(sys.argv[2]))",
            str(directory), json.dumps(options),
        ], cwd=root, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        groups = {}
        for path in (directory / "package/weights").glob("*.safetensors"):
            with safe_open(path, framework="numpy") as tensors:
                groups[path.stem] = tensors.get_tensor("flare.payload").tobytes()
        assert set(groups) == {"layer-000", "layer-001"}
        payloads.append(groups)
        addresses.append(json.loads((directory / "expected.json").read_text())["addresses"])
        result = subprocess.run([
            str(root / "helios/target/debug/test_runtime_package"), str(directory),
        ], cwd=root, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "192 graph launches" in result.stdout

    assert addresses[0]["weight"] != addresses[1]["weight"]
    assert payloads[0] == payloads[1]
    assert payloads[1]["layer-000"] != payloads[2]["layer-000"]
    assert payloads[1]["layer-001"] == payloads[2]["layer-001"]


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
@pytest.mark.parametrize("fixture,binary,expected,exec_handover", [
    ("runtime_package_fixture.py", "test_runtime_package", "192 graph launches", False),
    ("fixed_package_fixture.py", "test_fixed_runtime_package", "native fixed-address replay passed", False),
    ("fixed_package_fixture.py", "test_fixed_runtime_package", "native fixed-address replay passed", True),
])
@pytest.mark.parametrize("shard_size", [64 << 30, 64 << 20])
def test_package_replays_without_python(tmp_path, fixture, binary, expected, exec_handover, shard_size):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        "cargo", "build", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", binary,
    ], check=True, cwd=root, timeout=180)
    executable = str(root / "helios/target/debug" / binary)
    arguments = [sys.executable, str(root / "helios/tests" / fixture), str(tmp_path)]
    if exec_handover:
        arguments.append(executable)
    arguments += ["--shard-size", str(shard_size), "--legacy"]
    result = subprocess.run(arguments, cwd=root, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    package_path = tmp_path / "package"
    from ignition.preparation.package_file import read_index

    index, start = read_index(package_path)
    if fixture == 'runtime_package_fixture.py':
        from ignition.preparation.rebuild_manifest import validate_weights
        validate_weights(index.ranks[0])
        assert index.ranks[0].rebuild_weights.tensors[0].name == 'weight'
    else:
        assert not index.ranks[0].HasField('rebuild_weights')
    from safetensors import safe_open

    with safe_open(package_path, framework="numpy") as tensors:
        assert tensors.keys() == ["flare.layout", "flare.metadata", "flare.payload"]
        assert tensors.get_tensor("flare.payload").size == 0
        from ignition.preparation import runtime_package_pb2 as proto
        protected = proto.ProtectedIndex.FromString(tensors.get_tensor("flare.metadata").tobytes())
        shared = proto.PublisherMetadata.FromString(protected.groups[0].body).index
        signed = proto.RankProtection.FromString(protected.ranks[0].body)
        shared.ranks.add().CopyFrom(signed.bundle)
        assert shared == index
    document = index.ranks[0]
    storage = {allocation.file for allocation in document.allocations if allocation.HasField("file")}
    storage_end = max((document.resources[key].offset + document.resources[key].nbytes for key in storage), default=0)
    assert all(not span.filename for key, span in document.resources.items() if key not in storage)
    if index.version == 4:
        assert all(span.offset >= storage_end for key, span in document.resources.items() if key not in storage)
    else:
        assert all(document.resources[key].filename for key in storage)
    def contents(key):
        span = document.resources[key]
        from ignition.preparation.package_file import PackageFiles
        with package_path.open("rb") as source, PackageFiles(package_path, source, start, 0) as files:
            file, base, _ = files.get(span.filename)
            stored = file.read_at(span.nbytes, base + span.offset)
        resource = next((item for item in signed.executables.resources if item.offset == span.offset), None)
        if resource is None:
            return stored
        from helios.tests.signing import zstd_decompress
        assert resource.stored_bytes == len(stored)
        return zstd_decompress(stored)
    for module in document.modules:
        assert b".py\0" not in contents(module.file)
        for symbol in module.globals:
            assert b"/preparation/internal/kernel.py" not in contents(symbol.file)
    if fixture == "fixed_package_fixture.py":
        assert any(contents(symbol.file).rstrip(b"\0") == b"kernel"
                   for module in document.modules for symbol in module.globals)
        for name in (b"numeric_argument", b"integer_pointer_argument", b"pointer_table_argument",
                     b"transform_" + b"template_parameter_" * 32, b"no_arguments", b"assertFile_0"):
            assert name in document.SerializeToString()
            assert any(name + b"\0" in contents(module.file) for module in document.modules)
    if not exec_handover:
        result = subprocess.run([executable, str(tmp_path)],
                                cwd=root, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout



@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_cooperative_package_clamps_to_device_capacity(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        sys.executable, str(root / "helios/tests/cooperative_package_fixture.py"), str(tmp_path),
    ], check=True, cwd=root, timeout=120)
    subprocess.run([
        "cargo", "test", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package", "--lib",
        "oversized_cooperative_grids_replay_at_device_capacity", "--", "--ignored",
    ], check=True, cwd=root, timeout=180,
        env={**os.environ, "FLARE_COOPERATIVE_TEST_PACKAGE": str(tmp_path)})



@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires explicit CUDA test opt-in")
def test_captured_shared_memory_exceeds_function_allowance(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES before running GPU tests"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([
        sys.executable, str(root / "helios/tests/shared_memory_package_fixture.py"), str(tmp_path),
    ], check=True, cwd=root, timeout=120)
    subprocess.run([
        "cargo", "test", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package", "--lib",
        "captured_shared_memory_exceeding_function_allowance_replays", "--", "--ignored",
    ], check=True, cwd=root, timeout=180,
        env={**os.environ, "FLARE_SHARED_MEMORY_TEST_PACKAGE": str(tmp_path)})
