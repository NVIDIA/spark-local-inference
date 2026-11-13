# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only source records for testing native payload publication and signatures."""

from functools import lru_cache
from pathlib import Path
import hashlib
import struct
import subprocess
import tempfile

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from safetensors import safe_open

from ignition.preparation import runtime_core_pb2 as proto
from ignition.preparation.core_file import combine, write_sources


def rank_source(root, *, data=b'weight-bytes', index=None):
    root.mkdir(parents=True, exist_ok=True)
    index = proto.ModelIndex(version=4) if index is None else index
    if not index.ranks:
        index.ranks.add()
    rank = index.ranks[0]
    rank.architecture[:] = [12, 0]
    rank.context.SetInParent()
    rank.allocations.add(name='device-0', nbytes=len(data), alignment=256,
                         address=0x1000, file='weights', data_spans=[proto.Span(nbytes=len(data))])
    rank.modules.add(name='kernel', file='cubin')
    rank.functions.add(module='kernel', symbol='decode')
    rank.graphs.add(name='decode', resource='graph', node_count=1, functions=[0])
    index.contract.SetInParent()
    index.frontend.SetInParent()
    index.sampling_defaults.SetInParent()
    payload = []
    for offset, (name, content) in enumerate((('weights', data), ('cubin', b'cubin-bytes'), ('graph', b'graph-bytes'))):
        source = root / name
        source.write_bytes(content)
        rank.resources[name].CopyFrom(proto.Span(offset=offset * 4096, nbytes=len(content)))
        payload.append((offset * 4096, source, 0, len(content)))
    path = root / 'rank.json'
    write_sources(path, index, payload, hardware='gb202')
    return path


def publish(root, signer):
    source = rank_source(root / 'source')
    destination = root / 'core'
    combine(destination, [source], [], publisher=signer)
    return destination


@lru_cache
def schema():
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / 'schema.pb'
        subprocess.run(['protoc', '-I', str(root / 'helios/proto'), '--include_imports',
                        f'--descriptor_set_out={target}', 'core.proto'], check=True)
        files = descriptor_pb2.FileDescriptorSet.FromString(target.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for file in files.file:
        pool.Add(file)
    return lambda name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f'flare.core.{name}'))


def manifest(path):
    primary = path / 'model.safetensors' if path.is_dir() else path
    with safe_open(primary, framework='numpy') as file:
        return schema()('ProtectedCore').FromString(file.get_tensor('flare.manifest').tobytes())


def verify_signatures(path, public):
    envelope = manifest(path)
    assert envelope.version == 2
    assert envelope.signer == hashlib.sha256(public.public_bytes_raw()).digest()
    bodies = [(1, 0, 0, envelope.core_body, envelope.core_signature)]
    bodies += [(2, item.index, 0, item.body, item.signature) for item in envelope.deployments]
    bodies += [(3, item.deployment, item.rank, item.body, item.signature) for item in envelope.ranks]
    for kind, first, second, body, signature in bodies:
        fields = (b'flare-package-signature-v5', struct.pack('<I', 2), envelope.core_id,
                  envelope.signer, struct.pack('<I', kind), struct.pack('<I', first),
                  struct.pack('<I', second), body)
        public.verify(signature, b''.join(struct.pack('<Q', len(field)) + field for field in fields))
    return envelope
