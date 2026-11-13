# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU mechanism probe for queried capture IDs; not a model performance test."""
import argparse
import json
import os
from pathlib import Path

from qualify.graph_nodes import CaptureQuery, NodeScopes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('assign CUDA_VISIBLE_DEVICES explicitly')
    import torch
    torch.cuda.set_device(0)
    query = CaptureQuery()
    source = torch.arange(1024, device='cuda', dtype=torch.float32)
    temporary, output = torch.empty_like(source), torch.empty_like(source)
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.add(source, 2, out=temporary)
        torch.mul(temporary, 3, out=output)
    stream.synchronize()
    try:
        with torch.cuda.graph(graph, stream=stream):
            scopes = NodeScopes(query, stream.cuda_stream)
            with scopes.scope('probe', layer=None):
                with scopes.scope('add', layer=0):
                    torch.add(source, 2, out=temporary)
                with scopes.scope('multiply', layer=1):
                    torch.mul(temporary, 3, out=output)
            mapping = scopes.finish_capture()
        mapping.update(graph_exec_id=query.exec_id(graph.raw_cuda_graph_exec()),
                       name='probe', cupti_library=query.cupti_path)
        if mapping['unassigned_node_ids']:
            raise RuntimeError('probe has unattributed nodes')
        before = torch.cuda.memory_stats()
        with torch.cuda.stream(stream):
            for _ in range(3):
                graph.replay()
        stream.synchronize()
        after = torch.cuda.memory_stats()
        allocation_keys = ('allocation.all.allocated', 'allocation.all.freed',
                           'segment.all.allocated', 'segment.all.freed')
        mapping['allocator_stable'] = all(before[k] == after[k] for k in allocation_keys)
        mapping['output_equal'] = torch.equal(output.cpu(), (torch.arange(1024) + 2) * 3)
        mapping['expected_replays'] = 3
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(mapping, indent=2) + '\n')
        if not mapping['allocator_stable'] or not mapping['output_equal']:
            raise RuntimeError('probe replay check failed')
    finally:
        stream.synchronize()
        graph.reset()


if __name__ == '__main__':
    main()
