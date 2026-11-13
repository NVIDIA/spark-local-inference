# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect generated kernels in explicit ForwardCache artifacts without model weights.

The default reads serialized source and static-autotuner bindings on the CPU.
--load also creates CUDA kernel launchers and validates live bindings. It never
executes a model forward, warms kernels, or captures a graph.
"""

import argparse
import ast
import io
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path, nargs='+')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--load', action='store_true', help='load launchers on CUDA and validate live bindings')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply

    configuration = Configuration.load(args.config)
    devices = ()
    if args.load:
        from ignition.cli.preparation import Host

        available = {gpu.index: gpu for gpu in Host.discover().gpus}
        if args.gpu not in available:
            parser.error(f'GPU {args.gpu} is not visible')
        devices = (available[args.gpu],)
    apply(configuration, devices)

    import torch
    from torch._dynamo.aot_compile import AOTCompileUnpickler
    from torch._inductor.async_compile import CompiledTritonKernels
    from ignition.preparation.aot import BoundForward, StorageBackend, _guard_code_cache, _load_inductor
    from ignition.preparation.device_binding import bind_device
    from ignition.preparation.inductor_tuning import InductorSelections, _kernel_globals

    torch.set_num_threads(1)
    if args.load:
        torch.cuda.set_device(0)
    failures = 0
    for path in args.artifacts:
        with _guard_code_cache(path.parent / 'guards'):
            state = AOTCompileUnpickler({}, io.BytesIO(path.read_bytes())).load()
            deserializer, serialized = state['compiled_fn']
            if deserializer != StorageBackend.deserialize_compile_artifacts:
                raise TypeError(f'{path} does not contain a Flare storage backend')
            compiled, _, _, _, frozen = pickle.loads(serialized)
            entry, kernels, constants = pickle.loads(compiled)
        if kernels or constants:
            raise ValueError(f'{path} contains unsupported static native bindings')
        graph = entry.compiled_fw.result
        static = ({} if graph._triton_bundle is None else
                  {item.cache_key: item for item in graph._triton_bundle.static_autotuners})
        generated = {}
        for node in ast.walk(ast.parse(graph.source_code)):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == 'async_compile.triton'):
                name, = node.targets
                source = ast.literal_eval(node.value.args[1])
                generated[name.id] = CompiledTritonKernels.key(source)
        report = dict(artifact=str(path), graph_key=graph.cache_key, frozen=frozen,
            generated=len(generated), static_entries=len(static),
            missing_static=[dict(name=name, source_key=key) for name, key in generated.items() if key not in static],
            unused_static=[dict(name=item.kernel_name, source_key=key)
                           for key, item in static.items() if key not in generated.values()])
        if args.load:
            ordinal, = graph.device_idxs
            with (torch.no_grad(), torch._inductor.config.patch(compile_threads=1),
                  bind_device('cuda:0'), _guard_code_cache(path.parent / 'guards')):
                artifact, _ = _load_inductor(compiled, torch.device('cuda', ordinal), frozen=frozen)
            loaded = artifact._flare_graph
            module = _kernel_globals(loaded) if generated else {}
            report['live_bindings'] = {name: f'{type(module.get(name)).__module__}.{type(module.get(name)).__qualname__}'
                                       for name in generated}
            try:
                backend = SimpleNamespace(artifact=artifact)
                holder = SimpleNamespace(_artifacts=SimpleNamespace(compiled_fn=backend))
                InductorSelections('cuda:0').register(BoundForward(holder, None, True))
                report['registered'] = True
            except Exception as error:
                report['registered'] = False
                report['error'] = f'{type(error).__name__}: {error}'
                failures += 1
        print(json.dumps(report), flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
