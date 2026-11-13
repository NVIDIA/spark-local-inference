# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify PyTorch graph handles against the native CUDA replay worker."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import weakref

import torch


class GraphResources:
    def __init__(self):
        self.inputs = [torch.arange(size, device="cuda", dtype=torch.float32) + 1 for size in (4, 11)]
        self.outputs = [torch.zeros_like(value) for value in self.inputs]
        self.stream = torch.cuda.Stream()
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = []
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for index, (value, output) in enumerate(zip(self.inputs, self.outputs)):
                for _ in range(3):
                    output.add_(value * (index + 1))
        self.stream.synchronize()
        for index, (value, output) in enumerate(zip(self.inputs, self.outputs)):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
                output.add_(value * (index + 1))
            graph.instantiate()
            self.graphs.append(graph)
        for output in self.outputs:
            output.zero_()
        torch.cuda.synchronize()

    def worker(self, native, *, capacity=8, device=None):
        context, actual_device = native.current_context()
        return native.GraphReplayWorker(
            self,
            context,
            actual_device if device is None else device,
            [graph.raw_cuda_graph_exec() for graph in self.graphs],
            [(value.data_ptr(), value.numel() * value.element_size()) for value in self.inputs + self.outputs],
            queue_capacity=capacity,
        )


class CudaGraphReplayTests(unittest.TestCase):
    native = None

    @classmethod
    def setUpClass(cls):
        if cls.native is None:
            raise unittest.SkipTest("run this file with --extension and an explicitly assigned CUDA_VISIBLE_DEVICES")

    def assert_completion(self, worker, ticket, index, repetitions):
        completion = worker.poll(timeout_ms=10_000)
        self.assertIsNotNone(completion, "native replay did not complete within 10 seconds")
        self.assertEqual(completion.ticket, ticket)
        self.assertEqual(completion.graph_index, index)
        self.assertEqual(completion.repetitions, repetitions)
        self.assertIsNone(completion.error)

    def test_changed_inputs_variants_and_native_repetitions(self):
        owner = GraphResources()
        worker = owner.worker(self.native, capacity=2)
        try:
            expected = [value.cpu() * (index + 1) * repeats for index, (value, repeats) in enumerate(zip(owner.inputs, (7, 5)))]
            with mock.patch.object(torch.cuda.CUDAGraph, "replay", side_effect=AssertionError("Python replay was invoked")):
                tickets = [worker.submit(0, repetitions=7), worker.submit(1, repetitions=5)]
                with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
                    worker.submit(0)
                self.assert_completion(worker, tickets[0], 0, 7)
                self.assert_completion(worker, tickets[1], 1, 5)
                for actual, reference in zip(owner.outputs, expected):
                    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

                owner.inputs[0].fill_(13)
                owner.inputs[1].fill_(-9)
                torch.cuda.synchronize()
                expected[0].add_(13 * 17)
                expected[1].add_(-9 * 2 * 11)
                tickets = [worker.submit(1, repetitions=11), worker.submit(0, repetitions=17)]
                self.assert_completion(worker, tickets[0], 1, 11)
                self.assert_completion(worker, tickets[1], 0, 17)
            for actual, reference in zip(owner.outputs, expected):
                torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
            self.assertEqual(worker.outstanding, 0)
        finally:
            worker.close()

    def test_owner_lifetime_and_close_drain(self):
        owner = GraphResources()
        owner_ref = weakref.ref(owner)
        outputs = owner.outputs.copy()
        expected = [value.cpu() * (index + 1) * repeats for index, (value, repeats) in enumerate(zip(owner.inputs, (257, 131)))]
        worker = owner.worker(self.native)
        del owner
        gc.collect()
        self.assertIsNotNone(owner_ref(), "native worker did not retain its graph owner")
        tickets = [worker.submit(0, repetitions=257), worker.submit(1, repetitions=131)]
        worker.close()
        gc.collect()
        self.assertIsNone(owner_ref(), "closed native worker retained its graph owner")
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
        self.assert_completion(worker, tickets[0], 0, 257)
        self.assert_completion(worker, tickets[1], 1, 131)
        worker.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            worker.submit(0)

    def test_destructor_drains_before_owner_release(self):
        owner = GraphResources()
        owner_ref = weakref.ref(owner)
        output = owner.outputs[0]
        expected = owner.inputs[0].cpu() * 67
        worker = owner.worker(self.native)
        del owner
        worker.submit(0, repetitions=67)
        del worker
        gc.collect()
        self.assertIsNone(owner_ref())
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    def test_device_mismatch_does_not_invalidate_owner(self):
        owner = GraphResources()
        _, device = self.native.current_context()
        with self.assertRaisesRegex(RuntimeError, "context belongs to device"):
            owner.worker(self.native, device=device + 1)
        worker = owner.worker(self.native)
        try:
            ticket = worker.submit(0, repetitions=3)
            self.assert_completion(worker, ticket, 0, 3)
            torch.testing.assert_close(owner.outputs[0].cpu(), owner.inputs[0].cpu() * 3, rtol=0, atol=0)
        finally:
            worker.close()

    def test_owner_finalizer_can_reenter_close(self):
        finalized = []
        workers = {}

        class FinalizingResources(GraphResources):
            def __del__(self):
                workers["worker"].close()
                finalized.append(True)

        owner = FinalizingResources()
        worker = owner.worker(self.native)
        workers["worker"] = worker
        del owner
        worker.close()
        self.assertEqual(finalized, [True])
        workers.clear()

    def test_prepared_bundle_retains_entire_resource_set(self):
        from ignition.preparation.graphs import PreparedGraphBundle

        class AncillaryResource:
            pass

        owner = GraphResources()
        ancillary = AncillaryResource()
        prepared = PreparedGraphBundle(
            device=0,
            graphs={"short": owner.graphs[0], "long": owner.graphs[1]},
            buffers={"input_view": owner.inputs[0][1:], "output": owner.outputs[0]},
            resources=(owner, ancillary),
        )
        refs = [weakref.ref(value) for value in (prepared, owner, ancillary)]
        output = owner.outputs[0]
        expected = owner.inputs[0].cpu() * 19
        worker = prepared.start_worker()
        self.assertTrue(prepared.consumed)
        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
            prepared.start_worker()
        index = prepared.graph_indices["short"]
        del prepared, owner, ancillary
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in refs))
        worker.submit(index, repetitions=19)
        worker.close()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    def test_prepared_bundle_validation_and_corrected_retry(self):
        from ignition.preparation.graphs import PreparedGraphBundle

        owner = GraphResources()
        uninstantiated = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(uninstantiated, stream=owner.stream):
            owner.outputs[0].add_(owner.inputs[0])
        owner.outputs[0].zero_()
        prepared = PreparedGraphBundle(
            device=0,
            graphs={"decode": uninstantiated},
            buffers={"state": owner.outputs[0]},
            resources=(owner,),
        )
        with self.assertRaisesRegex(ValueError, "instantiated"):
            prepared.start_worker()
        self.assertFalse(prepared.consumed)
        uninstantiated.instantiate()
        worker = prepared.start_worker()
        try:
            ticket = worker.submit(0, repetitions=5)
            self.assert_completion(worker, ticket, 0, 5)
            torch.testing.assert_close(owner.outputs[0].cpu(), owner.inputs[0].cpu() * 5, rtol=0, atol=0)
        finally:
            worker.close()
        with self.assertRaisesRegex(RuntimeError, "already been consumed"):
            prepared.start_worker()

        for label, buffers, graphs, expected in (
            ("CPU storage", {"state": torch.zeros(4)}, {"decode": owner.graphs[0]}, "must reside"),
            ("noncontiguous view", {"state": owner.inputs[1][::2]}, {"decode": owner.graphs[0]}, "must be contiguous"),
            ("empty view", {"state": owner.inputs[0][:0]}, {"decode": owner.graphs[0]}, "nonempty"),
            ("duplicate executable", {}, {"a": owner.graphs[0], "b": owner.graphs[0]}, "unique"),
        ):
            with self.subTest(label):
                invalid = PreparedGraphBundle(device=0, graphs=graphs, buffers=buffers, resources=(owner,))
                with self.assertRaisesRegex(ValueError, expected):
                    invalid.start_worker()
                self.assertFalse(invalid.consumed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    arguments = parser.parse_args()
    extension = arguments.extension.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("flare_heliopause", extension)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native graph interoperability qualification")
    torch.cuda.set_device(0)
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    print(json.dumps({
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": properties.name,
        "device_uuid": str(properties.uuid),
        "capability": [properties.major, properties.minor],
        "extension": str(extension),
    }, sort_keys=True), flush=True)
    CudaGraphReplayTests.native = native
    unittest.main(argv=[sys.argv[0]], verbosity=2)


if __name__ == "__main__":
    main()
