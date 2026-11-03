# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify the two-frame native execution ABI with real captured CUDA graphs."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time
import unittest
from unittest import mock
import weakref

import torch


class Resources:
    def __init__(self, native):
        self.buffers = {
            "frame_index": torch.zeros(1, dtype=torch.int32, device="cuda"),
            "submission_sequence": torch.zeros(1, dtype=torch.int64, device="cuda"),
            "completion_records": torch.zeros((2, 4, 32), dtype=torch.int64, device="cuda"),
            "tokens": torch.zeros(4, dtype=torch.int64, device="cuda"),
            "slots": torch.arange(4, dtype=torch.int64, device="cuda"),
            "generations": torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda"),
        }
        self.consumed = torch.zeros(4, dtype=torch.int64, device="cuda")
        self.scratch = [torch.zeros((4, 32), dtype=torch.int64, device="cuda") for _ in (2, 4)]
        self.active = [torch.tensor([1] * count + [0] * (4-count), dtype=torch.int64, device="cuda") for count in (2, 4)]
        self.stream = torch.cuda.Stream()
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = []
        self.finalizer = None

        def launch(index):
            buffer, output, active = self.buffers, self.scratch[index], self.active[index]
            self.consumed.index_add_(0, buffer["slots"], active)
            output.zero_()
            output[:, 0].copy_(buffer["generations"])
            output[:, 1].copy_(buffer["submission_sequence"].expand(4))
            output[:, 2].copy_(self.consumed)
            output[:, 3].copy_(self.consumed)
            output[:, 4].scatter_(0, buffer["slots"], torch.where(active.bool(), buffer["tokens"] + 10 * (index+1), -1))
            output[:, 5].scatter_(0, buffer["slots"], active)
            output[:, 8].copy_(output[:, 5])
            output[:, 12:14].fill_(-1)
            output[:, 15].fill_(1)
            output[:, 16:].fill_(-1)
            output[:, 16].copy_(output[:, 4])
            buffer["completion_records"].index_copy_(0, buffer["frame_index"].to(torch.int64), output.unsqueeze(0))

        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for index in range(2):
                for _ in range(3):
                    launch(index)
        self.stream.synchronize()
        for index in range(2):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
                launch(index)
            graph.instantiate()
            self.graphs.append(graph)
        self.consumed.zero_()
        self.buffers["completion_records"].zero_()
        torch.cuda.synchronize()
        context, device = native.current_context()
        offset, bindings = 0, []
        for name, tensor in self.buffers.items():
            if name == "completion_records":
                continue
            offset = (offset + 7) // 8 * 8
            bindings.append({"buffer_name": name, "host_offset": offset, "nbytes": tensor.numel() * tensor.element_size()})
            offset += bindings[-1]["nbytes"]
        self.manifest = {"abi_version": 6, "completion_token_capacity": 16, "context": context, "device": device, "max_active": 4,
            "input_frame_bytes": (offset + 7)//8*8,
            "buffers": [{"name": name, "address": tensor.data_ptr(), "nbytes": tensor.numel() * tensor.element_size(),
                "dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape), "strides": list(tensor.stride())}
                for name, tensor in self.buffers.items()],
            "graphs": [{"name": f"decode{count}", "kind": "decode", "handle": graph.raw_cuda_graph_exec(),
                "sampling_mode": "general", "confidence_width": 0, "estimated_duration_ns": 0,
                "batch_capacity": count, "token_capacity": 1, "prefill_capacity": 0,
                "input_bindings": bindings, "tuned_bindings": []}
                for count, graph in zip((2,4), self.graphs)], "tuning_graphs": []}

    def __del__(self):
        if self.finalizer:
            self.finalizer()


class ExecutionTests(unittest.TestCase):
    native = None

    @classmethod
    def setUpClass(cls):
        if cls.native is None:
            raise unittest.SkipTest("run with --extension and an explicitly assigned CUDA_VISIBLE_DEVICES")

    def complete(self, executor, expected):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            event = executor.poll()
            if event is not None:
                self.assertEqual(event.fence, expected)
                for row in event.records:
                    self.assertEqual(len(row), 32)
                    self.assertEqual(row[8:16], [row[5], 0, 0, 0, -1, -1, 0, 1])
                    self.assertEqual(row[16:], [row[4]] + [-1] * 15)
                return event
            time.sleep(.001)
        self.fail("native completion event did not arrive within 10 seconds")

    def test_two_frames_partial_uploads_variant_order_and_grammar_barrier(self):
        owner = Resources(self.native)
        executor = self.native.GraphExecutor(owner, owner.manifest)
        upload = lambda name, values, offset=0: (name, offset, struct.pack(f"<{len(values)}q", *values))
        try:
            with mock.patch.object(torch.cuda.CUDAGraph, "replay", side_effect=AssertionError("Python replay callback")):
                self.assertEqual(executor.submit(0, [upload("tokens", [11,22,33,44]), upload("slots", [3,1,0,2])]), 1)
                self.assertEqual(executor.submit(1, [upload("tokens", [55], 8)]), 2)
                self.assertIsNone(executor.submit(1, []))
                self.assertIsNone(executor.submit(1, [], grammar_constrained=True))
                first = self.complete(executor, 1)
                self.assertGreater(first.gpu_ns[0], 0)
                self.assertEqual(list(first.gpu_ns[1:]), [None] * 15)
                self.assertGreaterEqual(first.wall_ns, first.gpu_ns[0])
                self.assertFalse(first.queued)
                self.assertEqual([row[5] for row in first.records], [0,1,0,1])
                self.assertEqual(first.records[3][:6], [4,1,1,1,21,1])
                self.assertEqual(first.records[1][:6], [2,1,1,1,32,1])
                self.assertIsNone(executor.submit(0, [], grammar_constrained=True))
                second = self.complete(executor, 2)
                self.assertGreater(second.gpu_ns[0], 0)
                self.assertGreaterEqual(second.wall_ns, second.gpu_ns[0])
                self.assertTrue(second.queued)
                self.assertEqual([row[4] for row in second.records], [53,75,64,31])
                self.assertEqual([row[2] for row in second.records], [1,2,1,2])
                self.assertEqual(executor.submit(0, [], grammar_constrained=True), 3)
                self.assertIsNone(executor.submit(1, []))
                third = self.complete(executor, 3)
                self.assertGreater(third.gpu_ns[0], 0)
                self.assertFalse(third.queued)
                self.assertEqual([row[2] for row in third.records], [1,3,1,3])
                self.assertEqual(executor.outstanding, 0)
        finally:
            executor.close()
        self.assertTrue(executor.drained)
        executor.close()

    def test_owner_retention_and_reentrant_close_after_async_submission(self):
        owner = Resources(self.native)
        executor = self.native.GraphExecutor(owner, owner.manifest)
        finalized = []
        owner.finalizer = lambda: (executor.close(), finalized.append(True))
        reference = weakref.ref(owner)
        executor.submit(1, [])
        del owner
        gc.collect()
        self.assertIsNotNone(reference())
        executor.close()
        self.assertTrue(executor.drained)
        self.assertIsNone(reference())
        self.assertEqual(finalized, [True])

    def test_drop_drains_and_releases_owner(self):
        owner = Resources(self.native)
        reference = weakref.ref(owner)
        executor = self.native.GraphExecutor(owner, owner.manifest)
        executor.submit(1, [])
        del owner, executor
        gc.collect()
        self.assertIsNone(reference())

    def test_validation_and_bad_upload_do_not_consume_frames(self):
        owner = Resources(self.native)
        invalid = copy.deepcopy(owner.manifest)
        invalid["buffers"][2]["shape"] = [4,2,32]
        invalid["buffers"][2]["strides"] = [64,32,1]
        with self.assertRaisesRegex(ValueError, "required buffer layout"):
            self.native.GraphExecutor(owner, invalid)
        executor = self.native.GraphExecutor(owner, owner.manifest)
        try:
            with self.assertRaisesRegex(RuntimeError, "reserved"):
                executor.submit(0, [("frame_index",0,b"\0"*4)])
            with self.assertRaisesRegex(RuntimeError, "unaligned"):
                executor.submit(0, [("tokens",1,b"\0"*8)])
            self.assertEqual(executor.outstanding, 0)
            self.assertEqual(executor.submit(0, []), 1)
            self.complete(executor, 1)
        finally:
            executor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("assign CUDA_VISIBLE_DEVICES explicitly")
    spec = importlib.util.spec_from_file_location("flare_heliopause", args.extension.resolve(strict=True))
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    torch.cuda.set_device(0)
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "device_uuid": str(properties.uuid),
        "capability": [properties.major, properties.minor]}), flush=True)
    ExecutionTests.native = native
    unittest.main(argv=[sys.argv[0]], verbosity=2)


if __name__ == "__main__":
    main()
