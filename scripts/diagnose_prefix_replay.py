# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare cold, split, and restored text prefill using the complete graph catalog."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def equality_summary(comparisons):
    total = 0
    failed_paths = []

    def visit(value, path):
        nonlocal total
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}"
                if key == "equal" and isinstance(item, bool):
                    total += 1
                    if not item:
                        failed_paths.append(child)
                else:
                    visit(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(comparisons, "comparisons")
    return {"comparison_count": len(comparisons), "equal_predicates": total,
            "passed_predicates": total - len(failed_paths), "failed_predicates": len(failed_paths),
            "failed_paths": failed_paths, "passed": total > 0 and not failed_paths}


def write_equality_receipt(receipt, output, *, require_bitwise=False):
    summary = equality_summary(receipt["comparisons"])
    receipt.update(require_bitwise=require_bitwise, equality_summary=summary)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    return int(require_bitwise and not summary["passed"])


def prefix_replay_sequence(admission, prefill, snapshot, observe, *, tokens, boundary, full_capacity, tail_capacity):
    """Keep monolithic execution separate from the independently computed checkpoint prefix."""
    if not 0 < boundary < tokens <= full_capacity or tokens - boundary > tail_capacity:
        raise ValueError("prefix replay lengths must fit both captured capacities")
    admission(1)
    prefill(0, tokens, full_capacity)
    observe("cold")
    admission(2)
    prefill(0, boundary, full_capacity)
    observe("checkpoint_prefix")
    snapshot(boundary)
    observe("immutable", 5, logits=False)
    admission(3, restore=True)
    observe("restored_before_tail", logits=False)
    prefill(boundary, tokens, tail_capacity)
    observe("restored_small_tail")
    admission(4, restore=True)
    prefill(boundary, tokens, full_capacity)
    observe("restored_full_capacity_tail")
    admission(5)
    prefill(0, tokens, full_capacity)
    observe("cold_repeat")
    admission(6)
    prefill(0, boundary, full_capacity)
    observe("split_prefix")
    prefill(boundary, tokens, tail_capacity)
    observe("split_small_tail")


def stage_state_transfer(host, *, source, destination, consumed, reset=False):
    """Stage one exact maintenance operation; page tables must already be populated."""
    if type(consumed) is not int or consumed < 0 or (reset != (consumed == 0)):
        raise ValueError("reset requires endpoint0; copy requires a positive exact endpoint")
    host["state_num_ops"].fill_(1)
    for name in ("source_slots", "destination_slots", "operation", "accepted_tokens"):
        host["state_" + name].zero_()
    host["state_qsa_slots"].fill_(-1)
    host["state_expected_consumed"].fill_(-1)
    for name, value in (("source_slots", source), ("destination_slots", destination),
            ("operation", int(reset)), ("qsa_slots", destination if destination < 5 else -1),
            ("accepted_tokens", int(destination < 5)), ("expected_consumed", consumed)):
        host["state_" + name][0] = value


class Layer3QSARecorder:
    """Retain attention operands outside the shared graph pool for fenced readback."""

    def __init__(self):
        self.records = {}

    def __enter__(self):
        import torch

        from ignition.recipes.qwen38.attention import PreparedAttention

        self.attention_type = PreparedAttention
        self.original_init = PreparedAttention.__init__
        self.original_call = PreparedAttention.__call__
        recorder = self

        def initialize(operator, module, *args, **kwargs):
            recorder.original_init(operator, module, *args, **kwargs)
            if module.layer_idx != 3:
                return
            graph_name, suffix = kwargs["name"].rsplit(".layer", 1)
            if suffix != "3.qsa" or graph_name in recorder.records:
                raise RuntimeError("layer3 QSA recording requires one operator per graph")
            source = args[1]
            tensors = {"source": source,
                **{name: getattr(operator.scratch, name)
                   for name in operator.scratch.__dataclass_fields__},
                "index_query": operator.index_query, "raw_index_key": operator.index_key}
            operator._diagnostic_qsa_tensors = tensors
            operator._diagnostic_qsa_record = {
                "tensors": {name: torch.empty_like(value) for name, value in tensors.items()},
                "metadata": {name: torch.empty_like(getattr(operator.metadata, name))
                             for name in operator.metadata.__dataclass_fields__},
                "raw_before": {name: torch.empty_like(getattr(operator.state, name))
                               for name in operator.state.__dataclass_fields__ if name.startswith("raw_")},
            }
            recorder.records[graph_name] = operator

        def run(operator, source):
            record = getattr(operator, "_diagnostic_qsa_record", None)
            if record is not None:
                record["tensors"]["source"].copy_(source)
                for name, destination in record["raw_before"].items():
                    destination.copy_(getattr(operator.state, name))
            output = recorder.original_call(operator, source)
            if record is not None:
                for name, destination in record["tensors"].items():
                    if name != "source":
                        destination.copy_(operator._diagnostic_qsa_tensors[name])
                for name, destination in record["metadata"].items():
                    destination.copy_(getattr(operator.metadata, name))
            return output

        PreparedAttention.__init__ = initialize
        PreparedAttention.__call__ = run
        return self

    def __exit__(self, kind, value, traceback):
        self.attention_type.__init__ = self.original_init
        self.attention_type.__call__ = self.original_call

    @property
    def nbytes(self):
        return sum(tensor.nbytes for operator in self.records.values()
                   for group in operator._diagnostic_qsa_record.values() for tensor in group.values())

    def read(self, graph_name, position_capacity):
        import torch

        operator = self.records[graph_name]
        record = {group: {name: tensor.cpu() for name, tensor in values.items()}
                  for group, values in operator._diagnostic_qsa_record.items()}
        metadata = record["metadata"]
        pages = set()
        compressed_pages = set()
        page_size = operator.state.main_k_cache.shape[1]
        compressed_page_size = operator.state.compressed_k_cache.shape[1]
        for request, length in enumerate(metadata["sequence_lengths"].tolist()):
            pages.update(metadata["main_block_table"][request, :(length + page_size - 1) // page_size].tolist())
            groups = length // 4
            compressed_pages.update(metadata["compressed_block_table"][request,
                :(groups + compressed_page_size - 1) // compressed_page_size].tolist())
        if any(page < 0 for page in pages | compressed_pages):
            raise RuntimeError("QSA recording found an unmapped live cache page")
        record["cache_after"] = {}
        for name, identifiers in (("main_k_cache", pages), ("main_v_cache", pages),
                                  ("compressed_k_cache", compressed_pages)):
            tensor = getattr(operator.state, name)
            ids = sorted(identifiers)
            values = (torch.stack([tensor[index].cpu() for index in ids]) if ids
                      else torch.empty((0, *tensor.shape[1:]), dtype=tensor.dtype))
            record["cache_after"][name] = {"page_ids": ids, "values": values,
                                            "pool_shape": tuple(tensor.shape)}
        record["raw_after"] = {name: getattr(operator.state, name).cpu()
                               for name in operator.state.__dataclass_fields__ if name.startswith("raw_")}
        record["constants"] = {
            "index_q_norm_weight": operator.binding_kwargs["index_q_norm_weight"].cpu(),
            "index_k_norm_weight": operator.binding_kwargs["index_k_norm_weight"].cpu(),
            "rope_cos": operator.rotary.cos[:position_capacity].cpu(),
            "rope_sin": operator.rotary.sin[:position_capacity].cpu(),
            "rms_norm_eps": operator.module.config.rms_norm_eps,
            "graph_name": graph_name, "token_capacity": operator.tokens,
            "main_page_size": page_size, "compressed_page_size": compressed_page_size,
            "context": operator.metadata.main_block_table.shape[1] * page_size,
        }
        return record


def checkpoint_endpoint(tokens, requested=None):
    """Select a positive consumed prefix strictly inside the complete prompt."""
    endpoint = tokens // 64 * 64 if requested is None else requested
    if type(endpoint) is not int or not 0 < endpoint < tokens:
        raise ValueError("checkpoint endpoint must be a positive token count below the prompt length")
    return endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--checkpoint-tokens", type=int,
                        help="exact prefix endpoint; defaults to the last complete 64-token block")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--record-qsa", action="store_true",
                        help="record layer3 QSA operands and mapped cache pages after each forward fence")
    parser.add_argument("--require-bitwise", action="store_true",
                        help="write all comparisons, then fail unless every recorded equality predicate passes")
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("assign CUDA_VISIBLE_DEVICES explicitly")
    if args.output.exists() or args.tensor_output.exists():
        parser.error("diagnostic output files must not already exist")

    import b12x.attention.paged._selected_forward as selected_attention
    import b12x.attention.qsa._contract as qsa_contract
    import b12x.attention.varlen._preparation as varlen_preparation
    import b12x.sequence.gdn_prefill._impl as gdn_implementation
    import flare_heliopause
    import torch
    from transformers import AutoTokenizer

    from ignition.components.sampling import AdmissionBuffers
    from ignition.preparation.qwen38 import MODEL_ID, REVISION
    from ignition.recipes.qwen38.attention import PreparedAttention
    from ignition.recipes.qwen38.bootstrap import PreparationConfig, prepare_model
    from ignition.recipes.qwen38.decoder import DecoderScratch, PreparedDecoder
    from ignition.recipes.qwen38.sampling import RequestAdmission
    from ignition.recipes.qwen38.state import HybridStatePool

    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    config = PreparationConfig(prefill_chunk_size=args.prefill_chunk_size)
    snapshot = args.snapshot or Path.home() / ".cache/huggingface/hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / REVISION
    request = json.loads(args.request_json.read_text())
    if any(not isinstance(message.get("content"), (str, type(None))) for message in request["messages"]):
        parser.error("this diagnostic requires a text-only request")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    options = dict(request.get("chat_template_kwargs") or {})
    for name in ("tools", "enable_thinking", "add_generation_prompt", "continue_final_message", "reasoning_effort"):
        if name in request:
            options[name] = request[name]
    options.setdefault("add_generation_prompt", True)
    messages = copy.deepcopy(request["messages"])
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
                if not isinstance(function["arguments"], dict):
                    parser.error("tool-call arguments must decode to a JSON object")
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options)
    if not 64 < len(prompt) <= config.prefill_chunk_size:
        parser.error(f"rendered prompt must have 65..{config.prefill_chunk_size} tokens")
    try:
        boundary = checkpoint_endpoint(len(prompt), args.checkpoint_tokens)
    except ValueError as error:
        parser.error(str(error))
    bucket = lambda tokens: next(size for size in config.prefill_capacities if size >= tokens)
    full_capacity, tail_capacity = bucket(len(prompt)), bucket(len(prompt) - boundary)
    extensions = list(Path(flare_heliopause.__file__).parent.glob("*.so"))
    if len(extensions) != 1:
        raise RuntimeError("expected one installed native extension for provenance")
    b12x_root = Path(qsa_contract.__file__).resolve().parents[2]
    b12x_sources = {str(path.relative_to(b12x_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(b12x_root.rglob("*.py"))}
    receipt = {"checkpoint": {"repository": MODEL_ID, "revision": REVISION},
        "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"], "request": request,
        "prompt_ids": prompt, "boundary": boundary, "full_capacity": full_capacity,
        "prefill_chunk_size": config.prefill_chunk_size,
        "tail_capacity": tail_capacity, "submissions": [], "comparisons": {},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_helper_sha256": hashlib.sha256((ROOT / "qualify/checkpoint_state.py").read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "gdn_implementation_sha256": hashlib.sha256(Path(gdn_implementation.__file__).read_bytes()).hexdigest(),
        "b12x_sources": {str(Path(module.__file__).resolve()): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (gdn_implementation, qsa_contract, selected_attention, varlen_preparation)},
        "b12x_source_root": str(b12x_root), "b12x_tree_sha256": b12x_sources,
        "math_modes": {"gdn_prefill_intermediate_dtype": "bfloat16",
                       "vision": "flash; vision graphs are not submitted"},
        "record_layer3_qsa": args.record_qsa,
        "native_extension_sha256": hashlib.sha256(extensions[0].read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "ignition/recipes/qwen38").glob("*.py"))},
        "diagnostic": "All model mutations use native graph submissions. CPU state readbacks occur after completed fences.",
        "checkpoint_source": "An independent execution of tokens [0, B) at the monolithic graph capacity supplies the live checkpoint at B. The monolithic [0, P) comparison remains separate.",
        "state_comparison": "All 77 continuation regions are compared; checkpoint-only qsa.* storage is retained raw and checked against logical live caches during snapshot/restore.",
        "checkpoint_transfers": []}
    observations = {}
    started = time.monotonic()
    def progress(message):
        print(f"{time.monotonic() - started:.1f}s {message}", flush=True)

    recorder = Layer3QSARecorder() if args.record_qsa else None
    with (recorder if recorder is not None else nullcontext()), prepare_model(snapshot, config, progress=progress) as model:
        qsa_contracts = []
        for resource in model._resources:
            if not isinstance(resource, list):
                continue
            for decoder in resource:
                if not isinstance(decoder, PreparedDecoder):
                    continue
                layers = [(index, attention.binding.state.caps)
                          for index, (_, attention, _, _) in enumerate(decoder.layers)
                          if isinstance(attention, PreparedAttention)]
                if len(layers) != 12 or any(caps != layers[0][1] for _, caps in layers):
                    raise RuntimeError("each decoder must bind twelve QSA layers with identical capacities")
                caps = dict(vars(layers[0][1]))
                for field in ("device", "dtype", "kv_dtype"):
                    caps[field] = str(caps[field])
                qsa_contracts.append({"mode": decoder.state.mode,
                                      "layer_indices": [index for index, _ in layers], "caps": caps})
        if len(qsa_contracts) != len(config.prefill_capacities) + 3:
            raise RuntimeError("QSA provenance requires every configured prefill and decode capacity")
        receipt["math_modes"]["qsa"] = qsa_contracts
        from qualify.checkpoint_state import checkpoint_transfer_checks

        state = next(value for value in model._resources if isinstance(value, HybridStatePool))
        decoders = next(value for value in model._resources if isinstance(value, list) and value
                        and isinstance(value[0], PreparedDecoder))
        cache = decoders[0].qsa_states
        continuation_regions = [region for region in state.layout.regions if not region.name.startswith("qsa.")]
        if len(continuation_regions) != 77:
            raise RuntimeError("ordinary prefix diagnostic requires all 77 continuation regions")
        receipt["continuation_regions"] = [region.name for region in continuation_regions]
        receipt["checkpoint_storage_regions"] = [region.name for region in state.layout.regions if region.name.startswith("qsa.")]
        scratch = next(value for value in model._resources if isinstance(value, DecoderScratch))
        host = {name: tensor.detach().cpu().clone() for name, tensor in model.buffers.items()}
        staging = AdmissionBuffers(**{name: host[name] for name in AdmissionBuffers.__dataclass_fields__})
        graph_indices = {name: index for index, name in enumerate(model.graphs)}
        executor = model.start_executor()
        model.runtime = executor

        def execute(name, fields, consumed):
            uploads = [(field, 0, host[field].numpy().tobytes()) for field in fields]
            allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
            fence = executor.submit(graph_indices[name], uploads)
            if fence is None:
                raise RuntimeError(f"{name} submission was unexpectedly refused")
            deadline = time.monotonic() + 120
            result = executor.poll()
            while result is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} did not finish in120 seconds")
                time.sleep(.001)
                result = executor.poll()
            epoch = int(host["admission_generations"][0] if name.startswith(("init", "restore")) else host["row_generations"][0])
            expected = [epoch, fence, consumed, 0, -1, 1, 0, 0]
            if result.fence != fence or list(result.records[0]) != expected:
                raise RuntimeError(f"invalid {name} completion: {result.records}")
            if torch.cuda.memory_stats()["allocation.all.allocated"] != allocations:
                raise RuntimeError(f"{name} replay allocated CUDA memory")
            receipt["submissions"].append({"graph": name, "fence": fence, "record": result.records[0]})

        def transfer_check(label, consumed):
            checks = checkpoint_transfer_checks(state, cache, slot=0, snapshot=0,
                pages=host["page_tables"][0].tolist(), consumed=consumed)
            if label.startswith("restore_"):
                checks.update({f"qsa.{layer}.restored_anchor":
                    int(value.raw_interval_start_positions[1].cpu().item()) == consumed - 1
                    for layer, value in cache.items()})
            receipt["checkpoint_transfers"].append({"operation": label, "consumed": consumed, "checks": checks})
            receipt["comparisons"][label] = {name: {"equal": value} for name, value in checks.items()}

        def admission(epoch, *, restore=False):
            staging.stage(0, 0, RequestAdmission(epoch, 1234, max_new_tokens=96, temperature=0), prompt)
            staging.admission_count.fill_(1)
            # Every maintenance replay receives its physical page mapping before launch.
            host["page_tables"].fill_(-1)
            host["page_tables"][0] = torch.arange(host["page_tables"].shape[1], dtype=torch.int32)
            stage_state_transfer(host, source=5 if restore else 1, destination=1,
                                 consumed=boundary if restore else 0, reset=not restore)
            name = "restore1" if restore else "init1"
            execute(name, model.signatures[name].inputs, boundary if restore else 0)
            if restore:
                transfer_check(f"restore_epoch{epoch}", boundary)
            host["row_slot_ids"].fill_(-1)
            host["row_slot_ids"][0] = 0
            host["row_generations"].zero_()
            host["row_generations"][0] = epoch
            host["live_rows"].fill_(1)
            host["sample_enabled"].zero_()
            host["vision_token_indices"].fill_(-1)

        def snapshot_prefix(consumed):
            before = state.backing[1].cpu()
            stage_state_transfer(host, source=1, destination=5, consumed=consumed)
            execute("snapshot1", model.signatures["snapshot1"].inputs, consumed)
            after = state.backing[1].cpu()
            receipt["comparisons"]["snapshot_live_source_unchanged"] = {"state": {"equal": torch.equal(before, after)}}
            transfer_check("snapshot_prefix", consumed)

        last_forward_graph = None

        def prefill(begin, end, capacity):
            nonlocal last_forward_graph
            count = end - begin
            host["query_lengths"].zero_()
            host["query_lengths"][0] = count
            host["token_ids"].fill_(248044)
            host["token_ids"][:count] = torch.tensor(prompt[begin:end])
            host["prefill_positions"].zero_()
            host["prefill_positions"][:count] = torch.arange(begin, end)[:, None].expand(-1, 3)
            name = f"prefill{capacity}"
            execute(name, model.signatures[name].inputs, end)
            last_forward_graph = name

        def observe(name, row=1, *, logits=True):
            observations[name] = {"state": state.backing[row].cpu()}
            if logits:
                observations[name]["logits"] = scratch.logits[0].cpu()
                if recorder is not None:
                    observations[name]["qsa"] = recorder.read(last_forward_graph, len(prompt))
                progress(f"{name}: argmax={observations[name]['logits'].argmax().item()}")
            else:
                progress(f"{name}: saved state row{row}")

        try:
            prefix_replay_sequence(admission, prefill, snapshot_prefix, observe,
                tokens=len(prompt), boundary=boundary, full_capacity=full_capacity, tail_capacity=tail_capacity)
        finally:
            executor.close()
            if not executor.drained:
                raise RuntimeError("native diagnostic executor did not drain")

        def compare(left_name, right_name):
            left, right = observations[left_name], observations[right_name]
            result = {}
            for region in continuation_regions:
                def view(value):
                    return value["state"][region.offset:region.offset + region.nbytes].view(region.dtype).reshape(region.shape)
                a, b = view(left), view(right)
                error = a.float() - b.float()
                result[region.name] = {"equal": torch.equal(a, b), "relative_l2": float(error.norm() / b.float().norm().clamp_min(1e-12)),
                    "max_absolute": float(error.abs().max())}
                if not a.is_floating_point():
                    result[region.name].update(left=a.tolist(), right=b.tolist())
            if "logits" in left and "logits" in right:
                a, b = left["logits"].float(), right["logits"].float()
                result["logits"] = {"equal": torch.equal(a, b), "relative_l2": float((a-b).norm()/b.norm()),
                    "argmax_left": int(a.argmax()), "argmax_right": int(b.argmax())}
            if "qsa" in left and "qsa" in right:
                def logical_rows(value):
                    metadata = value["qsa"]["metadata"]
                    return {(request, position): row for row, (request, position) in enumerate(zip(
                        metadata["request_ids"].tolist(), metadata["query_positions"].tolist()))
                        if request >= 0 and position >= 0}
                left_rows, right_rows = logical_rows(left), logical_rows(right)
                positions = sorted(left_rows.keys() & right_rows.keys())
                if not positions:
                    raise RuntimeError("QSA observations have no common logical rows")
                left_indices = torch.tensor([left_rows[position] for position in positions])
                right_indices = torch.tensor([right_rows[position] for position in positions])
                qsa_result = {"logical_rows": positions, "tensors": {}}
                for name in left["qsa"]["tensors"]:
                    a = left["qsa"]["tensors"][name].index_select(0, left_indices)
                    b = right["qsa"]["tensors"][name].index_select(0, right_indices)
                    error = a.float() - b.float()
                    qsa_result["tensors"][name] = {"equal": torch.equal(a, b),
                        "relative_l2": float(error.norm() / b.float().norm().clamp_min(1e-12)),
                        "max_absolute": float(error.abs().max()), "different_elements": int((a != b).sum())}
                result["qsa"] = qsa_result
            return result
        for left, right in (("cold", "cold_repeat"), ("checkpoint_prefix", "immutable"),
                ("immutable", "restored_before_tail"), ("checkpoint_prefix", "split_prefix"),
                ("cold", "restored_small_tail"), ("cold", "restored_full_capacity_tail"),
                ("restored_small_tail", "split_small_tail")):
            receipt["comparisons"][left + "_vs_" + right] = compare(left, right)
        receipt["memory_report"] = model.memory_report
        if recorder is not None:
            receipt["qsa_recording"] = {"layer_idx": 3, "graphs": sorted(recorder.records),
                "persistent_recording_bytes": recorder.nbytes,
                "cache_readback": "Only mapped live request pages; page IDs preserve the physical mapping.",
                "raw_state": "Pre-call raw state is copied in the graph; post-call raw state is read after its fence."}
    if any(hashlib.sha256((b12x_root / name).read_bytes()).hexdigest() != digest
           for name, digest in b12x_sources.items()):
        raise RuntimeError("b12x source changed during prefix replay qualification")
    if hashlib.sha256((ROOT / "qualify/checkpoint_state.py").read_bytes()).hexdigest() != receipt["checkpoint_helper_sha256"]:
        raise RuntimeError("checkpoint transfer oracle changed during qualification")
    torch.save(observations, args.tensor_output)
    receipt.update(tensor_output=str(args.tensor_output), tensor_sha256=hashlib.sha256(args.tensor_output.read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started)
    exit_code = write_equality_receipt(receipt, args.output, require_bitwise=args.require_bitwise)
    print(json.dumps({name: (result["logits"] if "logits" in result else
        {"all_regions_equal": all(region["equal"] for region in result.values())})
        for name, result in receipt["comparisons"].items()}, indent=2))
    print(json.dumps({"equality_summary": receipt["equality_summary"]}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
