# SPDX-License-Identifier: Apache-2.0

"""Measure complete Kimi DFlash2 graphs with generated weights and no core export.

Checkpoint files supply geometry and frontend assets only. Synthetic values do not
support model-quality, acceptance, serving-throughput, or runtime-comparison claims.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import monotonic


def _integers(text):
    values = tuple(int(value) for value in text.split(","))
    if not values or tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("expected distinct increasing comma-separated integers")
    return values


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="local target checkpoint metadata and frontend assets")
    parser.add_argument("--draft", type=Path, required=True, help="local DFlash2 checkpoint metadata; payloads are not read")
    parser.add_argument("--config", type=Path, required=True, help="Ignition preparation TOML")
    parser.add_argument("--output", type=Path, required=True, help="new benchmark JSON path; existing files are rejected")
    parser.add_argument("--kernel-cache", type=Path, help="diagnostic tuning/forward cache; defaults to the TOML cache")
    parser.add_argument("--seed", type=int, default=0, help="synthetic value seed (default: 0)")
    parser.add_argument("--gpus", default="0-15", help="sixteen participating GPU indices (default: 0-15)")
    parser.add_argument("--batches", type=_integers, default=(1,), help="request batches (default: 1)")
    parser.add_argument("--contexts", type=_integers, default=(512,), help="consumed-token contexts (default: 512)")
    parser.add_argument("--verify-prefixes", type=_integers, help="physical C1 prefixes (default: 8 for C1, full width otherwise)")
    parser.add_argument("--warmup", type=int, default=5, help="restored warmups per case and seed (default: 5)")
    parser.add_argument("--repetitions", type=int, default=20, help="timed restored replays per case and seed (default: 20)")
    parser.add_argument("--seeds", type=_integers, default=(0, 1, 2), help="request sampling seeds (default: 0,1,2)")
    parser.add_argument("--compile-workers", type=int, default=32, help="compiler processes per host (default: 32)")
    parser.add_argument("--autotune-effort", choices=("off", "default", "audit"), default="default")
    parser.add_argument("--no-stream-overlap", action="store_true", help="disable prepared independent-branch overlap")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run(options):
    from ignition.cli import bench, environment
    from ignition.cli.config import Configuration, Setting
    from ignition.cli.preparation import Host, derive, resolve_model, speculation
    from ignition.preparation.selection import BenchmarkSelection
    from ignition.recipes.kimi31.synthetic_weights import SYNTHETIC_SCHEMA

    target, draft = options.target.resolve(), options.draft.resolve()
    if not target.is_dir() or not draft.is_dir():
        raise ValueError("target and draft must be existing local checkpoint directories")
    if json.loads((draft / "config.json").read_text()).get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("synthetic benchmark requires a DFlash2 draft")
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output {options.output}")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("x"):
        pass
    if not 0 <= options.seed < 2**63:
        raise ValueError("synthetic seed must be a uint63 integer")
    if any(value < 128 or value % 128 for value in options.contexts):
        raise ValueError("contexts must be positive multiples of 128")
    if any(value not in (1, 2, 4, 8) for value in options.batches):
        raise ValueError("batches must select 1,2,4,8")
    if any(value < 1 or value > 8 for value in options.verify_prefixes or ()):
        raise ValueError("verification prefixes must be from 1 through 8")
    prefixes = options.verify_prefixes
    if prefixes is None:
        prefixes = (8,) if options.batches == (1,) else ()
    if prefixes and options.batches != (1,):
        raise ValueError("explicit verification prefixes require batch one")
    if options.warmup < 1 or options.repetitions < 1:
        raise ValueError("warmup and repetitions must be positive")
    if any(type(seed) is not int or not 0 <= seed < 2**64 for seed in options.seeds):
        raise ValueError("request seeds must be uint64 integers")

    configuration = Configuration.load(options.config)
    kernel_cache = None if options.kernel_cache is None else options.kernel_cache.resolve()
    if kernel_cache is not None:
        configuration.values.setdefault("paths", {})["kernel_cache"] = str(kernel_cache)
    model = resolve_model(configuration, str(target))
    if model.recipe.name != "kimi31":
        raise ValueError("synthetic benchmark requires the Kimi K3.1 recipe")
    speculation_args = argparse.Namespace(
        speculative_method="dflash2", speculative_graph_budget=16,
        mtp_draft_tokens=7, speculative_token_capacity=max(8, max(options.batches) * 8),
        no_mtp=False, no_mtp_adaptive_verification=True, mtp_only=False,
        draft_head_quantization=None, markov_head_quantization=None,
        vocabulary_parallel_sampling=False)
    proposal = speculation(speculation_args, model, configuration)
    plan = derive(configuration, model, Host.discover(), gpus=options.gpus,
        concurrency=Setting(max(options.batches), "flag"), speculation=proposal, prefill_chunk=4096,
        context_parallel=True, stream_overlap=not options.no_stream_overlap,
        compile_workers=options.compile_workers, tuning_gpus=None,
        autotune_effort=options.autotune_effort, collective_tuning="static", collective_ladder=None,
        lock_gpu_clocks=False, native_dense_activations=False,
        force_moe_decode_a16=None, shard_hyperconnections=False, joint_tuning=False,
        image_concurrency=None, mixed_prefill=False, prompt_logprobs=False,
        expert_scale_compression=None, online_mxfp8=True, language_model_only=True,
        mla_compute_mode=None)
    plan = replace(plan, benchmark_selection=BenchmarkSelection(
        batches=options.batches, contexts=options.contexts, width=8,
        verification_prefixes=prefixes))
    settings = bench.Settings(batches=options.batches, contexts=options.contexts, width=8,
        warmup=options.warmup, repetitions=options.repetitions, seeds=options.seeds,
        verification_prefixes=prefixes)
    metadata = {
        "synthetic_weights": {
            "schema": SYNTHETIC_SCHEMA, "seed": options.seed,
            "geometry_target": str(target), "geometry_draft": str(draft),
            "payload_bytes_read": 0,
            "csf_exception_policy": {
                "sampled_scale_rows": 38912, "zero_exception_rows_percent": 70.8,
                "mean_exceptions_per_row": 0.368, "maximum_exceptions_per_row": 4,
            },
        },
        "limitations": (
            "Generated values preserve tensor geometry and calibrated CSF exception density. These graph times do not "
            "establish model quality, speculative acceptance, serving throughput, or runtime parity."
        ),
    }
    print(json.dumps({"plan": plan.to_dict(), "synthetic_weights": metadata["synthetic_weights"]}),
          file=sys.stderr, flush=True)
    environment.apply(configuration, plan.gpus, plan.recipe.b12x_environment)
    preflight_started = monotonic()
    import torch
    from flashinfer import sampling

    torch.cuda.set_device(0)
    sampling.get_sampling_module()
    metadata["sampling_preflight_seconds"] = monotonic() - preflight_started
    result = bench._measure(model, plan, configuration, settings, bench._progress(options.verbose),
        compile_forwards=True, output=options.output,
        prepare_options={"synthetic_seed": options.seed, "draft_checkpoint": str(draft)},
        diagnostic_metadata=metadata,
        kernel_cache=kernel_cache)
    bench.write_result(result, options.output)
    return 0


def main():
    return run(arguments().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
