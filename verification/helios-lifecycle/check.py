#!/usr/bin/env python3
"""Check bounded Helios ownership, accounting, and rank protocols against Rust."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


MODEL = Path(__file__).resolve().parent
ROOT = MODEL.parent.parent
TEST = "scheduler::lifecycle_tests::scheduler_matches_lifecycle_model"
RANK_TEST = "execution::threaded::protocol_tests::rank_coordinator_matches_model"
CHECKS = [("Lifecycle", TEST, "LIFECYCLE_EDGE ", 14),
          ("RankProtocol", RANK_TEST, "RANK_EDGE ", 15),
          ("OffloadTransfer", "offload::transfer::lifecycle_tests::offload_transfers_match_model",
           "OFFLOAD_EDGE ", 16),
          ("CacheOwnership", "cache::lifecycle_tests::bounded_cache_ownership_matches_tla",
           "CACHE_OWNERSHIP_EDGE ", 29),
          ("Speculation", "scheduler::speculative_lifecycle_tests::scheduler_matches_speculative_accounting_model",
           "SPECULATION_EDGE ", 6)]
MODELS = [check[0] for check in CHECKS] + ["RankDispatch"]


def run(command, *, cwd=ROOT):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=300)
    for line in result.stdout.splitlines():
        if not any(line.startswith(check[2]) for check in CHECKS):
            print(line, flush=True)
    result.check_returncode()
    return result.stdout


def tla(value):
    if isinstance(value, list):
        return "<<" + ", ".join(map(tla, value)) + ">>"
    if type(value) is not int:
        raise ValueError(f"unexpected lifecycle value {value!r}")
    return str(value)


def check_observations(output, directory, java, jar, module, prefix, width):
    edges = [json.loads(line.removeprefix(prefix))
             for line in output.splitlines() if line.startswith(prefix)]
    if not edges:
        raise RuntimeError(f"{module} test produced no transitions")
    for before, action, after in edges:
        if len(before) != width or len(action) != 3 or len(after) != width:
            raise ValueError(f"invalid {module} projection geometry")
    for offset in range(0, len(edges), 2000):
        cases = ",\n".join(tla(edge) for edge in edges[offset:offset + 2000])
        (directory / "Observed.tla").write_text(
            "---------------- MODULE Observed ----------------\n"
            f"EXTENDS {module}\nVARIABLE edgeIndex, observed\n"
            f"Edges == <<\n{cases}\n>>\n"
            "ObservedInit == state = Initial /\\ edgeIndex = 1 /\\ observed = Edges[1]\n"
            "ObservedNext == /\\ edgeIndex' = IF edgeIndex < Len(Edges)\n"
            "                                  THEN edgeIndex + 1 ELSE edgeIndex\n"
            "                /\\ observed' = Edges[edgeIndex']\n"
            "                /\\ UNCHANGED state\n"
            "Corresponds == AllowedEdge(observed)\n"
            "ObservedSpec == ObservedInit /\\ [][ObservedNext]_<<state, edgeIndex, observed>>\n"
            "=================================================\n"
        )
        (directory / "Observed.cfg").write_text(
            (MODEL / f"{module}.cfg").read_text().split("SPECIFICATION", 1)[0]
            + "\n"
            "SPECIFICATION ObservedSpec\nINVARIANT Corresponds\n"
        )
        run(tlc(java, jar, directory, "Observed"), cwd=directory)
    print(f"Checked {len(edges)} distinct Rust transitions against {module}.tla", flush=True)


def tlc(java, jar, directory, module, config=None):
    config = config or f"{module}.cfg"
    return [java, "-XX:+UseParallelGC", "-Xmx1g", "-cp", str(jar), "tlc2.TLC",
            "-workers", "1", "-metadir", str(directory / f"states-{Path(config).stem}"),
            "-config", config, f"{module}.tla"]


def check_mutations(cargo, source, test, mutations):
    original = source.read_text()
    try:
        for name, needle, replacement, expected in mutations:
            if original.count(needle) != 1:
                raise RuntimeError(f"update the {name} mutation for {source}")
            source.write_text(original.replace(needle, replacement, 1))
            try:
                run(cargo + [test, "--"] + (["--exact"] if test == TEST else []))
            except subprocess.CalledProcessError as failure:
                if expected not in failure.output:
                    raise RuntimeError(f"{name} failed for an unexpected reason") from failure
                print(f"Detected Rust mutation: {name}", flush=True)
            else:
                raise RuntimeError(f"lifecycle verification missed the {name} mutation")
            finally:
                source.write_text(original)
    finally:
        source.write_text(original)


def check_model_mutation(directory, java, jar, module, needle, replacement, invariant):
    source = directory / f"{module}.tla"
    original = source.read_text()
    if original.count(needle) != 1:
        raise RuntimeError(f"update the {invariant} mutation for {module}")
    source.write_text(original.replace(needle, replacement, 1))
    try:
        run(tlc(java, jar, directory, module), cwd=directory)
    except subprocess.CalledProcessError as failure:
        if f"Invariant {invariant} is violated" not in failure.output:
            raise RuntimeError("model mutation failed for an unexpected reason") from failure
        print(f"Detected TLA+ mutation: {invariant}", flush=True)
    else:
        raise RuntimeError(f"model missed {invariant} mutation")
    finally:
        source.write_text(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--java", default="java", help="Java 11 or later executable")
    parser.add_argument("--tlc-jar", required=True, type=Path,
                        help="path to the TLA+ tools 1.7.4 tla2tools.jar")
    parser.add_argument("--target-dir", type=Path,
                        help="Cargo build directory, isolated from other worktrees")
    parser.add_argument("--mutations", action="store_true",
                        help="temporarily break covered Rust guards and require detection; use an idle worktree")
    args = parser.parse_args()
    jar = args.tlc_jar.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="helios-lifecycle-") as scratch:
        directory = Path(scratch)
        for filename in [f"{module}.{suffix}" for module in MODELS
                         for suffix in ["tla", "cfg"]]:
            shutil.copyfile(MODEL / filename, directory / filename)
        shutil.copyfile(MODEL / "Speculation16.cfg", directory / "Speculation16.cfg")
        for module in MODELS:
            run(tlc(args.java, jar, directory, module), cwd=directory)
        run(tlc(args.java, jar, directory, "Speculation", "Speculation16.cfg"), cwd=directory)
        cargo = ["cargo", "test", "--locked", "--manifest-path", "helios/Cargo.toml", "--lib"]
        if args.target_dir:
            cargo.extend(["--target-dir", str(args.target_dir.resolve())])
        for module, test, prefix, width in CHECKS:
            exact = ["--exact"] if test == TEST else []
            output = run(cargo + [test, "--"] + exact + ["--nocapture"])
            check_observations(output, directory, args.java, jar, module, prefix, width)
        for test in ["cache::lifecycle_tests::reclaimable_projection_matches_eviction_with_pins",
                     "scheduler::speculative_lifecycle_tests::packed_two_lane_completions_validate_atomically_for_bounded_geometry",
                     "scheduler::speculative_lifecycle_tests::terminal_noop_and_cancel_preserve_queued_speculative_storage",
                     "scheduler::cache_memory::tests::all_two_snapshot_alias_layouts_preserve_page_ownership_and_reservation_budget"]:
            output = run(cargo + [test, "--", "--exact"])
            if "1 passed; 0 failed" not in output:
                raise RuntimeError(f"{test} did not execute successfully")
        if args.mutations:
            for module, needle, replacement, invariant in [
                ("Lifecycle", "!.phase[slot] = IF waiting THEN Retiring ELSE Free",
                 "!.phase[slot] = Free", "QueuedReferencesRetainOwnership"),
                ("RankProtocol", "before.queue[i].ready = Peers",
                 "before.queue[i].ready # {}", "CommitRequiresEveryPeer"),
                ("RankProtocol", "Head(before.queue).done = Ranks",
                 "0 \\in Head(before.queue).done", "RemovalRequiresEveryRank"),
                ("RankDispatch", "!.failed = TRUE]", "!.failed = TRUE, !.released = TRUE]",
                 "FailureRetainsOwnership"),
                ("CacheOwnership", "!.nodes[child][6] = Max(U(before, child), Max(U(before, n), R(before, n)))]",
                 "!.nodes[child][6] = U(before, child)]", "InheritedLastUse"),
                ("Speculation", "Advance(st, ceiling) == Min(ceiling, Geometry(st)[7])",
                 "Advance(st, ceiling) == ceiling", "PendingUsesPhysicalCapacity"),
            ]:
                check_model_mutation(directory, args.java, jar, module, needle, replacement, invariant)
            check_mutations(cargo, ROOT / "helios/src/scheduler.rs", TEST, [
                ("premature retirement", "through: request.last_use,", "through: self.completed,",
                 "cancel released a slot referenced by queued work"),
                ("stale generation", "if record.generation != lane.request.generation as i64\n"
                 "                || record.submission_sequence != fence.0 as i64",
                 "if record.submission_sequence != fence.0 as i64",
                 "stale generation completion accepted"),
                ("unconfirmed drain", "if !shutdown.drained {", "if false {",
                 "unconfirmed drain accepted resource release"),
            ])
            check_mutations(cargo, ROOT / "helios/src/execution/threaded/protocol.rs", RANK_TEST, [
                ("partial readiness", "frame.ready.iter().all(|ready| *ready)",
                 "frame.ready.iter().any(|ready| *ready)", "commit must wait for every peer readiness"),
                ("partial completion", "frame.completions.iter().all(Option::is_some)",
                 "frame.completions.iter().any(Option::is_some)",
                 "publish must wait for every rank completion at the queue front"),
                ("ignored dispatch failure", "peer.send(Command::Commit(fence, graph))?;",
                 "let _ = peer.send(Command::Commit(fence, graph));",
                 "partial commit dispatch must report disconnected peer"),
            ])
            check_mutations(cargo, ROOT / "helios/src/cache.rs", CHECKS[3][1], [
                ("lost inherited fence", "child.last_use = child.last_use.max(node.last_use).max(node.ready_after);",
                 "child.last_use = child.last_use;", "compaction lost inherited last-use fence"),
            ])
            check_mutations(cargo, ROOT / "helios/src/offload/transfer.rs", CHECKS[2][1], [
                ("premature transfer aggregation", "if pending.remaining > 0 {", "if pending.remaining > 1 {",
                 "accepted rank replies must retain pending ownership until the final reply"),
            ])
            check_mutations(cargo, ROOT / "helios/src/scheduler.rs", CHECKS[4][1], [
                ("physical pending capacity", "lane.tokens.min(self.packed_token_capacity)", "lane.tokens",
                 "pending speculative advance must use physical capacity"),
                ("full lane reservation", "let pages = self.pages_through(request, through);",
                 "let pages = self.pages_through(request, lane.start_position + step.lane_advance_ceiling(lane) as u64);",
                 "queued lane must retain full target and draft page reservation"),
            ])
            for module, test, prefix, width in CHECKS:
                exact = ["--exact"] if test == TEST else []
                output = run(cargo + [test, "--"] + exact + ["--nocapture"])
                check_observations(output, directory, args.java, jar, module, prefix, width)


if __name__ == "__main__":
    main()
