---
name: helios-lifecycle
description: Check Helios scheduler, response ownership, checkpoint cache, offload, and threaded rank lifecycles with TLA+ and Rust conformance, and completion/state-view validators with Kani. Use when changing admission, output buffering, response lifetime, speculation, cancellation, fences, generations, cache compaction, snapshot aliases, offload aggregation, drain handling, readiness, or validation arithmetic.
---

# Helios lifecycle

Read `verification/helios-lifecycle/README.md` for the covered Rust functions,
state projection, bounds, assumptions, tool setup, and command. This check covers
scheduler ownership under an executor completion/drain contract and the actual
threaded rank bookkeeping helpers, checkpoint ownership, speculative accounting,
snapshot backing aliases, offload aggregation, and native response admission and
event ownership. Response checks also exercise worker completion, reasoning
continuation, and HTTP admission through terminal stream frames. CUDA execution,
the surrounding controller, and coordinated drain require their own execution checks. Kani harnesses cover
sequential validator properties. Use `docs/verification.md` to select the checks
for an affected property.

Run from the repository root:

```sh
python3 verification/helios-lifecycle/check.py --tlc-jar /path/to/tla2tools.jar
```

For completion-vector, prepared completion-contract, or state-view geometry changes, run
`cargo kani --manifest-path helios/Cargo.toml --lib` with the pinned setup in the
README. Kani covers `StateView::contains` and `fits_storage`; run the accompanying
`state_view_validation_resolves_buffers_and_preserves_storage_errors` test for
manifest lookup and diagnostics. Preserve the harness's full integer domains,
dtype cases, and empty/nonempty geometry inputs.

Supply `--java` or `--target-dir` when needed. Read every TLA+ result with pstack's
`statechart` skill. For a failure, identify the violated operation or invariant
and replay counterexamples through the covered production scheduler, cache,
transfer coordinator, or rank protocol methods. Correct a
modeling error from the implementation's actual behavior. Preserve a valid
failing property and report the runtime defect.

When covered Rust changes, review the corresponding transition, projection, and
rejection assertions. Use `maintain-formal-verification-skill` for semantic drift.
Run with `--mutations` in an idle worktree to check deliberate faults in the
model and real Rust guards, restore the source, and rerun. A passing abstract model alone does not establish
Rust conformance. Report the stated bounds, observed-edge check, mutation result,
and any unrun execution checks.
