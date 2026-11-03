# Commit design review

Every commit requires review of its staged diff against the principles below.
They summarize [the core design](core-design.md) and [repository guidance](../AGENTS.md).
Consult those specifications for details relevant to the change.

Helios's shared runtime policy and separation from model computation are
non-negotiable. Reject changes that violate this boundary even when tests pass,
performance improves, or the implementation is described as temporary. Delivery
pressure and intended cleanup do not waive the requirement. Resolve violations
before committing; a design acknowledgment cannot authorize them.

## Principles

1. **Keep Helios as simple as possible.** Model-specific execution behavior,
   composition, state-update computation, and optimization belong in Ignition and
   its captured programs. Limit native specialization to concrete cache storage
   requirements and graph cache bindings. Express differences through cache
   schemes and prepared layout metadata reusable across models. Native code
   manages declared model-state storage without implementing per-model algorithms.
   Introducing a captured-graph operation family requires prior explicit approval
   from project owner **Luke Alonso <lalonso@gmail.com>**. Shape and capacity
   variants are allowed within an unchanged operation contract. A new operation
   contract requires approval even if it reuses a graph kind or enum value.
   A design acknowledgment does not grant approval.
2. **Python prepares; Rust serves prepared graphs.** Serving has no Python
   callbacks, eager model fallback, runtime compilation, recapture, or steady-state
   GPU allocation. Admission reserves resources and rejects unsupported features
   and capacities explicitly.
3. **Preserve canonical model semantics and shared mechanisms.** Reuse Transformers
   forwards where their dataflow fits; replace the smallest coherent module needed.
   Recipes own model composition and checkpoint mapping. Reusable projections,
   experts, collectives, and state transfers belong in `ignition/components/`.
   Native transport, scheduling, cache ownership, input packing, frontend processing,
   and request handling stay shared; use prepared metadata and frontend assets
   to express model-dependent behavior. Admission, page reservation and growth,
   eviction, preemption, readmission, and offload must not branch on model or
   recipe identity or capability flags that encode model allowlists. Differences
   must follow concrete cache storage or execution constraints consumed by shared
   policy. Trace runtime fixes across supported cache schemes and apply them to
   every compatible contract; a single-model reproducer does not justify a
   single-model implementation.
   `PreparedContract` contains typed prepared inputs, storage, dependencies, and
   frontend metadata; there is no per-model native trait.
   Reject per-model boolean policy flags, including `supports_*`, `can_*`, and
   default-false opt-ins. Enums, callbacks, schema names, and metadata fields
   cannot serve as alternative model allowlists. Review each member and its
   consumers for a concrete meaning independent of model identity.
4. **Use FlashInfer/b12x and NVIDIA directly.** Target SM103/SM120/SM121, reuse proven
   operators and sampling, and avoid interchangeable backends or portability layers.
   Retain internal paths required to load and serve previously supported models.
   Package version validation is required by principle 13.
5. **Finish preparation before capture.** Constructors declare operations; one
   preparation batch tunes the complete shape catalog. Size scratch from tuned
   plans, refresh bindings, compile complete forwards, and warm every variant
   before capture. TP forwards compile once, sharded across ranks; every rank
   binds the same artifacts and captures.
   Captured startup tuning graphs may select persistent binding values from
   prepared candidates before serving; compilation and capture remain preparation work.
6. **Make storage and lifetimes explicit.** Concurrent work has disjoint scratch
   and preserved fork/join dependencies. Values surviving graph boundaries use
   persistent storage. Never reuse inputs, pages, or slots while queued work or
   transfers reference them. Drain native work before releasing resource owners.
7. **Preserve complete committed state.** Prefix reuse, branching, restore, and
   speculative rollback preserve every relevant history, partial page, and saved
   hidden vector. Distinguish consumed tokens, pending tokens, and visible output.
   Published checkpoints are immutable; inactive rows perform no state mutation.
8. **Budget physical memory across the lifecycle.** Bound loading, tuning,
   compilation, capture, and serving storage. Shared CPU/GPU DRAM is one budget.
   Account for duplicate representations and retained temporary owners. Release
   compiler resources before materializing preparation caches.
9. **Preserve checkpoint precision and numerical contracts.** Read quantization
   from checkpoint metadata. Never silently requantize or rescale weights to enable
   fusion. State transfers and metadata are exact; floating-point comparisons use
   operator-specific precision and measured reduction repeatability.
10. **Use explicit configuration.** Settings belong in recipe declarations,
    configuration files, or CLI flags; explicit user values win over defaults.
    Environment-variable behavior switches are prohibited except for the documented
    CUDA device selection and dependency setup policies.
    Read the model card and manually verify sampling defaults in each Ignition
    recipe; never derive sampling policy from checkpoint config files or another
    model. Persist complete sampling defaults and only consumed structural
    frontend settings in typed Protobuf fields. Runtime assets are limited to
    the tokenizer and chat template. Clamp request distribution parameters to
    recipe-declared model-author ranges, or to their defaults when ranges are
    absent. Seed remains request-controlled.
11. **Optimize the complete GPU computation.** Consider compatible projection and
    residual/normalization fusion, independent branch overlap, and TP vocabulary
    sharding. Preserve strides, precision, mutations, and scratch lifetimes across
    every affected capacity. Let b12x autotune rather than pinning plans to hide
    integration defects. Width-one speculative execution uses decode work while
    retaining the draft context needed to resume wider rounds.
12. **Validate the real path and report evidence honestly.** Exercise affected
    shapes, tails, inactive rows, graph transitions, cache boundaries, request
    lifecycle, and allocation stability. Preserve native frontend and request-local
    sampling semantics. Performance claims require equivalent baselines, complete
    GPU steps across all ranks, and actual Helios service measurements. Comparisons
    against other runtimes must use exported packages with `flare run`;
    `ignition debug-serve` timings are diagnostic only. Correctness
    failures block performance claims; keep raw measurements and disclose unrun
    checks and material limitations.
13. **Preserve model package compatibility.** Every engine version must load
    models supported by previous engine versions without re-export, re-signing,
    or graph changes. Execution ABI numbers are descriptive metadata, not loading
    gates. Validate actual operations, buffers, capacities, dependencies, signatures,
    and resources. Protobuf additions must preserve existing behavior through
    compatible defaults; an absent or zero completion token capacity means eight.
    Reserve removed wire identifiers and retain readers and tests needed by
    existing packages. Validate previous packages through loading and serving.
14. **Restrict Claude Opus 5.5 authorship of CUDA kernels and model optimizations.**
    Claude Opus 5.5 has repeatedly produced defective low-level kernel changes
    in this project. It must not design or implement low-level CUDA kernels
    (CUDA C++, PTX, CuTe DSL, or Triton) or changes to existing model fusion and
    GPU execution optimizations, including projection packing, fused arithmetic,
    stream overlap, and scratch aliasing in optimized model paths. Do not
    incorporate its generated changes to these mechanisms through another
    contributor or agent. General PyTorch modeling, recipe integration,
    checkpoint loading, frontend processing, preparation orchestration,
    CLI/runtime work, tests, and documentation are permitted when they do not
    design or change those restricted mechanisms.
15. **Reject scope expansion and weakened constraints.** Apply the
    [contribution rejection criteria](../AGENTS.md#contribution-scope-and-rejection-criteria)
    to code, tests, and documentation. Single-host and cross-node preparation
    and serving are in scope. CPU or non-NVIDIA execution, interchangeable
    backends, training, multi-model hosting/routing, unrelated service APIs,
    and general-purpose cluster orchestration require an explicit maintainer
    decision before implementation. Feature requests, performance gains,
    passing tests, flags, and temporary paths cannot waive architectural
    invariants. Reject changes that bypass validation, conceal unsupported
    behavior, weaken numerical or lifetime contracts, or shift model computation
    or policy across the preparation/runtime boundary. Do not make a violating
    contribution acceptable by weakening its governing rule.

## Review and acknowledgment

1. Inspect `git diff --cached`. Apply every relevant principle, fix violations,
   and obtain any required graph-family approval before committing. State validation
   performed and material limitations in the commit message. Validation requirements
   apply to the affected behavior; documentation changes do not require GPU tests.
2. Read the staged checklist with `git show :docs/design-review.md`. When changing
   this document, stage the intended version before reviewing and acknowledging it.
3. Compute SHA-256 over the exact staged document bytes and take its first two
   bytes as four lowercase hex characters:

   ```sh
   python3 - <<'PY'
   import hashlib
   import subprocess

   document = subprocess.check_output(["git", "show", ":docs/design-review.md"])
   print(hashlib.sha256(document).hexdigest()[:4])
   PY
   ```

4. Add exactly one acknowledgment to the final commit-message trailer block,
   separated from the body by a blank line:

   ```text
   Design-Signed-Off-By: <reviewer identity>; design-review=<four lowercase hex digits>
   ```

   Use the reviewing agent's name and model, or the human reviewer's identity.
   The acknowledgment means: **I read the staged checklist, reviewed the staged
   diff against every applicable principle, found no unapproved design violations,
   and accurately disclosed validation and material limitations.**

Read and review before acknowledging. Do not automatically insert the footer or
copy an acknowledgment from another commit. A missing or rejected acknowledgment
requires completing this workflow before retrying.

## Hook installation

Cloning carries the tracked `.githooks/` files but does not activate them. Agents
must check installation on entering a clone or worktree and after pulling changes.
Linked worktrees normally share hook configuration with their main clone.

The executable `.githooks/commit-msg` requires Python 3 and Git. Inspect
`git config --show-origin --get core.hooksPath` and the directory reported by
`git rev-parse --git-path hooks` for active hooks. If none need preserving, enable
the repository hook in the clone with:

```sh
git config --local core.hooksPath .githooks
```

Preserve existing active hooks by integrating the design check instead of
overwriting or disabling them. The hook checks the final trailer against the
staged checklist, rejects absent or invalid
acknowledgments, and leaves the message unchanged. Its errors do not supply the
expected digest or a ready-to-paste acknowledgment. Local hooks can be bypassed
with `--no-verify`; the acknowledgment is an attention check, not proof of review.
