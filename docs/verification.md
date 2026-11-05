# Choose verification for Flare changes

Choose the check from the property that can fail and the consequence of failure.
Use the smallest check that exercises that property on the affected path.
Formal verification complements ordinary tests and hardware qualification.

## Select the check

| Property or change | Primary check | Example |
|---|---|---|
| Observable behavior, regression, feature integration, or rejection semantics | Ordinary Rust or Python tests | A stale completion returns an error without changing another request. |
| Sequential Rust must satisfy a property for every input in a stated domain | Kani proof harness calling production code | Completion counts cannot produce an invalid slice; state-view extent arithmetic cannot accept overflow. |
| Safety depends on the order of events across actors or lifecycle transitions | TLA+ specification checked with TLC | Cancellation cannot free storage while a later frame or another rank still references it. |
| CUDA ordering, captured state, inactive rows, allocation stability, or resource lifetimes | Real execution tests on assigned GPUs | A delayed copy cannot permit pinned-frame reuse; rollback preserves complete continuation state. |
| Latency, throughput, or memory cost | The target benchmark or live trace | A scheduling change improves exported-package serving under the required workload. |
| Whether tests detect plausible implementation mistakes | Mutation testing | Changing an all-rank completion gate makes a relevant test fail. |

Start with ordinary tests for a concrete defect or behavior change. Add Kani when
the question is universal over sequential inputs and examples leave a material
gap. Add TLA+ when event ordering, ownership, or protocol state is the central
question. Combine them when a property crosses those boundaries. A contained
edit with adequate existing coverage does not require another test. Do not add
tests that merely restate implementation text or assertions.

Examples of scoped combinations:

- A malformed token vector needs direct rejection tests. Kani can additionally
  check every integer-field combination for safe validation.
- Cancellation with two submissions needs a Rust regression and a model of
  queued ownership. CUDA tests establish that the executor's completion and
  drain signals satisfy the model's assumptions.
- A package field change needs loading and compatibility tests. A protocol
  model is useful only if the field changes protocol states or ordering.
- Documentation and formatting changes normally need inspection and applicable
  existing checks, without additional test machinery.

## Write ordinary tests against behavior

Exercise the real affected entry point and assert its result, state changes,
resource ownership, or error. For rejection and cancellation, include relevant
state preservation and lifetime checks. A test that retries a stale plan cannot
establish a capacity barrier merely because some error occurs; check the relevant
error or arrange inputs that isolate the intended behavior.

Keep generated test sequences bounded and deterministic. State which inputs and
states are covered. Pruning on selected state fields does not establish exhaustive
coverage of the implementation's complete state. Mocks can isolate a contract,
but cannot establish the behavior they replace.

## Use Kani for sequential Rust properties

Put harnesses behind `#[cfg(kani)]` beside the production module they verify.
Call the actual function. Use `kani::any()` for inputs, `kani::assume()` for real
preconditions, and assertions for independent postconditions. Verify arithmetic
with a wider representation when that gives an independent specification.

Keep bounds, unsupported constructs, and fixture assumptions explicit. Retain
unwinding, undefined-function, overflow, and memory-safety checks. An insufficient
unwind bound, unsupported operation, compiler error, or exhausted solver is an
incomplete check. It is not a successful proof. Do not constrain an input merely
to exclude a counterexample.

Check useful positive cases or reachability as well as rejection properties, so
a validator that rejects everything cannot satisfy the complete contract.
State whether a result covers all bit patterns, bounded collections, selected
storage geometries, or only a particular valid-input class.

Kani does not establish concurrent thread interleavings, CUDA behavior, or the
correctness of an external system. Harness fixture construction and destruction
must be distinguished from the borrowed function being proved. Keeping an
artificial fixture alive can isolate a borrowed validator; that proof then says
nothing about destroying the fixture's resource owners. Never replace a required
production behavior with a stub and claim it was verified.

See the [Kani manual](https://model-checking.github.io/kani/) for supported
features and harness syntax. The checked Helios targets and commands are listed
in [Helios lifecycle verification](../verification/helios-lifecycle/README.md).

## Use TLA+ for states and ordering

Derive actions from actual ownership, message, enqueue, completion, and retirement
boundaries. A serialized scheduler method can be one action. Sequential sends to
several peers cannot become one atomic broadcast. Represent partial failure and
work that remains outstanding after cancellation or poison.

State safety invariants independently of the transition implementation. Check
liveness only with explicit progress and fairness assumptions. Set finite bounds
on actors, resources, identities, and work. Stop allocating at a bound; do not
invent identity wraparound. Preserve the invariants and assumptions when a model
finds a counterexample, unless source inspection demonstrates a modeling error.

A successful TLC run verifies the bounded specification. A TLA+ specification is
not automatically a proof of the Rust implementation. Connect it through tests
that call production code and compare observed transitions with the same model
relations. Replay counterexamples against actual code. Checking generated traces
is bounded conformance evidence; formal refinement requires establishing that
every reachable implementation transition maps to an allowed specification
transition. An unbounded theorem requires a proof beyond finite TLC exploration.

Use pstack's `statechart` skill to interpret model results and
`maintain-formal-verification-skill` when covered code changes without rechecking
the specification. Document the functions, state projection, bounds, and external
contracts needed to interpret the check.

## Use mutation testing to check sensitivity

Prefer `cargo-mutants` for routine Rust mutation coverage. Use a scoped set of
functions and relevant tests before considering a wider run. Keep caught,
surviving, unviable, and timed-out mutations distinct. Compilation failures do not
demonstrate test sensitivity. Inspect a surviving mutation before treating it as
a missing test; a generated change can be equivalent for the declared contract.

A targeted fault can calibrate an invariant that the tool's mutation operators
do not generate. Use scratch changes, require the expected behavioral failure,
restore source, and rerun. Source matching can locate a mutation, but source-text
checks cannot establish runtime correctness. Avoid custom mutation frameworks
when existing tools produce the required faults.

## Maintain the checks with the implementation

Review affected tests, harnesses, model actions, and projections when their Rust
behavior changes. Preserve the stated contract; do not weaken a check to obtain a
pass. Disclose unrun hardware cases and unsupported tooling. Proof-only changes
need the relevant proof and ordinary compilation checks; they do not by
themselves require a model export or a serving benchmark. Changes to actual GPU
execution or lifetime behavior require the real execution path.

Keep runnable tests, proof harnesses, model configurations, and raw tool output.
Do not add audit reports, receipts, source snapshots, hash inventories, or
development-provenance gates.
