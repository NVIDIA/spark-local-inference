# Helios lifecycle verification

The lifecycle check verifies that request slots and their private resources stay
owned until their final submitted use completes or shutdown confirms a drain.
Early reuse can corrupt another request's state.
Rank-control checks cover readiness and completion predicates guarding
tensor-parallel submission and queue removal.
Additional checks cover native response admission and buffered events,
prefix-checkpoint ownership, speculative scheduling arithmetic, snapshot backing
aliases, and multi-rank offload reply aggregation.

## Run the check

Install Java 11 or later and download the
[TLA+ tools 1.7.4 jar](https://github.com/tlaplus/tlaplus/releases/tag/v1.7.4).
The check uses Python 3, Cargo, `protoc`, and the build dependencies of Helios's
`server` feature, including CMake and a C compiler for the bundled libjpeg-turbo.
It requires no CUDA device or model checkpoint. The `Helios lifecycle` workflow
in `.github/workflows/helios-lifecycle.yml` pins Java 21.0.13.

From the repository root, run:

```sh
python3 verification/helios-lifecycle/check.py --tlc-jar /path/to/tla2tools.jar
```

Use `--java /path/to/java` when Java is outside `PATH`. Use
`--target-dir /path/to/build-directory` to isolate Cargo outputs from another
worktree. Tool failure, a timeout, a Rust assertion, or a TLC violation fails the
command. Temporary TLA modules and test inputs are removed on exit.

In an idle worktree, add `--mutations` to check deliberate Rust faults in
retirement fencing, generation validation, unconfirmed drain, rank readiness,
rank completion, commit dispatch, inherited cache fences, offload aggregation,
speculative pending bounds, and full-lane page reservation. The runner
temporarily edits the corresponding Rust source, requires each
fault to trigger its corresponding behavioral assertion, restores the source,
then reruns the unmodified implementation and its TLA+ conformance check.
It also checks that premature cancellation release in a temporary copy of the
model violates `QueuedReferencesRetainOwnership`.
Additional model faults check early rank commitment, early queue removal, and
ownership release after partial dispatch failure, lost inherited fences, and
speculative pending bounds, and publication before checkpoint-copy completion.

## Scheduler ownership

`Lifecycle.tla` checks two request slots, two queued submissions, four submission
identities, and two generations per slot. Its initial state represents two
requests ready for ordinary decode. A slot is free, live, or retiring. Each
submission retains its fence, participating slots and generations, and whether it
requires exclusive execution. Completion consumes the oldest submission.

| Model operation | Rust implementation |
|---|---|
| `Admit` | `RequestScheduler::admit` and `admit_planned` in `helios/src/scheduler.rs` |
| `Commit` | `RequestScheduler::commit` |
| `Complete` | `RequestScheduler::complete_inner` and `advance_retirement` |
| `Cancel` | `RequestScheduler::cancel` and `retire` |
| `UnconfirmedDrain`, `ConfirmedDrain` | `RequestScheduler::release_after_shutdown` |

The invariants require ordered completion, retained ownership and generation
identity for every queued reference, retirement through the last-use fence, and
exclusive feedback-dependent submissions. Submission and generation bounds stop
further allocation; identities never wrap. Failed drain is a permitted stutter.
Confirmed drain clears ownership but does not prohibit later admission through
the scheduler API.

The Rust test `scheduler::lifecycle_tests::scheduler_matches_lifecycle_model`
warms two real requests through initialization, prefill, and prompt snapshots.
It normalizes the completed warmup fences to zero, then explores operation
sequences through depth 12 and four further submissions. Re-admitted requests
use the ordinary preparation stages. Equivalent test states are pruned using
selected request, submission, retirement, resource, and statistics fields.
This is bounded conformance testing, not exhaustive Rust-state exploration.

Each operation emits its before state, action, and after state. The runner writes
these observed edges into a temporary TLA module. TLC checks every edge with
`AllowedEdge`, which uses the same transition definitions as `Next` in the
exhaustive model. Model and Rust state are connected through a 14-integer
projection containing submitted/completed fences, slot phases, generations,
last-use fences, queue fences, participant masks, and feedback barriers.
Queued generations are checked directly against the real submission lanes in
Rust; the projection reconstructs them from the retained slot generations.

Rust assertions also check disjoint free/live/retiring private pages and
snapshots, exact resource retention across cancellation, queued generations,
pending submission counts, stale handles and plans, malformed generation
completions, and duplicate or out-of-order completion rejection. Retained plans
exercise commit rejection with a full window or pending feedback barrier. Rejected
operations preserve the test's observed state.

## Response ownership

`ResponseLifecycle.tla` checks three reusable response cells, two host admission
permits, and one- or two-choice generation groups. Each response has independent
producer and reader owners, a bounded event queue, publication and consumption
counts, cancellation, and a terminal marker. A group retains its entire permit
reservation while any producer or reader in that group survives. Producer
retirement and reader release are separate actions; reading an event does not
release admission. Cells and group reservations can be reused without an
admission-count cutoff.

The invariants independently derive required admission from the owner sets and
compare it with observed semaphore availability. They also require event
conservation while a reader survives, bounded total retained events, and
cancellation of a producer whose reader was dropped. The producer contract in
this model allows two nonterminal events and one terminal event per response,
with queue capacity five. It does not prove the scheduler's output-limit
arithmetic, payload byte sizes, or Tokio's internal queue implementation.

`server::runtime::phase_tests::response_lifecycle_tests::native_responses_match_lifecycle_model`
replays operation sequences through depth ten against production
`ModelRuntime::generate_many`, `Incoming::emit` and `fail`, native event receivers,
and the `Incoming` and `Generation` destructors. It observes semaphore permits,
actual producer/reader ownership, and queue lengths. A 28-integer projection
contains available permits, three group reservation sizes, and each cell's group,
producer, reader, publication, consumption, queue, terminal, and cancellation
fields. All observed edges are checked against `ResponseLifecycle.AllowedEdge`.
Equivalent projected states are pruned; this is bounded conformance, not
exhaustive exploration of all Rust state. Repeated grouped admission and release
also exercises reuse of the same runtime.

The runner separately drives worker completion through synthetic completion
records to check that unread output releases model slots while retaining host
admission, including a reasoning continuation. An HTTP test checks all four
streaming protocols while terminal frames retain ingress admission after native
generation handles have been released. These checks exercise the production
host paths without a GPU. HTTP encoder state is outside the TLA+ projection.
The non-streaming JSON path releases admission before its serialized body is
consumed. The response ownership model does not cover that path.

Response mutations remove either producer or reader permit ownership, undercount
a generation group's admission, shrink the queue below its promised budget, and
release a model reservation when only its producer has retired. Each fault must
fail the corresponding Rust assertion or TLA+ admission invariant.

## Rank coordination

`RankProtocol.tla` checks two submissions, two graph identities, and three ranks.
Readiness is tracked for both peers; completion is tracked for every rank.
Preparing frames occupy execution-window slots. A second reservation requires
the preceding frame to be committed. Feedback-dependent frames occupy the window
exclusively. Queue removal requires every rank's completion for the oldest frame.

The Rust tests call the private helpers in
`helios/src/execution/threaded/protocol.rs` used by `ThreadedExecutor`.
The 15-integer projection contains submission/removal counts and each pending
frame's fence, graph, barrier, readiness mask, commitment marker, and completion
mask. TLC checks observed helper transitions against `RankProtocol.AllowedEdge`.

The commitment transition represents the marker set after successful peer
dispatch and local submission. CPU exploration checks its eligibility predicate;
it does not execute the CUDA calls around that marker. Peer completion insertion
requires the peer's oldest missing fence and matching graph. Rank-zero insertion
finds a frame by fence, and duplicate insertion replaces the stored completion
before returning an error. CUDA's local completion ordering and identity remain
external to these helper checks.

Queue removal happens before completion agreement and timing merging. The model's
removed fence represents bookkeeping removal, not successful publication to the
scheduler. Tests exercise a rejected agreement after removal.

`RankDispatch.tla` separately models three ordered peer sends, local submission,
independent rank completion, failure, publication, and release. A peer may finish
after receiving its commit while another peer has not received one. Failed
dispatch retains ownership even when accepted peers finish later. Successful
release assumes the runtime's external drain contract.
Failure can also follow local enqueue; a failed submission does not imply that
rank zero queued no work.

The concrete channel test calls the production `dispatch_commit` helper with
three real `mpsc` receivers. It checks success and disconnection at every send
position, including the exact accepted prefix, propagated error, and absence of
later sends. This is a channel-contract test; the dispatch model is not a
refinement proof of the complete executor.

## Prefix-checkpoint ownership

`CacheOwnership.tla` explores its complete finite graph with three one-time
checkpoint identities, fences zero through two, binary direct pins, and a
three-checkpoint chain or branch. Actions publish, pin, unpin, mark used, cancel
pending publication, compact or evict, advance completion, and publish a duplicate.
Invariants require immutable descendant page chains, disjoint owned pages,
inherited read fences, retirement retention, and completion-gated reclamation.

`cache::lifecycle_tests::bounded_cache_ownership_matches_tla` calls the production
`PrefixCheckpointCache` methods through depth ten. Successor branches copy every
cache field before calling an operation. The visited key includes the projection
and relative retention order. The 29-integer projection contains completed fence,
publication count, three original-page read fences, and eight fields per node
(status, parent, original parent, page mask, readiness fence, last-use fence,
direct pins, retirement fence). All observed edges are checked against
`CacheOwnership.AllowedEdge` in batches of at most 2,000 to bound TLC parser memory.

Rust assertions additionally check ordered page chains, cumulative pins,
transactional rejection, duplicate-publication caller ownership, and unique
returned resources. A separate test compares `reclaimable_if_pinned` with actual
pinning and eviction in 36 chain, branch, readiness, and pin configurations.
Victim selection is abstracted to any eligible victim; the model does not prove
demand/role retention ranking or identity/resource reuse.

The scheduler alias test independently covers all 225 pairs of contiguous ranges
within four pages, reservation budgets zero through five, both release orders,
and reacquisition. It checks actual snapshot reservation/release against
per-page owner counts, free sets, backing-page statistics, and retained page
budgets. It does not exercise physical CUDA memory aliasing.

## Checkpoint destination reservations

`CheckpointReservation.tla` checks two requests, three snapshot rows, and submission
fences zero through two. Each row is free, an unused anchor guard, an armed capture
destination, an in-flight copy, a published checkpoint, a retired destination, or
mandatory execution storage. A request may hold one guard while an earlier extra
capture owns another row. Mandatory allocation can revoke a guard; it cannot
reclaim an armed or copying destination before completion. Cancellation immediately
releases unused guards and retains outstanding destinations through their final
fence. Publication requires copy completion. Submitted and completed fences do
not wrap, and rows can be reused within those bounds.

`scheduler::checkpoint_lifecycle_tests` calls the production scheduler's planning,
commit, completion, and cancellation methods. It cancels an anchor capture during
its producing submission, while waiting to copy, during its copy submission, and
after publication. The tests check release after the relevant fence and published
checkpoint survival after cancellation. A separate guard test checks oldest-first
revocation and overlapping backing aliases through the production allocator and
revocation helper. Guard setup in that test assigns allocated rows directly to
live requests; it does not exercise optional-cut selection.

A 15-integer projection contains submitted/completed fences, the live-request
mask, and each row's phase, request owner, retention fence, and observed GPU-use
fence. TLC checks the observed edges with the same transition relations used by
`Next`. Scheduler commit and completion are atomic compositions of reservation,
arming, submission, publication, and reclamation actions. These traces cover
selected cancellation boundaries, not exhaustive Rust-state exploration. Optional
work arithmetic, victim ranking, physical CUDA aliases, and executor fence truth
remain outside this model.

## Speculative accounting

`Speculation.tla` checks two lanes, seven drafts per lane, target width eight,
physical packed capacity five, 64-token pages, and six submissions with at most
two queued frames. Requests start at consumed position 58 with one output token
and context extent 90. Completion explores every positive per-lane emitted-count
pair fitting physical capacity. Invariants bound pending progress, output and
context extents, and storage through each lane's full target ceiling.
`Speculation16.cfg` additionally checks fifteen drafts, target width sixteen,
packed capacity nine, and initial consumed position 54 under the same remaining
bounds. Both configurations cross a page boundary with the full lane ceiling
before pending advance crosses it.

The Rust fixture drives `RequestScheduler::plan`, `commit`, and `complete` across
one/two lanes, widths one through eight plus nine and sixteen, page sizes
64/128/256, and selected batch/draft/packed geometries and acceptance sequences.
Its six-vector projection
contains geometry, each lane's consumed/generated/scheduled positions and page
count, both queued lane ceilings, and submission/completion counters. Observed
commit/completion edges use the model's arithmetic relations.
Widths through eight use completion capacity eight; widths nine and sixteen
use completion capacity sixteen. Completion records and confidence sidecars
follow those declared capacities.

Separate direct tests reject malformed lane records and aggregate physical
overruns without partial mutation. They also check terminal no-op feedback,
cancellation, deferred page ownership, slot reuse, and stale generations while
speculative work is queued. Those lifecycle cases are outside this arithmetic
model. GPU draft/KV contents, arbitrary acceptance distributions, and progress
are outside these checks.

## Offload transfer aggregation

`OffloadTransfer.tla` checks two transfer keys and two/three ranks. Every accepted
subset is permitted, including a zero-acceptance rejection. Rank replies occur
once per accepted job in rank FIFO order, with every success/error assignment
and cross-rank interleaving. Pending ownership and outstanding rank counts must
remain until all accepted ranks reply. Exactly one aggregate then reports the
first error and sum of completed bytes.

`offload::transfer::lifecycle_tests::offload_transfers_match_model` calls production
`Transfers::submit`, `absorb`, and `poll`. Real channels inject full/disconnected
offers and rank replies; parked threads satisfy the worker-liveness check. Both
same-job/different-window and different-job/same-window keys are covered. Its
16-integer projection contains each key's accepted/replied masks, pending marker,
aggregate count, error and bytes, followed by rank outstanding counts and active
rank count. Duplicate in-flight submission and empty polling are checked too.
CUDA copies, physical slab ownership, worker channel lifetimes, shutdown drain,
and scheduler cancellation require their separate contracts and execution tests.

## Assumptions and limits

Scheduler methods are atomic relative to other scheduler calls because one
worker owns their mutable state. The executor must supply ordered aggregate
completion fences and truthful drain confirmation. This check verifies how the
scheduler consumes those signals; it does not verify their CUDA or rank-thread
implementation.

The rank-helper checks cover the extracted bookkeeping predicates. They do not
execute `poll_ranks`, `rank_loop`, poison handling, or `close` against CUDA, and
do not establish all-rank drain or collective progress. The socket coordinator
in `execution/parallel.rs` is a separate implementation that these checks do not
cover.

The model permits any nonempty live cohort with a compatible feedback barrier.
It does not prove scheduling-policy choices or fairness. It checks safety only;
failed drain can retain resources indefinitely. The initial decode fixture and
ordinary completion records do not cover speculative acceptance, kernel state
updates, prefix-cache compaction, snapshot aliases, offload transfers, or GPU
inactive-row writes. The separate CPU checks above cover host bookkeeping for
speculation, cache compaction, aliases, and transfer aggregation. They do not
establish kernel state updates, physical memory reuse, or inactive-row behavior.

## Maintain correspondence

Use [verification selection guidance](../../docs/verification.md) to decide
whether a change needs an ordinary test, a Kani harness, a TLA+ model, or real
GPU execution.

When a covered method changes, review its state updates and rejection paths
against the model operations and Rust projection. Change an operation only when
the implementation contract changes. Replay a counterexample through the actual
Rust methods before treating it as a runtime defect. Keep an implementation
failure distinct from an inaccurate abstraction or unsupported tooling.

Use pstack's `statechart` skill to interpret TLA+ results and
`maintain-formal-verification-skill` when covered code changes without updating
the checks. Neither skill is stored in this repository. A deliberate source mutation must fail the corresponding Rust test
or observed-edge check before the verification is used to guard that property.

## Sequential Rust proofs

Install the pinned Kani release and its matching compiler:

```sh
cargo install --locked kani-verifier --version 0.68.0
cargo kani setup
```

Run the sequential validator proofs from the repository root:

```sh
cargo kani --manifest-path helios/Cargo.toml --lib
```

The harnesses call production functions in `execution/transport.rs`, `model.rs`,
and `model/contract.rs`. They compile only under `cfg(kani)` and do not select a
serving path. Kani 0.68.0 uses CBMC 6.11.0 and nightly-2026-08-21. Automatic
safety and unwinding checks remain enabled. The completion harnesses use unwind
bound 20, the 65-token mask-boundary harness uses 66, and the state-view
harnesses use 16.

The vector safety harness supplies arbitrary `i64` fields and token IDs in
owned vectors of every length zero through sixteen, including inactive and
absent-vector cases. It checks that successful active-vector validation implies
the declared count, token, mask, and cursor bounds. An arbitrary in-range index
checks both emitted tokens and unused suffix entries. The positive harness
covers well-formed active vectors with length and width one through sixteen,
every permitted copy mask, and the default draft cursor. It prevents an
always-rejecting validator from satisfying the complete suite.

A separate mask-boundary harness uses 65-token storage, 63 or 64 offered drafts,
one arbitrary `u32` emitted token, and an arbitrary `i64` mask. It checks that
acceptance is equivalent to a nonnegative mask, including the maximum signed
mask at both draft counts. Other vector geometries above sixteen are outside
these proofs.

The completion-contract harnesses check `model::validate_completion` against
independent token upper-bound and history-cursor requirements. They supply any
positive `u32` vocabulary, any `u32` cursor lag, arbitrary `i64` completion status,
token IDs and consumed position, vector lengths zero through sixteen, optional
vectors/cursors, and speculative or decode operation kind. Cursor arithmetic
uses an `i128` oracle. Separate soundness and completeness harnesses check
acceptance and rejection against the same requirements. Unrelated metadata is
fixed; core validation is outside these harnesses. Counts, token lower bounds,
and status validity belong to the transport and scheduler validators. Cursor lags larger
than `u32::MAX` are outside this proof's domain.

The containment proof uses arbitrary target-width offsets, strides, and extents,
and representative equal or unequal buffer names. It compares the real
`StateView::contains` result with overflow-free `u128` arithmetic.

The storage-fit proof calls production `StateView::fits_storage`, which
`StateView::validate` uses after resolving its buffer. It supplies arbitrary
target-width offsets, view extents, byte strides, buffer extents, row counts,
and element strides, with every dtype and independent empty/nonempty shape and
stride vectors. An independent dtype-size table and `u128` arithmetic specify
both acceptance and rejection, including overflow and exact boundaries. Each
vector allocates one element before selecting its empty/single-element length;
allocation sites are independent of the geometry cases.

Run the accompanying manifest-lookup and diagnostic test:

```sh
cargo test --locked --manifest-path helios/Cargo.toml --lib \
  model::contract::tests::state_view_validation_resolves_buffers_and_preserves_storage_errors -- --exact
```

That test checks lookup before geometry validation, exact error classifications
and messages, malformed geometry, and equality boundaries. The numerical Kani
proof excludes manifest lookup and diagnostic construction. Shape and stride
vectors represent empty and single-element inputs; the predicate reads only
their lengths and first elements. Core validity, allocation failure,
resource-owner destruction, GPU state, and rank interleavings require their
separate checks.

## Mutation coverage with cargo-mutants

Install cargo-mutants 27.1.0, then run the scoped rank-helper selection:

```sh
TMPDIR="$PWD" cargo mutants --manifest-path helios/Cargo.toml --no-config \
  --file src/execution/threaded/protocol.rs \
  --re '(ready_to_commit|pop_completed_front|dispatch_commit)' \
  --test-tool cargo --test-workspace=false --gitignore=false --cap-lints=false \
  --jobs 1 --jobserver-tasks 4 --build-timeout 300 --timeout 30 \
  --output /tmp/flare-protocol-mutants --cargo-arg=--lib \
  -- execution::threaded::protocol_tests::
```

Run from the repository root. cargo-mutants 27.1.0 rewrites normal dependency
paths in its copied crate but leaves development and target dependency paths
relative. `TMPDIR="$PWD"` places the temporary copied crate beside `selene/`,
`selene-manager/`, and `eos/`, preserving those paths. The original source remains
unchanged, and the tool removes its temporary build tree afterward.

The selection is expected to leave no surviving mutation; inspect any survivor
as the [verification guidance](../../docs/verification.md#use-mutation-testing-to-check-sensitivity)
describes. A mutation that does not compile is unviable and does not count as
caught: constructing `Some(Default::default())` is unviable because `Pending`
has no `Default` implementation.
The tool does not generate every targeted fault, including changing `.all` to
`.any` or ignoring a failed commit send. The optional `--mutations` runner checks
those explicit faults separately.

For the storage-fit predicate and its manifest-boundary test, run:

```sh
TMPDIR="$PWD" cargo mutants --manifest-path helios/Cargo.toml --no-config \
  --file src/model/contract.rs --re 'StateView::fits_storage' \
  --test-tool cargo --test-workspace=false --gitignore=false --cap-lints=false \
  --jobs 1 --jobserver-tasks 4 --build-timeout 300 --timeout 30 \
  --output /tmp/flare-storage-mutants --cargo-arg=--lib \
  -- model::contract::tests::state_view_validation_resolves_buffers_and_preserves_storage_errors
```

This selection mutates the predicate into unconditional acceptance and
rejection, weakened conjunctions, omitted shape checks, and incorrect
comparisons, and is expected to leave no surviving mutation. The storage-fit
Kani equivalence assertion detects a strict-boundary fault (`<=` changed to `<`).
