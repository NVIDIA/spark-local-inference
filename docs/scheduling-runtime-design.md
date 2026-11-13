# Prepared-graph scheduling in Helios

Helios selects prepared physical graphs by expected useful value per execution
time. Request-level token and delivery prices provide fairness. A decode-age
continuation term prices the delay caused by a nonpreemptible graph. The
implementation is `helios/src/scheduler.rs` (commit, completion, and request
resources) and the modules in `helios/src/scheduler/`: `policy.rs` (candidates
and selection), `utility.rs` (value arithmetic and cohort selection),
`service.rs` (service prices), `cost.rs` (duration estimates), `refresh.rs`
(timing-refresh and acceptance-learning credit), `learning.rs` (acceptance
probes), `rounds.rs` (speculative round candidates and confidence forecasts),
and `speculation.rs` (acceptance estimates).
[`scheduling-utility.md`](scheduling-utility.md) defines the utility and price
flow. Python experiments are in `scripts/scheduling_sim/`; they do not run in
Helios. The diagnostic `ignition debug-serve --direct` path has its own Python
scheduler in `ignition/direct/`. It reproduces the selection specified here
only for decode rounds, and it estimates graph costs with a 7/8 moving average
that reverts to unmeasured after 30 seconds. The duration estimates, funded
refresh, and acceptance learning specified below describe Helios alone.

The native catalog contains prefill, scored prefill, ordinary decode, complete
speculative generation, mixed prefill/generation, vision, and state operations.
A complete speculative graph contains proposal generation, optional confidence
and prefix allocation, target verification, and accepted-state commit. Proposal
and verification are internal stages, not independently scheduled operations.
Ignition captures mixed graphs only when given `--mixed-prefill`, and only for
recipes that declare mixed capture; the Qwen3.8 Flash recipe
(`ignition/recipes/qwen38`, alias `qwen3.8-flash`) is the one that does.
Native loading retains eligible mixed graphs by default; `--no-mixed-prefill`
excludes them from the scheduling catalog, and `--mixed-prefill` requires an
eligible captured mixed graph. Neither serving flag changes captured storage.

## Speculative graph coverage

A speculative physical program is identified by request capacity B, captured
proposal depth D, per-request target capacity W including the pending token,
and aggregate target capacity P. All four extents are explicit; rectangular
programs declare P = B × W. W and P do not imply D. Width-one programs have D = 0
and maintain the draft context needed to resume wider rounds. Their execution
cost competes with wider programs at every captured request bucket.
For learned proposals D counts produced candidate positions, including greedy
proposals. For copy proposals it bounds the supplied prefix and contributes no
learned-proposal draws to sampling metrics.

Ignition captures power-of-two request buckets through padded concurrency.
The `--speculative-token-capacity R` envelope bounds aggregate target rows for
wider programs. Rectangular programs require B × W <= R; ragged programs require
P <= R. Each recipe captures its concrete supported shapes. Load-time selection
can reduce capacities only where those physical programs exist. Fixed selection
supports host-allocated ragged programs; device confidence allocation requires
adaptive selection.

For each selected physical program, the prepared value source determines who
allocates target rows. An exact `QueryLength` binding supplies host-selected
lengths whose sum fits P. The host uses completed acceptance observations and
measured complete-program costs, fixing those lengths before proposal draws.
A `SpeculativePrefixLimit` binding supplies immutable lane ceilings. The GPU
program generates its actual D proposals, evaluates causal confidence, and
allocates within P. It cannot change the cohort or request identities. Each
ceiling is bounded by `min(D + 1, W, request limit)`, but ceilings may sum above
P. Reservation covers every ceiling and the declared cache draft extent.

Device allocation repeatedly admits the best immediate causal frontier using
joint prefix probability, with ties resolved by shorter prefixes and original
request slots. Frozen host service prices rank cohorts, not the GPU's causal
frontier. Host and device allocation both price the complete captured program,
including unused physical rows. Neither path schedules a separate proposal
completion or retains proposals for later verification dispatches.

## Completed confidence and acceptance

For prefix-survival estimates `s_i`, expected output at width W is
`1 + sum(s_1..s_(W-1))`, bounded by actual proposal work, output limits, and
context capacity. Speculative selection scores raise each `s_i` to the exponent
`adaptive_verification_alpha`, which defaults to one. Each request tracks
acceptance by source. Accepted prefixes
supply positive observations; a rejection supplies zero to longer offered
prefixes. Termination censors the remaining suffix. Width-one rounds measure
cost without supplying acceptance observations.

A confidence-producing program declares `confidence_width`, equal to its actual
D. Its FP32 sidecar has shape `[2, request_slots, draft_capacity]`, covering every
produced prefix. Only completed rounds update host forecasts. Rank zero's
confidence drives host decisions; peer estimates must be valid probabilities
but need not agree bitwise. Exact request identities, consumption, emitted
prefixes, and committed state agree across ranks.

Confidence forecasts and verified acceptance are separate observations.
A request's forecast averages its completed joint-prefix probabilities,
starting from a population prior. That prior begins at `0.5^d` for the `d`-th
draft position and moves toward each completed prefix probability with weight
one sixty-fourth. Forecast and acceptance prefix estimates start with
weight two, cap at 128, and age with a half-life of 128 generated tokens.
Unobserved suffixes retain their means. Ordinary selection uses the historical
means; funded acceptance learning uses the bounded optimistic estimate
`min(1, mean + sqrt(ln(20) / (2 * aged_weight)))`.

A prepared program may also write `completion_speculative_survival`, cumulative
products of `min(1, p(token)/q(token))` for evaluated prefixes. These replace
binary acceptance observations at those depths. They affect future selection,
never the proposals or allocation of the completed program.

Cost samples use GPU replay intervals for the full graph, including proposal
work, allocation, verification, commit, and collectives, with the embedding-readiness
correction described below. Host delays and queue wait do not enter graph estimates.
A trace attaches physical B/D/W/P, selected cost, observation
age, selection reason, and exploration credit to each committed plan.
`flare_speculative_*` counters cover complete speculative programs rather than
separate producer and target calls.

## Useful service and selection

Prompt progress counts consumed uncached prompt tokens. Output progress counts
committed target tokens, including accepted speculative tokens and a replacement
or bonus token. Padding, rejected proposals, inactive rows, restored prompt
history, and state copies contribute no committed token service. A visible
speculative burst supplies one delivery regardless of its token count.

For a request with token rate `x` and delivery rate `z`, the utility gradient
prices are

```text
prefill token: 1 + f/x
output token: beta + f/x
output delivery: g/z
```

Only decoding requests have a delivery coordinate or output-age cost. Normalized
service prices follow the entropy-dual update in `scheduling-utility.md`. Actual
service updates history; expected acceptance scores choices. Conditional
acceptance is request-local and respects available proposals, grammar and stop
restrictions, output limits, and remaining context.

Let `T` be the graph duration in milliseconds and `A_i` the age of a decoding
request's last visible output. The reference interval `sigma_i` uses the fastest
batch-feasible delivery rate. For `K` delivery-bearing streams and a family of
`k` streams, choose the measured graph minimizing `T/min(batch_capacity,k)`;
then `sigma_i = K*T/min(batch_capacity,k)`. This corresponds to independent
uniform cohorts within each family and wall-time shares proportional to family
size. Prefill-only requests do not enlarge `K`. Multistage blocked streams use
an individual future-delivery reference.
For a ready inference graph, selection maximizes

```text
(sum of expected priced token and delivery service
 + sum over served decoders of c_i*sigma_i*(A_i+T)) / T
 - T/2 * sum over active decoders of c_i.
```

All values in one decision share a positive scaling factor to keep large prices
representable. `ScheduledPlan::selection_value_rate()` exposes the scaled score;
`selection_log_scale()` exposes the logarithm of that factor. An unknown duration
has an optimistic work score instead of a division by zero.

Every loaded physical batch capacity and selected verification width competes.
For a graph whose row budget does not constrain individual prefixes, duration
and excluded-stream waiting cost do not depend on cohort membership, so the best
cohort contains its highest additive lane values. Fixed host allocation retains
complete prefixes, shortened only by request limits; a constrained ragged graph
selects the highest-value cohort fitting both B and P. Adaptive ragged selection
evaluates cohort prefixes under three bounded orderings: total lane value,
pending-token delivery value, and value per target row. Each cohort is priced
using its resulting causal prefix allocation. This search does not enumerate
every possible cohort.

Selection is bounded to 64 request slots. Before ordinary selection, candidates for
the same graph kind, request cohort, and logical per-request prefixes discard any physical
shape whose capacities are all at least as large as an alternative, with at
least one strictly larger capacity. Mixed candidates also require the same prompt
row and logical prompt work. Speculative candidates also require the same
actual proposal depth. Timing estimates cannot make redundant padding preferable.
A batch-four candidate serving three requests competes with a batch-two
candidate serving two requests. Bounded cost-refresh probes are separate
from ordinary selection. Selection does not enumerate exponentially many
request subsets.

Each prefill capacity competes using the actual tokens it can consume. Atomic
image spans, system boundaries, and prompt length restrict logical work. A
4096-token graph processing a 64-token tail earns 64 prompt units and at most one
output token. There is no smallest-fitting-shape filter, duration ceiling,
forced oldest-request filter, or per-chunk prefill lateness cost.

With no active decoder, the age term vanishes. Fully occupied large prefill
graphs win when their useful value rate is highest. With decoding streams, a
smaller prefill can win near a throughput knee because it imposes less waiting.
Increasing a stream's output age raises its delivery value; it does not make
inefficient operations mandatory at a fixed deadline.

## Prerequisites and completion

Initialization, restore, vision execution, and intermediate snapshots enable
subsequent useful work. When a prerequisite immediately enables inference,
compare every pair of prepared prerequisite and following inference shapes.
Both cohorts are selected jointly. The first graph can also finish terminal
requests; the following graph can include already ready requests.

For fixed graph durations and capacities, lane values are additive. Enumerate
the number `k` of newly ready rows served by the following graph. Choose the
`k` largest such contributions, fill its remaining capacity with the largest
already ready contributions, and fill the prerequisite's remaining capacity with
the largest terminal-completion contributions. Sorted prefix sums give the best
choice over all `k` without enumerating request subsets. This solves the cohort
choice exactly for those two fixed shapes. At least one newly ready row must
participate in the following graph.

Score the sequence using total duration `T+F` and expected useful service,
with output ages resetting at the following graph's completion. Terminal
requests completed in the first graph stop accumulating age at time `T`.
Compare the productive first-stage prefix as well as the complete sequence.
Unused prerequisite capacity can advance additional rows at no extra physical
cost; equal value rates prefer advancing more requests.

The submitted plan retains the prerequisite's physical duration `T`; lookahead
time must not enter execution timing or cost learning. Only the first graph
executes before replanning. Setup never earns committed tokens.

For chains with more prerequisite stages before token production, an individual
continuation supplies value `V` in a further `F` milliseconds. Its cohort-ranking
contribution is `V*T/(T+F)`. Sum these contributions, divide by `T`, and apply
the cost of delaying active decoders. The continuation includes remaining image
operations. Unknown durations compare optimistically by useful value.

This is bounded lookahead, not an optimal schedule over all future execution
sequences. The fixed-pair cohort calculation is exact; longer multistage chains
use the individual continuation index. The stationary rate-utility result does
not establish optimality for those multistage decisions.

Image dependencies are grouped into whole-image windows bounded by prepared
feature storage. Vision prerequisites populate the active window; prefill stops
before the following window's first dependency. Atomic image spans remain within
one prefill chunk. A request spanning multiple windows reserves one private
snapshot row. At each window boundary, submitted work drains before snapshotting
the exact consumed state. Restore binds that state to a fresh request generation,
then the next window's vision graphs write compact feature offsets from zero.
The logical request retains its original prompt, seed, sampling settings, and
usage counters. Private window snapshots neither publish prefix-cache entries
nor require host caching. Snapshot and restore operations preserve
partial pages and every declared continuation history.

A terminal snapshot enables the visible response completion and frees the
request's execution resources. Terminal retirement has its own value and age
cost, independent of output-token and streaming-delivery prices. Its age starts
when validated completion identifies termination, or when preemption initiates
retirement. Queued work and final-token publication do not reset that clock.
Preemption retires a resident execution request without completing its response.

For snapshot duration `T`, terminal age `A`, and the snapshot family's
batch-feasible reference interval `sigma`, the lane contributes
`terminal_value + terminal_age_price * sigma * (A + T)`. Its age coefficient also
enters every candidate's common waiting cost. In a two-stage sequence, a terminal
row's age stops at its snapshot completion, including the saved waiting cost over
the remaining sequence. Nonterminal snapshots retain prerequisite utility.

Terminal work competes through the same value-per-time score, cohort selection,
and padding rules as inference. It has no priority override or admission-order
constraint. Unknown snapshot timings follow the discovery rules; no duration or
reference interval is invented. Snapshot execution earns no token or streaming
burst credit. Both balanced and throughput policies apply terminal value and
urgency. Successful snapshot completion and cache publication precede admission
of an immediate continuation after normal response completion. Error and
cancellation paths preserve fenced resource ownership regardless of utility.

## Duration estimates

Each prepared graph descriptor has one `estimated_duration_ns` field. Native
estimates start at zero, including descriptors supplied with a nonzero value.
Zero means indeterminate. Physical graph kind, batch capacity, token capacity,
proposal depth, mixed prefill capacity, and packed verification capacity identify
a cost entry; active row count does not change its identity.

Every selected complete speculative program enters the cost catalog with its
actual B/D/W/P. A smaller captured producer is a distinct physical program;
metadata must not relabel it as a deeper producer.

The first positive device observation sets the estimate. Further valid samples
while fresh use the median of the last nine replay observations, averaging the
middle pair when fewer than nine samples give an even count. An entry becomes stale after 30 seconds
without a valid observation. Its measured cost remains available for selection
and service-rate normalization. The first valid observation of a stale entry
replaces its estimate directly; elapsed time alone never makes a measured graph
unknown.

The raw device observation is the maximum positive rank-local CUDA replay interval.
When the prepared graphs expose frame-owned readiness timestamps, the cost sample
subtracts exposed embedding-delivery delay after aggregating the ranks' intervals.
This covers RAM and disk delivery, including CPU mailbox coordination. Service
charges and reported throughput use the raw replay and wall durations.
Graph estimates use these GPU observations for both isolated and queued executions.
Host dispatch, completion delivery, and polling time do not enter a graph's cost.
Missing or zero device intervals neither change an estimate nor make it fresh.

Prefill exploration excludes a larger unmeasured graph when a smaller prepared
capacity consumes the same tokens up to the next message or prompt endpoint.
Measured graphs remain eligible according to their observed costs.

An indeterminate candidate wins over a determinate candidate, then competes by
expected useful value. Equal values prefer shorter known time, more served
requests, less recently used requests, and smaller physical padding, in that
order. Unknown graphs require an empty
submission queue and completion feedback. A lone cold request consequently tries
its largest useful prefill or speculation extent without a solo-request rule.

Stale physical dispatches also require an empty queue and completion feedback,
while retaining their measured costs for ranking. Commit rechecks freshness
because a saved plan can outlive the estimate's freshness interval.

## Funded timing refresh

Ordinary valid completions earn `0.002 / 0.998` times their measured execution
duration, capped at 5 ms. For queued executions, credited time is the maximum
positive device interval; for isolated executions it is the larger of that
interval and wall time. Initial discovery earns no credit. Idle discards
positive credit and preserves debt.

Full credit permits one stale alternative to displace the ordinary measured
winner. Feasible alternatives are ordered by the later of their last valid
observation and last attempted dispatch, then by ordinary useful-value ranking.
Attempts advance only on committed execution. Invalid timing does not make
the estimate fresh, and its isolated wall time pays for the attempt.
Ordinary winners update their costs through normal service without spending
refresh credit.

A funded speculative refresh executes one complete captured program. It pays
that program's actual duration once, including proposal and verification work.
Cancellation suppresses publication without inventing a timing observation.
Another episode requires full credit after the preceding episode closes.

For credited ordinary duration `O`, refresh duration `P`, and the actual duration
`D` of the current or last refresh episode:

```text
P <= 0.002 / 0.998 * O + D.
```

The allowance bounds aggregate expense with one nonpreemptible episode overrun;
it does not cap individual decode stalls. Funding the first refresh requires
2,495 ms of credited ordinary execution. Repeated shorter busy periods separated
by idle can leave unused graph estimates stale indefinitely.

## Funded acceptance learning

Ordinary completions also earn a separate `0.01 / 0.99` share of raw execution
time, capped at 5 ms. Full credit permits an acceptance probe that offers a
request at least one depth beyond the ordinary choice, can beat the ordinary
choice under its optimistic yield estimate, and leaves future output in which
to use the observation. Candidates rank by optimistic score improvement per
physical execution time. Timing refresh takes precedence; only one probe
episode can be active. Equal gains prefer the least-informed added prefix,
then the shorter execution and smaller physical geometry.
Fixed-width execution does not probe acceptance.

An acceptance episode pays one complete graph interval, including readiness
stalls. Forecast observations stop at each lane's frozen `min(D + 1, W)`
ceiling. Timing and acceptance ledgers never charge the same execution. An
already-started episode can incur debt; another episode waits for full credit.
Idle clears positive credit and preserves debt. For ordinary duration `O`,
acceptance-probe duration `P`, and one episode's actual duration `D`, the bound is
`P <= 0.01 / 0.99 * O + D`. Initial cost discovery is accounted separately.
Cost-model estimates are median replay durations corrected for measured
readiness stalls; exploration budgets use raw physical time.
`flare_scheduler_execution_nanoseconds_total` and
`flare_scheduler_executions_total` expose ordinary, discovery, timing-refresh,
and acceptance-learning execution separately.

## Service accounting and publication

`commit` records resident runnable handles and generations, with token and
delivery normalizers from the applicable measured catalog. Requests admitted
after dispatch and requests waiting for external restoration are outside that
interval's participant set. A cancelled handle cannot charge a later request
reusing its slot.

`complete` validates every lane before updating graph costs, acceptance, service
prices, or committed state. Useful service comes from differences against
committed counters, never counters saved before earlier queued work completed.
Each successful frame with a positive device interval contributes one measured
interval to service history: the maximum rank-local device interval for a
queued frame, and the larger of that interval and the host wall interval from
dispatch through observed completion for an isolated frame. Unselected
participants receive zero service over that interval.
Accounting changes do not invalidate unrelated immutable plans.

Completion initially records zero deliveries. The first successful visible
publication for the same fence replaces that interval's delivery update with one
served delivery; subsequent tokens in the burst do not change the count. Stop
buffering and output parsing can postpone visible output. A publication must be
reported before processing another completion, so the correction never rewinds
through later service observations. Output age uses the actual host publication
time independently of device service accounting.

A first published burst initializes a missing delivery history to one delivery
over its measured frame duration. Token and delivery prices are stored in
physical service units. Phase transitions, acceptance updates, and timing
normalizer changes preserve those prices until actual service changes them.

## Planning and the execution queue

```rust
fn advance_to(&mut self, now_ns: u64) -> Result<()>;
fn plan(&self) -> Option<ScheduledPlan>;
fn commit(&mut self, plan: &ScheduledPlan, fence: Fence) -> Result<()>;
fn complete(&mut self, fence: Fence, records: &[CompletionRecord],
            timing: ExecutionTiming) -> Result<Vec<GeneratedToken>>;
fn published_output(&mut self, handle: RequestHandle, fence: Fence) -> Result<()>;
```

A speculative program with D > 0 whose target rows the device allocates must
complete through `complete_packed_round`, which also takes its confidence
sidecar. The `complete_with_survival` and `complete_packed_round_with_survival`
variants add the `completion_speculative_survival` rows.

A plan freezes physical shape, logical lane work, request epochs, estimated
execution duration, and selection score. Commit checks ownership, epochs,
resource availability, and feedback dependencies without rerunning selection.
Cancellation stops further selection immediately; pages and state slots remain
owned through their last submitted fence.

The transport has two frame slots. Only ordinary decode and complete speculative
generation may occupy both. Prefill, scored prefill, mixed execution, resume,
vision, and state operations drain the window. Unknown or stale costs, grammar,
host stops, host-fed pending tokens or committed cursors, history inputs, and
phase transitions also require completion feedback. Dependency checks apply to the actual selected
program. A finishing prefill exposes its decoder and checkpoint operation only
after completion.

Checkpoint retention uses the exact token endpoints supplied by the native
frontend. The last eligible completed message is a best-effort conversation
anchor; the prompt end substitutes when there is no eligible message. Earlier
messages are considered in reverse order, followed by a distinct prompt end.
Additional prefill splits and snapshot copies must fit a request-local allowance
of `scheduler_checkpoint_extra_work_fraction` times estimated uncached prefill
work, defaulting to 0.05. Unknown costs disable extras while preserving anchor
eligibility. Productive completed decode earns the same fraction of measured
time for a distinct terminal capture. There is no fixed capture count or
concurrency-based snapshot share.

An extra prefill cut is selected only with an obtainable snapshot destination.
If its cost expires after dispatch, its copy can be skipped after the split has
already executed. A row guarded for the later anchor protects it from earlier
optional captures; mandatory execution
may revoke an unused guard. Armed destinations and in-flight copies retain their
rows through GPU completion and cancellation. The alias-aware global allocator
uses demand recency, checkpoint role, and endpoint length to compare eligible
victims. See [complete-state checkpoint ownership](core-design.md#exact-endpoint-capture)
for publication, preemption accounting, and retention rules.

Feedback-dependent candidates remain visible while ordinary work is queued.
Selection uses projected prices and ages at the queue's expected end. When the
final winner requires an empty window, the scheduler submits nothing until
queued work drains, then replans. Lower-value queueable work cannot displace that
winner merely because it is immediately submit-ready. This gate also applies to
discovery and probe selections.

There is no millisecond cap on either a frame or the sum of queued frames.
Expected finish times provide the planning start time of an appended frame. An
overrun suspends further queueing until completion supplies feedback; it does
not make that graph ineligible in later decisions.

At submission, the scheduler freezes each queued generation program's expected
token and delivery service. Selection projects that service into copies of
request prices at the queue's expected end. A request with a queued expected
delivery has projected age measured from that expected delivery, rather than
from its preceding observed publication. Observed service history changes only
from validated completions, and visible output age resets only when output is
actually published. Projection never publishes tokens or commits sampled state.

A queued speculative lane can advance by at most `min(lane ceiling, P)` tokens.
Output limits and continuation positions account for that bound. Page reservations
cover the full execution ceiling and declared draft storage, and stay
pinned through the lane's last submitted fence.

## Policy preferences

`PolicyConfig` can be supplied before admission through `configure_policy`.
Defaults specify these preferences:

| Field | Default | Meaning |
|---|---:|---|
| `output_token_value` | 64 | Prompt-value units per output token |
| `token_fairness` | 16 | Token-rate log-utility coefficient, value/ms |
| `delivery_fairness` | 4 | Delivery-rate log-utility coefficient, value/ms |
| `decode_age_price` | 0.1 | Linear output-age cost coefficient, value/ms² |
| `terminal_value` | 256 | Value per terminal retirement, in prompt-token units |
| `terminal_age_price` | 1 | Linear terminal-wait cost coefficient, value/ms² |
| `adaptation_ms` | 100 | Service-price adaptation time |
| `estimate_ttl_ns` | 30,000,000,000 | Timing freshness interval |
| `adaptive_verification_alpha` | 1 | Exponent on prefix-survival estimates in speculative selection scores |

These weights express a throughput/cadence tradeoff. They are not graph timing
measurements or latency guarantees. Setting fairness or age coefficients to zero
removes the corresponding preference; intrinsic throughput alone does not ensure
starvation freedom. `flare run` defaults to `--scheduler-policy balanced`,
which uses these coefficients. `--scheduler-policy throughput` retains token
fairness and output-token value while setting delivery fairness and decode-age
price to zero. It favors token throughput and can select longer output bursts.
It changes scheduling preferences without changing captured capacities or
sampling policy. `--scheduler-terminal-value` and `--scheduler-terminal-age-price`
override the terminal coefficients after preset selection. Values must be finite
and nonnegative; zero disables the corresponding preference. Disabling both
removes terminal service value under throughput policy. These coefficients are
preferences, not latency bounds. `--mtp-adaptive-verification-alpha` sets
`adaptive_verification_alpha`; it must be finite and greater than zero, and
[the CLI design](cli-design.md) describes its effect.

## Mixed execution contract

A mixed graph declares `batch_capacity` total physical request rows,
`token_capacity` decode verification positions per row, and `prefill_capacity`
prompt tokens. The batch capacity is a power of two from 2 through 64; decode
width is 1 through the core's completion token capacity, which is eight when
the core declares none; prompt capacity is positive and aligned to 64. Other graph
families set `prefill_capacity` to zero. Core selection retains the explicit
`speculation_enabled` variant matching the loaded decode mode, including width
one with draft-state maintenance enabled. Fixed-width limits apply to mixed and
separate speculative graphs alike.

At most one active row performs prefill. `prefill_row`, an `int32[1]` input,
identifies its index in the dense active lane list; -1 means no prompt row.
`ScheduledStep::mixed` carries that role and the selected decode mode. Every row
has a logical query length. Packed token offsets are prefix sums of those
lengths, independent of physical slot IDs. Prompt tokens, positions, and image
metadata are uploaded at the prompt row's packed offset; decoder pending tokens
and proposals follow the model's captured input contract. Inactive rows and
unused token capacity must perform no state mutation. The maximum packed extent
is `prefill_capacity + (batch_capacity - 1) * token_capacity`; prompt capacity
is at least decode width, so a decode-only population also fits.

For each shape, selection compares the best decode-only cohort with the best
prompt row plus the highest-value decoders fitting the remaining row capacity.
A prompt-only population is legal. Every comparison uses the same useful value
per full graph duration; padding earns no value. Duration estimates are keyed
by graph kind, physical batch, decode width, and prompt capacity. Mixed shapes
also participate in service-rate references and prerequisite lookahead. For a
prerequisite followed by mixed inference, the prompt can be absent, already
ready, or enabled by the prerequisite. Each case leaves an exact two-stage
capacity problem for the decode and terminal rows.

Completion validates and updates each row by its phase. A prompt row commits
its full scheduled prompt extent and can publish one first output token; a
speculative row commits the accepted prefix. Prompt checkpoints, acceptance
learning, publication prices, cancellation generations, and fenced page
retirement follow the same contracts as in separate graphs. Mixed submissions require completion
feedback before another graph is queued. State-row reservation covers the
maximum verification width across separate and mixed catalogs.

A captured mixed forward must share the packed target execution across phases
and include proposals, accepted-state commit, output projection, collectives,
and sampling. Width one uses decode attention and direct updates while retaining
the draft context required by its selected mode. Recurrent-state rollback and
prompt absorption belong in the prepared model implementation. Helios executes
only supplied captures; it does not synthesize a mixed graph from serial prefills
and decodes.

Qwen shares target projections, hyperconnections, expert execution, and the
vocabulary projection across packed prompt and verification tokens. QSA consumes
per-row phase metadata. GDN and PLE gather their projected operands into fixed
prompt and decode workspaces for phase-specific convolution and recurrent-state
updates, then return the results to the packed target layout. Prompt consumption,
accepted verification prefixes, and draft-context absorption retain separate
state updates. Completion publication combines their disjoint physical slots.
Prompt image features use the prompt's packed offset. Workspaces are shared
across serial graph variants and allocated before capture.

Qwen allocates sampling, target-distribution, proposal, verification, and MTP
feedback buffers at the largest prepared capacity. Each variant binds contiguous
views of those typed buffers before compilation. Sequential proposal positions
reuse one workspace after copying their probabilities into the round's draft
probability buffer. Target probabilities, saved draft probabilities, verification
operands, and prompt residuals remain disjoint while simultaneously live. MTP
feedback rows and raw-cache backups are shared across serialized calls. Persistent
request state and overlapping b12x operations retain separate storage.

`ignition/components/capacity.py::mixed_shapes` bounds the joint catalog before
tuning and compilation when `--mixed-prefill` is enabled. Each batch bucket of at
least two rows has every prompt bucket at verification width one and one prompt
bucket for each wider width within the target-token envelope.
The latter bucket minimizes `max(P / P_min, P_max / P)`, balancing the worst
multiplicative distance to short and long prompt chunks; ties choose the smaller
bucket. The prompt bucket limits the prompt/depth combinations; the target-token
envelope limits the batch/depth combinations. Graph durations measured by Helios determine which
captured alternatives execute.

For each mixed batch bucket, speculative capture adds every prompt bucket at
width one and one graph per eligible wider width. Ordinary execution adds every
prompt bucket at width one without draft-state updates. Qwen concurrency four,
4096-token chunks, five draft tokens, and the default token capacity six add 22
speculative and 20 ordinary mixed graphs. Adaptive
speculative width one preserves draft state; loading with zero drafts selects
the ordinary catalog. Planning reports the same catalog used by preparation and
includes its maximum packed token extent in capture headroom.

## Verification

`helios/src/scheduler/policy_tests.rs` exercises the production selector with
measured synthetic catalogs: one, two, and sixteen long prefills; fitting and
padded tails; high- and low-acceptance speculation; heterogeneous cohorts;
padded batch capacities; long decode pairs; prefill/decode contention; graph
expiry and recovery; and output gaps. Mixed tests compare every legal cohort
against exhaustive useful-work/time enumeration, exercise fitting prompt tails,
ordinary and speculative decode rows, malformed completions, independent shape
expiry, and concurrent prompt/decode workloads. Native input tests check packed
prompt offsets, one query-length upload, row sampling masks, and role validation.
Scheduler lifecycle tests separately
exercise prescribed cache and resource transitions without assuming those
operations always outrank another runnable request. Run the scheduler and
runtime phase tests without a GPU:

```sh
cargo test --manifest-path helios/Cargo.toml --features server --lib scheduler::
cargo test --manifest-path helios/Cargo.toml --features server --lib \
  server::runtime::phase_tests::
```

Five- and sixteen-request startup traces cover zero, three, and seven accepted
drafts, with both untrained and preseeded acceptance histories. On the measured
synthetic catalog, each 64-token prompt must deliver its first output within
100 ms, subsequent gaps stay below 150 ms, and another hundred rounds give every
stream progress with a spread of at most one eight-token burst. A separate test
varies snapshot cost to check that enabling a shared decode batch wins only
when the complete sequence has sufficient value per time.

The runtime phase tests cover actual worker completion, publication, stop
buffering, and input packing. `tests/server/test_scheduler.py` exercises an idle
native HTTP service with four request slots and a 4096-token prefill capacity.
It checks complete responses for concurrent long prompts, two and four decode
streams, a long prompt arriving during decode, short/long prompt contention,
and cancellation followed by slot reuse. Run it for ordinary, fixed-width, and
adaptive speculative execution:

```sh
FLARE_RUN_CUDA_TESTS=1 FLARE_SCHEDULER_TEST_URL=http://127.0.0.1:18131 \
  .venv/bin/python -m pytest -q -s tests/server/test_scheduler.py
```

The HTTP tests print actual prompt/output token counts, client SSE arrival
gaps, and server graph-counter deltas. SSE arrival gaps are not GPU execution
times or per-token latencies. Mixed capture correctness requires cores that
contain mixed model forwards. Synthetic scheduler traces do not establish GPU
throughput.
