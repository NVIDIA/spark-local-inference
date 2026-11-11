# Request scheduling by useful work and service value

## Execution model

An interactive inference scheduler chooses a prepared graph and the requests
that occupy its lanes. A graph is a nonpreemptible service quantum. Its prefill
capacity, decode row capacity, verification width, and duration are properties
of the prepared execution. The scheduler chooses among these quanta; it cannot
choose an arbitrary time slice independently of the work.

The fairness entity is one request. Requests have known remaining prompt work,
observable generated output, an explicit output limit, and unknown EOS.
Speculative acceptance is stochastic. A tensor-parallel GPU group is one
execution resource when its ranks execute a model step together.

The Python implementation in [`scripts/scheduling_sim`](../scripts/scheduling_sim/)
contains a fixed-value dispatch baseline, a discrete-event simulator, and small
mathematical references. The [ongoing-service experiment](scheduling-utility.md)
adds proportional-fair rate histories and a decode-age continuation value.
The [completion-episode oracle](scheduling-oracle.md)
also compares exact conditional schedules with a continuation-value prototype.
The fixed-cost models use positive, known graph durations and supplied
conditional acceptance probabilities. Actual output lengths are hidden from
the dispatcher. These models have one graph in flight and publish output at
completion. They exclude cache capacity, admission, restore, host prerequisites,
and submission pipelining. The fixed-cost experiments exclude timing discovery and
runtime estimation error; the [timing-learning experiment](#learning-graph-timings)
models both with hidden execution costs and completion observations. The graph
timing table is synthetic unless supplied with `--catalog`.

The [native scheduler design](scheduling-runtime-design.md) applies the ongoing
service utility to Rust execution, including prerequisite operations, timing
discovery, and queued submissions. Python simulation results do not establish
native behavior or GPU performance.

Three kinds of material appear below. The fixed-value score, the
fair-allocation reference, the completion-episode oracle, and every
timing-learning estimator and exploration policy are implemented only in the
Python simulator. The sections on CPU scheduling, stream age, remaining work,
and the dynamic objective are analysis. A function they name is a reference
calculation in `analysis.py`, not a dispatch policy. Helios implements
the ongoing-service utility and decode-age continuation of
[`scheduling-utility.md`](scheduling-utility.md) and, from the timing-learning
comparison, only retained graph costs with funded remeasurement of stale
graphs.

## Connection to CPU scheduling

CPU fair scheduling distinguishes a request's entitlement from the granularity
at which it executes. CFS models an ideal processor shared among runnable tasks
and accounts for actual service with virtual runtime. EEVDF uses service lag
to determine eligibility and virtual deadlines to order eligible tasks. These
are useful foundations for accounting, rather than reasons to assign a real-time
deadline to each prefill chunk. See the Linux documentation for
[CFS](https://docs.kernel.org/scheduler/sched-design-CFS.html) and
[EEVDF](https://docs.kernel.org/scheduler/sched-eevdf.html).

For inference, the analogy has three qualifications:

- A decode stream remains runnable after producing output. Unlike an interactive
  CPU task blocked on input, the stream need not sleep between useful responses.
- Service cost is nonadditive. If two singleton decodes cost 15 ms each and their
  batch costs 17 ms, dividing those 17 ms between requests is an accounting
  convention, not a measured physical quantity.
- The quantum changes efficiency and yield. A wider speculative graph can take
  longer and produce disproportionately more output. A larger prefill can
  amortize graph overhead and use the hardware more efficiently.

A fair-service reference must therefore include the feasible batch rate vectors.
After defining that reference, a runtime policy can track deviation from it.
Copying CPU virtual-runtime arithmetic before defining service units leaves the
central allocation question unanswered.

Prefill has no externally visible per-chunk cadence. Its visible milestones are
first output and eventual completion. Decode has both useful token throughput
and a delivery cadence. Equal mean rates do not imply equal output gaps.

## One metric for graph selection

Let an action `a` take expected time `T(a) > 0`. For request `i`, let `P_i(a)`
be the actual prompt tokens consumed and `D_i(a)` the expected output tokens
emitted. A fixed-value baseline chooses the largest

```text
score(a) = sum_i w_i * (P_i(a) + beta * D_i(a)) / T(a).
```

`w_i` is a request weight, equal to one in the supplied workloads. One prefill
token has value one; `beta > 0` specifies the intrinsic value of an output token.
The CLI requires this value explicitly. The simulator does not infer a user
preference from hardware cost or select a production default.

All pure prefill, pure decode, and available mixed graphs compete on this score.
No duration ceiling, prefill size preference, or breadth-before-depth rule
precedes it. A graph consuming 64 useful prompt tokens earns 64 prefill units
whether its capacity is 64 or 4096. Inactive rows and rejected drafts earn zero.
Completing prefill emits one first token, counted once in `D_i`.

If conditional acceptance at draft position `j` is `q_ij`, a verification width
`k` has expected output, before EOS truncation,

```text
D_i(k) = 1 + sum_{j=1}^{k-1} product_{m=1}^{j} q_im.
```

The pending target token provides the first output; a draft contributes only
when the accepted prefix reaches it. The explicit remaining output budget caps
this expectation. A length-distribution model could additionally price hidden
EOS, but the fixed-value baseline does not have one.

For a fixed graph duration, fixed weights, and additive lane rewards, its best
decode cohort consists of the highest-valued rows up to capacity. Consequently,
ranking per-request contributions is exact for this baseline. A nonadditive
reward, context-dependent cohort timing, or an objective involving excluded
requests can invalidate that reduction.

This score gives the desired elementary behavior under appropriate timings:

- Fully occupied 4096-token prefills win when they have the highest useful
  prefill rate. Two long prefills do not need smaller chunks merely to alternate.
- For a 64-token tail, all sufficiently large graphs earn the same prefill work
  and first-token benefit. The fastest eligible graph wins.
- A wide speculative round wins when its additional expected accepted output
  outweighs its additional time. A 30 ms round is not excluded merely because
  a 17.3 ms alternative exists.
- A mixed graph with a smaller prefill can win on combined useful work per time.
  There is no universal crossover at 64, 256, 512, or 1024 tokens.

The invariants concern useful work and measured efficiency. They do not require
4096 tokens when a smaller graph has demonstrably better effective efficiency.
Likewise, a padded larger graph remains eligible if it is actually faster.

## What immediate value does not establish

Maximizing one action's expected value per time is not a proof of globally
optimal completion time, fairness, or interactivity.

Three executable counterexamples are in
[`tests/test_scheduling_sim.py`](../tests/test_scheduling_sim.py):

1. With prefill capacities 64, 128, and 256 costing 2, 3, and 4 ms, a 129-token
   prompt and `beta=1` select 128 tokens first. The remaining token needs another
   graph, taking 5 ms total. The padded 256 graph would finish in 4 ms.
2. A continuously backlogged decode with sufficiently high constant token value
   prevents a prefill from receiving any service over the tested horizon.
3. Five identical low-acceptance streams using batch four receive
   `[10, 10, 10, 5, 5]` tokens over ten rounds under oldest-service tie-breaking.
   Rotating which request was served most recently does not track accumulated
   entitlement.

A fixed set of completed requests earns a fixed total token reward. Maximizing
that reward divided by the time to finish all requests minimizes makespan;
request completion order and output cadence remain unspecified. Mean time to
first token and mean completion time are separate objectives.

For example, alternating full chunks between two equal long prefills and
finishing one before starting the other can have the same total prefill time.
Finishing one first improves mean time to first token if subsequent decode does
not alter the schedule. Alternation gives more even prompt progress. Choosing
between them requires a completion or fairness objective, not a different
hardware-throughput formula.

## A fair-allocation reference

For a frozen active set and unchanged execution conditions, express action `a`
as a vector of useful per-request rates:

```text
y_i(a) = E[useful tokens for request i in a] / T(a).
x_i = sum_a theta_a * y_i(a),   theta_a >= 0,   sum_a theta_a = 1.
```

`theta_a` is a fraction of wall time, not a fraction of graph launches. These
mixtures form the feasible rate region. The reference assumes indefinitely
backlogged requests with a fixed phase; finite prompt tails and phase changes
require a dynamic model.

A concrete family of allocation objectives is

```text
maximize F(x) = sum_i [v_i * x_i + f_i * log(x_i / x_ref_i)].
```

Here `v_i >= 0` is intrinsic token value, `f_i > 0` is a fairness coefficient,
and `x_ref_i > 0` fixes the units of the logarithm. The values `v_i` and `f_i`
are supplied explicitly to the reference. With `v_i=0`, this is weighted proportional
fairness. Concave rate utility and proportional fairness have established
foundations in [Kelly, Maulloo, and Tan's rate-control formulation](https://www.statslab.cam.ac.uk/~frank/rate.pdf).

The graph selector associated with the objective's gradient has the same
work-over-time structure:

```text
q_i = dF/dx_i = v_i + f_i / x_i
best action = argmax_a sum_i q_i * E[useful tokens_i(a)] / T(a).
```

This is a supporting-direction calculation for the allocation problem. It is
not yet an online algorithm: deciding how to estimate `x_i`, initialize arriving
requests, or retain service history changes transient behavior.

The reference has several useful properties:

- On a CPU-like resource that runs one request at a time at rate `r_i`, pure
  proportional fairness gives time share `theta_i=f_i/sum_j f_j`, independent
  of `r_i`. It recovers weighted processor sharing without equating token rates.
- Batch economies appear directly in `y(a)`. They require no invented division
  of a graph's physical time between its rows.
- Positive fairness coefficients keep every feasible request's reference rate
  positive. They establish no finite output-gap bound.
- The objective is increasing in every request's rate. A feasible allocation
  that is worse for every request cannot be optimal. This excludes wasted
  efficiency without requiring the largest physical graph on every dispatch.
- The intrinsic value term and fairness term have distinct meanings. Replacing
  `log(x_i)` with `log(beta_i*x_i)` adds a constant and cannot make decode more
  valuable. A unit conversion inside the logarithm is not a phase preference.

[`analysis.py`](../scripts/scheduling_sim/analysis.py) implements `fair_allocation`
with pairwise conditional gradients and scalar line search. Its `token_values`
argument is `v`, its `fairness_weights` argument is `f`, and it fixes
`x_ref_i = 1`, which changes the utility only by a constant. It returns time
shares, rates, utility, and a concavity gap that bounds utility suboptimality.
It fails explicitly if it cannot meet its requested numerical tolerance. This
is an offline reference over caller-supplied action vectors, not the dispatcher.

For five equal streams, suppose a four-row graph takes 21 ms and a graph holding
all five takes 29 ms. Uniformly rotating the five possible four-row cohorts
achieves `4/(5*21)` tokens/ms per request, versus `1/29` for the larger graph:
38.10 versus 34.48 tokens/s each. The reference selects the efficient smaller
batch and equal shares. It does not require sacrificing either fairness or
mean rate in this case. A delivery-gap objective could still distinguish these
allocations.

A runtime lag can eventually be defined as reference progress minus delivered
progress, integrated over the intervals when a request is runnable. Defining
how that reference changes on arrivals, departures, phase transitions, and
cost revisions is necessary before turning it into a dispatch weight.

## Stream age and nonpreemptible execution

An intrinsic token value rewards useful volume. Ending an output stall is an
additional benefit associated with a delivery event. Producing eight tokens at
one instant does not end eight distinct waiting intervals.

Let `A_i(t)` be time since the last published output for a decoding request.
A family of explicit interactivity costs is

```text
integral sum_{i in decode} c_i(A_i(t)) dt,
```

where `c_i` is increasing. A linear age cost accumulates during a graph as

```text
integral_0^T (A_i + u) du = A_i*T + T*T/2.
```

This applies to selected and unselected streams: both wait until graph
completion. At completion, the age of a served stream resets once, regardless
of burst size. Requests that arrive during execution also accumulate waiting
from their arrival. Prefill requires a first-token waiting cost if such a cost
is part of the objective; its internal chunk completions do not reset a stream
age clock.

`analysis.py::output_age_area` evaluates this identity. The fixed-value
simulator records
output gaps and integrated linear decode age without optimizing this cost.
The completion-episode oracle optimizes explicit time and waiting prices.
An increasing token weight based solely on time since last service is a
possible approximation, not a derivation of
this objective. In particular, multiplying the entire speculative burst by a
stall-clearing bonus can overvalue deep speculation for a neglected stream.

Average rate, accumulated service lag, and output age are separate state
variables. A burst can repay useful-progress lag while still leaving a poor
history of pauses. An age-only rule can repeatedly favor an inefficient stream
without a specified throughput tradeoff. Both effects belong in an explicit
objective before selecting a formula.

## Remaining work and completion value

The prefill token count is known, but its minimum remaining execution time
need not equal remaining tokens divided by one rate. The reference
`prefill_completion_ms` solves the isolated constant-cost case exactly:

```text
H(0) = 0
H(p) = min_{prefill graph g} [T(g) + H(max(0, p - capacity(g)))].
```

It ignores decode competition and assumes prefill timing is independent of
context. Its purpose is to expose greedy chunking errors, not to price an
entire mixed workload as isolated work.

For generated output `g` and a length prior `L`, expected remaining length is

```text
E[L - g | L > g].
```

`conditional_remaining` evaluates this expression for an explicit discrete
prior. A known fixed length gives decreasing remaining work. A geometric prior
has a memoryless residual. A heavy-tailed prior can predict more remaining
work after observing a longer prefix. For a continuous Pareto survival law
with exponent `alpha > 1`, above its minimum length,

```text
E[L - g | L > g] = g / (alpha - 1).
```

Thus an estimate of twice the attained output corresponds to one particular
Pareto exponent, `alpha=1.5`. It is not a distribution-free consequence of
observing that a request is still decoding. Explicit output limits truncate
any applicable prior. Learning a prior must also handle unfinished requests
as censored observations rather than completed short outputs.

Remaining length matters when the objective rewards completion or the future
capacity released by completion. It is not needed to evaluate the immediate
acceptance yield of a graph. Accurate graph timing does not remove this
uncertainty about the request's total duration.

## Dynamic objective and lookahead

Variable-duration decisions with stochastic next states fit a semi-Markov
control model. State includes remaining prompt work, attained output and any
length belief, acceptance information, service accounting, and stream ages.
Actions are complete graph executions. Expected transition reward can include
useful work and integrated waiting costs. This modeling framework and its
average-reward equations are described by
[Baykal-Gursoy](https://gursoy.rutgers.edu/papers/eorms0757-final11.pdf).

Under the conditions for an average-reward optimality equation, a continuing
workload has a relative value `h` and optimal reward rate `rho` satisfying

```text
h(s) = max_a { r(s,a) - rho*T(s,a) + E[h(s')] }.
```

Equivalently, optimal actions maximize

```text
[r(s,a) + E[h(s') - h(s)]] / T(s,a).
```

Dropping the future-state term produces the immediate value baseline. The
remaining term accounts for such effects as unlocking decode after a prompt
tail, ending a stall, completing a request, and changing future batch options.
A practical approximation needs tests against small solvable instances of a
chosen objective. The equation alone supplies neither its utility parameters
nor a latency guarantee. Average reward on an infinite horizon is also
uninformative for a finite batch followed by permanent idleness; finite-horizon
or completion-cost objectives apply to that experiment.

The [completion-episode oracle](scheduling-oracle.md) supplies those small
instances. It evaluates hidden-length beliefs, all legal cohorts, and acceptance
outcomes through completion. Its policy-improvement prototype keeps expected
net work plus continuation-value change over graph time as the dispatch score.
Normalizing negative net episode rewards by episode duration can reward wasted
execution time; that objective has an explicit counterexample. A fixed positive
time price supplies a separate reference for completion and waiting tradeoffs.

For a stationary sequence of actions with fixed rewards, a cycle's value rate
is a duration-weighted average of its action rates:

```text
(sum_j reward_j) / (sum_j time_j)
    = sum_j [time_j / sum_k time_k] * [reward_j / time_j].
```

It cannot exceed its best component. Looking ahead is useful because state,
future value, or fairness changes, not because averaging unchanged rates fixes
a starvation-prone linear objective.

## Running the experiments

From the repository root:

```sh
python -m pytest tests/test_scheduling_sim.py -q
python -m scripts.scheduling_sim --decode-values 1,16,64,256 \
  --output /tmp/flare-value-separate
python -m scripts.scheduling_sim --decode-values 1,16,64,256 --mixed \
  --output /tmp/flare-value-mixed
python -m scripts.scheduling_sim --workload prefill_five_decodes \
  --decode-values 64 --catalog /path/to/graph-times.json
```

A catalog is a JSON array. Durations are milliseconds for complete graph
execution. Unused capacities remain part of its measured cost:

```json
[
  {"name": "prefill-64", "duration_ms": 3, "prefill_capacity": 64},
  {"name": "prefill-4096", "duration_ms": 66, "prefill_capacity": 4096},
  {"name": "decode-b1-w8", "duration_ms": 30, "decode_capacity": 1, "width": 8}
]
```

The output directory contains aggregate measurements in `summary.csv`, individual
request measurements in `requests.csv`, selected actions and their scores in
`trace.jsonl`, and the timing input in `catalog.json`. The decode value and
random seed accompany each case. A graph started before the requested horizon
finishes in full; elapsed time can exceed that horizon by one graph.

The sixteen workloads cover solo and concurrent prefills, prompt tails,
low/high/heterogeneous acceptance, up to sixteen requests, mixed phases, delayed
arrivals, and different output lengths. Tests include exhaustive cohort scoring,
conditional acceptance sampling, padding, explicit output limits, uniform time
scaling, starvation counterexamples, exact prefill completion, and analytic
fair-allocation cases. Metrics include incomplete requests, integrated linear
decode age in milliseconds squared, and open output stalls at the measurement
boundary. A request already decoding at arrival has zero initial age; its wait
until the first simulated delivery counts. Prefill contributes no decode-age
cost until its first output. Busy time means execution-resource use,
not GPU occupancy or hardware efficiency.

## Learning graph timings

`python -m scripts.scheduling_sim.learning` compares timing estimators and
exploration policies over the same useful-work and service-price objective.
It is a separate entry point from the fixed-cost CLI,
`python -m scripts.scheduling_sim`. The timing simulator has one graph
in flight with synthetic execution costs. It does not model native prerequisites,
complete speculative B/D/W/P programs, queued submissions, cache storage, or
tensor-parallel synchronization. Its results do not establish GPU performance
or explain a particular service trace.

Helios implements one rule from this comparison: a measured graph cost stays
available after it becomes stale, and ordinary execution funds remeasurement
of stale graphs (`helios/src/scheduler/cost.rs` and `refresh.rs`, specified in
the [native timing policy](scheduling-runtime-design.md#funded-timing-refresh)).
The `retained-refresh` policy below models that rule. The posterior estimators,
change detectors, sampling and optimistic policies, targeted challengers, and
service-delay accounting described in this section exist only in the simulator.

The executor owns physical costs, correlated timing noise, regime changes, and
request outcomes. A learning policy receives duration-free graph geometry,
observable request views, and its own completed timing observations. The known-mean
reference alone receives conditional mean costs at dispatch, including the
current regime and previously observed noise state; it never receives the next
random execution outcome. It optimizes the next dispatch, not future exploration:
a known first-use penalty can keep it from trying a graph whose later executions
would be faster. Timing, acceptance, and policy sampling use separate
random streams. Acceptance draws are keyed by request, generated position, and
draft position; timing draws are keyed by physical graph and execution count.
Different schedules need not encounter identical stochastic trajectories.

The policy first prepares service prices and rate normalizers from measured
means. Unknown costs supply no normalizer. Those prices and delivery references
remain fixed while it evaluates candidate graphs and cohorts. A posterior
sample is shared by every candidate using the same physical graph. A nomination
of the mean incumbent's graph executes the incumbent's full cohort and counts
as ordinary service. Empty graph rows and rejected drafts earn no useful work.

### Retained estimates

Each physical graph keeps lifetime count, arithmetic mean, and variance, with
separate GPU and host-handoff component statistics. Eight observations form one
block by default; blocks of 32 are an experimental alternative. Partial blocks
immediately refine the ordinary dispatch mean. Complete block means refine an
uncertainty model for the mean cost. Idle time changes neither count nor
uncertainty, and never converts a measured cost into an unknown cost.

For block means `y`, the normal-inverse-gamma working model has prior
`kappa0 = 0.25`, `alpha0 = 2`, and `beta0 = m0²`. The frozen prior location `m0`
is the median measured graph mean in the same prefill/decode/mixed family, then
the incumbent mean if available, then 1 ms. This fallback is a statistical
prior, not a measured cost or a fairness normalizer. Ordinary dispatch uses the
raw observed arithmetic mean by default. The `mean_estimator = "ema"` alternative
uses a retained exponential mean with sample weight 1/8 for ordinary decisions.
It preserves lifetime and active arithmetic statistics, posterior updates, and
change detection. Its operational mean survives idle and regime declarations.
Posterior intervals describe the active arithmetic regime mean; they are not
confidence intervals around the exponential mean.

The `partial_blocks` alternative incorporates an unfinished block into the
uncertainty model. For `r` observations in a block of size `B`, its mean `z`
contributes weight `w = r/B`. Given the complete-block posterior parameters
`kappa`, `alpha`, `beta`, and `m`, the weighted posterior is:

```text
kappa' = kappa + w
m'     = (kappa*m + w*z) / (kappa + w)
alpha' = alpha + w/2
beta'  = beta + kappa*w*(z - m)^2 / (2*(kappa + w)).
```

Each observation replaces the partial contribution. Completing the block
promotes that contribution exactly once; overlapping partial means never count
as independent observations. This is a tempered likelihood with fractional
block weight. Partial blocks do not change complete-block detector references,
operational means, or lifetime history. Partial-sample interval
coverage depends on the timing distribution and prior location. Calibration
diagnostics include both matched and mismatched priors.

Exploratory mean-cost draws use the marginal
Student-t distribution conditioned on a positive cost. Its truncated mean need
not equal its location; with these prior parameters it is `1.8 * m0`.

The cumulative estimator retains all observations in its active mean. The
regime estimator also retains lifetime statistics, but changes its active
statistics after observed counterevidence. After at least eight active blocks,
three consecutive block means must lie outside a frozen central 99% predictive
interval on the same side and differ from the frozen raw mean by at least 10%.
The reference freezes before the first suspect block. Confirmation starts the
active regime from those three blocks without erasing lifetime statistics or
carrying the previous regime's sample count into the replacement prior.

The detector uses the symmetric NIG predictive interval for another block mean
as a working test. It is not the exact predictive distribution of a fully
positive-conditioned latent model. A broad interval can miss an early downward
shift, and correlated or heavy-tailed observations can invalidate nominal
coverage. The predictive-noise control samples individual execution variation,
whose uncertainty remains wider than uncertainty about the mean.

The `log-regime` estimator keeps the same NIG mean-cost posterior and configured
operational mean, with a separate detector operating on log durations.
After 64 valid observations in the active regime, it freezes the log mean,
sample variance, raw arithmetic mean, and lag-one correlation before examining
a group. Its working correlation is
`rho = min(0.99, max(0, r) + 2/sqrt(N - 3))`. For a group of
size `n`, the log-mean difference variance is
`s² * (F(n)/n + F(N)/N)`, where
`F(n) = 1 + 2 * sum((1 - k/n) * rho^k, k=1..n-1)`.
Student-t degrees are the largest power of two not exceeding `N/F(N)`, minus
one, with a floor of one.

A raw observation outside the central 99.99% working interval with at least
a 10% arithmetic-mean change can nominate the graph for confirmation.
Two consecutive groups of four observations must meet those conditions on
the same side of the same frozen reference. Their combined eight observations
must also pass a stricter test that spends a lifetime nominal error allowance:
for reference index `j` and the next power of two `J >= j`, its two-sided
allowance is `0.0001 / (J * (J + 1))`. The reference counter survives regime
changes. Confirmation seeds active statistics from those eight observations;
lifetime statistics retain every observation. The correlation model and its
nominal error allowances do not guarantee coverage for arbitrary timing noise.

### Foreground exploration allowance

By default, bootstrap is compulsory only when no feasible productive graph has a valid
measurement. It ranks candidates by useful value, then by least recent service
and smaller physical capacity;
unknown duration is never a numeric zero denominator. Bootstrap time is recorded
separately and earns no exploration credit. The `ttl` and `jitter` policies
instead select any unknown graph first and exclude unknown padded prefill
graphs. Discretionary
exploration can still try an oversized graph: equal useful work does not prove
that the smaller captured graph is faster.

The `initial_discovery = "native"` alternative gives feasible, genuinely
unmeasured graphs the native unknown-first priority and padded-prefill filtering.
It also masks service prices lacking an applicable measured normalizer.
Executions of never-measured graphs count as bootstrap, including newly feasible
graphs and retries after invalid timings. They record separate elapsed time and
earn or spend neither exploration allowance. Measured costs remain available after idle.
This option varies initial discovery independently of the operational mean and
subsequent remeasurement policy.

Funded policies start with zero credit. A valid ordinary completion earns
`alpha / (1 - alpha)` times its elapsed duration, capped at 5 ms by default,
with `alpha = 0.02`. A probe requires the full cap of credit, then pays its
entire actual elapsed duration, allowing negative debt. It earns no credit.
Missing or zero GPU timing does not update the estimate or earn ordinary credit;
observed host elapsed time still settles an executed probe and updates service
history. Commit and completion account for each operation once; repeated
planning reuses its decision and random draws.

For ordinary credited duration `F`, discretionary probe duration `E`, and the
most recent probe duration `D`, the experiment asserts after every completion:

```text
E <= alpha / (1 - alpha) * F + D.
```

This permits one nonpreemptible overrun; the 5 ms credit cap is not a graph
duration limit or a bound on individual request latency. Idle earns no credit
and discards positive credit while retaining debt. Consequently, separate busy
bursts shorter than 245 ms cannot fund a probe at the default settings.

An opportunity requires an alternative physical graph and sufficient credit;
ungated policies omit the credit condition. The counter advances on commit
even when exploration nominates the incumbent. Except in the `targeted` and
`retained-refresh` policies,
every sixteenth opportunity selects the least recently attempted feasible
alternative. Refresh can be disabled independently, and comparisons of funding
use equal refresh settings.
No synthetic work runs while the execution resource is idle.

### Targeted challengers

The `targeted` policy ranks alternative physical graphs by optimistic useful
value using the configured lower mean-cost quantile. Ordinary dispatch uses the
configured operational means and the service-price objective.
Prices remain frozen across all candidates in one decision.

A measured challenger must have a higher optimistic score than the ordinary
incumbent. Its estimated duration may exceed the incumbent by at most
`probe_extra_ms`, default zero. Its estimated utility loss is expressed as
incumbent service time:

```text
loss_ms = max(0, incumbent_score - challenger_mean_score)
          * challenger_mean_duration / incumbent_gross_value_rate.
```

The gross value rate includes useful work, service prices, and delivery value
before the shared decode-age cost. It remains positive even when the net
utility score is negative. The default `probe_loss_ms` limit is 5 ms.
The loss calculation uses the cohort that would actually execute. A graph
without observations still needs an optimistic utility improvement, but its
statistical prior is not treated as a measured duration for admission.

Periodic remeasurement examines graphs after `refresh_busy_ms`, default 5,000 ms,
of occupied execution since their last valid observation. Idle time does not
advance this clock. Due alternatives rotate by their last actual attempt;
one valid timing completes a periodic check. Invalid timings rotate attempts
without refreshing observation age. Pending change evidence receives
confirmation priority. A missing confirmation timing moves that graph into
the shared attempt queue with due checks until it supplies a valid timing;
its pending statistical evidence remains intact. These checks can bypass the
mean-cost admission limits and execute the selected graph's best cohort under
nominal costs.

By default, due checks take priority over ordinary challengers. Setting
`audit_every = 4` reserves every fourth contended opportunity for a due check
and gives the other three to admitted challengers. Healthy pending confirmations
retain priority. A graph awaiting a valid confirmation after a missing timing
participates only in the check queue. If only one queue has a candidate, it
uses the opportunity without advancing the contention counter. The counter
advances on commit, survives idle, and is unchanged by repeated planning.
This divides opportunities, not elapsed execution time.

Every challenger, periodic check, and confirmation uses the same earned-credit
condition and pays its full actual duration. The duration and utility limits
control estimated discretionary loss; they are not guarantees about the duration
of an unmeasured graph or the latency of a request. Traces expose each candidate's
mean and optimistic scores, estimated loss, extra blocking, occupied-time age,
and pending change evidence, along with the selected reason.

For equal-work alternatives without service prices or age costs, the utility-loss
formula reduces to `max(0, challenger_mean_duration - incumbent_mean_duration)`.
A zero extra-duration limit therefore overrides a positive utility-loss allowance.
With different useful yields, a longer graph can have lower utility loss than its
extra duration. The two admission limits are independent experiment parameters;
a nonbinding `probe_extra_ms` isolates selection by utility loss.

### Request service-delay accounting

An optional `probe_harm_ms` limit compares the proposed probe with the ordinary
incumbent's cohort. For each incumbent lane, a probe supplying the same next
service incurs estimated delay `max(0, probe_duration - incumbent_duration)`.
Otherwise its delay is the probe's full duration, representing the probe
followed by incumbent service. Next service means a decode delivery or at least
the incumbent's prefill progress. Delays are summed without offsetting one
request's delay by another request's gain.

The hard limit applies to challengers, periodic checks, and confirmations.
It exempts a lone live request. Under continuous contention, a stale estimate
can exclude every attempt to discover that a graph became faster. The cohort
search considers each feasible count of incumbent decode rows, using the
highest-value rows within and outside that group. It therefore retains safe
cohorts that unconstrained useful-value selection would discard.

The `harm_alpha` alternative replaces this hard limit with separate prefill and
decode balances, each capped at one request-millisecond and initially zero.
An ordinary completion earns `harm_alpha / (1 - harm_alpha)` times its elapsed
duration for each lane that commits service, credited to that lane's phase.
Final prefill belongs to prefill even when it also produces the first token.
Committed service and elapsed time can earn this allowance without a valid GPU
timing; the GPU-time allowance still requires valid timing.

A probe requires full balances for every phase represented in the frozen
incumbent, as well as the configured GPU-time allowance. Completion computes
the service delays using actual probe duration and committed progress, compared
with the incumbent's frozen estimated duration. Each affected phase pays its
sum, allowing debt. A lone request pays its own delay; there is no exemption.
Probes and bootstrap earn no allowance. Idle clears positive balances and
retains debt. For each phase, charged delay is bounded by its earned allowance
plus the charge of the most recent probe exposing that phase. An intervening
probe serving only the other phase does not replace this overrun term.
Bootstrap remains outside this bound.

Periodic checks and confirmations choose the cohort with the least predicted
service delay for that physical graph, using ordinary utility to break ties.
Challengers retain optimistic useful-value selection. A faster probe can incur
zero delay charge while still paying its full GPU-time cost.

This accounting measures delay relative to a frozen estimated alternative.
It is not an observation of counterfactual request latency. Timing errors,
future arrivals, and later scheduling decisions can change actual latency.
The hard limit and the earned-delay alternative cannot be enabled together.

### Policies and experiments

The compared policies are known conditional means; a 30-second timing-expiry
reference with fresh EMA weight 1/8; jittered TTL uniformly between 24 and 36
seconds; retained EMA; retained arithmetic-mean greedy; uniform epsilon exploration; normalized
softmax; optimistic lower mean-cost quantiles; Thompson mean sampling;
predictive execution sampling; and targeted challengers. The exploratory
policies have funded and ungated variants and cumulative, regime, and
log-regime estimator alternatives.

The `retained-refresh` policy models the retained-cost rule that Helios
implements: unknown-first discovery, service prices, useful-value ranking, and
funded remeasurement of stale graphs. Its cost estimate is an exponential mean,
whereas Helios uses the median of its last nine replay observations
(`helios/src/scheduler/cost.rs`). A measured cost
remains available after its 30-second lifetime. Fresh observations update its
exponential mean with weight 1/8; the first valid observation after expiry
replaces that mean. Expiry alone changes neither the measured value nor its
availability.

When the foreground allowance reaches its cap, `retained-refresh` remeasures a
feasible stale physical graph other than the ordinary incumbent. It orders
alternatives by the later of their last valid completion and last attempted
dispatch, then by ordinary useful-value ranking. Invalid timings therefore
rotate attempts without making an estimate fresh. The selected graph uses its
ordinary best cohort. Its entire actual elapsed duration is charged once,
including attempts with invalid GPU timing. An ordinary winner refreshes its
cost through normal service and needs no exploration allowance.

This policy uses the cumulative estimator only for recorded statistics; no
posterior, change detector, or request-delay ledger affects its decisions.
With a 5 ms cap, allowances of 0.2% and 0.5% require respectively 2,495 ms and
995 ms of ordinary service to fund the first stale remeasurement. Idle discards
positive credit, so repeated shorter bursts can preserve stale estimates
indefinitely. The aggregate exploration bound permits one graph-duration
overrun and does not bound individual decode stalls.

Retained EMA and greedy policies bootstrap only while no productive measured
choice exists. They can leave other graphs unmeasured indefinitely. The `ttl`
policy with `ttl_ms = 1e12` isolates the effect of expiry: it has unknown-first
discovery and EMA updates, and no measurement expires within the experiment
horizons.

Softmax samples physical graphs using their best cohort's utility `Q`:

```text
p(graph) proportional to exp((Q - Qmax) / (temperature * (Qmax - Qmin))).
```

Equal utilities give uniform probabilities. The annealing alternative uses
`temperature / sqrt(1 + opportunities / 100)`. Epsilon nomination is uniform
over all feasible physical graphs, including the incumbent. The `--screen-grid`
tuning grid uses epsilon `0.01/0.05/0.10`, temperature `0.03/0.10/0.30`,
optimistic quantile `0.01/0.05/0.10`, and untempered Thompson sampling.
`--sweep NAME=VALUE,VALUE` varies one numeric policy parameter at a time, such
as `alpha`, `credit_cap_ms`, `refresh_every`, or `block_size`; repeat the flag
for independent sweeps.

The scenario catalog distinguishes stationary core workloads, cold/idle
transients, recovery fixtures, and stress cases before measurements. It includes
full and tail prefills, 1/4/16 competing decoders, alternative efficiency and
batch/verification-width rankings, idle intervals of 29/31/120/3600 seconds,
1,000-observation histories, poisoned first observations, mean-preserving timing
noise, correlation, rare outliers, changed costs, context/occupancy mismatch,
expensive unknown graphs, and missing measurements.

```sh
.venv/bin/python -m scripts.scheduling_sim.learning \
  --preset screen --screen-grid --workers 12 \
  --reference ttl \
  --output /tmp/flare-learning-tuning

.venv/bin/python -m scripts.scheduling_sim.learning \
  --preset validate --configs /tmp/flare-learning-tuning/nominated.json \
  --reference ttl \
  --workers 12 --output /tmp/flare-learning-validation
```

Screening uses seeds 0 through 15 and nominates at most three nondominated
configurations. Validation uses frozen configurations and defaults to seeds
10000 through 10063. An explicit `--seeds START:END` selects another 64-seed
evaluation range disjoint from the tuning seeds. Freeze configurations before
examining their evaluation range. The `smoke` preset has explicitly shorter
windows and cannot qualify a policy. Validation subsets remain diagnostic.
`--output` names a directory that must not already exist; the command fails
rather than write into an existing one. It receives ordinary
summary/request/calibration CSVs, configuration inputs, and comparison results.
`--compress-requests` writes lossless `requests.csv.gz` instead of `requests.csv`.
`--trace` streams selected graphs, estimated/sampled costs, actual timings,
decision reasons, useful work, and credit state; tracing uses one worker.

`tests/test_scheduling_timing.py` covers the estimators in `timing.py`,
`tests/test_scheduling_learning_policy.py` the policies in `learning_policy.py`,
`tests/test_scheduling_learning_run.py` the execution model in
`learning_run.py`, and `tests/test_scheduling_learning_experiments.py` the
scenarios, acceptance criteria, and CLI.

`--comparison-configs` adds fixed controls from a JSON array of `id` and `config`
objects. Controls do not enter nomination, calibration multiplicity, or winner
selection. Separate comparisons apply the same throughput and latency bounds
against the TTL policy and supplied controls, and include their recovery counts.
`--reference known` is the default primary reference. `--reference ttl` instead
uses the 30-second timing-expiry reference for nomination, primary throughput
and latency gates, and accepted-policy ranking. The other reference remains a
diagnostic comparison. Retained-cost policy evaluations use
`--reference ttl`; the metric thresholds and recovery requirements are
identical for both reference modes.

Stationary runs learn for 30 occupied seconds and measure for another 120,
including each boundary-crossing graph in full. Finite request templates are
replenished without resetting estimates. After measurement, a bounded drain
collects completion and latency endpoints; unresolved tails remain censored.
Busy useful throughput includes probe and bootstrap time. Elapsed throughput
also includes idle intervals. Prefill's first output contributes TTFT; a
speculative output burst contributes one decode delivery event.

Every applicable core workload must retain at least 98% of the selected
reference's useful throughput and have at most 105% of its per-seed p99 TTFT and decode gap, without
extra unfinished or starved requests. Confidence bounds resample paired whole
seeds and compare ratios of mean per-seed metrics, not a pooled p99. Selecting
among `K` frozen configurations uses one-sided confidence `1 - 0.05/K`. Each
applicable per-seed tail metric requires at least 100 endpoints and no unresolved
censoring. A favorable aggregate does not rescue a failing workload.

Recovery fixtures have two or four choices and a unique best useful-work rate
at least 20% above the runner-up. Recovery is the first completion ending a
10-occupied-second window with at least 95% of ordinary execution time on the
best graph, sustained at subsequent completion windows. Probe time is charged
separately; this condition alone does not limit probe stalls. At least 61 of 64
seeds must recover within 60 occupied seconds. Mature unused-graph drift is also
reported separately as a stress case. Step-change fixtures switch costs at
30,000 occupied milliseconds. Mature fixtures seed 1,000 observations per graph
at the common initial clock. Their recovery times depend on this change phase
relative to each policy's remeasurement schedule.

Mean intervals are checked at fixed sample counts under the matched block
likelihood. Nominal 95% coverage must reach 90%, and median interval width after
1,000 samples must be at most one quarter of its width after 16. A stationary
regime-reset rate of at most 1% means zero affected seeds out of 64 empirically;
it does not certify a population rate below 1%. Correlation and other model
mismatches remain visible as stress results.

Among configurations passing every gate, selection maximizes geometric-mean
core throughput, then minimizes worst decode-tail ratio and probe expense.
If none pass, no policy is selected. Heldout failures do not change the gates
or authorize retuning on heldout seeds.

## Runtime policy choices

The ongoing policy uses separate intrinsic output value, token fairness,
delivery fairness, and decode-age coefficients. Their concrete defaults and
units are specified in the [native design](scheduling-runtime-design.md).
First-token waiting has no independent age penalty: prompt work competes by
its useful progress and token-service price. A burst earns its accepted token
volume and one delivery. The delivery reference accounts for feasible batching;
changing a rate normalizer preserves marginal service prices across phases.
Helios adds a value and an age price for terminal retirement, which the
simulator does not model.

The runtime uses no EOS-length prior. Exact conditional completion schedules
are an offline reference. Native prerequisite selection uses bounded
lookahead to include a following shared inference graph, with individual
continuation values for longer chains. The stationary allocation results do not
prove that approximation optimal. The runtime queues at most two frames. When
it selects the second, it projects the queued frame's expected service into
copies of the service prices; observed service history changes only at
validated completion. It pauses dispatch at feedback dependencies.

Unknown duration cannot be divided into reward as a numerical zero. The native
contract uses zero to mean indeterminate, requiring an explicit exploration
choice. A measured cost stays available when stale; foreground execution funds
remeasurement as specified in the [native timing policy](scheduling-runtime-design.md#funded-timing-refresh).
Those bounds do not establish efficient or fair behavior during exploration.
Native tests cover zero-cost startup, timing freshness, recovery, and phase
transitions on synthetic catalogs. They do not establish GPU throughput or
latency.
