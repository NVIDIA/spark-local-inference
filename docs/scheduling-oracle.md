# Exact scheduling over completion episodes

[`scripts/scheduling_sim/oracle.py`](../scripts/scheduling_sim/oracle.py) computes
optimal conditional schedules for small finite workloads. It is a reference
for testing scheduling objectives and approximations. Its policy-improvement
prototype compares graph executions by expected useful value per millisecond,
including a continuation value for the state after execution. The oracle and
its prototypes exist only in the Python simulator. Helios has no
completion-episode solver and no output-length prior.

## Workload and information

Every request is present at the start. A request has known remaining prefill,
an observed output count, an initial output age, and a finite discrete prior
over its total output length. The scheduler observes accepted output and EOS
after each graph; it never sees a sampled future length or acceptance outcome.
A live request with `g` generated tokens conditions its prior on `L > g`.
Priors and acceptance outcomes are independent across requests; length and
acceptance are also independent within a request in this model.

An action chooses a prepared graph, at most one prefill request, and any subset
of decode requests fitting its capacity. Every cohort is enumerated. Partial
and empty physical lanes retain the complete graph cost. Pure and mixed graph
variants have equal eligibility, with no duration limit. Prefill consumes the
minimum of remaining prompt work and physical capacity. Completion emits one
first token. Decode emits an accepted prefix truncated at EOS. All output
becomes visible at graph completion.

Graph timing is fixed and known. Every action advances at least one request;
idle actions are absent. Finite prompt sizes and bounded output-length priors
therefore give an acyclic decision graph that terminates in all requests done.
The state contains per-request remaining prompt, generated output, completion,
and stream age. Different output histories with the same sufficient state
share the same continuation calculation.

The explicit state limit raises an error when exceeded. There is no approximate
fallback. Long prompts with small available chunks or many request/acceptance
combinations can exceed practical time, memory, or Python recursion limits.
The event simulator is the tool for workloads up to sixteen requests;
the exact oracle targets small instances.

## Metrics and objective

All request contributions are multiplied by their declared weight. Supplied
workloads use weight one. A complete episode measures:

- `P`: consumed prompt tokens.
- `D`: emitted output tokens.
- `T`: elapsed graph time.
- `F`: integrated number of requests still in prefill. With initial arrival
  time zero, this is the sum of time to first output for prefill requests.
- `C`: integrated number of unfinished requests, equal to their summed remaining
  completion times from the root.
- `A`: integrated decode age. A decoding stream at age `a` incurs
  `a*t + t*t/2` during a graph lasting `t`, whether or not it is selected.

At a delivery, decode age resets once regardless of the burst size. Completing
prefill starts the decode clock; internal prefill boundaries do not reset one.
Finished requests incur no further waiting cost.

Define useful work and net work:

```text
W = P + beta*D
R = W - c_first*F - c_finish*C - c_age*A.
```

`beta` is output-token value. Waiting prices have units of useful-work units per
millisecond, except `c_age`, which has units per millisecond squared. All prices
are explicit experimental inputs, not production defaults. In `oracle.py`,
`Objective` holds `beta`, `c_first`, `c_finish`, and `c_age` as `decode_value`,
`prefill_wait_price`, `unfinished_wait_price`, and `decode_age_price`.
`Metrics` holds `P`, `D`, `T`, `F`, `C`, and `A` as `prefill_tokens`,
`output_tokens`, `elapsed_ms`, `prefill_wait_ms`, `unfinished_wait_ms`, and
`decode_age_ms2`. The time price `rho` defined below is the `time_value`
argument.

The priced oracle chooses the policy maximizing

```text
J(pi) = E[R - rho*T],             rho > 0.
```

`rho` is the value of occupying the GPU for one millisecond. It is an opportunity
price, not a maximum graph duration. A production derivation would need to
connect this price to the active workload and its fair-service reference.

For one action's expected net work `r(s,a)` and duration `t(a)`, backward dynamic
programming solves

```text
V(done) = 0
V(s) = max_a [r(s,a) - rho*t(a) + E[V(s')]].
```

Each action has one choice before its stochastic outcomes. Successor choices
may depend on observations. Optimization occurs outside the expectation over
that action's outcomes, preventing clairvoyant selection.

Because all policies finish the same requests, expected `W` is fixed. At a
fixed `rho`, the priced oracle therefore minimizes the priced time and waiting
costs. Changing `beta` alone adds the same constant to every complete schedule.
These episodes test completion and cadence; they cannot establish the desired
intrinsic phase preference under continuing arrivals. That preference and
sustained fairness belong to the [rate-allocation model](scheduling-theory.md).

## Why episode normalization is a separate experiment

The ratio solver maximizes `E[R] / E[T]`. This differs from `E[R/T]` when
completion time is stochastic. It iterates a trial rate `q`, solves the additive
problem `max E[R-q*T]`, and updates `q` to the selected policy's ratio until its
residual is within tolerance. This is the parametric method of
[Dinkelbach](https://doi.org/10.1287/mnsc.13.7.492).

With zero waiting prices, fixed expected work makes this a minimum expected
makespan reference. With negative net rewards, episode normalization can instead
reward dilution of an accumulated waiting penalty.

For example, one request has 64 prefill tokens and one eventual output token.
Another request is already decoding, needs one output, and has age 1000 ms.
Decode takes 1 ms. Two prefill graphs perform the same work in 1 and 100 ms.
Set `beta=1`, `c_age=1`, and other waiting prices to zero. Decode followed by
prefill earns 66 useful units and incurs 1000.5 units of age cost regardless of
which prefill graph runs. The ratio criterion prefers `-934.5/101` to
`-934.5/2`, selecting the slower graph. A positive time price prefers the
1 ms graph because extending the episode earns nothing and costs time.

The executable counterexample is
`test_normalizing_negative_net_value_by_episode_time_can_reward_waste`.
The CLI labels ratio results separately; they are not the priced optimum.

## Policy improvement through value per time

Let `pi` be any completing baseline policy. Evaluate its remaining surplus
exactly under the fixed time price:

```text
V_pi(s) = E_pi[remaining R - rho*remaining T | s].
```

Use that continuation value to score each executable graph:

```text
score(a | s) = [r(s,a) + E[V_pi(s')] - V_pi(s)] / t(a).
```

For the baseline action, the policy-evaluation equation makes this score
exactly `rho`. Choosing the maximum therefore gives

```text
r(s,a) - rho*t(a) + E[V_pi(s')] - V_pi(s) >= 0.
```

Sum this inequality along a complete trajectory of the improved policy.
Continuation terms cancel in expectation, and terminal value is zero. Thus
the improved policy's expected surplus is at least `V_pi(root)`, the baseline's
surplus. This proof applies to the priced objective, not to every reported
throughput or latency metric independently.

`Oracle.rollout` implements this operation. Its default baseline maximizes
immediate expected useful work per graph time, including the same hidden-length
belief available to the oracle. A supplied baseline allows repeated improvement.
The continuation values are frozen for each improvement pass; substituting an
unrelated rate at each state does not preserve this proof.

The value term distinguishes which stream an action serves. Directly subtracting
the next graph's integrated waiting cost does not: all live streams keep aging
until completion. For equal-duration actions, that immediate cost is identical
regardless of cohort. The important difference is which ages reset in the
successor state. An old stream becomes more valuable through its effect on
future waiting, without multiplying a stall-clearing bonus by speculative
burst size.

Exact evaluation of the baseline policy is expensive and requires finite
length priors. This prototype identifies the quantity an efficient scheduler
would approximate. It does not supply a bounded-cost runtime implementation.

## Tests and experiments

The tests in [`tests/test_scheduling_oracle.py`](../tests/test_scheduling_oracle.py)
compare both solvers with exhaustive schedule/policy-tree enumeration on small
instances. Separate transition tests check EOS conditioning, acceptance-prefix
probabilities, independent batched observations, inactive rows, first-token
accounting, and age accumulation during a request's own execution. Seeded
stochastic tests check that each improvement pass lies between its baseline
and the exact priced optimum.

Run from the repository root:

```sh
python -m pytest tests/test_scheduling_sim.py tests/test_scheduling_oracle.py \
  tests/test_scheduling_continuation.py -q
python -m scripts.scheduling_sim.oracle_sweep \
  --decode-values 1,64 --time-values 0.1,1 --age-prices 0,0.01,0.1 \
  --output /tmp/flare-oracle-sweep.csv
python -m scripts.scheduling_sim.oracle_sweep --case penalty_dilution \
  --decode-values 1 --time-values 1 --age-prices 1 --include-ratio
```

The sweep uses eleven explicitly defined synthetic workloads, including prompt
tails, two 8192-token prefills, concurrent prefill/decode, heterogeneous
acceptance, mixed captures, a stalled stream, and uncertain output length.
Decode lengths are deliberately small enough for exact enumeration. The CLI
prints each policy's first action and expected episode measurements, and writes
ordinary CSV when requested. `--decode-values`, `--time-values`, and
`--age-prices` supply `beta`, `rho`, and `c_age`; `--prefill-wait-price` and
`--unfinished-wait-price` supply `c_first` and `c_finish` and default to zero.

The sweep labels its methods `oracle`, `value_rate`, `myopic_wait`, `rollout`,
`rollout_twice`, and `serial_value`, plus `episode_ratio` with
`--include-ratio`. `serial_value` comes from `SerialReference` in
[`continuation.py`](../scripts/scheduling_sim/continuation.py). It applies the
value-per-time score above with a cheaper baseline: finish the requests one at
a time in input order, each by its isolated immediate useful-work rate. It
composes that baseline's value from per-request tables over remaining prompt
and attained output, without enumerating joint acceptance outcomes.
[`tests/test_scheduling_continuation.py`](../tests/test_scheduling_continuation.py)
compares the composition with complete conditional evaluation and checks, on
small seeded instances, that the resulting policy's surplus lies between the
serial baseline and the priced optimum. `helios/src/scheduler/valuation.rs`
contains the same serial composition in Rust as `serial_metrics`. It is
compiled only for tests, and the Helios selector does not call it.

Each sweep row reports `surplus_gap`, the amount by which the method's expected
surplus falls below the priced optimum for that configuration. A method matches
the optimum when its gap is below `1e-7` useful-work units. The gap compares
synthetic objectives; it measures neither inference performance nor fairness
under ongoing arrivals. Repeated policy improvement has no general optimality
guarantee.

Continuation value corrects schedules that immediate useful work per time gets
wrong. `test_completion_oracle_and_rollout_resolve_the_greedy_prefill_tail`
asserts one case: over prefill graphs of 64, 128, and 256 tokens costing 2, 3,
and 4 ms, a 129-token prompt finishes in 4 ms under the oracle and one
improvement pass and in 5 ms under the immediate score.
`test_mixed_completion_value_can_outweigh_lower_immediate_rate` asserts a
second case with mixed graphs.
