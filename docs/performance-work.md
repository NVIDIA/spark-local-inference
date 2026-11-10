# GPU performance work

The [core design](core-design.md#performance-acceptance) defines service
baselines and the native replay overhead limit. Measure each optimization
against a baseline with the same checkpoint, GPU assignment, activation
precision, graph capacity, sampling settings, cache conditions, and completed
work. Correctness failures block performance claims.

Use `ignition export` and `flare run CORE` for performance comparisons
against other runtimes. `ignition debug-serve`, including native, `--direct`,
and `--eager` execution, is for fault diagnosis only. Its preparation resources,
cache backing, and loading lifecycle differ from core serving, so its timings
cannot establish apples-to-apples serving performance.

`ignition bench` supports Qwen3.8-27B DFlash2, Kimi K3.1 Pro DSpark, and
Nemotron Bolt LiLiCorr. It measures complete prepared speculative graph rounds
or, with `--prefill --prefill-chunk-size N`, exact prefill chunk graphs
for kernel and model iteration without export or HTTP. It uses real tokenized
contexts, restores complete state outside CUDA event timing, and records raw
samples by case and seed. Its score is the equal-weight arithmetic mean of
per-case medians; lower is better. Keep GPU classes separate. The
[benchmark CLI](cli-design.md#gpu-graph-latency-benchmark) defines the default
matrix and timed boundary. Graph latency does not establish serving throughput
or comparative performance against another runtime.

Before changing model or preparation code, read
[model development](model-development.md) and
[preparation development](preparation-development.md). Inspect fusion boundaries,
precision, sharding, overlap, and selected kernel plans during implementation.
Reuse FlashInfer/b12x mechanisms directly. Attribute gaps to emitted kernels,
memory traffic, dependencies, or host work before changing architecture.
Preserve correctness when changing layouts, fusion boundaries, scratch reuse,
or stream overlap.

## Measurement

Use `qualify.gpu_replay` for complete GPU graph duration, including vocabulary
projection, collectives, and sampling even when a reference engine executes them
outside its main graph. Under tensor parallelism, measure the interval spanning
every rank. Collect batch-one and batch-four kernel
timelines with operator and layer attribution, kernel counts, summed kernel
durations, and graph elapsed time. Host graph-launch traces alone cannot
attribute GPU work.

Verify improvements through an exported core loaded by
`flare run CORE --embedding-table-storage ram`, including Helios's asynchronous
row delivery. Measure disk delivery separately when it is the deployment
configuration. For matching batch and verification width, graph rounds per
second are the change in `flare_speculative_rounds_total` divided by the
change in `flare_speculative_gpu_nanoseconds_total`, multiplied by
`1e9`. Emitted-token throughput also depends on draft acceptance.

Separate GPU replay time, native scheduling and submission, and HTTP latency.
Compare equivalent shapes and execution configurations. Reuse valid baselines
and use kernel traces or component measurements to attribute gaps before checking
gains through the actual Helios service. Record raw measurements with their
configuration, rerun noisy comparisons, and retain workload regressions. The
single-token weight-traffic estimate is a bandwidth bound; the minimal replay
harness in `qualify/replay_floor.py` supplies the measured floor.

Verify allocation stability and scratch/input lifetime under repeated
native replay, including poisoning reusable storage after its last permitted use.

## Fusion and precision

Fuse hyperconnection residual injection with the following RMSNorm using
FP32 gains and caller-owned output. Preserve PLE injection and inactive-row
mask boundaries. Pack compatible GDN QKV/Z and B/A projections and sparse
attention Q/K/V projections once during loading, including scale rows and TP
slices. Downstream kernels must accept their strides at every capacity.

Preserve BF16 intermediate rounding, FP32 normalization statistics, ATen
mean-reduction order, and FP32 sigmoid evaluation before multiplication and
BF16 rounding. Retain native ATen LayerNorm where decomposition changes
normalization or affine rounding.

Read formats and calibrated scales from checkpoint metadata. Routed MoE
gate/up fusion requires the dedicated loader's shared-scale validation.
Investigate a layer-wide gate/up activation scale only where the checkpoint
and operator contract permit it; retain the down projection's calibrated
scales. Never rescale or requantize weights to enable fusion.

## Complete-forward compilation

Compile prepared forwards with `torch.compile` after b12x tuning and before
capture. Opaque b12x calls expose their reads, writes, aliases, and native
launches; Inductor optimizes surrounding tensor operations. Preserve normal
functionalization, using byte storage for mixed-dtype aliases and typed
storage for aliases of one dtype.

Complete text forwards include MTP absorption, speculative verification, and
commit. Vision encoders compile their canonical tensor operations around prepared
projections and attention. Maintenance graphs capture their prepared operations
directly. Chain verification exposes token and acceptance-counter mutations.
Conditional penalty histograms include only preceding candidate IDs, bounded
by verification width and the conditional-prefix offset.

`qualify.gpu_replay --check-compiled` compares logits and completion records
over 256 steps, plus final request, hybrid, and KV state, against uncompiled
graphs on the same model and buffers. This bytewise diagnostic requires
deterministic kernels. Plans with variable reduction order use component
bounds derived from measured repeatability.

Check graph breaks, escaping output lifetimes, shared scratch, native replay
allocations, and graph duration. The
[PLE delivery test](../tests/components/test_ple_delivery_cuda.py) exercises
asynchronous delivery; the
[PLE component tests](../tests/recipes/test_qwen38_ple.py) compare consumer
arithmetic across graph variants.

## Autotuning and kernel bands

Use b12x's numerical policy and autotuning, including dynamic expert routing
and atomic reductions where the model contract permits them. Measure GDN
recurrence schedules, QSA partitions, MoE reductions, and hyperconnection
projection precision independently. Preparation metadata must agree with the
actual submitted queries.

Measure every captured speculative shape. Physical target rows, `P`, determine
whether expert execution stays in the small-row band or selects grouped kernels.
Rectangular rounds use `P = B × W`; ragged rounds can use fewer target rows.
The [MTP design](mtp-design.md) specifies
round-cost and acceptance measurements. Do not infer a speedup from draft
depth or a component benchmark alone.

## Independent branch overlap

Declare simultaneous scratch through `ScratchArena.register_parallel`.
Reserve the complete candidate envelope during tuning, then trim and refresh
bindings to the selected peak. Serial layers and graph variants reuse storage;
concurrent branches use disjoint regions.

Qwen's GDN input projections overlap through a prepared auxiliary stream at
capacities up to sixteen rows. Shared experts overlap routing and routed
experts through capacities of 256 rows, joining before gating and addition.
QSA index projection and selection overlap QKV projection, normalization,
rotary embedding, and KV writes through capacities of sixteen rows. The
selection join precedes attention.

Forks and joins must survive compilation. Their dependencies preserve inputs,
scratch, cache mutations, and final KV-pool bindings. Rust launches the
captured graph; its internal streams require no host intervention. Measure
complete graph latency, including synchronization and resource contention, at
each relevant row capacity and with each candidate node-construction order.

## Compressed scale loading

The [compressed scale layout](tp-checkpoint-layout.md) supports TP1, TP2, and
TP4 without changing packed weights or calibration. Measure partition reads,
GPU regrouping, native preparation, and upload separately from replay.
Scale-expansion tuning includes live producers and consuming expert work.
Poison reusable scale scratch between calls and verify alternate graph
capacities through 4,096-token prefill.

## Cross-model scratch planning

Use the complete prepared graph catalog when diagnosing fixed device memory.
The relevant result is available cache capacity and graph latency from an
exported core under the intended serving configuration. `ignition debug-serve`
can isolate a fault, but its memory and timing include preparation resources.

1. Establish a baseline with the same checkpoint, GPU, tensor-parallel width,
   concurrency, prefill chunks, and speculative capacities that the change will
   target. Record retained weights, explicit cache and request state, graph-pool
   storage, operator workspaces, allocator inactive blocks and padding, available
   cache capacity, and complete graph time. Distinguish physical backing from
   reserved virtual address space and preparation-only allocations from serving
   allocations.
2. Trace the largest excess allocation to its owner and last consumer. Inspect
   the compiled schedule and stream joins when a lifetime is unclear. Check
   whether the bytes are useful live data, a constructor-retained example, an
   opaque operator allocation, a graph-local tensor, or allocator fragmentation.
   Compare the measured allocation with the largest required live overlap before
   changing a pool size.
3. Put graph-local activations, metadata, outputs, and workspaces in the prepared
   forward. Pass their tensors to b12x operators and expose reads, mutations,
   aliases, and side-stream dependencies. Let Inductor plan reuse within the
   forward and let serialized graph variants share the CUDA graph pool. Keep
   immutable model tensors in `CoreStorage` persistent destinations and keep
   cache, request, completion, and transport state in explicit storage. Retire
   declaration examples and tuning workspace before compilation. A second manual
   temporary planner needs a demonstrated lifetime or compiler constraint.
4. Re-export the full catalog and validate native replay at the affected shapes,
   including alternating graph variants, inactive rows, speculative rollback,
   vision transitions, and tensor-parallel joins where supported. Zero or poison
   disposable preparation storage before replay; every graph must write scratch
   before reading it. Check numerical references, repeated-replay allocation
   stability, fixed memory, available cache capacity, and graph latency. Keep a
   resource reduction when latency is equal within measurement noise; investigate
   any remaining gap by owner rather than assuming Inductor planned it away.

Review temporary lifetimes across every supported recipe and shared component.
Measure prepared fixed memory and available KV capacity before changing each
recipe. Activations, metadata, sampling intermediates, expanded scales, and
operator workspaces belong inside the compiled forward. Expose their reads,
writes, and aliases to Inductor. Follow [serving temporary lifetimes](preparation-development.md#serving-temporaries-and-concurrent-lifetimes).

Constructors may use ordinary declaration examples to select plans. Inspect
component fields, closures, prepared argument tuples, priming owners, and library
caches for references that keep those examples alive after tuning. Release them
and retire preparation workspace before compilation. Shared declaration examples
need distinct semantic roles for simultaneously live operands.

Size speculative scores, counts, probabilities, and retained proposals from the
actual prepared batch and verification width. Independent maxima can describe a
shape that no graph uses. Keep proposal probabilities, target probabilities,
validity, and correction records live through their final verification or commit
consumer. Inductor can reuse their storage afterward.

Persistent cache history, snapshots, request state, native inputs, completion
frames, transport buffers, and request-owned image features retain explicit
storage. Write escaping values there before a graph completes. Serialized graph
variants share one CUDA graph pool only after every auxiliary branch joins.
Carry each branch's source, outputs, encoded planes, and workspace through an
explicit pending record, and expose the join's main-stream dependency.

Preserve model-specific retention constraints. Qwen image validation precedes
active metadata construction; prefill retains target residuals until feedback
absorption. Recurrent correction records and verification compressor values
remain live through accepted-state commit. DeepSeek's bounded decoder suffix
uses destinations disjoint from the full prefill; candidate capacities follow
that suffix geometry. Layer histories must be committed before their source
intermediates can be reused. Vision upload pixels retain their declared
precision and validation before conversion into encoder activations.

Step, Muse, and MiMo retain caller-owned graph-buffer paths. Their
buffer cursors may reuse storage only after the last consumer and must
keep concurrent expert branches and draft-context values disjoint.
Tensor-parallel ranks must bind equivalent layouts, including storage
reserved for designated-rank sampling.

An allocation inventory is not the exact live-memory peak. Account separately
for graph-private pools, persistent storage, opaque library allocations,
alignment, and allocator slack. Derive overlapping lifetimes from the compiled
schedule and stream dependencies. Preserve graph-wide synchronization workspace
through its final consumer.

Validate numerical correctness, allocation stability, and native replay across
decode, chunked prefill, vision, and supported speculative capacities. Alternate
graph variants and poison disposable declaration buffers. Record fixed memory,
KV capacity, and complete graph latency before and after performance changes.
