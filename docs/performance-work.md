# GPU performance work

The [core design](core-design.md#performance-acceptance) defines service
baselines and the native replay overhead limit. Measure each optimization
against a baseline with the same checkpoint, GPU assignment, activation
precision, graph capacity, sampling settings, cache conditions, and completed
work. Correctness failures block performance claims.

Use `ignition export` and `flare run PACKAGE` for performance comparisons
against other runtimes. `ignition debug-serve`, including native, `--direct`,
and `--eager` execution, is for fault diagnosis only. Its preparation resources,
cache backing, and loading lifecycle differ from package serving, so its timings
cannot establish apples-to-apples serving performance.

`ignition bench qwen3.8-27b` measures complete prepared speculative graph rounds
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

Verify improvements through an exported `.safetensors` package loaded by
`flare run --embedding-table-storage ram`, including Helios's asynchronous
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
harness supplies the measured floor.

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

Review scratch lifetimes across every supported recipe and shared component,
including Qwen, GLM, DeepSeek, MiMo, Kimi, Step, and Nemotron.
Measure prepared fixed memory and available KV capacity before changing each
recipe. Inspect attention, rotary, activation, expert, and vision temporaries
for caller-owned outputs and shared storage across sequential graph variants.
Keep persistent state and overlapping branches disjoint; include stream joins
and image feature consumption when determining lifetimes.

Size speculative score, count, probability, and retained draft buffers from
the actual prepared (batch, verification-width) catalog. Independent maxima
for batch and width can describe a shape that no graph uses. Inspect both
oversized shared workspaces and separate allocations retained by serialized
graph variants or proposal steps; keep simultaneously live distributions
disjoint.

Vision working buffers and mutable validation metadata use graph-buffer cursors;
request-owned image features and upload inputs retain independent storage.
Speculative graphs allocate filtering and retained-probability buffers for their
actual batch and width, sharing backing across serialized variants. Proposal and
target stages execute within one graph; retained proposal values remain disjoint
from target scratch until verification consumes them. Graph-buffer cursors reuse
available space in earlier chunks without moving existing views.
Proposal filtering, target filtering, and verification reuse sequential-stage
intermediates. Target probabilities and validity retain separate storage through
verification, as do retained draft probabilities.

Qwen, DeepSeek V4, MiMo, Nemotron, and Step vision encoders reuse storage across
attention, feedforward, and merger stages. Qwen also reuses patch staging and
consumed QKV storage. Kimi reuses its fused-projection workspace across these
stages. Residuals, shared projection outputs, metadata, and live projection
operands stay disjoint. Qwen validates FP32 upload pixels before sanitizing them
directly into the encoder's BF16 patch buffer.

Cursors for sequential stages reuse storage beyond the buffers retained
throughout a forward. Qwen's GDN, QSA, PLE, MoE, and hyperconnection temporaries share this
storage. Block inputs, outputs, residuals, and injection coefficients remain
disjoint from these temporaries. Target and MTP share a pool. Prefill retains the
target residual until feedback packing; MTP feedback reuses normalized-activation
storage and is consumed before the mixer overwrites it. Verification residuals
and acceptance state retain independent storage. GDN's chunked-prefill operands are
allocated only for the SM103 execution path that consumes them.
DeepSeek V4, MiMo, and Step share attention and FFN intermediates. Concurrent
shared/routed expert branches allocate from one cursor to retain disjoint views.
DeepSeek V4.1 shares activation storage across text, draft, and vision graphs.
Prefill layers commit window state before reusing private buffers; sequential
indexer batches reuse packed queries and scales. The bounded decoder suffix
gathers into destinations disjoint from the full prefill. Candidate rows follow
the suffix capacity when hierarchical selection starts after that boundary.
Verification compressor values and gates remain separate through acceptance.
GLM's mHC carry and Kimi's named private buffers still require a stage-lifetime
review. Nemotron reuses a cursor across serialized heterogeneous blocks.
DeepSeek V4.1, MiMo, and Kimi draft-context projections retain dedicated
intermediate tensors;
review their consumption and history updates before sharing decoder storage.

Allocated scratch bytes are an upper bound, not a measurement of simultaneously
live values. Typed chunk rounding, tensor views, graph-created temporaries,
operator workspaces, retained speculative results, and upload storage all affect
the allocation total. Derive a live-memory peak from read/write lifetimes and
stream dependencies before describing an allocation inventory as that peak.

Use the shared scratch planner where the lifetime contract fits. Validate
numerical correctness, allocation stability, and native replay across decode,
chunked prefill, vision, and supported speculative capacities. Record fixed
memory, KV capacity, and complete graph latency before and after each change.
