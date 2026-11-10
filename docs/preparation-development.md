# Preparation development

Ignition prepares weights, tunes operators, plans storage, compiles forwards,
and captures graphs before Helios serves them. Follow
[repository guidance](../AGENTS.md) and the relevant
[core preparation design](core-design.md#2-model-preparation) sections.
For running preparation, use [model preparation and serving](model-preparation.md).
Recipe and execution-component changes also require
[model development](model-development.md).

## Operator declaration, tuning, and binding

- Use b12x's autotuning and warmup facilities during preparation. Do not pin
  kernel plans or tuning choices without a demonstrated requirement.
- Constructors declare b12x requests through `ignition.preparation.tuning.declare`;
  they must not compile or prime default plans. Submit all operators across the
  complete serving shape catalog to one preparation batch, then bind selected
  plans through `register_finalizer`. Scratch sizing and serving bindings must
  follow tuning.

`RecordedPreparationSession` uses `PreparationSession(prime=False)`. Selected
plans still materialize, construct prepare-call bindings, retain owners, and
run restore/close hooks, but skip the call's reset, activation production, and
execution. Factories must construct required resources without depending on
those execution hooks. Tuning candidates and retained benchmark calls still
prime. Complete-forward warmup in `capture_forwards` runs every selected variant
before the session's capture guard forbids further kernel resolution.

## Startup progress

- Show startup work through `ignition.progress.phase`. Weight loading and kernel
  preparation retain a phase activity bar while their detailed b12x panels are
  transient and share the phase's Rich console. Use `DirectWeightLoader` for
  loading progress, including draft-model weights. Only the root TP rank
  renders progress.
- Heliopause reports native export phases on stderr without the Flare banner,
  including scanned, hashed, copied, installed, and reused byte counts. Capture
  and frontend rows render only on rank zero. The enclosing Python export phase
  uses `live=False` while native rows animate and retains its elapsed result.
  Keep bulk export progress in native code so transfers do not call back into
  Python.

## Complete-forward compilation and capture

- **Compile complete prepared forwards.** Use
  [capture_forwards](../ignition/preparation/capture.py), which restores compiled
  forwards before tracing and compiles missing variants with static shapes and
  autotuning. Workers perform Dynamo tracing, storage lowering, and Inductor
  compilation from tensor and selected-plan metadata; CUDA resources remain in
  the owning rank. Preserve native mutations, storage aliases, and stream dependencies;
  bind process-local kernel handles and caller-owned tensors at startup. Complete
  b12x preparation, tuning, binding refresh, and compiled warmup before capturing
  each serving graph. Retain compiled kernel owners until native replay has drained.

## Runtime package export

`ignition.preparation.runtime_package` hands live graph handles, typed storage
and execution metadata, tensor-view roles, compiler signatures, frontend assets,
and retained owners to `flare_heliopause`. Recipes describe file-backed table
ranges and generated frontend text without creating package files or payload
offsets. Heliopause records CUPTI resources, traverses graphs and pointer-bearing
storage, downloads retained bytes, serializes resources, and owns asset writes,
hashing, zstd level-3 executable compression, signing, conversion, rank payload
transfer, and publication. Each executable resource is one zstd level-3 frame
with a signed decoded length and compressed-chunk commitments.
Keep captured buffers, contexts, graph templates, and kernel owners alive until
native work confirms a drain; an unsuccessful drain retains their owners.

Native transfer pools bound file buffers and pinned CUDA download storage from
available host memory and preparation rank count. Preserve direct bulk I/O,
exact unaligned tails, cache verification before reuse, and hash-before-write
publication. `--no-weights` must fail on any missing exact stored-data blob;
correctness and cache checks remain required. See
[runtime package transfers](runtime-package-design.md#native-export-transfers-and-progress)
for the transfer budget and progress contract.

`--reuse-weights` pins the cached default package and its weight objects through
publication. Native export authenticates its metadata, checks preparation inputs
and named weight layouts, and rebinds the stored weight initializers to compatible
allocation placements. Only declared numeric weight ranges may bypass live
storage reads; pointer-bearing storage and undeclared initial state retain their
normal capture and export path. Cached weight objects are linked without reading
or hashing their payload bytes; export checks their container lengths and pinned
file identities against the authenticated metadata. This mode trusts the cached
weight contents and does not detect preexisting payload corruption. Missing or
incompatible metadata is an error, with no live-weight fallback.

## Persistent model tensor storage

Model weights and immutable constants use the packed destinations in
`ignition.preparation.persistent`. `persistent_empty`, `persistent_zeros`,
`persistent_copy`, and `persistent_cat` place final tensors in a `PackageStorage`
arena with aligned, disjoint storage descriptors. Intentional aliases remain
views of the same parent tensor. The arena grows its physical backing at the
device's allocation granularity while preserving every tensor address.

Shared checkpoint loaders allocate final destinations there by default. Pass
`temporary=True` for inputs discarded by concatenation, quantization, or another
conversion. Packing and quantization helpers accept an allocation callback for
their retained output planes; intermediate arithmetic stays in transient storage.
Do not move already-bound tensors or put disposable conversion outputs in the
persistent arena. Outside package preparation, the helpers use ordinary Torch
storage for standalone loading.

Library caches that outlive a model retain their own storage and must not hold
pointers into its packed arena. Storage placement does not classify a tensor as
numeric weight data; immutable pointer tables retain opaque metadata and pointer
relocation handling.

The arena exports through fixed device allocations and remains alive through
native replay. `PreparedModel.close()` releases it after confirmed drain and graph
cleanup. Leaving the allocation context does not release model-owned storage.
Preparation failures before that ownership transfer drain and release their
mappings; a failed drain retains them. Tensor-allocation accounting includes the
arena's used bytes, and physical memory accounting includes its mapped backing.

## Scratch storage and concurrent lifetimes

Compiler-local intermediates use Inductor's static memory planner with a combined
pool. Captured graph variants share a CUDA graph pool when their executions are
serialized and their outputs live in explicit caller-owned buffers. Functional
components allocate their intermediates inside the compiled forward so the
compiler can track their lifetimes. Bound operator interfaces still require the
explicit scratch ownership described below.

Dense Qwen target and draft forwards pass packed batch metadata, proposal
state, recurrent correction records, and sampling intermediates as local tensor
values. Their prepared operators retain kernel plans, weights, geometry, native
inputs and outputs, and persistent cache state. Declaration examples are released
after tuning, and the operator arena is retired before compilation. These forwards
do not use `GraphBufferPool`. The shared CUDA graph pool backs Inductor's planned
temporaries across serialized graph variants.

Retained rotary frequencies, fixed page indices, and scalar quantization constants
use persistent tensor storage. Attention placeholder operands belong to the
forward's declared scratch rather than hidden plan allocations. Constructors may
use temporary examples to select kernels, but those examples must not become
serving storage through closures, argument tuples, or module fields.

CUDA libraries can retain workspace without a Python tensor owner. Captured GEMMs
may embed cuBLAS workspace pointers inside packed kernel arguments. Those
allocations remain required for replay and must not be released as preparation
caches. Account for them separately from inactive allocator blocks and alignment.

- Supply caller-owned scratch through the kernel planning interfaces. Reuse
  storage across sequential work; preserve separate storage for overlapping
  operations and persistent state. Aliasing must follow actual lifetimes and
  stream dependencies, including overlap captured inside a single graph.
- **Share scratch across the complete graph catalog.** Vision, prefill, decode,
  speculative, and resume graphs execute serially and should reuse caller-owned
  activation and private buffer pools. Allocate per-variant views with
  `GraphBufferPool` cursors; keep simultaneously live values disjoint within a
  graph, including retained verification state and overlapping expert branches.
  Use the shared operator arena for kernel workspaces and one CUDA graph pool for
  compiler-created temporaries. Inspect constructor allocations as well as captured
  allocations; sharing the CUDA graph pool alone does not share recipe scratch.
  Constants and request-owned state must retain independent storage.
  Prime indexed operators with private valid metadata: aliased serving scratch
  is initialized by the complete graph, not by each operator's constructor.
  Keep shared buffer-pool layouts identical across TP ranks, including storage
  reserved for designated-rank sampling, so shared compiled forwards retain the
  same storage aliases on every rank.

## Physical cache backing and preparation memory

- **Bound preparation cache backing by tuning and capture needs.** Keep virtual
  serving capacity separate from physical preparation backing. Back only the
  KV pages touched by tuning and capture, plus the null row, simultaneously live
  request/verification rows, and disjoint snapshot destinations. Do not physically
  allocate the planned prefix-cache capacity or fill remaining VRAM with KV during
  export; Helios sizes serving caches after graph upload. Check every fixture,
  state-clear operation, and snapshot/restore path against the backed extent.
  Describe the full virtual shapes with CUDA tensor metadata while compiling. Pass a
  `materialize` callback to `capture_forwards` to bind a bounded physical working
  set after compiler workers exit and before warmup and capture. Validate compiled
  guards against those tensors. Helios sizes the serving cache independently;
  temporary compiler contexts must not reduce its budget.
- **Release preparation temporaries at phase boundaries.** Retire loading,
  conversion, tuning, and warmup pools when their temporary owners are gone;
  retain graph pools and captured addresses through export and replay. A live
  tensor can pin otherwise unused space in a shared allocator segment; retiring
  a pool cannot release a segment that still holds a weight. Account for
  simultaneous scratch, cached allocator segments, physical cache backing,
  and graph-executable/driver allocations. Report allocated and reserved tensor
  bytes separately from device-free memory. Validate the complete requested graph
  catalog near capacity; a smaller concurrency does not establish builder fit.
- **Derive preparation memory policy from hardware.** Shared preparation code
  detects integrated GPUs and budgets compiler workers and tuning races against
  available system RAM. Recipes pass byte headroom, not hardware flags. Forward
  compilation uses memory-bounded parallel workers on both integrated and discrete
  GPUs; artifact binding starts no additional compiler pool.

## Tensor-parallel compilation

PCIe communication declarations participate in b12x preparation alongside local
kernels. Components submit the actual operation, shape, dtypes, layouts, and
serving bindings. b12x owns the candidate benchmark operands, resets, arithmetic
checks, captured timing, and selection cache. Equivalent declarations share a
winner; component names do not identify separate tuning workloads.

`CollectivePreparation` relays b12x's collective tuning contributions in TP rank
order. Each candidate executes on the whole TP group and is scored by the slowest
rank. `ReplicaPreparation` keeps these exchanges within each serving TP replica;
independent kernel candidates retain corresponding-rank sharding across replicas.
Preparation-time search ends before graph handoff. With `--collective-tuning
runtime`, the required `--collective-ladder PATH` file supplies final compiled
geometries for eligible plain BF16 PCIe reductions and column gathers. Gather
buckets depend on total per-rank input bytes, independently of row width. These pinned plans do not
race geometries during preparation, including with `--autotune-effort off`.
Captured tuning graphs select among the supplied submode/active-CTA profiles
before serving begins; native rank coordination only launches those graphs and
waits for their completion. See the [ladder format](../ignition/preparation/ladders/README.md).

- **Compile tensor-parallel forwards once, sharded across ranks.** Pass the
  preparation coordinator to `capture_forwards`. Each forward variant compiles
  once, on a rank that selects it, from that rank's own forward; on one host, a
  rank without variants left, or without room for a compiler, compiles another
  rank's variant from that rank's traced forward metadata. Compiler pools are
  budgeted for their host's ranks. Ranks on one host share its cache directory;
  ranks on other hosts receive the artifacts they lack. Every rank loads the same
  artifacts, binds its tensors and native handles, and participates in ordered
  warmup and local capture. Inductor's first owner-process warmup may tune the
  same generated kernel independently on several ranks. At each complete-forward
  boundary, ranks adopt one measured selection and compiled kernel for each exact
  generated source and candidate contract, with only the CUDA ordinal normalized.
  The first canonical selection remains authoritative across the complete catalog.
  A second warmup verifies the frozen selections before capture; cached artifacts
  retain those selections on single-rank and tensor-parallel reloads. No selection
  exchange occurs inside a forward or a native collective. Keep shard offsets and native handles in runtime
  bindings when they do not change the computation. For distinct rank behavior,
  select a variant key through `compilation_variant` on each rank. Every
  selected variant must compile before collective warmup; ranks must not retrace
  or compile on a binding mismatch.

## Shape and graph-transition validation

Packed verification catalogs declare physical row capacities independently of
logical per-request widths. Use `packed_speculative_shapes` to distribute a
graph-count budget across the request buckets while preserving each bucket's
physical endpoints. Size target workspaces from the largest physical capacity
and sampling/state workspaces from their declared logical widths. Validate
capacities that are not multiples of batch size, including a single aggregate
draft row, inactive lanes, and replay transitions between physical capacities.

- **Prepare the whole serving shape range.** Include batch-one decode, packed
  decode, partial/inactive rows, and configurable chunked prefill through at
  least 4096 tokens. Test alternating graph variants with shared scratch and
  changing cache/recurrent state. Small-batch success does not establish stride,
  alias, or workspace correctness for prefill.

Follow [model development](model-development.md#speculative-decode) for width-one
speculative execution and [GPU performance work](performance-work.md) for complete
GPU-step measurements and repeated native replay checks.
