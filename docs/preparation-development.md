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
  each serving graph. Warm the complete selected catalog before retaining a serving graph.
  Core-backed eager warmups release their transient pools before capture, so
  their workspaces do not coexist with retained graph intermediates. Ranks
  synchronize before each warmup and each capture. Retain compiled kernel owners
  until native replay has drained.

`ignition.runtime_patches.apply()` installs process-local Dynamo, Inductor, and
Triton fixes after dependency environment setup and before compilation or artifact
loading. It requires Torch 2.14.1+cu132 and Triton 3.8.0, checks replacement source
contexts, and leaves dependency files unchanged. Spawned preparation ranks and
tuning replicas install the fixes before GPU setup. Dependency worker hooks cover
b12x's offline compiler initializer, Inductor's subprocess sidecar and job workers,
and direct spawned Inductor workers. The lightweight Triton bootstrap does not
import Torch or initialize CUDA.

The runtime patches defer mutation-region recomputation during single-node
auto-functionalization decomposition, directly construct eligible no-clone
custom-op decompositions, avoid scheduler membership-set copies, and defer
input-mutation stream checks until a trace records an event. Event recording
restarts tracing with complete mutation checks. Required clones and stream
ordering checks remain active.

Static Triton launchers preserve cooperative CUDA attributes and discard
process-local launcher bindings when serialized. Their cache filenames follow
Triton's 150-character limit without truncating CUDA symbol names. Scheduler
template replacement rebuilds dependencies after mutation renaming, and Triton
helper compilation derives host identity without launching a subprocess.

The runtime patch implementation contributes to both Ignition forward cache
keys and Torch's compiler cache tag. Changing it invalidates compiled forwards
and Torch graph caches. Captured cores execute independently of these Python
patches.

## Runtime core export

`ignition.preparation.runtime_core` hands live graph handles, typed storage
and execution metadata, tensor-view roles, compiler signatures, frontend assets,
and retained owners to `flare_heliopause`. Recipes describe file-backed table
ranges and generated frontend text without creating core files or payload
offsets. Heliopause records CUPTI resources, traverses graphs and pointer-bearing
storage, downloads retained bytes, serializes resources, and owns asset writes,
hashing, zstd level-3 executable compression, signing, conversion, rank payload
transfer, and publication. Each executable resource is one zstd level-3 frame
with a signed decoded length and compressed-chunk commitments.
Export capture copies each graph's nodes, argument bytes, attributes, and
annotated edges into native storage without instantiating an executable.
Ignition retains the external CUDA memory pools and captured kernel owners,
then resets the capture object to release its template. Diagnostic captures
retain executables; their post-instantiation hooks can copy the same metadata
before PyTorch releases their templates. Native export resolves allocations
and validates bindings against the copied metadata. Keep captured buffers,
contexts, copied metadata, pool owners, kernel owners, and any diagnostic
executables alive until native work confirms a drain; an unsuccessful drain
retains their owners.

Native transfer pools bound file buffers and pinned CUDA download storage from
available host memory and preparation rank count. Preserve direct bulk I/O,
exact unaligned tails, cache verification before reuse, and hash-before-write
publication. `--no-weights` must fail on any missing exact stored-data blob;
correctness and cache checks remain required. See
[runtime core transfers](runtime-core-design.md#native-export-transfers-and-progress)
for the transfer budget and progress contract.

`--reuse-weights` pins the cached default core and its weight objects through
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
`persistent_copy`, and `persistent_cat` place final tensors in a persistent
arena owned by `CoreStorage`, with aligned, disjoint storage descriptors.
Intentional aliases remain views of the same parent tensor. The arena grows its physical backing at the
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

## Serving temporaries and concurrent lifetimes

Recipes with compiler-planned temporaries allocate intermediate tensors inside
prepared forwards and pass them explicitly between components. Activations,
packed batch metadata, sampling intermediates, CSF expansion planes, and operator
workspaces follow this rule. Prepared components retain selected plans, weights,
geometry, and workspace requirements.
Their functional calls accept invocation tensors and return tensors or structured
tensor records.

Expose every temporary input, output, workspace, and alias to compilation.
Custom operators receive their destination and workspace tensors as arguments
and declare their reads and mutations. Allocate those tensors with ordinary
Torch operations in the forward. Inductor's static memory planner uses their
dependencies to reuse storage within a combined pool. A temporary allocated
inside an opaque operator cannot participate in that planning.
Tensors retained by constructors appear as external storage inputs to compilation.
Their owners keep those allocations alive across graph variants.

Constructors use example tensors to declare layouts and prepare kernels.
The [declaration tensor helper](../ignition/preparation/examples.py) shares examples
by semantic role, shape, dtype, and device. Give simultaneously live operands
distinct roles. Clear the example dictionary after binding finalizers complete
and CUDA work drains.
`ScratchArena` supplies preparation workspace for candidate tuning and binding
finalizers. After tuning and finalization, synchronize CUDA work and call
`arena.release_preparation()` before compiling forwards. Drop example tensors
from component fields, prepared argument tuples, closures, and retained priming
owners. Release callbacks discard library declaration buffers, including expanded
CSF scale examples. Keep immutable layout metadata needed to bind invocation
tensors.

Weights, rotary frequencies, fixed page indices, and scalar quantization
constants use [persistent tensor storage](#persistent-model-tensor-storage).
Native batch inputs, completion frames, request state, cache history, snapshots,
saved hidden vectors, image features, and registered transport buffers retain
explicit storage. Their lifetimes extend beyond one forward or include native
consumers outside its dependency graph. Write escaping results into that storage
before the graph completes.

For auxiliary-stream work, pass every live source, output, encoded plane, and
workspace through an explicit pending record. The fork waits for input
production. The join receives the pending operands and the main-stream result
that must precede it, then waits for the auxiliary stream. Custom operators
declare these dependencies so Inductor keeps storage live through the join.
`ScratchArena.register_parallel` sizes simultaneous preparation workspaces.
Serving overlap uses invocation-local tensors.

Capture serialized graph variants with the same CUDA graph pool through
`capture_forwards`. Prefill, decode, speculative, maintenance, and vision graphs
reuse that pool after all work from the preceding graph has joined. Every value
that survives a graph boundary belongs to explicit external storage. Retain the
pool, compiled kernels, graph executables, and native resources until replay
drains. CUDA libraries can retain allocations referenced by captured kernel
arguments without Python tensor owners. Preserve those resources through drain.

Recipes that retain graph-buffer pools keep their caller-owned scratch until
their serving forwards expose the required reads, writes, and dependencies.

The diagnostic `--no-compile` path executes the same prepared functional calls
during warmup and capture. CUDA graph capture retains their local allocations
without Inductor's lifetime planning. Native replay executes the captured work
without Python calls or steady-state allocations.

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
- **Release preparation temporaries after their consumers finish.** Retire loading,
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
  Kernel tuning admits candidates under one scratch limit shared by participating
  ranks, after reserving headroom, tuning growth, and pending persistent storage.
  Envelope sizing and candidate races use the same admissible configurations;
  concurrent branches retain their combined scratch requirement.

Compiled-forward warmup reuses one transient Torch pool across the graph catalog.
Torch can split and recycle its free blocks between synchronized forwards. The
pool is retired before native physical-retention references are released;
allocations with surviving owners keep their complete mapped segments.

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
  budgeted for their host's ranks and close when their local and borrowed work
  finishes, before ranks exchange completion results. Ranks on one host share its cache directory;
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
  Load, warm, freeze, and save one forward at a time in catalog order. Release
  its portable compiler state, generated source, bundled tuning metadata, and
  tuning registration after the cache write;
  retain canonical kernel selections, execution guards, live kernel owners, and
  bound storage. Standalone guards retain their full check tree and callbacks;
  discard the unused Dynamo diff tree and duplicate debug lists. Artifact loading
  omits guard construction stacks and Inductor diagnostic graph text and line maps;
  executable guard predicates, generated source, and launchers remain intact. Guard wrappers
  retain their invalidation managers; object finalizers refer to those managers
  weakly, so discarding every wrapper owner releases the guard tree and runtime
  globals even while guarded types remain alive.
  Complete warmup for the entire catalog before capturing any graph.
  After successful capture and any required instantiation, release the compiled
  Python wrapper and guards while retaining graph, kernel, and storage owners
  through drain. Grouped forwards release shared guards after their final capture;
  recipes exposing eager forwards retain guarded callables.

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
  least 4096 tokens. Test alternating graph variants in a shared CUDA graph pool and
  changing cache/recurrent state. Small-batch success does not establish stride,
  alias, or workspace correctness for prefill.

Follow [model development](model-development.md#speculative-decode) for width-one
speculative execution and [GPU performance work](performance-work.md) for complete
GPU-step measurements and repeated native replay checks.
