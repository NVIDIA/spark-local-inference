# Tensor-parallel ranks in one process

A tensor-parallel runtime core loads into one Helios process with one CUDA
context and execution thread per rank. Each rank owns a fixed device arena,
mapped host regions, modules, graphs, and replay streams. Peer buffers have
the same addresses during preparation and serving.

The [runtime core design](runtime-core-design.md) defines serialization and
loading. The [Flare core design](core-design.md) defines model contracts and request
scheduling. Preparation uses Python rank processes; core serving uses
Rust rank threads.

## 1. CUDA contexts and peer access

CUDA unified addressing gives device allocations, mapped host allocations,
and registered host regions distinct addresses in the process. A pointer
identifies storage; the allocation records its owning device.

A context owns streams, events, modules, functions, and graph executables.
`cuCtxSetCurrent` binds a context to the calling thread. Each execution thread
binds its rank's context before issuing CUDA work. Modules and functions are
loaded per context; Flare preserves one graph per rank.

The core allocator creates physical storage with `cuMemCreate`, maps it
with `cuMemMap`, and grants access with `cuMemSetAccess`, using one access
descriptor per reader device. Access applies to the mapping. Core serving
uses these VMM grants; it does not import legacy CUDA IPC handles.

Peer traffic crosses the PCIe fabric according to device topology. Process
boundaries do not change that route. The b12x PCIe barriers use system-scope
stores to peer signal slots and loads from local slots; their atomics operate
on local memory. Preparation must validate that each collective's worker grid
can remain resident on every participating GPU. The collective transport and
its signal and payload buffers stay alive until all ranks drain.

## 2. Address layout and preparation

Every preparation process reserves every rank's device arena, and the
allocator assigns each rank a disjoint host range and reports its base to
Ignition. For rank `r`, each range spans 1 TiB:

| Region | Base |
|---|---|
| Device | `0x0800_0000_0000 + r × 2^40` on Linux; `0x0400_0000_0000 + r × 2^40` on Windows |
| Mapped host | `0x0c00_0000_0000 + r × 2^40` when every rank's device range ends at or below `0x0c00_0000_0000`; otherwise, on Linux, `0x2000_0000_0000 + r × 2^40` |

On Linux the device ranges of up to four ranks fit below the first host base;
larger rank counts use the second. Initialization fails when the device and
host ranges would overlap. The
[preparation allocator](../heliopause/src/allocator.rs) defines these bases
(`BASE`, `HOST_BASE`, `host_base`) and serves these
disjoint arenas. Owners export VMM physical allocations as POSIX file
descriptors. Peers import the handles, map them at the owner's recorded
address, and grant their own device access. A preparation rank therefore
captures the addresses that the serving process will use.

Collective pointer tables, kernel parameters, and opaque tensor maps can
retain their captured bytes. Peer allocation records identify the owner;
only the owner stores a payload. The loader reserves all address ranges
before mapping storage and checks that every requested address was honored.
Address collisions fail startup.

N-gram tables are stored once in the core. A shared reader gathers rows
for all ranks; each rank copies them into its own compact GPU buffers.
Portable mapped staging and readiness flags are shared, while request
mailboxes belong to their rank's context. RAM and disk table storage use
the same rank graphs.

## 3. Loading and rank coordination

[NativeModel](../helios/src/runtime_core/model.rs) validates GPU count,
architecture, rank layout, peer dependencies, and selected graph coverage
before serving. Rank loading stages allocate fixed storage, load payloads,
load kernels, and instantiate and upload graphs in parallel, joining rank
threads between stages. Runtime-sized rows and pages use a common capacity
that fits every rank. Payload staging has a process-wide bound.

The scheduler coordinates rank zero and peer execution threads through
[the threaded executor](../helios/src/execution/threaded.rs). A step carries
one fence, graph selection, grammar state, and upload set. Each rank validates
its inputs before commit, launches its graph, and transfers completion
records. The coordinator retires the fence only when all ranks agree on
request epochs, consumed-token counts, and status. Rank zero delivers
sampled output.

At most two coordinated steps are outstanding. A second step is admitted only
after the first has been committed, and only when both are ordinary
decode or complete speculative steps without grammar state; every other
operation requires an empty window (`can_submit_pending` in
`helios/src/execution/threaded/protocol.rs`). A 120-second completion
deadline bounds stranded collectives. Input frames, completion frames, graph
owners, and collective storage remain retained until every referencing
rank finishes. Captured GPU collectives perform the reductions; host
channels carry control and completion information.

## 4. Preparation diagnostics

[rank_group.py](../ignition/preparation/rank_group.py) starts one Python
process per preparation rank and connects the ranks through Gloo. Each
process owns its interpreter, tensors, and prepared kernels. Each forward
variant compiles once, on a rank that selects it; ranks on one host share its
forward cache directory, and ranks on other hosts receive the artifacts they
lack ([tensor-parallel compilation](preparation-development.md#tensor-parallel-compilation)).
Every rank binds local resources, warms the
ordered catalog, and captures its graphs.

Heliopause's qualification interface can replay these in-process graph
bundles while their Python owners remain alive. The
[process coordinator](../helios/src/execution/parallel.rs) uses Unix sockets
for prepare, ready, commit, and completion messages. Heliopause's
[rank supervisor](../heliopause/src/processes.rs)
terminates the group on peer failure or drain timeout. This diagnostic
lifetime ends before Python resource owners are released.

Loading a core whose ranks share one host reconstructs resources in Rust and
uses rank threads (`RankPeers::Threads` in `helios/src/execution/ranks.rs`).
No Python interpreter, peer process, or control socket participates
in request execution. Ranks on other hosts run in their own Helios processes
and use the process coordinator over TCP control sockets
(`RankPeers::Processes`); see [multi-host serving](multi-host.md).

## 5. Failure and shutdown

Shutdown stops admission, drains all execution and transfer streams, and
joins rank workers before releasing graphs, peer mappings, and contexts.
A failed drain retains resources that may still be referenced by GPU work.
A fatal device fault or stranded collective requires process termination.

Process-image diagnostics require every rank thread to remain outside CUDA
driver calls throughout checkpoint and restore. The
[process-image constraints](runtime-core-design.md#7-process-images)
apply to all contexts in the process.

## 6. Driver behavior the design depends on

One process that holds a context per device, with each VMM mapping granted
to both devices, has these properties on two GPUs:

- A buffer read from the peer context over PCIe has the same contents as
  when it is read from its owning context.
- A captured graph whose recorded pointers address storage on both devices
  replays unchanged, both under serialized launches from one thread that
  rebinds contexts and under concurrent launches from one thread per rank.
  Concurrent launches by two rank threads, each graph writing 16 MiB to
  buffers on both GPUs, lengthen each graph's replay by about 1.5 percent
  over serialized launches.
- After an external process checkpoint and restore, fixed addresses, buffer
  contents, and graph replay are unchanged. Rank threads keep their context
  bindings and launch without rebinding, and a graph captured after the
  restore can address a buffer allocated in the peer context.
- A rank thread blocked on a channel receive through the checkpoint window
  satisfies the process-image thread rule.
- No IPC handle or second process is required.

Not qualified: four ranks, graphs that contain b12x spin-barrier collectives,
and fixed-address import of a shareable handle in a second process.
