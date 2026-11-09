# Tensor-parallel ranks in one process

A tensor-parallel runtime core loads into one Helios process with one CUDA
context and execution thread per rank. Each rank owns a fixed device arena,
mapped host regions, modules, graphs, and replay streams. Peer buffers have
the same addresses during preparation and serving.

The [core design](runtime-core-design.md) defines serialization and
loading. The [core design](core-design.md) defines model contracts and request
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

Every preparation process reserves every rank's device arena and host range.
For rank `r`, each range spans 1 TiB:

| Region | Base |
|---|---|
| Device | `0x0200_0000_0000 + r × 2^40` |
| Mapped host | `0x0c00_0000_0000 + r × 2^40` |

The [preparation allocator](../heliopause/src/allocator.rs) serves these
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

One coordinated step is outstanding at a time. A 120-second completion
deadline bounds stranded collectives. Input frames, completion frames, graph
owners, and collective storage remain retained until every referencing
rank finishes. Captured GPU collectives perform the reductions; host
channels carry control and completion information.

## 4. Preparation diagnostics

[rank_group.py](../ignition/preparation/rank_group.py) starts one Python
process per preparation rank and connects the ranks through Gloo. Each
process owns its interpreter, tensors, and prepared kernels. Rank zero
distributes compiled forwards; every rank binds local resources, warms the
ordered catalog, and captures its graphs.

Heliopause's qualification interface can replay these in-process graph
bundles while their Python owners remain alive. The
[process coordinator](../helios/src/execution/parallel.rs) uses Unix sockets
for prepare, ready, commit, and completion messages. Its rank supervisor
terminates the group on peer failure or drain timeout. This diagnostic
lifetime ends before Python resource owners are released.

Production core loading reconstructs resources in Rust and uses rank
threads. No Python interpreter, peer process, or control socket participates
in request execution.

## 5. Failure and shutdown

Shutdown stops admission, drains all execution and transfer streams, and
joins rank workers before releasing graphs, peer mappings, and contexts.
A failed drain retains resources that may still be referenced by GPU work.
A fatal device fault or stranded collective requires process termination.

Process-image diagnostics require every rank thread to remain outside CUDA
driver calls throughout checkpoint and restore. The
[process-image constraints](runtime-core-design.md#7-process-images)
apply to all contexts in the process.

## 6. Driver probe measurements

`cr-probe p2p` from `~/projects/flare-research/probe`, driven by
`scripts/cr-single.sh` with `SUBCOMMAND=p2p`, `GPU` set to the UUIDs of GPU 8
and GPU 10, and `MODE=plugin` or `MODE=toggle`, once without and once with
`PROBE_ARGS=--threads`; results under `results/p2p-plugin-084920`,
`results/p2p-toggle-084927`, `results/p2p-threads-plugin-114431`, and
`results/p2p-threads-toggle-114435`. One process retains the primary context
of each device and enables legacy peer access in both directions. For each
rank it reserves 64 GiB at `0x7000_0000_0000` (rank 0) or `0x7100_0000_0000`
(rank 1) as the first address-space operation of that context, both honored;
maps a 16 MiB VMM buffer and a 32 KiB partial-sum buffer owned by the rank's
device with access descriptors for both devices; allocates a 16 MiB
`cuMemAlloc` buffer reachable through peer access; loads the probe kernels as
a library in that context; and creates a non-blocking stream and two timing
events. Each rank's graph is a stream capture of four `fill_pattern` launches
(1,024 blocks of 256 threads, 16 MiB each) into its own VMM buffer, its own
`cuMemAlloc` buffer, the other rank's VMM buffer, and the other rank's
`cuMemAlloc` buffer, each with a distinct seed. Twenty serialized rounds
launch rank 0's graph and then rank 1's, each timed with events and
synchronized, from one host thread that rebinds the context. Each of the four
buffers is then checksummed with `checksum_region` (4,096 blocks of 256
threads, order-independent 64-bit sum) once from the owning context and once
from the peer context, and the first 64 words of rank 0's VMM buffer are
compared against rank 1's seed, since rank 1's graph ran last. The process
then writes `ready`, is checkpointed and restored externally, reads `go`,
synchronizes both contexts, repeats the rounds and checksums, and finally
captures a new graph on rank 0 that fills a fresh `cuMemAlloc` buffer
allocated in rank 1's context, verifying its first 256 words.

With `--threads`, each rank also has its own thread. The thread binds the
rank's context once, launches the rank's graph twenty times concurrently with
the other rank's thread before the serialized rounds, blocks on a channel
receive through the checkpoint window, and after the restore launches twenty
more times without rebinding its context before the serialized rounds repeat.

All four probe configurations passed. In every run the VMM buffers sit at
`0x7000_0000_0000` and `0x7100_0000_0000`, the device footprint is 710 MiB,
the image is 1.5 GB, all eight checksums agree between the owning context and
the peer context, and all agree before and after the restore.

| Quantity | plugin | toggle | plugin, rank threads | toggle, rank threads |
|---|---:|---:|---:|---:|
| Serialized replay, median before → after | 0.6697 → 0.6708 ms | 0.6698 → 0.6700 ms | 0.6701 → 0.6695 ms | 0.6699 → 0.6699 ms |
| Concurrent replay per rank thread, before → after | | | 0.6796, 0.6796 → 0.6791, 0.6797 ms | 0.6796, 0.6796 → 0.6796, 0.6796 ms |
| First driver call after restore | 0.035 ms | 0.025 ms | 0.026 ms | 0.023 ms |
| New cross-device graph after restore | passes | passes | passes | passes |
| Checkpoint stages | dump 2.0 s, restore 2.7 s | lock 1.36, device-to-host 1.60, criu dump 0.17, criu restore 0.23, host-to-device 2.35, unlock 1.18 s | dump 1.08 s, restore 1.04 s | lock 0.47, device-to-host 0.63, criu dump 0.20, criu restore 0.16, host-to-device 1.09, unlock 0.35 s |

Reading a buffer from the peer context over PCIe yields the same checksum as
reading it from its owner, and a graph whose baked pointers cross the GPUs in
both directions replays unchanged in the same process and after an external
checkpoint and restore. Two rank threads launching concurrently, with both
graphs writing both GPUs, add about 1.5 percent to each graph's replay over
the serialized rounds. The threads' context bindings survive the checkpoint
and restore: the second concurrent pass launches without rebinding and reports
no error. Threads blocked on a channel receive during the window satisfy the
thread rule. No IPC handle, job file, or second process is involved.

The probe uses two GPUs and runs no spin-barrier collective. Four ranks, b12x
collectives inside the graphs, and fixed-address import of a shareable handle
in a second process are not covered by it.
