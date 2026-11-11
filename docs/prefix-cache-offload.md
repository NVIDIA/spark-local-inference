# Tiered prefix cache offload

Helios keeps immutable prefix checkpoints in GPU memory and evicts them under
pressure. Host retention stores complete model checkpoints and reusable CPU
input records in memory, on disk, or in both tiers. Checkpoint disk transfers
use bounded registered host staging; CPU input records use a separate bounded
ordinary-memory window. A GPUDirect Storage transport is designed, not
implemented (section 2.6).

Every checkpoint tier stores immutable page chains and a complete
endpoint row across all ranks. Recipes define the bytes needed to resume;
storage moves those bytes without interpreting the model. A completed restore
publishes an ordinary GPU checkpoint, so branching, continuation, and retirement
use the GPU cache's semantics. Scheduler-owned transfer tickets retain
buffers independently of the requests waiting for them.

Common host file I/O, memory mapping, scheduling-priority and filesystem
capacity helpers live in `eos/` (`flare-eos`). Helios, Selene and its manager use Eos;
Helios does not depend on Selene for non-cache OS operations. CUDA library
loading and terminal sizing remain in `helios::os`. Eos has Unix and Windows
implementations. The sections below name the Linux mechanisms (`mmap`,
`O_DIRECT`, `io_uring`, `flock`, `fallocate`, `SCHED_BATCH`); on Windows the
same paths use `VirtualAlloc`, unbuffered overlapped files on an I/O completion
port, an exclusively opened `.lock` file in the record directory, and
below-normal thread priority, and record files are not preallocated.

The host tiers themselves live in the `selene/` crate (`flare-selene`): the
`HostStore` trait, the index, slot pool, record format, disk thread, the
memory, disk, and composite stores, CPU input records, and the pool client.
Selene has no CUDA dependency. `helios/src/offload/` supplies the CUDA side:
the per-rank transfer threads (`transfer.rs`), host-memory registration
(`registrar.rs`), and the glue that joins rank processes (`shard.rs`) and
starts the pool at load time (`pool.rs`).

Host offload requires explicit configuration: `flare run` retains no host
records unless a memory size, disk directory, or pool is configured. Derived
default sizes and a default record directory are designed, not implemented
(section 2.11). Host offload does not change the contracts of preparation,
captured graphs, or the runtime core format. The
[core design](core-design.md#4-prefix-checkpoints-and-complete-model-state)
defines checkpoint contents and ownership; the [CLI design](cli-design.md)
defines configuration precedence.

## 1. Scope and configurations

- Offered GPU checkpoints are copied to the first host tier in local and pool modes.
  A later matching request restores a retained checkpoint the GPU no longer holds.
- Live request state, in-flight speculative rounds, and anything not published
  as a checkpoint are never offloaded.
- Every model contract is covered through the core's runtime-sized
  allocation groups. The offload path contains no model-specific code.
- Offload work never blocks the model worker thread or the model stream.
  Section 2.10 states the isolation mechanisms and section 4 the measurements
  that gate derived defaults.

The local tiers form a `MemoryStore`, a `DiskStore`, or a `CompositeStore` of
both; the optional pool attaches to whichever is configured.

`--prefix-cache-memory SIZE` (or `[prefix_cache] memory`) is the single warm
host-memory setting. It defaults to zero; zero disables warm retention while
leaving configured disk and pool tiers available. With no local tiers or pool,
including an explicit memory size of zero, offload is disabled and startup
omits the `Prepare host prefix cache` phase. When a disk directory is
configured but yields no store, because it cannot be created or its limit
resolves to zero, and no other tier is configured, that phase reports
`disabled` with the reason.

| Warm memory | Local disk | Pool | Behavior |
|---|---|---|---|
| 0 | none | none | Disabled. |
| > 0 | none | none | `MemoryStore`; retain checkpoints in host memory. |
| 0 | enabled | none | `DiskStore`; stream through automatic intake. |
| > 0 | enabled | none | `CompositeStore`; retain warm records, spill under pressure. |
| 0 | none | enabled | Pool only; the manager's arena supplies intake, with no local slot pool. |
| > 0 | none | enabled | Warm memory, then pool on retirement. |
| 0 | enabled | enabled | Local disk, then pool on retirement; automatic local intake. |
| > 0 | enabled | enabled | Warm memory, then local disk, then pool on retirement. |

All combinations apply on one host. Ranks on separate hosts support zero warm
memory with disk and/or pool; nonzero warm memory is refused because that tier
holds every rank's bytes in one process. Each disk peer uses its host's intake.

Automatic registered intake is additional to the warm budget whenever warm
memory or local disk is enabled. It needs no setting: its slot count is
`2 × max_active × row_slots + 2 × WINDOW_PAGES`, where `WINDOW_PAGES = 64`.
The warm checkpoint budget rounds down to whole slots. If host headroom cannot
hold the full intake, it shrinks toward two complete windows,
`2 × (64 + row_slots)`, and one warning reports the memory needed for full intake.
Prepared image inputs use one eighth of the warm budget, rounded down to 4 KiB,
in ordinary CPU memory; the rest holds checkpoint slots. Text-only cores use
all warm bytes for checkpoints. CPU input serialization also has a separate
bounded window derived from prepared image buffers, including with warm zero.

With zero warm memory and disk enabled, intake uses about 0.5–8 GB of host
memory; the amount depends on checkpoint geometry and selected concurrency.

On an integrated-memory device, such as a DGX Spark, GPU memory and host
memory are one physical pool, and the GPU prefix cache takes the
memory that `capacities` leaves after weights and live state. A retained
memory store there would duplicate bytes within the same pool, so disk-only
retention avoids that duplication. Its intake exists because direct I/O needs a
CPU-mapped destination and the page planes are mapped only on the GPU side.
As designed, the GPUDirect Storage transport of section 2.6 removes payload
staging; its indexes and I/O control structures occupy host memory.

Host checkpoint memory uses equal page-record-sized slots. Rows occupy several
slots (for example, 23 for DeepSeek and 135 for Qwen TP1), with each rank's row
pieces in its share of each slot. A row smaller than a slot uses a whole slot.
Slot placement does not affect disk record bytes or checksums.

## 2. Design

### 2.1 Checkpoint bytes

Each rank bundle declares runtime-sized allocations with a `Sizing` record in
`helios/proto/runtime_core.proto`: exactly one in the `STATE_ROWS` group and
one or more in the `KV_PAGES` group. Every `KV_PAGES` allocation on a rank is
indexed by the same physical page id, so one logical page occupies `unit_bytes`
at `base + id × unit_bytes` in each of them. These allocations are called page
planes below. The `STATE_ROWS` allocation holds one row per state slot. A GPU
checkpoint in `helios/src/cache.rs` owns its tail pages, meaning the complete
pages since its parent, and one snapshot row. The recipe stores everything else
that defines the endpoint inside the row: the partial tail page, recurrent
matrices, convolution histories, raw selector rings, sliding windows, the last
hidden vector, and the consumed count.

The complete bytes of a checkpoint are therefore:

- for each rank, for each tail page id, for each page plane, `unit_bytes` bytes
  at that id;
- for each rank, the row at the snapshot's state slot.

Neither contains physical ids. Restore and snapshot graphs take page tables and
slot ids from Rust uploads, and the row validates its own consumed count
against the expected count supplied at restore. Pages and rows can be restored
into any free ids. Qwen has 36 page planes (key, value, and selector for each
of 12 sparse-attention layers) plus draft planes; GLM has latent and index
planes per indexed-attention layer plus draft planes; DeepSeek has latent and
index planes per cache source layer. Draft planes at the first chain page hold
an unused draft page and are copied regardless. Allocations with `PEER` memory
belong to another rank and are excluded, so each byte is copied once under
tensor and context parallelism.

`CacheGeometry` in `selene/src/geometry.rs`, one per rank, records the page
token count, the row plane (name, address, `unit_bytes`), the page planes in
name order, the row count, the page count, and the page ranges that share
physical bytes with snapshot rows. `allocate_caches` in
`helios/src/runtime_core/memory.rs` collects it before consuming the sizing
records, and `execution_manifest` in `helios/src/runtime_core/serving.rs`
attaches it to the rank's `ExecutionManifest` as `cache: Option<CacheGeometry>`.

A layout fingerprint is the 16-byte BLAKE2b digest of the rank count, the page
token count, and for each rank the ordered page plane names and unit sizes and
the row plane name and unit size. The disk namespace combines this fingerprint
with a 32-byte identity derived from the core's model identity and the
selected continuation-state mode: ordinary execution, learned speculation,
or copy speculation. `offload_identity` in
`helios/src/runtime_core/model.rs` computes that identity with BLAKE2b and
the domain `flare-prefix-state-v1`. Ordinary execution omits draft-state
updates, so its records cannot be loaded into speculative execution even when
the physical layouts match. Changes to positive draft limits, concurrency,
and verification policy that retain speculation do not change the state mode.
Permanently fixed width one selects ordinary execution. Token-prefix
keys retain the core's model identity within this storage namespace.

The fingerprint is required because model identity does not encode every
row-layout parameter.
Preparing draft history can add row regions and page planes. Cores that
share a model identity can therefore have incompatible checkpoint bytes.
Unit sizes do not depend on the concurrency or verification width selected
at load time, so the fingerprint is stable across `flare run` options for
one core.

### 2.2 Store interface

The worker uses Selene's `HostStore` trait from `selene/src/store.rs` directly.
Its core operations are:

```rust
pub trait HostStore {
    fn lookup(&mut self, history: &dyn History, resident: u64, before: u64) -> Option<HostHit>;
    fn store(&mut self, checkpoint: &PublishedCheckpoint) -> Result<StoreOutcome>;
    fn fetch(&mut self, hit: HostHit, ticket: InboundTicket) -> Result<()>;
    fn poll(&mut self, events: &mut Vec<StoreEvent>) -> Result<()>;
    fn stats(&self) -> StoreStats;
    fn close(&mut self) -> Result<()>;
}
```

The worker calls `store` for each offered checkpoint, without waiting for GPU
pressure. The scheduler offers message checkpoints immediately, including those
followed by a prompt checkpoint less than one physical page later. A continuation
can branch at the message endpoint and cannot use the prompt checkpoint's row.
Terminal checkpoints are deferred until a request resumes at that exact
checkpoint. A preempted terminal counts as resumed. GPU captures, prefill cuts,
retention, and GPU eviction policy are unchanged; deferred terminal points can be
lost to GPU eviction before reuse, and are not announced to the pool until offered.
Compaction also offers a deferred branch point as its heir and refreshes
descendant offers with their new parent.
`Queued` holds one GPU pin until `Stored` or `Deferred`; `Present`
requires a committed local copy. `Retry` accepts no copy. The worker retains only
checkpoint IDs for retries after store progress and discards IDs that have left
the GPU cache. `Deferred` releases the copy pin and schedules the same retry when
a queued publication cannot obtain host capacity. A failed transfer releases its
pin without retrying. GPU eviction uses the GPU cache's LRU, leaf and compaction
rules. `fetch` copies a looked-up chain into the ticket's page IDs and state slot.
Submissions never wait; `poll` drains completions and failures.
A `PublishedCheckpoint` names the GPU checkpoint, its key, its nearest offered
ancestor's key, its bucket digest, every page of
its chain from the root with the tail count since that offered ancestor,
and the state slot of its row. `MemoryStore`, `DiskStore`, and `CompositeStore`
implement the trait for serving, `PooledStore` wraps them when a pool is
attached (section 2.15), and a test double implements it for worker tests. The
worker holds `Option<Box<dyn HostStore>>`. The trait's remaining methods have
default bodies: `reused` refreshes a retained record when a request reuses its
GPU checkpoint, and the export, eviction-notice, and pool-lookup hooks of
section 2.15 do nothing unless a pooled store overrides them.

Three components are shared by the implementations:

- `Index`, the in-memory tree of section 2.3. Every store owns one.
- `SlabPool`, one pool of equal page-record-sized slots, section 2.5.
  The memory store and `Staged` share its mapping, registration and free list;
  there is no separate staging pool.
- `Transfers`, the per-rank copy threads of section 2.5. Each tier owns its
  transfer queues; a composite has independent memory and disk transfer sets.

### 2.3 Index and lookup

The index is a tree keyed by exact-endpoint `PrefixKey`, with the same shape
as `PrefixCheckpointCache`: each node has a parent key, a child count, a tail
page count, a location in its store, a transfer state, a pin count, and an
access tick. It is owned by the model worker thread, single-writer, with no
locks. A bucket map from the digest of the aligned 64-token boundary at or
below each endpoint to the node keys ending in that block supports lookup.

Lookup takes the request's `PrefixHistory` and the consumed count `a` of the
deepest ready GPU checkpoint. It walks the history's aligned boundaries from
the longest down to the block containing `a`, including the root bucket for
endpoints inside the first block, probing the bucket at each one. A candidate
node qualifies when its row exists, its consumed count `c` exceeds `a`, and
`history.key_at(c)` equals the node's key, which costs one hash of at most 64
tokens. The candidate is feasible when walking its parent links collects
every node whose consumed count exceeds `a` and stops at a parent at or below
`a`, or at the root. The result is that chain in ascending order; its tail
page counts sum to `c / page_tokens - p / page_tokens` for the stopping
parent's count `p`, and the restore skips the first `a / page_tokens - p /
page_tokens` of them, leaving exactly the pages `publish` requires. The first
feasible candidate wins. The deepest hit is usually within the last few
blocks because a new user message is short relative to the history, so the
probe count is about the new tokens divided by 64.

Page content is determined by the token prefix ending at the page, so a
suffix of a host chain's pages combines correctly with the GPU ancestor's
pages, in the same way a GPU checkpoint's pages combine with a child computed
later from a different request. A checkpoint published while its parent key
is absent from the host index is stored as a root record holding its complete
page chain, so a skipped or evicted ancestor never makes its descendants
unrestorable.

Eviction follows `PrefixCheckpointCache::evict`: remove the least recently
used unpinned leaf; when every leaf is pinned, compact the least recently used
unpinned single-child ancestor by dropping its row and keeping its pages. A
compacted node stays in the tree as page-only, no longer matches lookups, and
continues to supply chain pages. A node is evictable only when its store's
policy allows it: the memory store standalone evicts any unpinned node, the
disk store evicts any unpinned node, and the composite restricts the memory
store to durable nodes.

Disk drops a non-root record's row once its only child commits, keeping the
pages, unless a child less than one physical page later has been indexed.
Those close ancestors retain their rows for divergent continuations, even if
that child is later removed. The close-child marker is in-memory; recovery
reconstructs it from surviving parent-child links. DRAM retains close-ancestor
rows during redundant-row thinning; ordinary capacity-driven record eviction
and compaction still apply. Other redundant rows remain in DRAM while its
free retention space exceeds two continuation rows. Reserved transfer intake
does not count as retention headroom. When that threshold is reached, or an
incoming record needs more space than is free, DRAM reclaims eligible
redundant rows before evicting other records. Readers and spills delay row
removal until their pins are released. DRAM releases row slots, disk truncates
at the row segment, and later spills omit removed rows. A rewind or later
fork restores its retained endpoint when available; otherwise it restores the
nearest row-bearing ancestor and re-prefills the remaining tokens.
With lazy write-behind, a crash or an incomplete shutdown drain can lose the
DRAM-only descendants of a page-only disk prefix, leaving recovery to an older
row-bearing ancestor. Startup prunes childless page-only records and any
page-only ancestors they leave childless.

An outbound for a key already present or in flight is skipped. An inbound for
a key already in flight attaches the request as another waiter.

### 2.4 The three stores

`MemoryStore` owns an index, a slab pool with retention, and a transfer
handle. `store` allocates slabs, evicting under the tree policy when the pool
is full, and copies the checkpoint out; the record stays resident until
evicted. `fetch` copies resident slabs in. Lookup hits are always resident.

`DiskStore` owns an index rebuilt from its directory at startup, a file
backend, and a transport. The transport is the `DiskTransport` trait in
`selene/src/disk_store.rs`, implemented by `Staged` and, for ranks on separate
hosts, by `shard::Sharded` (section 2.14):

- `Staged` shares the host slot pool with the memory store and owns an
  independent transfer handle and the disk thread. `store` streams checkpoints
  through intake windows into record files; `fetch` streams them back to the GPU.
  With warm zero and disk enabled, the pool contains only intake slots.
  `write_from(key, metadata, buffers)` and `read_into(key, buffers)` move records
  between files and caller-owned registered slots for the composite.
Section 2.6 specifies a second transport, `Direct`, which is not implemented.

`CompositeStore` owns a `MemoryStore` and a `DiskStore` and implements the
trait by orchestrating them:

- `store` with the staged transport runs `memory.store`; when the copy-out
  lands, local and pooled instances both park the record until the clean floor
  or allocation pressure needs its slabs. `write_from` writes those same memory slabs.
  Completion makes the memory copy durable. If disk reservation fails, the
  complete record stays parked in DRAM, restorable and announced there, with
  its parent pinned until reservation succeeds. Its chain metadata is indexed
  without charging file space. If available and evictable memory slabs cannot
  hold the record, `disk.store` streams it through bounded staging without
  retention. Indexed writes become usable when their bytes are complete.
- `lookup` runs on the disk index, including uncharged parked metadata and
  records whose writes are in flight. A complete DRAM copy can supply them.
- `fetch` walks the chain. A node resident in the memory store uploads from
  its slabs. With the staged transport, a node that is not resident gets slabs
  from the memory store, evicting durable unpinned residents if needed, is
  read with `read_into` into those slabs, and uploads from them; it is then a
  durable memory resident like any other. An ancestor whose endpoint shares
  its parent's last page owns no pages, so when it is not the endpoint nothing
  of it is read and an empty resident copy stands in for it. When the chain
  exceeds available and evictable retention slabs, lookup selects a complete
  disk chain and `disk.fetch` streams it through bounded staging directly to
  the GPU destinations. Such a restore neither requires the chain to fit in
  RAM nor evicts memory residents merely to attempt an impossible admission.
- The memory store's eviction is restricted to durable records. Disk
  removal also removes the memory copy. A parked copy is never evictable until
  its file commits. Filesystem-full or quota errors retain it and retry with a
  short backoff. Terminal disk I/O errors may discard it and are counted.
- The staged transport and memory store share slots. Warm slots supply spill
  and read-through buffers when retention fits; reserved intake handles records
  and chains that cannot be retained. Intake is additional to the warm budget.

### 2.5 Transfers, slabs, and record files

`Transfers` in `helios/src/offload/transfer.rs` runs one thread per rank. Each
thread makes its rank's context current through
`Stream::open_lowest_priority`, owns one stream at the lowest priority and one
event, and receives jobs over a channel as page id lists, a state slot, and
slab addresses. The thread expands them into device and host addresses from
its rank's geometry, so descriptor construction never runs on the worker
thread. It issues batched copies through `cuMemcpyBatchAsync`, records its
event, created with blocking synchronization so the wait sleeps instead of
spinning a core, waits on it, and replies with the job id and rank. Each batch
call
carries at most 16 MiB and at most 4,096 entries. These bounds limit submitted
work; section 4 measures the resulting delays on shared copy engines. A
failure is reported only after the thread has synchronized its stream, so
destination ids are quiescent when the scheduler frees them. Host-observed
completion, through the event wait and the channel message, orders later
model-stream work after the copies; the model stream never waits on an
offload event.

Transfers move records in windows of at most 64 pages plus the multi-slot row.
The pool contains warm checkpoint slots plus automatic intake slots. It is one
4 KiB aligned anonymous mapping (`MAP_PRIVATE`, `MAP_ANONYMOUS`, `MAP_POPULATE`,
advised for transparent huge pages), registered once at startup. Rows use each
rank's page share across as many slots as needed; their pieces need not be
contiguous. CPU readers traverse pieces in rank order, preserving disk bytes
and streaming xxh3 checksum values. After copy-out, Selene compares local ranks'
**unpadded** plane and row bytes. Each record has two decisions: all of its pages
share one representation, and its row has an independent representation. Identical
shares pack into existing slots and unused slots are released; differing pages
keep the entire page segment per-rank. A single-rank store makes no comparison.
Restores fan one stored
share out to every local rank. Disk spills use DRAM's decisions without comparing
again, then hash the stored bytes once. Disk-only stores decide from the first
completed page window; a later differing page abandons that store after its jobs
drain, without publishing a partial or mixed record.

Memory-only TP>1 split pools are per-rank because their slot identities are not
CPU-readable. Pool records are also per-rank, including disk files, checksums
and announcements. Memory stores containing replicated records refuse pool
announcements; recovered replicated files stay local. `/metrics` reports packed
and per-rank record stores by tier, plus abandoned disk-only stores. A record
counts as packed when its pages or row are replicated; promotions count as a new
DRAM store. These decisions do not reduce GPU copy-out traffic.

Memory allocation keeps at least `intake_slots` free slots by evicting the
coldest ready, unpinned, durable records that, in pool mode, are not announced.
Maintenance starts writes of
the coldest dirty records, or retirements to the pool, when `free + clean` falls
below `2 × intake_slots`. Memory-only records are already durable. Maintenance
never waits for `Staged`; independent memory and disk copier queues share the
pool. No more than twice the intake plus one largest record is written ahead
of need, so a large warm tier does not eagerly write its contents to disk.
Neither publication nor admission waits for I/O. GPU-only retries remain
unpinned so they cannot prevent admission. If a copier cannot drain at close,
the mapping is leaked rather than reused under outstanding copies.

Registering host memory maps it into every device's page tables. On an RTX PRO
6000 that costs each device about 2 MiB of device memory per GiB registered,
126 MiB for 64 GiB, whichever ranks copy the bytes. A memory store without a
disk tier or pool sharing never reads records on the CPU, so its pool is split
by rank instead. Each rank's share of all slots is one
`cuMemCreate` allocation of pinned memory on the host NUMA node closest to the
rank's device, mapped into one reserved range and opened by `cuMemSetAccess`
to that device and the host only. Each device then pays only for its own part:
on two GPUs an 8 GiB pool costs each 8 MiB instead of 16 MiB. Slab addresses
stay identities, where each slab would begin in a pool of whole cross-rank
records inside the reserved range, so the memory store and its index address
slabs the same way in a split pool and a registered pool; each transfer
thread holds its rank's `RankPart` and places a
split slab's planes at their offsets relative to the rank's first plane.
Composite and disk stores, whose disk threads read and write whole records,
keep the registered mapping. Cache sizing leaves each device 1/512 of the
host memory it maps, its share of a split pool or the whole registered pool
including intake, plus 16 MiB for transfer streams.

A record file lives at
`<directory>/<identity prefix>-<layout fingerprint>/<consumed>-<digest>.ckpt`,
with the consumed count as 16 hexadecimal digits and the digest as 32. The
first 4 KiB is a little-endian header: magic `FLARECKP`, format version, flags,
the storage namespace identity, the layout fingerprint, the key (consumed count and
digest), the parent key, the bucket digest, the page token count, the tail
page count, the rank count, the per-rank plane sizes and row bytes, the page
segment offset, the row segment offset, the row checksum, and the creation
time. Versions 1 and 2 encode each plane size as `u64`; version 3 uses `u32`
plane sizes so wider rank/plane catalogs fit the same header block. Row sizes
remain `u64`. New records use version 3 when every plane fits `u32`, otherwise
the `u64` format is retained. All encodings still require the header to fit 4 KiB.
Versions 2 and 3 use flag bit 31 for a replicated page segment and bit 30 for a
replicated row; version 3 also permits non-replicated records. Page offsets use
one fixed stride for the whole record. Version-1 and version-2 records still
recover and preserve their encoding on rewrite; cross-rank agreement compares
canonical header fields independently of the file's encoding version. The
layout fingerprint is unchanged. A binary that does not support a record's
version treats it as invalid and **deletes it during recovery**; downgrading
therefore starts those records cold. A checksum segment follows,
one 64-bit xxh3 over each stored page record, padded to
4 KiB, so a restore that reads only a suffix of the pages verifies each page
it read. The page segment follows, page-major: page `i`, then rank `r`, then
planes in geometry order, with every plane padded to 4 KiB. The row segment
follows, rank-major, with every row padded to 4 KiB. Per-plane padding is
what lets the `Direct` transport of section 2.6 address each plane at an
aligned file offset,
so one file format serves both transports; the cost is small because the
large planes are multiples of 4 KiB and only small index planes pad.
The row exists when the file length covers the row segment; compaction
truncates the file at the row segment offset and changes nothing else. Writes
go to a temporary name in the same directory (preallocated to its final length
with `fallocate` on Linux, so direct writes overlap), use direct I/O, optionally
`fdatasync`, and rename into place. Deletion is `unlink`. Checksums are
64-bit xxh3 over each page record and over the row record to detect torn or
corrupted files; the staged transport computes and verifies them on the disk
thread.

The loader acquires an exclusive nonblocking lock on the record directory
before scanning it. The store retains that directory descriptor through all
I/O and shutdown. Another process using the same namespace fails startup
before scanning or deleting files; independent namespaces can share a base
directory. The disk store's startup scan reads each file's header, validates the magic,
version, identity, fingerprint, and name, derives row presence from the size,
seeds the access order from the modification time, deletes temporary files,
and deletes any node whose parent is missing together with its descendants.
Leaf-first eviction never unlinks a parent before its children, so orphans
arise only from interrupted operations or external changes.

A process killed mid-restore releases its directory lock without changing
committed record files; the next scan can reuse those files. Publication and a
completed GPU-to-RAM copy
do not imply disk durability: files commit under pressure or an enabled shutdown
drain, and SIGKILL loses RAM copies and queued writes. Only records renamed
into place survive a process restart; partial temporary files are discarded.
Waiting after restart cannot recover a write the killed process never committed.
A RAM record reserves a disk-index entry and disk budget, but its
file is written only under memory pressure or an enabled shutdown drain. The disk-record
count includes these reservations; it does not count only committed files. An
idle wait does not flush them. A default clean stop or crash loses these RAM-only records;
records whose file commits completed remain recoverable.

`--prefix-cache-drain-on-shutdown[=DURATION]` saves DRAM-only records to
local disk for a warm restart. It is off by default because larger DRAM tiers
extend stop time. A bare flag drains until done; `=90s` limits new writes, and
`=off` disables a configured drain. `[prefix_cache] drain_on_shutdown` accepts
`"on"`, `"off"`, or a positive duration in ms, s, m, or h (bare numbers are seconds).
Recent endpoints drain first, after their required ancestors; committed files
are preserved. Limits and write failures do not fail shutdown; accepted transfers
still drain before memory is released. Size a container's stop grace as Flare's
usual stop time plus DRAM bytes ÷ measured disk write rate.

Host record expiry is off by default. Enable it with
`--prefix-cache-max-age DURATION|off` or `[prefix_cache] max_age = "2h"`;
positive durations accept ms, s, m, h, or bare seconds. `off` disables expiry.
Publication, reuse, restore and child publication refresh the record and its
required ancestors in memory only. Recovery uses file mtime, from the last write
when a record was spilled cold or drained. After a restart, a record reused from
disk but never rewritten may expire up to one max age early. The cost is a
recompute, never wrong output. With expiry off, hits do not refresh last use;
enabling it later may expire old files on the first sweep.
In pool mode, instance hits do not refresh manager copies, whose max age counts
from their last put or restore; this is acceptable for a cold-tier copy because
an instance that keeps the prefix hot re-cascades it on eviction.

At most once a minute, the model-thread poll walks from the oldest entry,
skipping interior nodes and records that cannot be dropped, and stopping at
the first fresh leaf or 64 expired leaves. Expired leaves are dropped outright;
they are never sent down the cascade. Page-only ancestors are removed with their
last child; in-flight copies and exports retain their storage. LRU positions change only for the accessed key.
A fresh former interior node left as a leaf by pressure eviction can delay a
sweep by up to one max age. The manager sweeps every namespace on its store
tick, using each index's own once-a-minute limit. Expiry does not apply to the
GPU tier.

The disk thread in `selene/src/disk.rs` owns an `io_uring` ring over files
opened with `O_DIRECT`,
chunks transfers at 8 MiB, and computes and verifies checksums. Before accepting
payload reads,
it checks the opened file's header against the indexed record, including all
layout and size fields. Changed headers cannot enlarge reads into registered
slabs. A failed ring submission drains every kernel-owned buffer reference
before reporting completion; shutdown retains the worker and its buffers if
that drain cannot finish before the deadline.

`helios/src/cuda.rs` binds, for offload: `cuMemcpyBatchAsync`, resolved as the
driver symbol `cuMemcpyBatchAsync_v2`, with the CUDA 13
signature (destinations, sources, sizes, count, attributes, attribute indices,
attribute count, stream) and one attribute whose source access order is
stream-ordered; `cuMemHostRegister_v2` with the portable flag;
`cuMemHostUnregister`; `cuEventSynchronize`; `cuStreamCreateWithPriority`;
`cuDeviceGetAttribute` for the integrated-memory check and the host NUMA node,
which the loader's driver binding also provides; and `cuMemAddressReserve`,
`cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease`,
`cuMemAddressFree`, and `cuMemGetAllocationGranularity` for split pools.

### 2.6 GPUDirect Storage transport

Designed, not implemented. The repository contains no cuFile bindings, no
`Direct` implementation of `DiskTransport`, and no flag or configuration key
that selects a transport; every disk record moves through `Staged`. This
section specifies the design.

`Direct` implements `DiskTransport` over `libcufile.so`, bound through
`libloading` the way `helios/src/cuda.rs` binds the driver, in
`helios/src/offload/cufile.rs`. The library loads lazily, so a host without it
is unaffected. FlashInfer's b12x weight loader
(`flashinfer/experimental/b12x/loader/_gds_reader.c`) uses the same API and is
the reference for call patterns. The record file format of section 2.5 serves
both transports.

- Startup probe: `cuFileDriverOpen`, then `cuFileDriverGetProperties` to confirm
  the direct path. The probe reports whether the direct path is available and,
  if not, why. Compat mode, in which cuFile bounces through its own host
  pool, is refused for serving because it is an uncontrolled host copy; the
  store reports the reason and uses the staged transport instead. The direct
  path requires the `nvidia-fs` kernel module, a filesystem cuFile supports with
  `O_DIRECT`, and a PCIe path on which peer DMA between the NVMe devices and
  the GPUs is permitted: ACS control cleared on the bridges and an IOMMU that
  is off or in passthrough. `gdscheck -p` reporting the NVMe path as `nvfs`,
  and `gdsio` completing transfers with compat mode forbidden, confirm a host
  outside Flare. Where the probe accepts the direct path, `Direct` is the
  default transport once the section 4 thresholds hold for it.
- Registration: every page plane allocation and the row allocation on each
  rank is registered once at startup with `cuFileBufRegister`, so transfers
  address the planes and rows in place with no copy. These are VMM allocations
  mapped with `cuMemMap`. If the cuFile release does not support registering
  them, the transport stages through `cuMemAlloc`-backed buffers.
- Files: each record file is opened with `O_DIRECT` and registered with
  `cuFileHandleRegister` for the duration of a transfer. Plane offsets and
  sizes are 4 KiB aligned by the file format, so every I/O takes the direct
  path.
- Transfers: each rank's transfer thread submits one cuFile batch per window
  through `cuFileBatchIOSetUp`, `cuFileBatchIOSubmit`, and
  `cuFileBatchIOGetStatus`, with one entry per plane, and reports completion
  the way it reports a memcpy batch. Writes read the GPU pages directly, so
  the pinned GPU checkpoint is the only lifetime requirement. Reads land in
  the ticket's page ids and state slot directly, so the inbound completes when
  the batch completes on every rank. The batch API is used instead of the
  stream-ordered `cuFileReadAsync` and `cuFileWriteAsync` because it reports
  per-entry failures.
- Composite store: `store` fans out to the memory copy and the direct write
  concurrently, and `fetch` warms the memory store through the publication
  path.
- Checksums: a prepared GPU reduction over record planes preserves the payload
  verification that the staged transport performs on the disk thread.
- Isolation: the same bounds apply, at most 16 MiB per batch and the lowest
  stream priority, and cuFile's own worker threads are confined with the same
  scheduling priority as the disk thread.
- cuFile configuration: Flare supplies its own `cufile.json` through
  `CUFILE_ENV_PATH_JSON` before opening the driver, as fixed policy like the
  JIT and preload variables: compat mode disallowed, `max_direct_io_size_kb`
  raised so a 16 MiB entry is one I/O, and `max_batch_io_size` raised from
  its default of 128 entries within cuFile's limit, since one 64-page window
  of Qwen's 36 planes is 2,304 entries.

### 2.7 Scheduler integration

`helios/src/scheduler.rs` and `helios/src/scheduler/sharing.rs` provide:

- `RequestPhase::Inbound`, which `RequestScheduler::plan` never selects.
- Admission in two steps. `plan_admission(&spec)` validates the request,
  builds its history, performs the GPU lookup, and returns an `AdmissionPlan`
  with the history and the lookup without reserving anything. `admit_planned(
  spec, plan, inbound)` reserves and creates the request; `admit(spec)` is the
  composition with no inbound. With an inbound hit at consumed count `c` and
  `n` chain pages under GPU ancestor `A`, the request reserves pages and
  snapshots as if its prefix depth were `c`, and the scheduler creates or joins
  an inbound ticket for the key: `n` pages and one snapshot popped from the
  free lists through `ensure_resources`, a pin on `A`, the plan's history, and
  the waiting request handles. The request starts in `Inbound` with no prefix,
  `consumed_tokens` equal to `c`, and `admission_reused_tokens` equal to `c`.
- `finish_inbound(key, success)`. On success it publishes with parent `A`, the
  ticket pages as the tail, the ticket snapshot, and `ready_after` equal to the
  completed fence, so the checkpoint is ready immediately. A created checkpoint
  is pinned once per waiter, and each waiter receives the prefix, its page
  chain as shared pages, the snapshot, and phase `Restore`. An existing
  checkpoint returns the ticket's pages and snapshot to the free lists, which
  is safe because no graph has read them, and attaches the waiters to the
  existing id. The ticket's pin on `A` is released. On failure the ticket's
  resources return to the free lists, every waiter is retired as a cancelled
  request that never executed, and the waiters' handles are returned to the
  caller for re-admission. The same retirement and return apply when the
  published checkpoint is not ready until a later fence.
- Cancelling a waiter removes it from the ticket. The ticket completes
  regardless, because the restored checkpoint is useful on its own.
- `take_publications()` returns GPU-ready checkpoints offered since the previous
  call, each once, with their id, key, nearest offered ancestor's key, bucket
  digest, whole page chain, tail count since that ancestor, and row state slot.
  Message checkpoints enter the list at capture; deferred terminal checkpoints
  enter on reuse or compaction.
- `SchedulerStats` reports `host_lookups`, `host_hits`, `host_reused_tokens`,
  and `inbound_waiting`.
- `release_after_shutdown` drops tickets.

Ticket resources are owned by the scheduler rather than by any request, so
cancellation during inbound needs no special retirement rule, and several
requests sharing a prefix share one transfer.

### 2.8 Worker integration

`Worker` in `helios/src/server/runtime.rs` holds the optional store.
`ModelRuntime::start` constructs the configured store on the model thread
after the executor exists, from the per-rank contexts, devices, and cache
geometry in the local and peer manifests and from the pools and indexes the
loader created. Process-peer tensor parallelism accepts the option only for a
disk-only store whose ranks each keep their own records; see
[section 2.14](#214-ranks-on-separate-hosts).

The worker loop admits requests, completes executor frames, and publishes
finished requests. `service_store` polls host completions, applies inbound and
outbound results, delivers eviction notices, reports publications through
`store`, and retries capacity refusals after host progress. The worker then
plans and submits model work. Copy-out pins are taken before the next admission, so a
checkpoint cannot be evicted between submission and its pin. A pooled store
receives logical eviction notices, including compaction; the scheduler queues
no notices for a store without a pool. Every store
call from the worker returns without waiting.

Admission consults the store after `plan_admission`: `lookup` runs above the
GPU hit, the policy in section 2.10 decides whether to fetch, and
`admit_planned` receives the inbound request. A newly created ticket starts
the inbound job through `fetch`. On inbound failure the worker cancels each
waiter and pushes its `Incoming` to the front of the pending queue with host
lookup disabled for that attempt.

Shutdown closes the executor, then the store, which drains its streams, joins
its threads, and finishes pending disk writes within a bound, before
`release_after_shutdown`. If either drain fails, the store and its resource
owners remain alive together with the model resources; host registrations
and device allocations are never released beneath outstanding transfers.

### 2.9 Loader integration

`NativeModel::load` in `helios/src/runtime_core/model.rs` skips host-store
preparation entirely when no memory, disk, or pool tier is configured. Otherwise,
after `allocate_caches`, the `Prepare host prefix cache` phase derives the
storage namespace from the selected serving configuration, resolves settings,
locks the record directory, maps and registers slab pools with rank zero's
context current, or creates a memory-only store's split pool with each rank's
context current, and scans the directory. `manifests` attaches a `Binding` to each rank's
`ExecutionManifest`; the binding shares one prepared bundle, holding every
rank's geometry and context, which the model worker takes once to construct
the store and its transfer threads.

### 2.10 Policy and isolation

Policy:

- Outbound: copy each offered checkpoint into the first host tier. In local
  composite mode, spill to disk; in pool mode, move records down the
  eviction cascade only when the preceding tier needs room. Only queued copies
  pin GPU checkpoints. A capacity refusal waits for host progress before retry;
  admission uses ordinary cache eviction and the cache's pin and retirement fences.
  A GPU-resident hit refreshes the matching host record's recency. This does not
  recreate an evicted host copy or protect an endpoint from tree compaction.
- Inbound: fetch when the estimated transfer time is below half the estimated
  prefill time of the tokens gained. Transfer time is record bytes divided by
  the tier rate; prefill time is tokens gained divided by prefill throughput.
  Initial rates are 20 GB/s for memory, 3 GB/s for disk, 400 MiB/s for a
  remote level, and 8,000 tokens per second for prefill. Transfer rates use
  the median of the last three completed transfers at that level; the
  worker supplies measured prefill throughput from metrics.
- Eviction: the tree policy of section 2.3 under each store's evictability
  rule.

Offload limits interference through bounded work and explicit ownership:

- The model stream never waits on an offload event. Host-observed completion
  orders transfers and scheduler visibility. Shared bandwidth and copy engines
  still require the interference measurements in section 4.
- Copy engine queueing: offload batches are bounded to 16 MiB per call and the
  offload streams are created at the least stream priority. Every other Helios
  stream uses the default priority, which CUDA defines as the least priority
  as well, so stream priority does not separate offload work from model work.
  Creating the model and copy streams at a greater priority requires the
  section 4 measurements to show a benefit. Measure delays to
  input uploads, completion downloads, and Engram row copies on shared copy
  engines through per-token latency. Batch size and stream priority alone do
  not bound those delays.
- CPU: transfer threads sleep on event waits and channel receives; the disk
  thread computes checksums and submits I/O. Both use `SCHED_BATCH` with a
  positive nice value to reduce contention with the model worker, HTTP runtime,
  and frontend workers. Adding CPU affinity requires a serving-latency
  measurement that shows a benefit.
- Page cache: all record I/O is direct, so disk writes neither grow the page
  cache nor evict the core file's mappings.
- Memory: every slab pool is fixed at startup and sized by the defaults in
  section 2.11. Descriptor arrays, rings, and events are preallocated.
- Backpressure: every queue is bounded. When the disk thread falls behind,
  records awaiting their write stay pinned; when a pool is exhausted by pinned
  records, copy-out is skipped and counted, never waited for. Inbound waits
  are per request and visible as the `Inbound` phase.
- A noted publication submits no copy job. A queued copy carries page ids;
  address expansion runs on the transfer threads.
- The `Direct` transport design applies the same per-batch byte limit to
  NVMe DMA. Its section 4 measurements cover
  contention for GPU memory bandwidth and its effect on serving latency.

### 2.11 Configuration and defaults

- Memory size: `flare run --prefix-cache-memory SIZE` or `[prefix_cache] memory`
  sets the warm budget, shared by checkpoints and prepared CPU inputs as described
  in section 1. Zero and an unset value disable warm retention; automatic intake
  serves disk retention regardless. There is no minimum warm size or row-count warning.
- Disk directory: `--prefix-cache-disk DIR` or `[paths] prefix_cache` names
  the directory under which
  the model's record directory, `<identity prefix>-<layout fingerprint>`, is
  created. A directory that cannot be created selects no disk store; the
  startup line gives the reason only when no other tier is configured.
  An unset directory selects no disk store.
- Disk limit: one tenth of the free space on the directory's filesystem at
  startup. `--prefix-cache-disk-limit SIZE` or `[prefix_cache] disk_limit`
  overrides it; zero selects no disk
  store and does not create its directory.
  Checkpoint and CPU input records share this limit and the model's locked
  record directory.
- Disk records move through registered host staging and batched stream copies.
- Intake size is automatic; only shrinking it for host headroom emits the
  warning described in section 1.
- Pool: `--prefix-cache-pool ADDRESS` and `--prefix-cache-pool-ca FILE`,
  the CA only with the pool; see
  [section 2.15](#215-sharing-through-a-pool). Unset, the instance uses its
  local tiers only. Neither option has a configuration-file key.
- Shutdown drain and record expiry: `--prefix-cache-drain-on-shutdown` and
  `--prefix-cache-max-age`, or `[prefix_cache] drain_on_shutdown` and
  `[prefix_cache] max_age`, as section 2.5 describes. Both default to off.
- No flag or configuration key selects a disk transport; `[prefix_cache]`
  rejects unknown keys, including `disk_transport`.
- `LoadOptions` carries `prefix_cache: PrefixCacheOffload` with the
  memory size, directory, disk limit, shutdown drain, max age, and pool.
- The `Prepare host prefix cache` startup phase reports the slot pool's size
  and slot count, CPU input limits, the record directory, the disk limit, the
  scanned record count, and the `staged` transport.
- Metrics expose `flare_host_cache_slots_in_use`, `flare_host_cache_slots_total`
  (including intake), and `flare_host_cache_intake_slots`. Used slots include
  pages, every row piece, and staging allocations from the same local pool.
  Rows are counted in these slot gauges; no separate row gauges exist.
- The GPU prefix cache is derived, as the CLI design specifies. Host
  retention is explicit because host
  memory and disk are shared with other software.
- Derived defaults. Designed, not implemented: a default record directory
  beside the core file, named after it with the `.safetensors` extension
  replaced by `.prefix-cache`; derived memory and disk sizes, with the
  integrated-memory check of section 1 selecting disk-only retention; and
  policy rates seeded from measured values. Enabling them requires every isolation
  threshold in section 4 to hold. A threshold that fails is attributed to its
  channel in section 2.10 and corrected before any default changes.

### 2.12 Invariants

- Only published, GPU-ready checkpoints are copied out; their pages and rows
  are immutable, so device-to-host copies need no model-stream ordering.
- A checkpoint is pinned in the GPU cache for the whole copy-out, which
  excludes both leaf eviction and ancestor compaction.
- Inbound destinations come from the scheduler's free lists, which hold only
  ids whose last GPU use has passed a completed fence.
- A node is pinned in its index while any transfer reads or writes its slabs
  or file.
- A transfer failure is reported only after the thread's stream or batch is
  drained.
- Ticket resources belong to the scheduler until `finish_inbound`.
- A restored endpoint's row and pages describe the same token prefix. GPU
  ancestor pages are retained, and only the missing page suffix is uploaded.
- Rows and pages are stored byte-exact; no compression or requantization.
- Record files are keyed by model and execution-state identity plus layout
  fingerprint; incompatible state is never restored. Each record directory
  has one process owner through its complete I/O lifetime.
- In the composite, memory contents are a subset of disk contents, counting
  writes in flight, and memory evicts only durable records.
- Slab allocation never blocks; an unsatisfiable request fails its job.
- The model worker thread never waits for offload work, and the model stream
  never waits on an offload stream. A peer connection that stops accepting
  frames fails the job; it never delays the worker.
- A transfer whose completion can no longer be reported stops the store and the
  worker, so its ids and pins are neither reused nor released.
- Offload copies are bounded per issue, and offload threads run below the
  serving threads' priority.
- Steady-state serving allocates no device memory, no pinned memory, and no
  threads for offload; slabs, descriptor arrays, rings, and events are created
  at startup, as are the cuFile registrations of the `Direct` transport design.

### 2.13 Reusable CPU inputs

Source-description records map an encoded input key to its semantic identity
and geometry. Prepared-input records map that semantic identity to serialized
CPU tensors. Image source keys include encoding, encoded bytes, and preparation
settings. Semantic identities cover the decoded representation consumed by the
profile, including alpha, CMYK planes, and resize mode where relevant. The
versioned identity domain prevents reuse across incompatible preprocessing
semantics.

The frontend describes every input and validates the complete request budget
before admission. It does not resize images or build patch tensors. Admission
selects inputs whose complete token dependencies extend past the reused
checkpoint. Whole images are grouped into windows that fit prepared feature
storage. A bounded CPU worker materializes only inputs in the active window,
using retained tensors when available. Vision scheduling waits for its inputs
while unrelated requests remain runnable. Each window's features are packed
contiguously from zero; canonical prompt positions and feature indices remain
unchanged. Chunked prefill consumes one window before a private GPU
snapshot/restore boundary permits reusing its feature storage.

CPU records use ordinary host memory. Active leases remain charged until their
last owner releases them. A bounded serialized-record window covers queued
writes and disk-read leases, including disk-only configurations. Its capacity
comes from the largest prepared graph's image input buffers and codec overhead.
The disk input index has a fixed 4,096-record bound and a 1 MiB host allowance.
The loader accounts for these allocations before sizing GPU caches.

CPU preparation holds at most one image tensor result across queued work,
completion delivery, and packing. Cold decoded sources use a separate shared
one-use allowance of `max_image_pixels * 8` bytes, charged by their actual
buffer capacities. Sources exceeding the remaining allowance retain their
encoded bytes and decode again only if vision computation needs them.
Packing, prefix coverage, cancellation, and request destruction release the
corresponding owners. Both transient allowances are accounted for even when
host retention is disabled.

Versioned `.input` files store bounded lengths and checksums independently of
the `.ckpt` checkpoint format. Writes replace records atomically.
Description lookups read metadata without loading tensor payloads. Failed or
invalid records are cache misses; the request's original source reconstructs
the required input. Concurrent lookups share disk work, and bounded source
locks prevent repeated decoding of a retained source during concurrent
frontend preparation.

Startup fails if scanning or cleanup cannot establish the owned files' disk
usage. If a failed write leaves a temporary file that cannot be removed, its
reservation remains charged and input disk writes stop until restart.

Both record kinds share disk reservations. Pressure can reclaim either
unpinned checkpoint leaves through the model worker or CPU input files through
the input worker. CPU input records do not change the meaning of checkpoint
ancestry, transfer pins, or GPU fences. Evicting a CPU input never invalidates an otherwise
complete checkpoint; evicting a checkpoint never invalidates prepared CPU
tensors. Preemption and failed restores retain original sources and derive a
fresh execution plan from the prefix actually restored.

With host retention disabled, fully covered inputs skip tensor
preparation and vision replay as well. Source decoding is necessary to establish
semantic identity. Cores use these paths without re-export.
Per-image graph capacities and live feature storage stay fixed; total image
tokens across a conversation are bounded by context rather than one window.

### 2.14 Ranks on separate hosts

When a multi-host core's ranks run as separate processes, no process can
copy another host's GPU memory, so each host keeps the records of its own rank
in its own record directory. A file holds one rank's planes and row in the
ordinary record format, so every host's files of one record have the same name
and length. The layout fingerprint covers that rank's geometry, its rank, and
the world size, so a host never restores records another rank wrote.

Rank zero's `DiskStore` indexes, sizes, evicts, and plans every record as on a
single host. The disk limit is the smallest limit among the ranks, because a
record every host must hold is bounded by the host with the least room. Its
transport, `shard::Sharded`, runs every store, fetch, truncate, and unlink on
rank zero's `Staged` transport and sends the same job, with the same file names,
GPU ids, and state slot, to each peer over a dedicated TCP connection. Each peer
runs the job on its own `Staged` transport and reports its completion. Rank
zero reports a job only after every rank finished it, so a fetch completes only
once every rank's GPU holds its pages and row, and a checkpoint stays pinned
until every rank has copied it out. When some ranks wrote a record and another
failed, rank zero unlinks the file from the ranks that wrote it before
reporting the failure, so no host keeps a record the index lacks. A cleanup
unlink that fails is reported with that failure rather than discarded.

A partially successful replacement invalidates the previous record: successful
ranks have overwritten it, even when cleanup fails. Its index entry becomes
unreadable, so descendants can restore only when a GPU-resident prefix already
covers that entry. Parent links remain until the descendants are evicted, and
the full replacement size stays charged until reclamation. Removing a file
already deleted by partial-write cleanup succeeds. When every rank fails before
replacing its file, the previous record remains readable.

Sends are queued to a writer thread per connection, so the model worker never
waits for a peer. A peer that stops reading fills a bounded queue and fails the
job instead of stalling the worker. A connection that fails while a job is
outstanding stops the transport: it reports no completion for that job and
accepts no new one, and the worker stops with it. The peer may still be copying
into the ids of the job it already received, so reporting the job as failed
would return those ids to the scheduler's free lists while the copy runs. A job
that no rank completes within the job timeout stops the transport the same way.

The ranks join their stores after connecting their RoCE transport and before
the execution protocol starts. A peer starts its own transport before it
reports its scan, so a rank that cannot open its transfer threads, disk thread,
or record directory fails the join instead of interrupting serving after rank
zero has treated it as joined. Every rank must have a disk store, or none:
startup fails when the settings differ. Rank zero opens one listener on a
wildcard address and tells each peer the local address that peer's rendezvous
connection uses, so a multi-homed rank zero is reachable on the interface each
peer already selected. A peer presents the nonce rank zero sent over that
connection, its rank, and its layout fingerprint, and rank zero requires the
connecting address to be that rank's rendezvous peer. Peers then send rank zero
a summary of the records their scan kept. Rank zero keeps a record only when
every rank holds a file with the same header, ignoring the layout fingerprint
and row checksum, keeps its row only when every rank holds the row, and drops
records whose parent it drops. Every rank deletes or truncates its files to
that agreement before rank zero builds its index. Rank zero and each peer also
compare the layout fingerprint of their own rank's geometry at join, so a
core whose ranks have different plane or row sizes fails startup instead of
silently never matching.

A memory tier keeps every rank's bytes in one process, so a multi-host core
supports disk-only retention; startup rejects a nonzero
`--prefix-cache-memory`. A peer's record service holds its host intake pool and
transfer threads until rank zero closes its store after draining every rank,
or until rank zero's connection ends. When the peer's own drain is not
confirmed, the peer retains its transport, transfer threads, disk thread, and
registered staging together and exits with the error, rather than releasing
memory its transfers may still use.

Rank event framing and aggregation preserve filesystem-space pressure so a
failed copy can defer until progress after every rank drains.

### 2.15 Sharing through a pool

`--prefix-cache-pool ADDRESS` attaches the instance to a pool manager. See
[address forms, trust and startup warnings](../selene-manager/docs/network-operations.md#addresses-and-startup-warnings).
The manager's
`--name` salts scope and instance identities; clients learn it from Hello. A
renamed pool has distinct scope and instance identities. Record files carry no
pool name: a manager started with a different `--name` on the same `--root`
recovers the existing records under its own name, and an instance's local
record directory does not depend on the pool name. Without the
pool flag none of the sharing code below runs: the worker builds no pool lookup history and
sees no export requests. Local and pooled instances use the same copy-at-publish
and GPU eviction policy from section 2.10. A pooled instance reads
its own memory and disk tiers first; the pool supplies what they lack.

The loader attaches once before sizing and allocating GPU caches. The
`Wait for prefix cache pool` startup phase waits up to ten seconds for the
manager's first Hello. HTTP admission opens after loading and worker readiness;
if the manager is absent, the instance starts on local tiers and never attaches
later. Connection retries during that startup window are silent. At the deadline,
one warning says the manager did not answer, local tiers will be used, and the
instance will not attach later. An instance that attached at startup retries
its attachment after a lost connection.

Before any CUDA registration, the complete Hello plan must fit both budgets:
the manager's arena plus staging on this host, or the client's data staging for
a remote manager. Host supply is available physical memory less future local
host allocations, a 2 GiB server allowance and five percent of total RAM for the
system. Every GPU is charged one device byte per 512 registered host bytes,
rounded up separately for each region, as for local registered tiers. Integrated
GPUs also share the host-byte charge once per host. The check preserves runtime
headroom, explicit memory limits and minimum execution/KV capacity; accepted
charges reduce the remaining cache budget. A plan exceeding either budget
registers nothing and produces one warning naming its sizes and the reason,
then serving uses local tiers. Failure to observe host supply also refuses the
pool with that reason.

The startup registrar requires identical CUDA and host addresses. Its owning
registrations survive a manager restart when the region generation, paths and
sizes are unchanged, and remain alive through copy drain at close. The client
drains copies from the previous session before reconnecting. The manager holds
reused regions until previous clients reconnect or exit, then sharing resumes
without an instance restart. If the generation, paths or sizes change, sharing
is disabled with one warning; the instance continues locally and needs a restart
to register the new regions.

Selene's `PoolBuilder` owns pool settings, scope and instance identity, attachment,
and registration-plan accounting. The native loader supplies GPU identities,
cache geometry, remaining budget figures, a copier and a registrar. Selene also
owns rank framing, pool handshakes and record relay work; native joins bind those
resources to prepared CUDA storage and retain them through shutdown.

The worker's part is a set of defaulted `HostStore` hooks, all run on the
model worker thread and none waiting on the network. The worker consumes the
same completion events the store produces.

- **Exports.** The store reports an `ExportRequested` event through `poll`.
  `service_store` answers it in the same iteration: it refuses an export id
  it already serves (`Busy`), finds the ready checkpoint by
  key (`NotFound`), describes the chain
  from the requested start page (`ChainGap` when nothing can be exported),
  checks headroom (`NoHeadroom`), pins the checkpoint with `pin_checkpoint`,
  records the export in a map from export to checkpoint, and then calls
  `export_pinned`. The pin therefore exists before the store hears of the
  export, and eviction, which also runs on the worker, cannot interleave.
  `ExportEnded` is the only release: it removes the map entry and unpins once,
  and a second end for the same export finds nothing. An export the store never
  ends, because its copies could not be drained, keeps its pin. A store that
  rejects a pinned export gets the pin back at once. The store, which knows the
  record sizes, refuses an export larger than the staging the manager reserved
  (`NoWindows`).
- **Headroom.** `RequestScheduler::export_fits` is read-only. It asks whether,
  with the checkpoint pinned, the free pages and rows, those awaiting
  retirement fences, and what eviction could still reclaim cover the next step
  of every running decode and the request admission would take next, the first
  eligible one, parked before pending.
  `PrefixCheckpointCache::reclaimable_if_pinned` runs the
  cache's own leaf and compaction rules on the pin-adjusted counts, so a
  branching ancestor or a directly pinned one keeps its row and the rest does
  not; snapshot aliases go through the same `projected_fits` arithmetic that
  admission uses, and the next request's pages come from `admission_pages`,
  which admission also uses. A pin is never revoked once a copy has started.
- **Evictions.** Every `PrefixCheckpointCache::evict` call goes through
  `RequestScheduler::evict`, which queues the key of each
  retired checkpoint, and for a compaction the offered descendant that took
  its pages. The
  worker passes them to `HostStore::evicted` before the next `store`. The
  notice is sent when the checkpoint is logically retired, not at its fence;
  the pin of an export in flight keeps a leaf from being evicted and
  compaction never moves pages. The queue is bounded like the publication
  queue; when it drops its oldest entries the worker calls
  `HostStore::evictions_lost`, so the pool client announces holdings again.
- **Lookups.** When a request reaches the worker, the worker builds its prefix
  history once and calls `begin_lookup`; the store returns a ticket whose state
  never waits for a reply, and the ticket stays on the request while it queues.
  Admission passes it to `lookup_for`, which answers from the local tiers and
  adds pool candidates only if the ticket is ready. A pending lookup holds
  admission for at most its grace period: 2 ms locally, or four measured control
  round trips for a network manager, bounded between 2 and 250 ms. After that,
  admission can proceed with local results. An admitted text request retains a
  pending ticket until prefill starts. A late answer with a longer prefix retries
  ordinary admission once initialization or restoration has drained, provided
  no prefill has been submitted. This rechecks history, cost and capacity through
  the same admission path, without extending the grace period. Requests with
  conditioned inputs keep their original decision because image preparation may
  already own their slot. A request that waits on capacity keeps its ticket;
  completion clears its capacity block, allowing another attempt without a
  resource change. A cancelled request drops its ticket. Readmission without the
  store after a failed restore starts no new lookup.
- **Restore deadlines.** Claims allow 10 seconds of setup plus transfer time at
  a conservative 8 MiB/s; the manager clamps setup to its configured limit
  (30 seconds by default), then adds the same byte allowance. The requester
  allows 12 seconds plus that byte allowance once a claim slot is available.
  Waiting restores also budget for the restores ahead of them. The manager's
  independent three-second no-progress timeout still fails stalled transfers;
  destination memory remains held until writes drain.
- **Eviction below the instance.** Host records move from DRAM to local NVMe,
  then to the manager, with source retention until acceptance. In pooled mode,
  DRAM demotes cold records to disk, and the last local tier to the manager,
  before they fill. DRAM follows the free and clean floors described above;
  disk uses byte-based headroom. Disk headroom includes
  pending unlink and truncate bytes, which stay charged until completion.
  Store, eviction, spill completion, handover completion, reconnect and hashing
  progress trigger one coalesced maintenance pass per model-thread poll.
  Selection is bounded by the spill and move caps, without sorting all
  parked records. Full queues delay demotion and never discard a parked copy.
  Refusals, submission errors and repeated handover failures keep records local,
  release selection and retry only after manager progress or a short backoff.
  Manager loss releases waiting selections; started moves keep their pins until
  their drained terminal event. While a manager is unavailable or busy, local
  disk LRU makes room; with DRAM alone, durable DRAM LRU is the local end.
  Local mode uses the same parked/spill path and LRU unlink at the end of its
  chain, with the same completed-maintenance budget accounting. Only end-of-chain eviction,
  expiry, infeasible footprints (`Oversize`) and terminal disk I/O errors discard data.
  A host offer announces the GPU holding and starts its first host copy;
  GPU eviction sends its eviction notice and never starts a late copy.
- **Retention metrics.** `flare_host_cache_held_total{tier}` counts record
  holding events. `flare_host_cache_dropped_total{tier,reason}` counts local
  end-of-chain, oversize and terminal I/O drops (`end_of_chain`, `oversize`,
  `io_error`). Tiers are `dram` and `nvme`. The `parked_records` count is
  exported as `flare_host_cache_parked_records`, a gauge.
  `flare_host_cache_store_skipped_total` counts publications that started no
  copy, including duplicates and retries.
  `flare_host_cache_expired_total{tier}` counts direct age removals. Slot gauges
  `flare_host_cache_slots_in_use`, `flare_host_cache_slots_total` and
  `flare_host_cache_intake_slots` report the shared local pool, including
  every row piece and slots held by in-flight copies.
  `flare_host_cache_retry_abandoned_total{reason}` counts retries abandoned
  when a GPU checkpoint disappears, labelled with the last retry cause
  (`inflight`, `slabs`, `queue`, `budget`). The same counter reports terminal
  losses under the `oversize` and `io_error` reasons;
  `flare_host_cache_store_reasons_total{reason}` counts attempts.
- **Publication outcomes.** `Present` requires a committed local copy; a lookup
  hint is not sufficient. `Oversize` includes required ancestor files that
  cannot coexist within the disk limit even without unrelated records, and
  `Retry` means that capacity or queue room is temporarily unavailable.
- **One source.** The store chooses the longest matching prefix, then the
  closest level. It restores the whole suffix after the GPU-resident prefix,
  including the endpoint row, from that source. A failed pool restore may use
  a local copy of the same endpoint after the pool's writes drain; otherwise
  the destination is discarded and the request recomputes.

Loading attaches the pool to the local tiers, if any, as a `PooledStore`; with
no memory or disk tier every restore comes from the pool. The pool has its own
copier, a third set of rank threads that names the manager's shared-memory
windows by host address, so the registrar for those regions refuses any
registration whose device address differs from its host address. The manager's
pool name, model identity, layout fingerprint and per-rank plane sizes define
the scope, so instances share only identical record layouts, and
the instance id comes from the GPUs' UUIDs and is stable across restarts.

With a manager on the same host, the instance holds no pool staging of its
own: the manager's arena and staging regions are its
memory, mapped and registered once, in the startup registration described
above. Startup budgets the
copier's streams and events on each device, like each local tier's. The
registered regions are charged separately through the Hello plan: their host
bytes against the host budget and one device byte per 512 host bytes, about
2 MiB per GiB, against every GPU.

A manager on another host shares no memory, so its restores and puts pass through
**staging of the instance's own**: slots of one registered region, one window each, made
in the startup registration and kept across reconnects. Their count is the instance's two
claim slots, the manager's open-put limit and the instance's two export slots times the
manager's window credits (sixteen with the defaults)
and their size is the largest of its window limit (4 MiB), one page record and one row
record, so at least 64 MiB of registered host memory,
charged to the startup budget as the Hello plan's data staging.
A claim arrives window by window on the manager's data
channel into those slots, each page and the row is checked against the checksums stored
with the record before it is copied to the GPU, and the slot is freed only when its copy
has drained; a put hands filled slots to the channel and commits after the last one has
left. Such an instance is also a source: the manager grants an export's windows on a
channel the instance dials after pinning, the instance copies each into a slot and writes
it with its checksums, and the manager relays it to the requester, so instances on
different hosts (or on and off the manager's host) share their GPU and host caches
through the manager.
Connection security follows the [network settings](../selene-manager/docs/network-operations.md#addresses-and-startup-warnings).
With the `rdma` cargo feature and a manager started with `--rdma`, which the
manager refuses to combine with TLS, unsliced
claims can use RDMA WRITEs into private slots lent by the instance. Puts and rank
legs use TCP. An instance without a device, or after one failed RDMA transfer,
uses the stream.

Copy priority is fixed: neighbour GPU, host DRAM, host NVMe, then remote NVMe.
There are no synthetic startup transfers or borrowed GPU scratch ids.
Successful restores refine transfer costs without changing that priority;
failed transfers and row-only restores do not update the rates.

A pooled instance can span hosts. Rank zero owns the pool session and decisions;
each peer runs an agent that moves only its own rank's bytes. The pool identity
covers every rank's GPU UUID, and its layout is built the same way as
single-process TP of the same width. Every
rank approves and registers its pool memory before cache sizing; a refusal on
any rank disables the pool for the whole instance.
Multi-host local retention is disk-only: peer agents read their shards
under rank zero's holds, while the record links of section 2.14 mirror writes and
restores. The pool link shares the record listener's launch token and address
checks; deployments must secure the cluster fabric, since these links carry
session secrets and transfer tokens in plaintext.
Records reach the manager's store only when they cascade off the end of a
local chain under pressure, and an instance's own records leave the manager's
index when its session ends. Restarting with wiped local disks therefore
restores nothing unless records had already cascaded to the manager. This holds
on a single host too.

## 3. Components and required test coverage

Each mechanism of section 2 has one owning module. Its tests cover the behavior
listed with it; section 4 gives the commands.

- **Cache geometry and layout fingerprint.** `selene/src/geometry.rs`.
  `allocate_caches` collects the geometry, the rank's `NativeResources` stores
  it, `execution_manifest` attaches it, and `same_layout` in
  `helios/src/execution/threaded.rs` compares it across ranks. Coverage: a
  synthetic `Core` with one row allocation, several page planes, and a peer
  allocation yields the expected planes, and the fingerprint is the same
  across different `LoadOptions`.
- **CUDA bindings.** `helios/src/cuda.rs`; `Stream::copy_batch` takes slices of
  destinations, sources, and sizes and enforces the per-call bounds. Coverage:
  an ignored GPU test allocates two device buffers, registers an anonymous
  mapping, round trips a scattered batch in both directions byte-exact, and
  reads a file with `O_DIRECT` into the registered mapping.
- **Slot pool.** `selene/src/slab.rs`. Mapping is separate from `register` and
  `unregister`, so CPU tests run without a driver. Coverage: exhaustion, reuse,
  alignment, multi-piece rows, release on completion without retention, and
  intake reservation.
- **Index.** `selene/src/shared/index.rs`. Coverage, with 64, 128, and 256
  token pages: unaligned endpoints, feasibility through the resident ancestor,
  a chain broken by a page-only node that supplies its pages, pinned and
  non-evictable nodes surviving eviction, compaction removing lookup matches
  but not chain pages, and LRU order.
- **Record files.** `selene/src/shared/record.rs`. Coverage, in a temporary
  directory: round trip, aligned plane offsets for every geometry, row presence
  by size, rejection of a wrong identity or fingerprint, temporary file
  cleanup, orphan removal, and a rebuilt index equal to the one that wrote the
  files.
- **Disk thread.** `selene/src/disk.rs`. Coverage: write, read, truncate, and
  unlink jobs on a temporary directory, skipped when the filesystem refuses
  direct I/O; a corrupted segment fails verification.
- **Transfers.** `helios/src/offload/transfer.rs`. Coverage: an ignored GPU
  test loads the core named by `FLARE_OFFLOAD_TEST_CORE`, which may be any
  exported core with sized allocations, writes patterns into chosen page ids
  and a cache row, copies them to registered slabs, poisons the device
  storage, copies them back into different ids, and compares byte-exact.
- **Memory and disk stores.** `selene/src/memory.rs` and
  `selene/src/disk_store.rs`. Coverage: the transfer GPU test repeated through
  each store's `store` and `fetch`, and CPU tests of each store's event
  sequencing with a transfer stub.
- **Composite store.** `selene/src/composite.rs`. Coverage, with transfer and
  disk stubs: a memory hit, a disk hit that becomes resident, a fetch of a
  record whose write is in flight, memory eviction restricted to durable
  records, and disk eviction removing the memory copy; the GPU test repeated
  through the composite.
- **Store assembly and settings.** `selene/src/offload.rs` resolves settings,
  sizes the slot pool, and assembles the configured stores.
- **Scheduler.** `helios/src/scheduler.rs` and
  `helios/src/scheduler/sharing.rs`, tested in `helios/src/scheduler/tests.rs`.
  Coverage: inbound requests are never scheduled; success attaches with the
  correct page chain, snapshot, consumed count, and `Restore` phase; a
  full-prompt hit resumes; a checkpoint the cache holds frees the ticket;
  failure frees the ticket and reports waiters; two waiters share one ticket;
  cancellation during inbound leaves ticket resources intact; free page and
  snapshot counts balance across every path; publications report the exact
  tail pages and state slots.
- **Worker, loader, flags, and metrics.** `helios/src/server/runtime.rs`,
  `helios/src/runtime_core/model.rs`, `flare-cli/src/cli.rs`, and
  `helios/src/server/metrics.rs`, with one dashboard line in
  `helios/src/server/dashboard.rs`. Coverage: a `HostStore` test double covers
  requeue after a failed restore; the service scenarios of section 4 cover the
  loop order, admission, and shutdown.
- **Ranks on separate hosts.** `selene/src/shard.rs` and
  `selene/src/offload/sharded.rs`, joined to rank streams by
  `helios/src/offload/shard.rs`.
- **Pool client.** `selene/src/pool/`, started at load by
  `helios/src/offload/pool.rs`. The manager and its test targets are described
  in [selene-manager](../selene-manager/README.md).
- **CPU input records.** `selene/src/input.rs`.
- **`Direct` transport.** Designed, not implemented (section 2.6). Its
  required coverage: an ignored GPU test registers a `cuMemMap` allocation,
  writes it to a file with one batch, reads it back into a second allocation,
  and compares byte-exact; a CPU test covers the probe's decision table over
  recorded property values; the store and composite GPU tests are repeated
  with the direct transport, and cuFile statistics confirm that every byte
  took the direct path.

## 4. Validation

A correctness failure in any check below blocks performance claims.

The host-tier CPU tests (index, slot pool, record files, disk thread, and the
memory, disk, composite, and pooled stores against copiers that touch no
device) are in the `selene/` crate:

```sh
cargo test --manifest-path selene/Cargo.toml --features test-support
```

Tests needing direct I/O create their directories under
`FLARE_OFFLOAD_TEST_DIR`, or the system temporary directory when it is unset,
and skip when that filesystem refuses direct I/O. The scheduler, worker,
namespace, and shutdown regressions are Helios library tests:

```sh
cargo test --manifest-path helios/Cargo.toml --all-features --lib
```

`--all-features` enables the `roce` and `rdma` features, which link
libibverbs; `--features runtime-core,server` builds the same offload tests
without it.

Core-backed GPU tests verify exact page and continuation-row bytes, including
multi-window transfers, RAM eviction, bounded disk streaming, and restarts.
They are ignored by default and locate a core through
`FLARE_OFFLOAD_TEST_CORE`, following `helios/src/runtime_core/event_tests.rs`.
Each loads the whole core, so they run one at a time. Choose an idle GPU and
name a test directory on a filesystem supporting direct I/O:

```sh
CUDA_VISIBLE_DEVICES=0 FLARE_OFFLOAD_TEST_CORE=/path/to/core.safetensors \
FLARE_OFFLOAD_TEST_DIR=/path/to/offload-tests \
cargo test --manifest-path helios/Cargo.toml --all-features offload:: -- --ignored --test-threads=1
```

Build the native CLI and run the HTTP regressions against a Qwen TP1 core
containing ordinary and MTP5 graphs:

```sh
cargo build --release --manifest-path flare-cli/Cargo.toml
CUDA_VISIBLE_DEVICES=0 FLARE_RUN_CUDA_TESTS=1 \
FLARE_OFFLOAD_TEST_CORE=/path/to/qwen-tp1-mtp5-core.safetensors \
.venv/bin/python -m pytest tests/server/test_prefix_cache_offload.py -q \
  --basetemp /path/to/offload-tests/http
```

The HTTP tests launch and stop their own servers, isolate ordinary and
speculative records, restore prefixes exceeding RAM retention, reject corrupted
payloads, and check GPU prefix reuse with offload disabled. Pytest owns and clears
its `--basetemp` directory; use a dedicated test path.

`qualify/host_cache.py` launches Flare with and without a host store and runs
these scenarios, each with every store that supports it (memory, disk, and
composite; `--skip-memory`, `--skip-disk`, and `--skip-composite` omit one):

```sh
python3 -m qualify.host_cache --core /path/to/core.safetensors \
  --gpus 0 --artifacts /path/to/host-cache-qualification
```

- Memory hit after GPU eviction: the host hit counter increments, the reused
  token count equals the terminal endpoint, and one restore and the prefill
  launches cover only the new tokens.
- Disk hit after a server restart with the same core and directory: prefill
  launches cover only the new tokens (`prefill_launches` in
  `qualify/server.py`), with the composite and with the disk store alone.
- Branching: two continuations of one evicted prefix admitted together share
  one inbound ticket and both complete.

Every continuation asks for nine facts spread across the whole restored
context and must answer each one correctly, after GPU-cache restores and host
restores alike. Sampled continuations are evaluated by their factual answers;
generated tokens are not compared verbatim. The memory scenario
sizes its pool to hold more complete checkpoints than the GPU cache, because
the filler prompts that evict the GPU cache publish a checkpoint each.

Namespace acceptance: ordinary and speculative execution use separate
namespaces; compatible positive draft limits, concurrency, and verification
policies reuse records; incompatible row or page layouts miss.

Multi-rank acceptance on one host: Qwen TP2, GLM TP4, and DeepSeek TP4 pass
the same scenarios and the isolation measurements below, with identical
results on every rank and transfer byte counts matching the geometry. DeepSeek
is also run with Engram tables in disk mode, which shares the NVMe and PCIe
paths with record I/O.

`qualify/multihost_cache.py` covers the ranks-on-separate-hosts configuration.
It starts one `flare run` process per rank on one host, each on its own GPU
with its own record directory, so the sharded transport runs between processes
as it does between hosts:

```sh
python3 -m qualify.multihost_cache --core /path/to/core.safetensors \
  --gpus 0,1,2,3 --artifacts /path/to/multihost-cache-qualification
```

The core's rank count must equal the number of GPUs. The scenario requires a
GPU-cache hit without a host hit, a host restore after evicting the GPU copies,
and a restore after restarting every rank. After each phase it checks that
every rank holds the same record files, because a record that one rank lacks is
unusable everywhere.

Isolation measurements gate the derived defaults of section 2.11 and any
change of default transport. `qualify/host_cache_isolation.py` runs each
workload against a server without a host store, the baseline, and then against
a server with the configured host store under the same core, flags, and GPU,
and writes raw observations beside its summary. Pass `--disk DIR` to measure
the composite; without it the script measures the memory tier alone. Rerun
noisy results.

```sh
python3 -m qualify.host_cache_isolation --core /path/to/core.safetensors \
  --gpus 0 --artifacts /path/to/host-cache-isolation \
  --disk /path/to/records --seconds 600
```

| Measurement | Workload | Metric | Threshold |
|---|---|---|---|
| Decode latency under copy-out | concurrency 4, continuous short exchanges of 256-token prompts and 64-token answers for ten minutes, the maximum publication rate | per-token latency p50 and p99 from HTTP observation timestamps; tokens per second | within 2% of baseline |
| Decode latency under inbound | three requests decoding at 64K context while a fourth restores a 100K-token prefix from disk | the same | within 2% of baseline |
| Prefill throughput under copy-out | 32K-token prompts admitted back to back | prefill tokens per second | within 2% of baseline |
| Speculative round time | the first workload | round GPU time from `/metrics` | within noise |
| Inbound latency | 8K, 32K, and 100K token prefixes, through each store and transport | admission to `Restore` submission | reported beside the prefill time of the same prefix |
| Copy-out bandwidth | per rank, each transport | bytes per second from store counters | reported |

The `Direct` transport design adds two measurements when it is implemented:
inbound latency against the staged transport on the same prefixes, and fabric
bandwidth from each NVMe root complex to each GPU root complex.

## 5. Known limitations

- Disk compaction of a record is not propagated to its memory copy, whose row
  slots stay allocated until the whole copy is evicted.
- A publication whose key names a stale memory copy pinned by an upload
  in flight is skipped rather than written again.
- Shutdown deadlines apply separately to pipelines, disk workers, and CUDA
  transfers, so total shutdown time can exceed one drain deadline.
- `bytes_out` counts every memory copy that landed, including copies the disk
  limit then refused; `stored` counts only records that reached disk.
- Ranks on separate hosts keep records on disk only; their memory tier is
  not supported.

## 6. Designed, not implemented

- The GPUDirect Storage transport of section 2.6.
- Derived default sizes and the default record directory of section 2.11.
- A split operation in the GPU cache, the inverse of ancestor compaction, so a
  compacted ancestor's row can be restored onto pages its child holds
  without copying pages.
- Page-granular matching in the GPU cache, which depends on rows being cheap
  to refetch.
- Prefetch of likely continuations, and compression.
