# Runtime packages: the preparation boundary and native serving

Flare serves a prepared model from a runtime package containing everything a Rust
process needs to rebuild the model's CUDA graphs and storage without a Python
interpreter. The [package format](package-format-design.md) defines a packed
`.safetensors` file and a directory form with `model.safetensors` plus immutable
payload companions. The Flare model cache stores the directory form. Serving
requires execution ABI 5 and prepared model contract 3; other versions require
re-export. Preparation happens in Python with PyTorch, Transformers, and b12x. Serving requires the NVIDIA driver,
the package, the Rust `flare` executable, and its system library dependencies
(including libturbojpeg). Python and preparation tools are not required.
The [core design](core-design.md) defines the model contracts, scheduling,
and cache behavior that the served process implements; this document defines
how the prepared state leaves the Python process and how it is reconstructed.

## 1. Components

![Flare component relationships](images/component-architecture.png)

| Component | Role | Implementation |
|---|---|---|
| ignition | Internal preparation command: plans, exports, and prepares cached packages for native serving | The Python command line under `ignition/cli/` |
| heliopause | CPython extension imported by Ignition. It owns preparation storage, export recording, graph serialization, payload transfer, and signed package publication | The `heliopause/` Rust crate; Python distribution `flare-heliopause`, import `flare_heliopause` |
| Helios | Rust runtime library. It rebuilds a package into a live CUDA context and serves it | The `helios/` crate, including `src/runtime_package/` and the server |
| flare | Public serving command. It links Helios to load a package and run the server | The `flare/` Rust crate; section 5 |

Preparation knowledge, recipes, planning, checkpoint acquisition, the kernel cache,
compilation, kernel optimization, and capture stay on the preparation host.
Prepared tuning graphs can select persistent runtime binding values during
native startup. Their candidate kernels, launch geometry, fixtures, and scratch
are already compiled and captured in the package. Ignition imports
`flare_heliopause` and passes graph handles, allocation declarations, tensor-view
metadata, and resource owners through `ignition/preparation/runtime_package.py`.
The Rust exporter in `heliopause/src/export/` reads the graph and storage,
records the package resources, and publishes the files while releasing the GIL.
`ignition export` writes the package, closes the
prepared model, and exits. `flare run PACKAGE` loads it in a separate Rust
process. Graph handles and Python resource owners do not cross the process boundary. Helios has no CPython dependency;
heliopause owns the CPython hooks and preparation allocator and uses Helios's
CUDA interfaces. Heliopause has no executable or embedded interpreter. `flare` and Helios are
the public deployment surface. Ignition and heliopause are internal preparation
tools. Dependency direction is `flare → Helios` and `heliopause → Helios`;
Helios has no dependency on either preparation tool.

### Distribution builds

From the repository root, build public executables with:

```sh
cargo build --locked --release --manifest-path flare/Cargo.toml
```

The executable is `flare/target/release/flare` (`flare.exe` on Windows). Docker and
native-serving CI use this build. The release profile strips symbols and retains
Rust panic messages, source locations, and unwinding behavior. Builds use the
stable toolchain without unstable compiler flags. Source paths, required dynamic
imports, API routes, metrics, and operational errors remain observable. Publisher
stamping through `python -m ignition.release` preserves the embedded
verification-profile section.

## 2. The preparation boundary

Ignition records four things while preparing a model, and refuses to
write a package when any of them is incomplete.

Ignition supplies captured graph handles, allocation and cache declarations,
tensor shapes, strides and numeric roles, compiler parameter signatures,
frontend metadata, and resource owners. File-backed embedding tables supply
checkpoint ranges; generated tokenizer and template assets supply text.
Heliopause owns CUPTI callbacks, graph introspection, pointer reachability,
device-to-host transfers, graph and package serialization, asset materialization,
cubin debug stripping, hashing, compression, signing, conversion, rank payload channels,
and publication. Export retains graph templates, storage, contexts, and their
Python owners until native CUDA work has drained. A failed drain retains those
owners. Final assembly consumes rank indexes and open source file ranges
independently of the prepared model.

**Storage.** PyTorch's persistent device allocations come from one address range
reserved before model loading: 1 TiB per rank from `0x0800_0000_0000` on Linux or
`0x0400_0000_0000` on Windows, backed on
demand with `cuMemCreate` and `cuMemMap` at allocation granularity. Packages
reserve GiB-aligned ranges covering stored allocations and the declared maximum
extents of dynamic caches; unused gaps in the preparation arena are omitted. PyTorch
allocates through a `MemPool` whose pluggable allocator is the Rust arena in
`heliopause/src/allocator.rs`, so weights, state, pages, and the
graph pool all live inside the range. Ordinary preparation and capture use
separate PyTorch pools backed by that arena. Mapped host buffers, including
n-gram request mailboxes, use fixed addresses from `0x0c00_0000_0000` when
the device rank ranges fit below it. Larger Linux layouts start host storage
at `0x2000_0000_0000` to avoid overlap. Each rank has a disjoint 1 TiB host
range; the native allocator reports the selected base to Ignition.
Linux maps and registers host pages; Windows uses host-backed CUDA VMM.
Both expose the recorded address to the CPU and GPU. A CUPTI subscriber logs
driver allocations and removes records when their context is destroyed.
Persistent device allocations outside the arena are grouped into complete
VMM pages and placed within reservations aligned to the device's allocation
granularity. Reservations without retained allocations are omitted. Only their
live byte spans are considered for payload storage; page gaps are zero. Before
copying, the exporter checks that each external allocation belongs to the
serving context. On Linux integrated GPUs, preparation reserves the low address
window through 256 GiB without physical backing. Windows retains the driver's
low allocation pool. Export on either integrated platform rejects reachable
external device storage below 256 GiB, where context mappings can prevent replay.

The model exporter retains allocations reachable from graph arguments, copy
and memset endpoints, module globals, native execution buffers, and declared
peer storage. It follows pointer-bearing allocation contents while writing
tensor payloads. Checkpoint transfers record numeric tensor spans with weak
storage ownership; expired or resized storage cannot reuse those declarations.
Live floating-point tensor views supply additional numeric spans, excluding
integer aliases and holes in strided layouts. Native input/output buffers are
numeric by their execution contract. These spans are leaves in the traversal:
their storage is retained when referenced, but their values are not interpreted
as addresses.

Numeric-span discovery merges overlapping allocation coverage, excludes
zero-initialized storage, and deduplicates identical live tensor views. Integer
aliases crossing coverage boundaries still suppress floating-view inference in
their covered bytes. Strided layouts
retain their holes; fragmented persistent views without bounded span metadata
remain conservative rather than expanding an unbounded interval catalog.

Preparation uses temporary PyTorch pools backed by the package allocator during
loading and tuning. Retiring a pool releases whole free segments before forward compilation;
live tensors retain their storage and addresses. Address candidates within those
pools must fall within requested allocation sizes, including in retired pools
with surviving tensors. Inactive blocks and the padding between a live block's
requested size and its allocator-rounded size are cleared and excluded from
payload reads. Their stale contents cannot retain other allocations, and stale
address candidates cannot retain those unused bytes' backing segments. This
filter excludes the private graph pool: its intermediates remain live through
captured replay even when their Python tensors have been released. Explicit
resource bindings retain their allocations independently of candidate scanning.

Compiler-declared scalar arguments are also leaves. Triton signatures are
checked against the driver's parameter sizes; kernels with integer-to-pointer
conversions, inline assembly, or opaque calls retain conservative argument
scanning. Untyped structs, tensor maps, globals, and integer/byte storage without
a numeric role are scanned for address candidates. This can retain false
positives and retains complete backing segments, including unused bytes within
a reachable segment. The preparation metadata does not enter the serving index.
Omitted arena allocations keep their virtual ranges reserved and physically unmapped,
so an unexpected access faults instead of reaching another allocation.

Before export, model recipes declare disposable KV and scratch storage,
including target and MTP page pools, operator workspaces, decoder and vision
intermediates, and sampling and verification temporaries. The graph memory
pool holds only graph-local intermediates and is also cleared, as are unused
bytes in the preparation allocator. Weight tensors, MTP weights, rotary
tables, pointer tables, and required initialized metadata retain their contents.

Serving accepts payload format 2 and signed format-6 logical-resource containers,
with execution ABI 5 and prepared model contract 3 required in both carriers.
Packages with another execution ABI or prepared contract require re-export with
Ignition and fail loading before graph replay. Unsigned rank staging cannot load.
The canonical format uses signed content IDs, payload-relative initializers,
and unsigned physical locations; see [payload packages](package-format-design.md).

`ModelIndex.contract` is a typed `ServingContract`. It declares serving defaults
and a versioned `ModelContract` containing vocabulary bounds, vision capacities,
attention-history storage, page-table views, graph value sources and dependencies,
completion cursors, and frontend syntax. Field 4 is reserved; arbitrary Protobuf
`Struct` configuration is not accepted. Execution ABI 5, model contract 3, and
payload format 2 are independent version domains.

Two prepared-contract fields fail validation when absent or out of
range. `vision` must decode to a well-formed `VisionCapabilities`: a
text-only package encodes the explicit all-zero value (`feature_slots`,
`features_per_request`, `max_images`, and `patches_per_feature` of 0
with empty `patch_capacities`), and an absent or null `vision` fails
the contract decode (a null submessage raises `invalid type: null,
expected struct VisionCapabilities`). Every graph contract also carries
`embedding_tables`, whose entries index the package's embedding-table
catalog and are validated against the catalog size at package
validation: a package that ships no embedding-table resources must
give every graph an empty `embedding_tables`, and any entry outside
the catalog fails with `graph requires an absent embedding-table
resource`.


Semantic bindings name a destination buffer and an enum-selected value source.
Indexing distinguishes scalar, submission row, active row, physical request row,
and packed-token storage. Offsets and padding are explicit. Omitted transfers
retain device bytes. Persistent state views contain byte offsets, strides, and
extents within declared buffers; graph state accesses describe reads and writes.
Paged key/value and latent histories refer to actual cache planes. Continuation
rows hold recurrent state, sliding windows, partial pages, and other captured
model histories. Shared cache allocation sizes the physical backing independently
of model identity. Native code implements no attention state-update computation.

Frontend enums independently select tool/reasoning parsers and image preparation
and layouts. Marker strings, feature tokens, and atomic
image spans are data. Enum zero and unknown identifiers are rejected; enum
numbers are stable and removed values must be reserved. Frontend identifiers
cannot select scheduler, cache, text-graph packing, or completion policy.
Message checkpoints derive from exact rendered prefixes without role filters.

The index stores complete recipe-declared sampling parameters in
`DefaultSamplingParameters` and structural frontend settings in
`FrontendSettings`. Sampling defaults are manually verified against model-card
recommendations for the default serving mode; checkpoint configuration does not
select sampling policy. Missing values reject loading rather than inheriting
another model's settings. `DefaultSamplingParameters.ranges` contains optional
inclusive `min` and `max` bounds for temperature, top-p, top-k, min-p, and the
repetition, presence, and frequency penalties. Each supplied range must contain
its existing default, and both endpoints must be valid sampler values. Numeric
request values are clamped to these bounds; omitted ranges fix parameters at
their defaults. Packages without range fields use this fixed-default policy. Seed
remains request-controlled.

`FrontendSettings` contains EOS token IDs, the special-token strings used by the
chat template, and optional image settings. Qwen image settings contain resize
area limits, learned-position geometry, and image/video token IDs. MiMo image
settings contain resize limits and channel normalization. DeepSeek's image profile
has fixed geometry validated by Ignition. Preparation reads and validates structural
checkpoint config files, then exports only these consumed fields. The only frontend
payload assets are `tokenizer.json` and the Jinja chat template; raw generation,
tokenizer, processor, and model configuration files are not included.

Payload descriptors bind destination ranges to signed content objects. Unsigned
locations supply physical placement. Payload aggregation and initializers are
defined in [payload packages](package-format-design.md). The metadata declares
rank bundles, peer-memory ownership, serving configuration, frontend assets,
allocations, kernels, and graph programs for each prepared deployment.

Graph bodies, cubins, module globals, packed tensor data, tokenizer data, and
other frontend assets live in separate indexed payload spans. A decoded graph body contains its declared number of nodes followed by its
edges. Each record is a little-endian u32 length followed by one Protobuf
`Node` or `Edge`, with a 1 MiB record limit. Kernel arguments, node attributes,
and edge data are Protobuf byte strings. Graph bodies are independent; the
loader need not read an unselected graph to locate or select another graph.
Rust bindings are generated by `prost-build`; Ignition uses checked-in Python
bindings generated by `protoc` from the same schema.

Allocation extents are separate from stored payloads. Each allocation has
ordered, disjoint `data_spans` containing destination byte offsets and lengths.
These are memory destinations and must leave `filename` empty.
Its payload concatenates those spans without padding. Declared disposable
ranges are excluded before device reads; remaining ranges are checked for
zero-filled 64 KiB blocks, which are also omitted. An entirely zero allocation
has an empty span list and no payload. Helios zeroes omitted regions and
copies only the stored spans to their declared offsets. Runtime-sized row and
KV allocations carry no tensor payload. The loader accepts the indexed file
format; incompatible packages must be exported again with Ignition.

Each preparation rank writes its resource file before the model file is
assembled. After all ranks finish exporting and peer ownership is recorded,
Ignition drains and closes the prepared models, joins peer processes, and
releases registered host buffers. Final file assembly requires only rank files
and their indexes. N-gram table payloads are appended directly from checkpoint
file ranges during rank export, without materializing their contents in
preparation memory.

### Native export transfers and progress

The exporter in `heliopause/src/export/` performs bulk I/O while the GIL is
released. CUDA downloads and storage commits each use a two-slot pinned ring
with 8 MiB per slot, reserving 32 MiB per preparation rank.
File transfer workers share bounded pools of 8 MiB buffers with 4 KiB alignment
padding, including a reserved scratch pool for unaligned source ranges.
Each transfer pool budget starts from the smaller of 1 GiB and one sixteenth
of available host RAM. Capture divides this budget across preparation ranks and
reserves both pinned rings before sizing file buffers. Capture releases its
storage-transfer pool and pinned rings before materializing frontend assets.
Publication after model closure uses the full file-transfer budget. Concurrent
preparation and rank-transfer processes budget their pools independently.
Metadata and cubin images occupy separate retained storage.

Rank transfers borrow established control and fabric sockets. The binding retains
their Python owners until all native workers have joined, including cancellation
and error paths. Callers lend these sockets exclusively: they must not close them
or perform other reads and writes until the transfer returns. Native export
returns authenticated channel sequence counters for Python to restore before
resuming use.

Linux bulk reads and writes use `O_DIRECT` with `io_uring`; Windows uses
unbuffered overlapped I/O. Aligned bulk transfers bypass the filesystem page
cache. Unaligned source ranges use aligned scratch reads, and small headers,
leading bytes, and trailing bytes use positioned I/O where needed. Output files
retain their exact declared lengths. Direct-I/O errors fail the operation.

Payload publication verifies source bytes and existing cache objects before reuse.
A cache miss installs an immutable object without replacing existing entries.
`--no-weights` requires exact cached weight payloads, including tuned weights;
missing or corrupt required payloads fail publication. Executable and initial-state
payloads may be installed during this export.

Native stderr rows omit the Flare banner. Rank-zero capture reports storage
clearing, graph inspection and serialization, module/global capture, storage scans,
weight materialization, and frontend assets. Coordinators report export peer waits,
rank transfers, and preparation-resource release before publication.
Publication reports payload assembly, validation, frontend verification, each rank's
executable compression and installation, each rank's data verification and
installation, shared data, metadata signing, and final package commit.

Native phases animate and update elapsed time independently of Python. Byte totals
count plaintext executable inputs or data source occurrences, including duplicate
payloads verified before reuse; they do not claim physical disk throughput or
cache bytes written. Ignition prints one final export success or failure result
with the total elapsed time.

### File placement and addressing

Payload packages use signed content IDs and unsigned locations. Signed format-6
logical-resource containers use the archive reader and must declare execution
ABI 5 and prepared model contract 3. Payload conversion accepts only payload
packages; converting logical-resource containers requires re-export with Ignition.

Canonical payload packages do not use logical filenames, layout segments, or
key-free conversion of signed logical ranges: they use signed content ids,
payload-relative initializer slices, and unsigned `LocationSet` entries, with
`flare sign` required only when payload partitioning or signed semantics change.
See [payload packages](package-format-design.md) for the present-state rules on
canonical forms, packing, publication, locations, and conversion.

### Preparation rebuild metadata

Exports through `ignition export` record an optional
`ModelIndex.rebuild` manifest in the signed metadata of the primary shard.
It identifies the recipe, preparation source identity, qualified checkpoint
repository and revision, tensor-parallel size, concurrency, chunk size,
speculation, and selected preparation settings. A local checkpoint override
is marked explicitly and carries no asserted upstream revision. Runtime
collective preparation records the complete offline ladder as
`RebuildManifest.collective_ladder_json`; rebuilding does not depend on the
original ladder file path. Helios does not interpret that preparation input.

Each recipe also supplies its named weight owners. After reachability pruning,
the exporter records retained tensor views in `BundleIndex.rebuild_weights`:
owner paths, packing-owner types, dtypes, shapes, element strides, and
allocation-relative byte offsets. The containing bundle identifies the rank.
Allocation resource spans and the package layout locate those bytes in the
payload shards; repacking and re-signing preserve the references. Aliased views
share storage. Ignition describes live tensor metadata; Heliopause resolves its
addresses against retained allocations and creates no weight copy.
Meta tensors, discarded storage, and disposable scratch are listed as
unavailable rather than treated as recoverable weights.

These catalogs describe prepared storage, whose names and packing can differ
from the original checkpoint. They do not serialize Python objects or execute
imports. A package-based graph rebuild still requires a compatible recipe
importer that validates the catalog and supplies any unavailable inputs;
Ignition does not yet provide that importer or a package rebuild command.
`--no-weights` continues to prepare from the source checkpoint and reuse cached
payloads. Offloaded embedding planes retain their existing typed table metadata.

Speculative rebuild metadata records the resolved physical graph-count budget
as `RebuildSpeculation.graph_budget`. Packed speculative preparation requires
an explicit positive budget; rebuilding a catalog must use that recorded value
rather than selecting a recipe default.

Flare exposes the manifest and per-rank tensor counts through `flare inspect`;
`--all` includes the tensor catalogs. Absent manifests are reported as null.
Helios ignores rebuild metadata during loading and replay, including unfamiliar
catalog versions. Serving still requires execution ABI 5 and model contract 3.
Changing signed rebuild metadata requires signing the package again.

### Publisher authentication and executable access

Every final package carries three ordered Ed25519 signature groups for what
every rank shares, and one signed `RankProtection` per rank for that rank's
resources:

| Group | Authenticated contents | Payload checks |
|---|---|---|
| 1 | Model index without rank bundles, publisher namespace, source URL, development marker, model license name and link when recorded, rank count, and SHA-256 commitments to groups 2–3 | Before publisher acceptance or CUDA initialization |
| 2 | Frontend asset spans and hashes | Before those contents are consumed |
| 3 | Ranges and SHA-256 of shared embedding tables, including padding | With `flare run --verify-weights` |
| Rank *r* | Rank *r*'s bundle; its cubin and graph spans, decoded and stored sizes, and compressed chunk hashes; its module global spans and hashes; and the ranges and SHA-256 of its stored allocations and embedding partitions, including padding | Bundle before loading; resources as for groups 2–3 |

Stored data is ordered by owner: shared tables, then each rank's allocations and table partitions.
Each owner's signed parts describe its contiguous ranges across files. Every signature
of the ranks a process serves is mandatory even when the stored-data scan is
omitted. A process verifies every rank, except a host serving one rank of a
multi-host package, which verifies the shared groups and its own rank only, so
a copy holding only that rank's bytes verifies completely and other ranks'
bytes or signatures do not affect it. Peer transfer sends ranges separately for
each logical file for `format: flare` packages, including headers and signed
padding. Snapshot transfer sends the complete metadata primary and only the whole
blobs selected for a rank. Agent protocol 3 stores package files in a rank-specific
directory. Content-addressed files are hash-verified on arrival and hardlinked
from the agent blob cache, which survives package-directory eviction. A peer needing companions rejects older agents
before sending any files; monolithic transfer retains protocol 1/2 compatibility.
Resource classification follows actual consumers in the signed index. Missing,
overlapping, unclassified, or incorrectly classified spans fail loading. The
128 MiB metadata limit applies to the complete signed envelope. The package
contains the signing-key fingerprint, not the verification key. Keys come from
Helios's embedded publisher profiles or explicitly supplied profile files.

The exporter writes each cubin or graph as one zstd frame at level 3.
`ExecutableResource.stored` addresses the frame and `decoded_nbytes` is its
signed decoded length. `chunk_nbytes` is 4 MiB and `chunk_sha256` holds the
SHA-256 of each compressed chunk, with count
`ceil(stored.nbytes / 4 MiB)`. The stored and decoded lengths are each bounded
by 4 TiB. The decoder accepts one bounded zstd frame, caps its window at 8 MiB,
and rejects truncated or extra frames, trailing bytes, and decoded length
mismatch.

Signatures authenticate the stored frame and its signed decoded length. Helios
authenticates each compressed chunk against its signed commitment before feeding
it to the decoder; it does not require hashing the entire combined executable
payload first. Acceptance is a trust and license decision: the value passed to
`authorize` equals the signed publisher namespace exactly. It is not key
material. Weights and tables stay uncompressed and retain direct-I/O compatibility
and storage deduplication.

`PackageClaim::read` returns what a package claims before verification: signer
fingerprint, namespace, official source, and development marker, syntax-checked
so a caller can resolve the signer's identity from them. `VerifiedPackage::open`
authenticates the index and publisher binding against the supplied profiles,
for every rank or for one rank of a multi-host package, and exposes the recorded
license. `VerifiedPackage::authorize` requires the acceptance string to equal the
signed publisher namespace exactly, records it, and optionally checks the
stored-data contents. Only its `AuthorizedPackage` result can enter
`NativeModel::load` or `NativeResources::load`. Executable resources are
hash-checked, decompressed, and length-checked before CUDA receives them. Frontend assets retain
signed digest checks. Open package descriptors remain alive through native
use, including deferred frontend and embedding-table reads; replacing the
pathname cannot redirect those reads. Package files must remain immutable while
loaded. Signatures authenticate bytes and do not sandbox executable code.
Compression does not prevent a user controlling the runtime from recovering code.

Source builds contain no embedded publisher profiles:
`helios/src/runtime_package/publishers.json` is an empty array. Helios reserves an
ELF `.flare_publishers` section for release-provisioned profiles.
Embedded profiles verify signatures when the publisher's repository cannot be
consulted; they confer no verified identity. The private release command stamps
a JSON profile array into a separately named executable:

```sh
.venv/bin/python -m ignition.release flare/target/release/flare \
  --output /path/to/distribution/flare --publishers /private/publishers.json
```

Each profile has `name`, `namespace`, and `public_key` (32 integer bytes), plus
an optional boolean `development`. The public key belongs to the Ed25519 key
used by Ignition. Names in profiles are claims; profile import does not grant
execution trust or verify identity. A key with conflicting profile attributes is
rejected.

Flare establishes identity from the publisher's own repository. On every load
that is not offline it fetches `publisher.json` from the root of the signed
official source on huggingface.co and requires the file to list the signing key
under the package's namespace. A file that omits the key rejects the package
outright, remembered or not, which is how a publisher revokes a key. An absent
file or an unreachable repository leaves the identity unverified unless an
earlier load confirmed it, and loading then needs the key from a profile or
remembered acceptance. Flare displays the publisher,
how its identity was established, the source, the recorded license and its
link, and the full key fingerprint. First use asks for the exact namespace,
which agrees to the license terms at the link, and remembers the key, its
namespace, its verification key, and how its identity was established.
Unattended approval requires both `--dangerously-approve-key FINGERPRINT` and
`--dangerously-accept-publisher NAMESPACE`; both must match even if trust was
previously remembered. Trust never transfers to a replacement key merely
because it uses the same namespace. `flare trust revoke FINGERPRINT` removes
remembered acceptance. Identity rests on control of the Hugging Face namespace,
the same account the model itself is published from.

## 3. Address rules

Parameter bytes are replayed verbatim, so a rebuilt process must reproduce
every recorded address. Three driver behaviors, measured on driver 610.57.04
with the `cr-probe` experiments under `~/projects/flare-research`, fix how
that is done.

- The driver satisfies a `cuMemAddressReserve` request from its current
  64 MiB block whenever the request fits, ignoring the requested address, and
  maps a fresh block at the requested address only when the request does not
  fit. Four 16 MiB fixed requests in a fresh process are all moved and the
  fifth is honored; a first request of 64 MiB is moved and a first request of
  128 MiB, 1 GiB, 64 GiB, or 1 TiB is honored anywhere from 8 TiB to 127 TiB.
  A package therefore uses large enclosing reservations and maps allocations
  at their recorded offsets inside them; it never relies on small fixed
  reservations. Preparation reserves a 1 TiB arena; runtime packages use
  GiB-aligned enclosing ranges for retained arena storage and external spans.
- Host memory mapped with `MAP_FIXED_NOREPLACE` and registered with the driver
  receives the same device address in every process.
- Tensor-map parameters embed a global address and are 128 opaque bytes;
  module constant and global symbols hold whatever preparation wrote. With
  identical addresses both replay unchanged.

Conditions that break address reproduction fail loudly: the loader compares
the reserved address with the recorded one and the registered host pointer
with the recorded device pointer before it loads any contents.

## 4. Rebuild and serving

`NativeResources::load` in `helios/src/runtime_package/mod.rs` rebuilds a
package in this order: create a context with the recorded flags, limits, and
cache configuration; reserve the recorded range and compare addresses; back
each device allocation at its recorded address and map each host region;
upload contents or zero-fill; apply storage relocations; load modules and
resolve functions, checking each kernel's parameter layout against the
package; set function attributes; add nodes, attributes, and edges; and
instantiate each graph with its recorded flags. The execution manifest is
then rebased onto the live addresses and validated, and the server starts
from this reconstructed contract. `flare run PACKAGE` in
`flare/src/main.rs` starts the shared server directly from the typed contract.

Before CUDA initialization or serving worker threads, Flare restricts
`CUDA_VISIBLE_DEVICES` to `--gpus`, resolving indices within an inherited mask.
When omitted, the list contains the first N visible GPUs for the package's N
local ranks, or one GPU per host for a multi-host package. Package loading finishes
its progress threads before setting visibility. Hub revision detection queries
NVML in a separate process without initializing CUDA.
Rank order follows the selected list; Helios receives contiguous device ordinals
starting at zero, while loading progress retains the CLI's GPU labels. Flare
sets `CUDA_FORCE_PRELOAD_LIBRARIES=0` and `CUDA_DISABLE_JIT=1`. The loader requires
compiled cubins; missing native GPU code fails instead of compiling PTX.

`LoadOptions` selects runtime concurrency, maximum actual MTP draft count, and
adaptive or fixed prefix selection before CUDA initialization. Every required
power-of-two request bucket has width-one execution. Wider programs fit the
prepared aggregate target envelope. A package must include its configured
maximum prefill chunk; intermediate prefill capacities are optional.

Execution metadata identifies complete speculative programs by B/D/W/P: request
capacity, captured proposal depth, per-request target capacity including the pending
token, and physical aggregate target rows. P is explicit for rectangular programs
as well as ragged programs. D is independent of W and must describe the captured
computation. The loader selects supported reductions from actual programs, rejects
uncaptured capacities, and verifies identical selected catalogs across ranks.
Each program's function dependencies allow kernel selection before graph-body
reads. Graph names do not select semantics.

`QueryLength` inputs bind exact host-selected lengths. `SpeculativePrefixLimit`
inputs bind immutable ceilings and leave actual lengths to a captured GPU causal
allocator. Fixed prefix selection supports host-allocated ragged programs; GPU
allocation requires adaptive selection. A confidence output declares
`confidence_width` and uses FP32 `[2, request_slots, draft_capacity]` frames.

`flare run` accepts `--max-concurrency`, `--mtp-draft-tokens`, and adaptive/fixed
selection flags. Unspecified values retain package defaults. A concurrency limit
of three requires the batch-four bucket. Zero drafts selects ordinary prefill,
decode, mixed, resume, and prefix-cache operations. The optional
`speculation_enabled` field selects complete serving modes for shared operations.
Preparation includes ordinary programs and width-one speculative programs that
preserve draft context. Selection removes unused variants before reading graph
bodies; captured weights and storage remain shared. There are no standalone
proposal or verification operations or separate packed-speculative operation.

Unselected graph templates and executables are never reconstructed. Functions
used only by omitted graphs are dropped, as are unused modules without globals.
Modules with globals retain their persistent contents. Weights, MTP weights,
and fixed scratch remain intact because opaque kernel parameters and persistent
device storage can embed their addresses. `SchedulerLimits.max_active` controls
admission, while `request_slots` preserves the captured physical buffer strides,
and completion-frame extent. The scheduler selects from the
loaded graph catalog instead of deriving a capture budget.

Continuation storage and KV pools use runtime-sized allocation records. Each
record declares its capacity group, bytes per unit, fixed virtual base, alignment,
and maximum units. The package reserves the full virtual extent and stores no
tensor payload or prescribed physical size for these records. Per-rank 1 TiB
reservations place fixed resources in the lower half and runtime-sized regions
in the upper half. Maximum extents are chosen during preparation and validated
for overflow and overlap before loading.

Qwen preparation backs only a prefill chunk plus two pages per captured request
row unless its programmatic configuration supplies an explicit page count.
Forward compilation sees the full virtual tensor shapes. Physical backing is
allocated after compiler workers exit and before warmup and capture; it does
not constrain the serving cache or the checkpoint's context limit.

Shared cache records identify the owning rank, reader ranks, and full virtual
extent. Preparation exports a shareable handle for each physical chunk added
while growing the cache. Helios maps the owning rank's selected physical prefix
and grants every reader access to that prefix; peer records allocate no storage.

Helios maps physical prefixes after loading fixed storage, kernels, and graph
executables and uploading every selected graph. It leaves 64 MiB for driver
bookkeeping, execution streams, and events, plus the package's `reserve_bytes`
when configured. Ignition exports `memory.reserve` as this optional serving
reserve, which defaults to zero; preparation's warm-up and capture headroom does
not reduce the serving budget. All ranks use the same
row and page counts, limited by the rank with the smallest remaining budget.
On integrated GPUs such as GB10, that budget is Linux `MemAvailable`, capped
by CUDA's device total. CPU allocations, registered buffers, and GPU allocations
share physical DRAM; reclaimable page cache contributes to available memory,
while swap does not. Discrete GPUs use CUDA's free device-memory query.
Dedicated rows comprise one null row and the selected request capacity times
the recipe's execution rows per request. KV pages receive the remaining budget,
within their virtual maximum. The planner accounts for mapping alignment and
requires at least 8,192 cached tokens per active request.

Immutable snapshot capacity is bounded by borrowable KV backing and the prepared
virtual extent. At load, CUDA virtual mappings expose the same allocation chunks
through the page planes and the snapshot part of the continuation range. Small
interleaved runs of dense page planes keep each snapshot's conflicting page range
close to its byte cost. Planes whose allocation granules span more pages than a
row's byte cost remain dedicated to KV; the densest planes always participate.
Across ranks, the scheduler uses the union of conflicting ranges.
Ranks loaded in separate processes must report identical cache capacities and
snapshot exclusions during the control-channel handshake; a mismatch rejects
startup before replay. Before cache allocation, hosts agree on the smallest row
and page capacities that all ranks can back.
An owned snapshot excludes all conflicting page ids; overlapping exclusions
are reference-counted. A snapshot fitting entirely in execution-allocation
alignment padding excludes no pages. Execution rows never alias KV pages.

Admission reserves the request's KV pages. System, prompt, and terminal snapshots
acquire backing at their capture boundaries, evicting unpinned checkpoints when
needed. If active requests or transfers hold all suitable backing, optional
retention is skipped. An inbound host restore reserves both its row and pages
before transfer; if its snapshot prevents admission, the request recomputes its
prompt. Snapshot and page ownership remain exclusive through graph
and transfer completion, including cancellation and duplicate publication.
Once generation drains, unused output-page reservations are released before
retaining the final checkpoint.

The loader writes the chosen physical row and page counts into `memory_capacity`.
Maintenance kernels consume those device scalars; kernels preserve fixed row
strides and base addresses. Request-to-row bindings separate transport slots
from execution rows. Virtual addresses and all physical mappings are fixed at
load; changing ownership requires no CUDA allocation or remapping. Model recipes declare row
contents and KV geometry; the memory planner does not depend on recurrent or
attention architecture.

The prepared context extent determines page-table shapes and captured workspace
sizes. The runtime sequence limit is the smaller of that extent and loaded KV
capacity after the model's per-request draft-page allowance. Runtime input packing
preserves the prepared strides. Existing packages retain their prepared ceiling;
raising that ceiling requires export.

The `flare_snapshot_backing_pages` metric counts page ids excluded by owned
snapshots, including rows awaiting retirement. It counts overlapping exclusions
once. `flare_free_snapshots` counts unowned virtual slots; their backing may be
occupied by KV pages.

The loading display separates allocation capacity from payload bytes.
`Allocate model memory` reports GPU and host capacity. `Load weights` counts
only bytes present in allocation payload spans, including initialized model
metadata, and excludes disposable storage and elided zero blocks.

The loader reads only the bounded index before selection. It decodes selected
graph bodies one record at a time, retains dependency bookkeeping for one
graph, and destroys each CUDA graph template after instantiation and upload.
Cubins are mapped from their file spans while loading; no anonymous copy of
the full kernel catalog is required. Frontend assets are read from their own
file spans. The index and loader metadata are released after startup.

Cooperative kernels use resident worker grids and must tolerate a reduction in
their largest grid dimension, whether X, Y, or Z. If restoring the cooperative
launch attribute returns `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`, the loader
queries the kernel's maximum active blocks per SM for its captured block
dimensions and dynamic shared memory.
It multiplies that occupancy by the target device's SM count, reduces the largest
grid dimension to fit while preserving the other two, and retries the attribute
once. A grid that cannot fit by reducing one dimension rejects loading. Fitting
cooperative grids and ordinary launches retain their captured dimensions without
occupancy queries.

The loader releases clean package file pages before CUDA context creation and
after payload loading. On integrated GPUs, distribution and verification can fill
shared DRAM with file cache that the CUDA driver does not count as free memory.
Payload loading opens the package with `O_DIRECT`, bypassing the page cache.
Each rank uses four file-reader threads and four pinned staging buffers of
`4 MiB / rank_count`. Total staging is 16 MiB plus at most one 4 KiB alignment
block per buffer. Reads cover complete 4 KiB blocks; only declared payload
bytes are copied to their destinations. A partial final file block is accepted
when it contains the entire requested payload; missing payload bytes fail loading.
Each rank uploads device chunks asynchronously on its own CUDA stream, with
completion events guarding buffer reuse. Host-resident payloads are copied
from staging into their final mapped allocations. Readers join and uploads
drain before relocations, module loading, or graph execution.

The serving process contains no Python interpreter. Loader tests inspect
`/proc/self/maps` after reconstruction. Startup measurements separate context
creation, address reservation, payload transfer, module loading, graph upload,
and cache allocation.

## 5. The Flare command

Flare resolves its argument to a package, calls the linked Helios library,
supplies the bind address and API key, handles signals, and reports readiness.
Flare owns the terminal banner. Helios supplies loading stages with measured
progress, cycling bars for indeterminate work, results, and elapsed time.
Helios validates the package before reconstruction:
the package's GPU architecture matches the selected device, the recorded
address range is free, free device memory covers the recorded footprint, and
the package's execution contract version is one the Helios build
understands. It does not pin a driver version or a Helios build, because sm_120
SASS loads on any driver that supports the architecture and a package stays
valid across Helios builds that keep the contract.

Flare acquires prepared packages from the Hub and resolves offline packages
through the shared model cache. Model recipes, preparation planning, and kernel
preparation belong to Ignition.

## 6. Tensor-parallel ranks as threads

Tensor-parallel serving uses one Rust process with one CUDA execution thread
per rank. Rank
zero's thread also runs the scheduler. In-process channels carry prepared
steps to peer threads; every rank validates the same fence and uploads before
commit. Completion records must agree on request epochs, token consumption,
and status. A 120-second deadline bounds a stranded collective. Shutdown
confirms drain on every rank before releasing any shared storage.

Preparation uses one Python process per rank. Each process reserves every
rank's 1 TiB arena; rank r owns device base `0x0200_0000_0000 + r * 2^40`
and host base `0x0c00_0000_0000 + r * 2^40`. b12x collective allocations
use the native package allocator. POSIX descriptor passing shares VMM physical
allocations, and importers map each allocation at its owner's recorded address.
Collective signals and generation-tagged payloads are reset after all ranks
drain. Imported mappings are recorded as peer dependencies and stored only in
the owning rank's bundle.

Helios checks that rank allocations do not overlap, creates every device
context, loads rank storage, and grants the declared peer access through
`cuMemSetAccess` before loading modules and instantiating execution state.
Address reservations are established before allocation. Allocation, payload
loading, kernel loading, and execution preparation each run concurrently across
ranks, joining all rank threads between stages. Loading retains a process-wide
16 MiB pinned staging bound and displays aggregate progress for each stage.
Serving uses ordinary shared pointers; no IPC import,
Python process, or socket protocol is involved. Each model and TP size requires
independent numerical and serving validation. The
[rank-thread design](same-process-tp.md) specifies coordination and failure
handling.

## 7. Process images

A CRIU image taken with the driver's process checkpoint support restores a
loaded Helios process bit-exactly: every state class a prepared model holds
was verified, including tensor maps, module symbols, pinned and registered
host memory, two devices with peer access in one process, and legacy IPC
across a process tree launched inside a cuda-checkpoint job. Images are not
the production path. Restore moves the same bytes a package load moves, once
into anonymous host pages and once to the device, so it is bandwidth-bound in
the same way; measured at a 70 GB footprint, restore from an NVMe image took
29 s and from RAM about 13 s. An image is valid only on the driver build that
produced it, bakes in the Helios binary, needs root and CRIU on the host,
holds the whole device footprint in host memory during dump and restore, and
is corrupted if any thread touches GPU-side driver state during the
checkpoint window, including a thread polling `cuEventQuery`.

Images remain a validation tool: image a loaded Helios once, and after a
kernel, driver, or loader change compare a fresh load's replay against the
restored image. The probes, orchestration, and measurements live under
`~/projects/flare-research`.

## 8. Open items

- Full-model relocation requires complete pointer and tensor-map bindings in
  parameters, persistent storage, and module globals. The loader accepts typed
  relocations, but the model exporter preserves addresses for opaque resources.
- The sm_120 TMA tile path links against driver-provided code and reserves
  about 3.6 GiB of device memory per context when loaded; the memory plan
  should account for it once b12x kernels that use the path are identified.
- Legacy allocator-segment packages pack weights in address order, so changes
  in allocation order or size rename later blobs of a rank. Each independently
  exported tensor-parallel size stores a complete copy of its weights.
  [Payload-based model packages and the Flare model cache](package-format-design.md)
  specifies stable layer payloads, payload-based authentication, packed and
  directory canonical forms, and one package containing multiple prepared TP
  deployments. Download materializes selected layouts into a shared Rust-owned
  cache whose entries use the directory form. Aligned W4A16, QMMA, and IQ2
  producer mappings have focused GPU byte-comparison coverage. Unsupported
  mappings retain independently exported bytes, including shard-dependent
  quantization data. Derived TP materialization has byte-exact fixture coverage
  but no full-model TP2 reshard qualification.
