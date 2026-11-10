# Payload-based model cores and the Flare model cache

This document specifies the core format and cache contract.
[Runtime cores](runtime-core-design.md) describes the implemented format;
[TODO.txt](TODO.txt) tracks implementation. Payload format version 2 is current.

The format separates immutable payload bytes, authenticated execution metadata,
and physical container placement. Weight identity depends on prepared weight
bytes, not allocation addresses, graph catalogs, capture capacities, executable
compression, or filenames. One core describes all exported tensor-parallel
deployments. Acquisition produces a ready directory in the Flare model cache;
serving loads its selected deployment without resharding or Python execution.

The model artifact is called a **core** in public commands, APIs, and source
names. This terminology does not change the binary format. Existing field
numbers, signatures, authentication domains, and stored payload bytes retain
their meanings, so previously exported cores need neither re-export nor
re-signing. Compatibility readers retain required historical format markers.

## Canonical forms

The **first canonical form** is one self-contained `.safetensors` file. The
**second canonical form** is a directory containing `model.safetensors` and
readably named companion `.safetensors` files. The Flare model cache always uses
the second form.
These form names describe physical organization. Legacy directory entries retain
their legacy signed schema as specified under loading and compatibility.

Both forms use the same metadata, payloads, signatures, and location schema.
Packing and unpacking change only container placement and safetensors headers.
They require neither a signing key nor a GPU. A size-sharded arrangement is
representable, but is not a canonical form. Directory export does not split or
merge layers according to a size threshold. The packed form has no inherited
64 GB shard limit; lengths and offsets use checked 64-bit arithmetic.

Commands accept a directory core as either its directory or its
`model.safetensors` primary. A primary with external payload locations is
directory form. An explicit directory argument also selects directory form when
all payloads happen to be embedded in its primary. A file with no external
payload locations is packed form.

### Directory form

```text
model/
  model.safetensors
  weights/
    misc.safetensors
    layer-000.safetensors
    layer-001.safetensors
    ...
    engram-0-weights.safetensors
    engram-0-scales.safetensors
    tuned/
      gb202-tp1/rank-0/layer-000.safetensors
    tp2/
      rank-0/layer-000.safetensors
      rank-1/layer-000.safetensors
  deployments/
    gb202-tp1/
      rank-0/
        kernels.safetensors
        graphs.safetensors
        initial-state.safetensors
    gb202-tp2/
      rank-0/
        kernels.safetensors
        graphs.safetensors
        initial-state.safetensors
      rank-1/
        kernels.safetensors
        graphs.safetensors
        initial-state.safetensors
```

Names illustrate semantic roles; absent roles produce no empty file. A model
whose smallest exported deployment is TP2 uses rank subdirectories for its
canonical weights. A layer file includes its attention, MLP or expert weights,
norms, and stable scales. Other ordinary weights share `misc.safetensors`,
including token embeddings, output projections, and components outside declared
layers. Aliased storage is stored once within the captured ownership catalog.
Tuning-dependent layouts occupy separate payloads. TP-specific files exist only
for byte-distinct layouts required by selected deployments.

Offloaded DeepSeek Engram and Qwen PLE tables remain separate from ordinary
weight groups. Each full table uses `weights/engram-<index>-weights.safetensors`
and `weights/engram-<index>-scales.safetensors`, with checkpoint row ordering
independent of TP size. Every host receives the full table; local ranks share
its reader and row cache. Disk backing requires only bounded staging and row
cache memory. DeepSeek retains its TP-sharded projection and reduction, while
its graph requests use canonical full-table row IDs on every rank. Existing
cores with partition-local table rows retain their recorded execution layout.

Hardware names in deployment IDs and cache-entry names use device identifiers
such as `gb202` and `gb10`, producing names such as `gb202-tp1` and `gb10-tp2`.
Device identity comes from the target hardware, not from an SM-version-to-name
guess. Numeric SM architecture requirements remain in compatibility metadata;
loaders validate that metadata independently of descriptive names.
Payload exports record the captured device name in the signed core's
`hardware_identity`. Cluster deployment IDs may be arbitrary identifiers.
Cores exported without a device name remain usable; when their authenticated
metadata cannot identify the device, acquisition uses an
`unknown-device-tp<N>-<UTC minute>` cache label. The downloader never substitutes
its local GPU's name for the core's target.

Every core file in this tree is a valid safetensors container. There are no
raw `.bin` companions and no root JSON index. The one primary contains the
deployment catalog, including every rank and its graph catalog. Existing
`publisher.json` identity discovery may accompany a published repository as
described in [signing](signing.md); it is not a core index or required to
address payloads. Trust and namespace acceptance remain external to container
placement, including when importing a standalone packed file.

### Packed form

```text
model.safetensors
  flare.manifest
  flare.locations
  flare.derivations                 # Optional.
  frontend.tokenizer
  frontend.chat-template
  weights.embedding
  weights.layer-000
  weights.layer-001
  ...
  deployments.gb202-tp1.rank-0.kernels
  deployments.gb202-tp1.rank-0.graphs
  deployments.gb202-tp1.rank-0.initial-state
  ...
```

These are tensor names inside the container. Each payload is a one-dimensional
U8 tensor containing its exact stored bytes, including packed weight bytes or
compressed program bytes. Packing embeds payloads, not nested safetensors files.
Tensor names and filenames are readable locators, not content identifiers.

A packed core is self-contained for the deployments it exports: each required
payload is present or derivable from payloads present in the file. A directory
distribution may have the same source-complete representation. A ready cache
entry instead contains every final payload needed to load its selected
deployments directly. Exporting a ready directory as a packed file includes its
selected closure; packing a source-complete distribution preserves its derivation
closure. Neither conversion silently fetches missing sources. A core retaining
metadata for unselected deployments advertises those deployments as unavailable
when their closure is absent.

An explicitly selected deployment with an incomplete closure is an error, never
silently dropped. Selection comes from the caller; unsigned availability hints
cannot reduce it. Conversion preserves the input's available selections unless
the caller explicitly requests a subset.

## Per-file schema

Safetensors supplies framing and offsets. Protobuf supplies typed metadata.
Header metadata identifies the format version; it is an untrusted dispatch hint
checked against the authenticated format version. Parsers reject duplicate
entries, invalid extents, overlapping tensor storage, unsupported types, integer
overflow, and metadata beyond explicit limits before allocating from them.
Metadata retains a 128 MiB per-tensor limit; bulk payloads do not share that limit.

| File | Tensor | Contents |
| --- | --- | --- |
| `model.safetensors` | `flare.manifest` | U8-encoded core envelope, signed shared body, deployment bodies, and rank bodies. |
| `model.safetensors` | `flare.locations` | U8-encoded unsigned physical locations and locally available deployment selections. |
| `model.safetensors` | `flare.derivations` | Optional U8-encoded bounded byte-copy/fill recipes for reconstructing signed payloads. |
| `model.safetensors` | `frontend.tokenizer`, `frontend.chat-template` | Referenced frontend payloads; format and interpretation come from signed frontend metadata. |
| `weights/*.safetensors` and nested weight files | `flare.payload` | U8 stable weight, tuning-dependent weight, or TP-specific weight payload. |
| `deployments/<id>/rank-<n>/kernels.safetensors` | `flare.payload` | U8 concatenated zstd-compressed cubin resources, addressed by signed slices. |
| `deployments/<id>/rank-<n>/graphs.safetensors` | `flare.payload` | U8 concatenated zstd-compressed graph resources, addressed by signed slices. |
| `deployments/<id>/rank-<n>/initial-state.safetensors` | `flare.payload` | U8 retained initialization bytes outside stable weight payloads. |

The packed primary adds named payload tensors in place of companion files.
Directory companion wrappers contain only their payload and format framing;
they contain no deployment-specific metadata. Their deterministic encoding allows
one immutable cache object to be hardlinked into several model entries.
Several cubin resources can occupy slices of one kernels payload, and several
graph resources can occupy slices of one graphs payload. Each such companion
still contains exactly one payload tensor.

### Semantic records

The following records define the schema responsibilities. Byte counts, offsets,
and extents are unsigned 64-bit values with checked conversions at every host
and CUDA API boundary. Identifiers have explicit namespaces; display names are
not substitutes for identifiers.

| Record | Fields and meaning |
| --- | --- |
| `CoreBody` | Format version, core identity, publisher identity, shared frontend contract and payload descriptors, deployment IDs and commitments to their serialized bodies. |
| `DeploymentBody` | Deployment ID, target architecture, TP topology, prepared contract and capacities, rank IDs and commitments to their serialized bodies. |
| `RankBody` | Rank ID, context and address reservations, allocations, initializers, modules, functions, graph catalog, events, peer/transport bindings, referenced payload descriptors, and optional prepared-weight catalog. |
| `Payload` | Content ID equal to SHA-256 of exact stored bytes and byte length. Conflicting lengths for one ID are rejected. |
| `PayloadSlice` | Payload ID, byte offset within the payload tensor, byte length, and use role: weights, tuned weights, initial state, kernels, graphs, or frontend. |
| `Allocation` | ID, memory kind, recorded address, byte length, alignment, declared runtime sizing, and supported typed relocations. |
| `Initializer` | Allocation ID, destination offset, source `PayloadSlice`. |
| `ExecutableResource` | Preassigned resource ID, kind, stored compressed `PayloadSlice`, signed decoded length, 4 MiB chunk metadata, and compressed chunk commitments. |
| `Module` / `Function` | Cubin resource reference; module reference, symbol, parameter layout and launch attributes. |
| `Graph` | Operation and capacity selectors, typed input bindings, compressed graph resource reference. Graph bytes encode nodes, edges, parameters, and dependencies. |
| `PreparedWeightView` | Semantic tensor name, stable group, dtype, shape, strides, allocation/view binding, source tensor identities and logical ranges, concatenation order, padding/fills, typed packing parameters and locality tile, and quantization statistic domain. |

Payload descriptors are authenticated wherever consumed, with equality enforced
across shared and rank bodies. Each body is signed over its exact serialized
bytes with a versioned domain separator. Parent commitments bind child bodies;
verification must not parse and reserialize before checking a signature. A rank
can authenticate its execution closure without reading another rank's payloads.
Metadata and its signatures are not members of their own payload digest catalog.
Shared consumers resolve descriptors from the authenticated core body; rank
consumers resolve from that body or their authenticated rank body. An unrelated
rank's descriptor cannot authorize a selected rank's resource. Verification of
rank metadata includes its deployment and core commitment chain.
Roles belong to authenticated uses, not content identity: identical bytes used
in different roles can share storage. Each consumer validates the permitted role
and resource structure for its binding; sharing bytes grants no extra authority.

Prepared weight catalogs encode source tensor identities and half-open logical
ranges per axis, concatenation axis and order, destination padding regions and
fill values, and typed packing parameters. Packing parameters include format,
block/group size, scale atom geometry, tile order, row rotation where applicable,
and the locality tile on which slicing commutes with packing. Quantization
statistics declare their tensor, expert, or shard domain. These records describe
how to match byte regions; Helios serving does not interpret them. Exact target
byte comparison remains required even when these declarations agree.

Rank metadata stores a prepared weight catalog inline in `prepared_weights`
or as one zstd level-3 frame in `compressed_prepared_weights`. The compressed
form includes its decoded byte length; both forms together are invalid.
Exporters use compression when it reduces the encoded size. Readers retain
support for inline catalogs. Rank signatures authenticate compressed bytes
before catalog decoding. Decoding requires an exact length, a window no larger
than 8 MiB, and one complete frame without trailing bytes. Reconstructing a
rank's inline metadata must fit within 128 MiB; selected ranks together must
fit within 512 MiB. The signed envelope retains its 128 MiB limit. Keyless
container conversion preserves compressed catalogs without decoding them.
Verification checks the size bounds for all selected ranks before decoding
any catalog. Consumers decode one catalog at a time; verified rank records
retain the compressed form.

An unchanged checkpoint tensor may contribute several affine byte pieces to a
retained owner, including expert concatenation and scale swizzles:

```text
WeightSource {
  tensor: string
  ranges: [{ begin: u64, end: u64 }]
  concatenation_axis: u32
  pieces: [WeightPiece]
  piece_encoding: string
}
WeightPiece {
  source_offset: u64
  destination_offset: u64
  contiguous_bytes: u64
  extents: [u64; 0..6]
  source_strides: [u64; 0..6]
  destination_strides: [u64; 0..6]
}
```

With empty `piece_encoding`, source offsets address raw bytes of the named
checkpoint tensor. A nonempty encoding identifies a deterministic encoded byte
coordinate system for that source tensor. Destination offsets address bytes of
the retained owner before payload grouping and zero elision. Strides and
contiguous lengths are bytes, while extents count repeats.
The three outer-axis arrays have equal lengths. Checked arithmetic bounds each
piece within its owner; the producer validates source-coordinate bounds. Raw
pieces assert unchanged checkpoint bytes. Encoded pieces assert byte identity
within the named encoding; they never instruct the materializer to encode,
quantize, convert dtypes, or perform arithmetic. Producers record the finalized
retained layout after preparation mutations.

For example, `w4a16_mma_k16_n64_v1` identifies 512-byte weight tiles spanning
16 input channels and 64 output channels in the W4A16 MMA packing. Each named
checkpoint tensor has its own full-source grid in input-tile-major,
output-tile-minor order. A tile at `(k, n)` begins at
`(k * full_output_tiles + n) * 512`. Expert and fused-projection placement affect
destination offsets, not the source tensor's coordinates. Partition boundaries
must preserve whole tiles. Scales with shard-dependent re-encoding cannot use
the weight encoding to assert byte identity.

Absent or empty `pieces` preserves the source-range and packing semantics of
previously serialized catalogs. Absent `piece_encoding` means raw checkpoint
coordinates. The combiner requires matching source tensor and encoding
identities before intersecting source regions,
translates owner regions through allocation initializers, and lowers supported
relations to the bounded copy operations described below. The six provenance
axes describe layout; they do not extend the materializer's operation set.

Allocation geometry and execution bindings are independent of disk grouping.
One allocation may initialize from several layer payloads and state payloads.
Several allocations or ranks may read the same payload or overlapping source
slices. Destination initializer ranges must be disjoint and within allocations;
source slices must be within their declared payloads. Allocations start zeroed,
and omitted regions therefore have explicit zero semantics. Nonzero initialized
bytes must be supplied. Aliased tensor views continue to refer to their shared
allocation storage.

The format preserves the prepared model contract, graph dependencies, peer
bindings, cache geometry, and frontend semantics described in
[the model contract](../helios/model-contract.md) and
[the execution manifest](../helios/manifest.md). It does not introduce a graph
operation family or model-specific native policy. Fixed virtual addresses remain
supported for opaque captured resources; complete model relocation is not a
prerequisite for stable disk payloads.

### Physical location records

`flare.locations` contains records of the following shape:

```text
PayloadLocation {
  payload_id: bytes[32]
  file: relative path             # Empty means this primary.
  tensor: string
}
```

Each selected materialized payload resolves to exactly one location. Multiple
semantic consumers may share it. A file/tensor location supplies the whole
payload; `PayloadSlice` supplies any subrange needed by a consumer. There is no
synthetic logical-file address space or global concatenated offset scheme.

Locations are outside the signature. Resolution checks relative paths, prohibits
traversal and symlink escape, validates safetensors framing and declared lengths,
and opens immutable files beneath the core root. Open handles remain pinned
through deferred reads and asynchronous uploads. A pathname-reopening LRU cannot
replace those handles. Local availability claims are hints: readiness follows
from validating the selected deployment's actual payload closure.
Directory locations name entry-relative hardlinks such as
`weights/layer-000.safetensors`, never paths into the global blob store. Export
reuse queries the store by payload ID and constructs those links before publishing.

## Authentication and conversion

Payload identity excludes the safetensors header, tensor name, filename, and
containing file. Signed metadata binds payload identities, lengths, roles,
allocation-relative destinations, and all execution and deployment semantics.
Changing a file boundary or tensor name without changing a payload is keyless.
Changing payload partitioning, bytes, or signed execution bindings requires a
publisher signature. Arbitrary size splitting is representable by placement of
existing payloads; cutting one payload into different payloads changes identity.

Packing and unpacking preserve signed bodies, signatures, and stored executable
bytes exactly. They validate their inputs and final closure. Identity promotion
through `ignition sign` remains a signing operation and is not equivalent to
container conversion.

The exporter preserves kernel and global symbol names in cubins and their
execution bindings. Cubin debug sections and assertion source paths are stripped.
Each executable resource is written as one zstd frame at level 3.
`ExecutableResource.stored` addresses the frame and `decoded_nbytes` is its
signed decoded length. `chunk_nbytes` is 4 MiB and `chunk_sha256` holds the
SHA-256 of each compressed chunk, with count
`ceil(stored.nbytes / 4 MiB)`. The stored and decoded lengths are each bounded
by 4 TiB. The decoder accepts one bounded zstd frame, caps its window at 8 MiB,
and rejects truncated or extra frames, trailing bytes, and decoded length
mismatch.

Signatures authenticate the stored frame and its signed decoded length. Helios
authenticates each compressed chunk against its signed chunk commitment before
feeding it to the decoder; it does not require hashing the entire combined
executable payload first. Acceptance is a trust and license decision: the value
passed to `authorize` equals the signed publisher namespace exactly. It is not
key material.

Downloads and locally materialized payloads are hashed against their signed
content IDs before installation. Cached loading retains the optional full weight
scan: `--verify-weights` hashes allocation initializers and shared or rank-sharded
embedding-table planes. Ordinary loading validates their signed descriptors and
physical bounds without hashing their contents. The full scan hashes actual
payload bytes, not digest lists. A signed
digest without a byte scan is not evidence that local weight bytes were checked.
Executable authentication remains mandatory. A consumer using only a slice of a
payload may need the whole payload for full digest verification; download planning
and progress account for those extra bytes. Full verification hashes each
distinct referenced payload once, even when many initializers consume its slices.

## Stable weight extraction

Recipes and reusable weight components declare semantic owners and groups.
The prepared-weight catalog records views and transformation information, while
Heliopause resolves those views against retained capture storage. Automatic
object traversal alone does not establish that a tensor is immutable weight
storage. Graph state-access declarations alone do not prove that opaque kernels
never write a proposed weight range.

Stable owner bytes depend only on checkpoint bytes, declared packing, and the
TP partition. They exclude pointers, tensor maps, counters, warmup residue,
capacity-dependent padding, and tuning-dependent layouts. Shard-local quantized
weights may be stable within one TP deployment without being shareable across
different TP sizes. The exporter never requantizes to force sharing.

Owners are ordered deterministically within their declared layer or the
`misc` group for weights outside layers. All stable families in a layer share its payload. Group sizes never
cause adjacent layers to merge. A change to one layer cannot shift another
layer's payload boundaries. Tuning-dependent representations are grouped
separately and name their deployment and rank when necessary.

Storage extraction preserves precise occupied byte intervals, including strided
views and aliases. A strided view's bounding extent does not grant ownership of
its holes. Overlapping views are resolved deterministically to one stored copy
of each shared byte, and all view bindings are retained. Export must account for
every retained live byte as stable weight, tuned weight, initial state, or
explicit zero/disposable storage. Unclassified retained bytes go to initial
state; they are never silently discarded.

Zero-block elision uses fixed 64 KiB blocks relative to stable owner intervals,
not allocator placement. Elided bytes reconstruct as zero through allocation
initialization. Stable group ordering, overlap resolution, and elision rules are
part of the packing contract. State bytes and their elision are independent of
stable weight ordering. Disk payload grouping does not require separate GPU
allocations for each group.

## Tensor-parallel deployments and materialization

The canonical weight representation uses the smallest TP size independently
exported and validated for that model and hardware target. That size may be 1,
2, 4, 8, or larger. A core cannot manufacture an unexported TP deployment:
every supported size has its own prepared kernels, graphs, allocations, and
contract. Hardware targets with incompatible prepared bytes do not implicitly
share a canonical source set.

Sharing follows exact byte relations:

| Relation | Representation |
| --- | --- |
| Identical payload | Shared payload ID. |
| Contiguous region | Signed `PayloadSlice` referencing canonical bytes. |
| Affine rearrangement | Materialized target payload from bounded one-, two-, or three-dimensional byte copies and fills. |
| Different quantization or incompatible packing | Independently stored TP-specific payload. |

Input-sharded columns and some tiled W4A16 layouts require rectangular copies.
Scale atom boundaries can prevent sharing even when the corresponding weight
rows share. Quantization based on a shard's absolute maximum can change actual
values and requires per-TP bytes. A layer containing both reusable weights and
shard-dependent scales need not retain a complete independent layer payload.
The combiner stores the uncovered target ranges in a signed TP-specific
exception payload and reconstructs the layer from canonical copies plus copies
of those exception bytes. Exception companions use
`weights/exceptions/<deployment>/rank-<rank>/<name>-exceptions.safetensors`,
where `<name>` is the target's semantic filename without its extension, or
`exception-<ordinal>` when no semantic filename exists. A colliding path is
scoped under `weights/shared/<deployment>/rank-<rank>/` with a numeric prefix.
A layer without any verified reusable bytes remains independently stored.

Exception extraction changes the authenticated payload catalog and requires a
signing key. It streams exact ranges from the independently exported target with
bounded memory, preserves their destination offsets in the reconstruction
recipe, and verifies the complete reconstructed target. It does not infer fill
values or reuse the entire target as its own source. Helios performs no model arithmetic, dtype
conversion, quantization, or kernel-layout inference during materialization.

A direct subrange reference creates neither a payload nor a derivation: the
initializer names the canonical payload and its slice. A derivation exists only
when a distinct stored target payload must be materialized.

### Derivation schema

`flare.derivations` describes how to obtain a target already identified by signed
payload metadata:

```text
Derivation {
  target_payload_id: bytes[32]
  operations: [Copy | Fill]
}
Copy {
  source_payload_id: bytes[32]
  source_offset: u64
  destination_offset: u64
  contiguous_bytes: u64
  extents: [u64; 0..2]
  source_strides: [u64; 0..2]
  destination_strides: [u64; 0..2]
}
Fill {
  destination_offset: u64
  byte_length: u64
  pattern: bytes
}
```

The contiguous run plus up to two strided outer axes represents up to three
dimensions. Offsets, strides, and contiguous lengths are measured in bytes;
extents count iterations. At outer indices `i` and `j`, copy `contiguous_bytes`
from `source_offset + i * source_strides[0] + j * source_strides[1]` to the
corresponding destination expression; absent axes contribute zero. All three
outer-axis arrays must have equal lengths. More complex tilings use multiple
bounded operations rather than a higher-dimensional instruction or numerical
transform. Expanded destination ranges must be disjoint and exactly cover the
target, including explicit padding fills. All arithmetic, source bounds, output
bounds, operation counts, pattern sizes, and dependency depth are bounded before
work begins. Cycles and unsupported operations are rejected. Recipes reference
payload IDs only; they cannot request arbitrary paths, network URLs, or code.

Derivations are untrusted acquisition hints. They cannot alter signed execution
bindings, and their complete output must match the signed target length and
digest before publication. A derivation can therefore be added without a key
when it reconstructs an already signed target. Source identities and required
lengths must resolve through authenticated descriptors or a validated dependency
closure. No derivation is executed by the serving loader.

All operation offsets address stored payload bytes after zero elision, not the
allocation's expanded byte image. A fill supplies bytes present in the target
payload, including stored padding; it does not insert allocation-only zero holes.
The combiner translates owner and allocation regions through both exports'
initializer maps before describing copies. `flare.derivations` resides in the
primary in either canonical form.

### Combining independently prepared deployments

The Rust combiner consumes independently exported deployments and their prepared
weight catalogs. It checks compatible model/checkpoint and hardware identities,
matches semantic owners, intersects logical source regions, and proposes byte
relations from the declared packing and padding. Each deployment retains its own
preparer source revision, and those revisions may differ; declared checkpoint,
recipe, and preparation semantics must agree. Engram vocabulary, table encoding
geometry, and hash methods, orders, multipliers, and primes must agree.
TP-specific plane partitions and hash row offsets remain per deployment with
that deployment's own payloads retained. Shared tables still require exact
equality.
The combiner compares every candidate against the independently
exported target bytes. Only exact equality permits sharing or a derivation;
unsupported relations retain the exported bytes.

Constructing a core catalog or replacing a rank's initializer bindings with
canonical slices changes signed semantics and requires signing the affected
bodies and their parent commitments. Constructing reconstruction hints for
unchanged signed targets does not. Combining requires no GPU and does not
substitute its byte comparisons for qualification of each exported deployment.
Its report distinguishes directly shared, locally derivable, and independently
stored bytes, including reasons for failed sharing candidates.

Tiled W4A16, QMMA, and IQ2 weight producers describe aligned encoded tiles;
shard-dependent scales require independent bytes. Focused GPU tests compare
independently prepared TP1 and TP2 owners, including MXFP4-CSF gate/up rotation
and both compressed-scale and expanded-scale NVFP4-CSF preparation. Nemotron
Bolt's IQ2 metadata also
passes complete TP1 export and native text, streaming, prefix-reuse, and
concurrent-request tests. A signed fixture containing W4A16 owners,
scales, and globals passes export, combination, keyless packed/directory
conversion, and cache materialization with exact initializer and global bytes.
Conversion preserves signed bodies and stored executable frames. All fixture weight bytes are copied from
canonical payloads; only scales and global scalars occupy exception payloads.
The piece mappings account for different shard-local FC1 row rotations.
Full-model qualification covers TP1 only; no full-model TP2 reshard has been
qualified.

Native rank distribution passes cold and warm staging between two ARM hosts
over CX7 with a signed TP2/TP4 fixture. The receiving host stores only its
selected TP2 rank and shared payloads, and the warm run reuses the verified
cache entry. This fixture exercises transport and publication without GPU
loading; its synthetic graphs do not qualify model execution.

### Download behavior

`flare download --tp N` selects a prepared deployment and completes all fetching,
resharding, verification, and cache installation before reporting it ready.
`flare run` may invoke the same acquisition path for a remote or incomplete
source, but reports that work as download/acquisition before native model loading.
Startup of a ready cache entry performs no resharding.

Download transport keeps a shared tree per download parent at
`<parent>/transport/hub/v1`, marked by a `version` file and holding
content-addressed complete blobs, resumable `.part` files, quarantined suspect
bytes, transient adoption copies, per-content transfer locks, and repository
refs and snapshot hardlinks.
Blob content is scoped by endpoint and repository so Hub ETags are never shared
across repositories; revisions of one repository share content. An acquisition
lease excludes reclamation for the lifetime of every acquisition. The tree is
separate from the model cache and survives across acquisitions for
cross-revision reuse.
A successful installation reclaims its resolved commit's snapshot entries and
the consumed complete blobs whose last hardlink they held. Failed partials,
quarantined suspect bytes, and content referenced by another commit remain.
`flare models downloads` inspects the tree and `flare models gc --downloads`
reclaims remaining unused content under the per-content locks. Model-cache collection preserves unlocked
staging trees that contain the older Hugging Face layout so their resumable
partials survive until download-cache reclamation. [The CLI design](cli-design.md#download-cache)
defines the command interface and reclamation rules.

Selecting TP1 and TP2 produces one primary with both selected deployments and
the union of their required payloads. Identical and directly referenced bytes are
shared; distinct materialized layouts consume additional storage. Adding TP2
reuses locally held canonical sources and already materialized target payloads.
Derivation sources are retained by default for later selections. Source eviction
must preserve every selected deployment's loadable closure.

The materializer streams bounded source buffers into destination files and
overlaps acquisition, copies, and writes where dependencies allow. It hashes
source streams and final targets. Out-of-order destination writes may require a
final sequential target read; that cost belongs to download. Local alternative
TP layouts may supply canonical regions through verified invertible affine
mappings. Such reuse requires exact coverage and byte verification; unsupported
inverse relations fetch canonical sources instead of guessing an inverse.
Coverage and invertibility are computed from the checked copy extents and
strides, not asserted by an untrusted flag. Only injective copied regions can be
inverted; fills cannot recover a source. Composed mappings must cover every
requested source byte and remain within the same operation and depth bounds.

Progress reports network bytes, local source reads, materialized bytes,
verification work, and elapsed acquisition time separately. Performance targets
apply to the complete acquire-and-ready path, including verification and writes.

## One Rust-owned model cache

The configured `paths.core_cache`, default `~/.cache/flare/cores`, owns
prepared model storage. Hugging Face remains an acquisition source; its checkpoint
cache is not mutated into a prepared-model cache. Existing Ignition export and
Flare/Orbit prepared-core storage converge on this one implementation. Native
executable caches with distinct purposes remain separate.

```text
~/.cache/flare/cores/
  blobs/<payload-sha256>.safetensors
  models/<model-key>/<entry-id>/
    model.safetensors
    weights/layer-000.safetensors           # Hardlink to an immutable blob.
    deployments/gb202-tp2/rank-0/graphs.safetensors
    ...
  refs/...
  sources/...                             # Acquisition selectors to cache entries.
  origins/...                             # Per-version acquisition origins.
  staging/...
  locks/...
```

Model keys encode source identity without trusting user-supplied path components.
Entry IDs distinguish immutable published selections; mutable refs select an
entry. The payload store uses hashes internally, while model entry filenames
remain descriptive. Blob keys hash payload bytes, not safetensors wrappers.
Frontend payloads can remain embedded in the small primary.

Each model's mutable ref selects its default version. A successful export or
download automatically switches that ref to the resulting entry, including when
a download reuses an existing cache entry. The switch occurs only after the
complete selected deployment closure is ready. Preparation, acquisition, or
verification failure leaves the default unchanged; a failure while writing the
ref itself is reported as indeterminate under entry publication below.
Concurrent operations serialize their ref
updates; the last successful ref update selects the default. Switching the default
retains preceding entries and does not change versions held by running processes.
An explicit version selection overrides the default during resolution.

Unqualified model names resolve the local default before repository lookup.
HF-acquired versions retain their repository, requested revision, resolved commit,
filename, TP selection, and signed-envelope binding in a per-version origin record.
A moving acquisition selector cannot erase an earlier version's origin. Online
runs check that recorded source and automatically acquire and select an available
update; an unchanged commit transfers no core bytes. Locally exported versions
have no download origin. Explicit versions and offline runs do not check for
updates. Origin records are removed with their versions.

Offline repository resolution uses only acquisition references and entries in
the Flare model cache. Successful downloads record the repository, requested
revision, resolved commit, filename, deployment selection, and the SHA-256 of the
exact signed envelope authenticated during acquisition. A branch resolves its
last successfully acquired commit; an unavailable deployment must not select an
older commit implicitly. Offline resolution verifies the selected entry without
reading or importing the Hugging Face cache. The original four-field source
record remains readable but has no expected envelope and is rejected offline
until an online acquisition refreshes it; a record whose entry envelope differs
is rejected before the default changes. Source references do not retain entries
against eviction; a missing entry requires acquisition again.

Payload installation writes a deterministic one-payload safetensors wrapper into
staging, verifies the actual bytes and length, and publishes it atomically under
the payload ID. Concurrent writers coordinate through stable lock files and reuse
an already verified immutable object. Hardlinks connect that object to all entries
that consume it. A store and its entries must be on the same filesystem.
Exporting to another filesystem uses copying or a supported filesystem clone;
an unavailable hardlink must not result in a partially published entry.
Copies on another filesystem do not share storage with the originating cache.
Such outputs are standalone export artifacts, not entries owned by that cache's
refs, leases, or garbage collector. Import into another cache uses that cache's
own publication transaction and object ownership.
Filesystem file-size or space limits cause an explicit preflight/export error;
they do not silently repartition a signed payload or change the canonical form.

Hardlinked files are never edited in place. Adding a deployment builds and
publishes a replacement entry, then atomically switches its ref. Readers holding
the preceding entry continue using its open handles. A ready entry is published
only after metadata, selected payload closure, and required materialized targets
are complete. Interrupted staging is not a model entry. Publication includes
the file and directory synchronization needed for crash-safe visibility.

Leases pin entries and their payloads for loading, serving, acquisition, and
Orbit transfer. Garbage collection coordinates with publication and leases;
it removes an unreferenced payload only when no entry or active operation owns
it. Eviction unlinks model entries and releases their references. It neither
mutates shared inodes nor removes data still used by another model or rank.
The design must permit reclamation of abandoned staging and unreferenced blobs.

Ref resolution and lease acquisition form one operation coordinated with eviction:
acquire the relevant lock, resolve the ref, pin the entry and required objects,
then release the coordination lock. Use process-owned file locks, released by the
kernel on process death, rather than persistent Boolean lease markers or timed
expiry of live readers. Keep lock inodes stable while concurrent processes may
refer to them. Staging recovery checks these locks before removing abandoned work.

Serving holds an entry-scoped lease; the immutable entry's complete payload set
remains pinned without a separate lease descriptor per payload. A staging
transaction likewise holds one lease over its acquired objects. Published entry
locations and live staging ownership determine reachability, not mutable refs or
filesystem link counts alone. Publication coordination protects the interval
between installing an object and linking it into staging. Eviction exclusively
locks an entry before removing it; garbage collection cannot race that operation
or a publication. A ref selects an entry but does not enumerate all retained
entries. External hardlinks may keep disk space allocated after cache reclamation.

### Entry names

A published cache entry is named for its selected deployments:
`<platform>-tp<N>-<UTC timestamp>`, for example `gb202-tp1-202610170923`.
An entry containing several TP sizes joins their labels in ascending order,
without duplicates: `gb202-tp1+tp2-202610170923`.
The platform is the device identifier recorded by preparation (`gb10`,
`gb202`, or `gb300`), never an SM architecture name. Downloads use the authenticated device
identity; cores without one use `unknown-device`, including union names such
as `unknown-device-tp1+tp2-202610170923`. The timestamp records entry creation in
UTC with minute resolution.

A second export completing in the same minute for the same platform and size
takes the next free numeric suffix: `-2`, then `-3`, and so on. Reservation and
the collision check happen in the cache before staging begins, so concurrent
exporter processes never choose the same name and an existing entry is never
overwritten. Reusing an existing entry keeps its original timestamp; only a new
entry receives a new one.

Within one rank deployment, residual initial state stored as a shared cache
object and locally staged module globals use distinct canonical files
(`initial-state.safetensors` and `initial-state-globals.safetensors`); both carry
the initial-state role and each initializer names the payload that supplies it.

Entry names are display metadata. Payload identity, core identity,
signatures, and selected tensor-parallel unions are unchanged by renaming, and a
rebuilt core does not invalidate an entry whose bytes are unchanged.
Successful export updates the model's default after the complete cache entry
has been published. `ignition export` accepts no standalone destination or form
selection. `ignition convert` creates standalone packed or directory cores
from the printed cache entry path; a failed standalone conversion leaves that
entry and its default reference unchanged. A failure while writing the default
ref may occur after the ref rename, so the export reports publication as indeterminate, retains the
complete entry, and does not rewrite or clear any default: another exporter's or
user's concurrent selection is never rolled back to a snapshot. Automatic
cleanup therefore never removes a selected entry; otherwise a published entry is
removed only by an explicit cache removal. Abandoned staging trees are reclaimed
by collection. Download entry naming uses the same scheme.

`--no-weights` export requires the stable and tuned weight payloads it names to
exist in the cache. It still prepares the checkpoint and reads captured storage
to establish exact content identity; missing or changed weight data fails before
entry publication, with the missing groups named.
Initial-state payloads are not weights: a graph or capacity change may produce
new state bytes, which such an export writes normally while the weight payloads
reuse their cached objects. The export avoids temporary allocation files and
duplicate weight writes and does not trigger download or TP materialization.
Changed graphs, compressed executable resources, and initial state may still
produce additional payloads.

### Ownership and integration

| Component | Responsibility |
| --- | --- |
| Helios | Shared Rust core reader/writer, payload hashing, signature authentication, executable decompression, container conversion, byte derivation, immutable cache storage, locks, leases, publication, hardlinks, and reclamation. |
| Heliopause | Capture and export integration; own all export file management using Helios core/cache operations, including staging, destination creation, conversion, hardlinks, publication, and cleanup. |
| Flare | Source acquisition, HTTP/Hub resolution, deployment selection, user-facing download/cache commands, and Orbit integration using Helios facilities. |
| Ignition | Python model preparation and semantic weight declarations; thin calls into Heliopause for core and cache work. |

Helios has no Python or preparation-tool dependency. Core/cache modules are
usable without creating a CUDA context. Callers provide acquired byte streams
to the shared writer/materializer; HTTP policy stays in Flare. Python does not
implement a second serializer, signature scheme, resharder, or cache manager.
Serving replay does not perform acquisition, filesystem copying, compilation,
recapture, or steady-state GPU allocation.
Heliopause writes export staging and payloads through this shared publication
transaction and holds leases through rank transfer and entry publication.
Ignition passes prepared resources, semantic declarations, destinations, and
export options to Heliopause. Heliopause chooses temporary paths, manages rank
files, checks destination conflicts, and cleans up failed exports. These rules
apply to explicit destinations and cache-only exports.
Configured export staging remains temporary workspace; it is not an independent
persistent model store.

## Loading and compatibility

The loader resolves the selected signed rank metadata, validates allocations and
initializers, pins all required files, and schedules contiguous payload slices
through bounded pinned staging and asynchronous upload. Preserve the implemented
process-wide staging bound across ranks; adding ranks must not multiply it.
Preflight the open-file limit against the selected closure. Coalescing adjacent
reads may improve throughput without changing initializer semantics or handle
lifetimes. GPU storage and resource owners outlive all queued work.

Format version dispatch is explicit. Payload format version 2 identifies the
executable representation written by export, conversion, and signing.
Legacy readers accept `ProtectedIndex` envelope 6. Absent Protobuf fields use
their declared defaults.

Legacy cache entries also use a primary-plus-companions directory, retaining the
legacy primary's authenticated semantics and accepted companion encoding. Their
compatibility reader does not pretend those signatures authenticate payload-based
metadata. Legacy immutable companions use container-digest keys under
`blobs/legacy/`; payload-format companions use payload-digest keys under `blobs/`.
Both use the same Rust publication, hardlink, lease, and reclamation facilities.
Legacy cores remain readable and servable; conversion and signing accept
only payload-format cores.
Legacy companions remain direct siblings of the primary under the exact
container-digest basenames expected by the legacy layout. They are not moved into
the payload format's `weights/` or `deployments/` subdirectories.

Existing export and download cache contents can be read and imported through
the appropriate reader. Import verifies or reconstructs the payload identities
needed by this cache before publishing an entry. It must preserve existing model
files, user configuration, and active leases; no automatic destructive cache
reset is part of adoption. Removing duplicate internal cache machinery follows
migration of its actual export, Hub, local-core, and Orbit consumers.

## Implementation dependencies and validation

Implement the shared semantic schema, payload reader/writer, and compatibility
reader before wiring consumers to the format. Add pack/unpack and authentication
using the same primitives. Implement immutable cache publication and leases before
moving Heliopause export, Flare acquisition, or Orbit onto it. Stable owner
extraction enables useful weight reuse; independently prepared TP catalogs and
verified derivations then enable download-time resharding. CLI reporting and
cache eviction must use the same shared lifecycle.

Required validation covers behavior, not source-text checks:

- Repeated export, graph-only changes, allocation address changes, and capacity
  changes preserve stable weight payload IDs. Changing one layer affects only
  its stable group; tuning-dependent changes affect their separate groups.
- Strided and aliased owners preserve all live bytes, including unclassified
  residual state. Sparse initializer loading reproduces independently prepared
  allocations and their zero regions.
- Packed/directory round trips preserve payload bytes, signed bodies, and
  stored executable frames without keys. Malformed locations, path escapes, corrupt bytes,
  invalid slices, arithmetic overflow, and overlapping destinations fail.
- Signature tests distinguish keyless placement changes from changed payloads
  and execution bindings. Rank authentication does not need other ranks' data.
- Materialization reproduces independently exported TP targets byte for byte,
  including rectangular copies, padding, per-TP exceptions, and supported local
  inverse mappings. Each selected deployment passes real model qualification.
- Cold, warm, interrupted, concurrent, and multi-TP acquisition exercise atomic
  publication, shared hardlinks, source retention, leases, and reclamation.
  Failed verification never publishes a ready entry. Cross-filesystem exports
  exercise the explicit copy/clone path.
- Serialized old-layout fixtures preserve absent-field semantics.
- Measure complete download/materialization time and bytes, disk amplification,
  loading time, peak host/pinned/device memory, and file count on representative
  dense and MoE models. Compare against independent exports on the same hardware.
  Loading throughput should remain within measured noise of contiguous-blob
  loading; weight reuse and resharding claims require measured acquisition bytes.

The material costs are explicit: incompatible TP layouts occupy distinct cache
bytes, layer payloads increase file counts, slice verification may read an entire
layer, and random destination writes can require a second hashing pass. These
costs belong to acquisition and loading measurements, not hidden serving work.
