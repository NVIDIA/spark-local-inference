# Offline collective ladders

`--collective-tuning runtime --collective-ladder PATH` selects a joint PCIe
kernel's compiled geometry from a JSON file. Preparation compiles that geometry
without racing alternatives, regardless of `--autotune-effort`. Captured tuning
graphs select among the file's submode and active-CTA profiles when the core
loads. Serving replays the same compiled kernel with the selected profile.

From the repository root, add these options to `ignition export`:

```sh
--collective-tuning runtime \
--collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

The option is required whenever the resolved collective policy is `runtime`,
including configuration-file and recipe defaults. Static policy rejects the
option. File contents travel with the preparation plan and core rebuild
settings; workers do not reopen the path.

## Version 2 format

```json
{
  "version": 2,
  "name": "example_tp4",
  "compute_capability": [12, 0],
  "world_size": 4,
  "dtype": "bfloat16",
  "inplace": false,
  "cache_policy": "cold",
  "min_bytes": 1024,
  "buckets": [
    {"max_bytes": 1048576, "implementation": "joint_bandwidth", "threads": 512, "blocks": 64,
     "profiles": {"bulk_push": [32], "bulk_pull": [8]}}
  ],
  "all_gather": {
    "min_bytes": 128, "buckets": [
      {"max_bytes": 1024, "implementation": "gather_joint", "threads": 32, "blocks": 4,
       "profiles": {"word_exchange": [4]}}
    ]
  }
}
```

Payload bytes count the full per-rank input before reduction. Bounds must be
16-byte aligned. The first bucket includes `min_bytes` through `max_bytes`;
subsequent buckets include payloads above the preceding maximum through their
own maximum. Selection does not extrapolate outside those bounds.

Each bucket declares one kernel family, thread count, and physical CTA grid.
`joint_latency` permits `full`, `payload_rsag`, and `pull`; `joint_bandwidth`
permits `payload_rsag`, `bulk_push`, `bulk_pull`, and `pull`. Each profile lists
active CTA counts between one and the physical grid size. Several counts may
belong to the same submode. `cache_policy` controls the captured profile race's
cache treatment and accepts `cold` or `warm`.

The ladder applies to eligible plain BF16 PCIe reductions. TP size, compute
capability, and input/output aliasing must match. Other collective transports
and fused operations retain their preparation policy. Runtime profile selection
cannot change the compiled family, threads, or physical grid.

`all_gather` contains one ladder keyed by total per-rank input bytes. Row width
does not select a bucket; it remains part of the compiled kernel's addressing
geometry. Gather modes are
`barrier` and `word_exchange`; `gather_joint` permits both,
`gather_selected_barrier` permits only barrier, and
`gather_selected_word_exchange` permits only word exchange. Word exchange is
limited to TP2/TP4 and at most 16 KiB per rank. Gathers require disjoint
input/output storage; the top-level `inplace` field applies only to reductions.
An explicit `null` is valid for a reduction-only ladder. Validation rejects
version 1 files and files without an explicit `all_gather` field.

Ignition applies this ladder to contiguous, disjoint PCIe column gathers,
including BF16, FP32 bit views, and E4M3 payloads. Captured startup producers
validate exact payload bytes and select among the supplied profiles. Gathers
with equal byte sizes use the same recommendations regardless of row width.
RoCE gathers and reduction-only ladders retain their preparation policy.

### Thread-block clusters

A bucket may set `"cluster_size": 8` to launch eight-CTA CUDA thread-block
clusters. Omitting the field, or setting it to `1`, retains an ordinary launch.
Clustered buckets require SM120 or SM121 with at least 64 SMs, 512 threads per
CTA, and 64 physical CTAs. Preparation checks that the compiled function can
keep eight clusters resident. Ordinary profiles retain their active-CTA counts.
Phased full all-reduce is named `full_phased`; phased reduce-scatter followed
by all-gather is named `payload_rsag_phased`. The latency family admits both;
the bandwidth family admits `payload_rsag_phased`. Each phased profile must select
exactly eight active CTAs.

Phased execution uses the first complete cluster to write one peer destination
at a time. All its CTAs participate in each phase barrier; other clusters return
without changing collective state. System-scope publication fences remain
required for peer-visible writes. Phased execution uses captured kernel launch
attributes and does not require cooperative launch.

`gb202_pcie_tp4_clustered.json` adds phased RSAG to 36 compatible buckets used
by packed DeepSeek V4.1 TP4 verification graphs with eight, sixteen, or thirty-two
requests, seven draft positions, and up to 128 target rows. It retains every
ordinary candidate, kernel family, thread count, and physical grid from
`gb202_pcie_tp4.json`. Its 256-thread buckets use ordinary launches.
`gb202_pcie_tp4_clustered_full.json` adds one further choice: phased Full in the
bucket containing 80 KiB. This also covers the target reductions for one request
verified at eight rows.

The compiled cluster geometry also applies when startup selects an ordinary
profile; startup selection cannot disable clustering. Cluster scheduling can
affect those profiles, so retaining every ordinary candidate does not guarantee
unchanged latency. Qualify these catalogs on the intended GPU topology and
complete serving workload; component improvements do not establish a throughput gain.
`gb202_pcie_tp4.json` is the catalog without clustered launches.

Select the catalog with both phased modes during export:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --collective-tuning runtime \
  --collective-ladder ignition/preparation/ladders/gb202_pcie_tp4_clustered_full.json
```

Use `gb202_pcie_tp4_clustered.json` in the same flag to exclude the phased Full
choice at 80 KiB. `--collective-tuning static` uses preparation-time autotuning
and accepts no ladder; these catalogs require the explicit `runtime` policy.
Changing the catalog requires core export because cluster geometry is compiled
into its kernels and captured launch attributes. A core's geometry and profiles
are fixed at export.

## GB202 PCIe TP4 recommendations

`gb202_pcie_tp4.json` contains 210 buckets from 16 bytes through 8 MiB.
The 16–1008-byte bucket uses one active CTA of the 64-thread latency kernel
in `full` mode for readiness barriers and small validation reductions; this
profile has correctness coverage but no measured performance ranking. The
remaining 209 buckets cover 1 KiB through 8 MiB at 16 intervals per doubling.
The ladder targets cold-cache, out-of-place BF16 reductions on four SM120 GPUs
in three PCIe topology classes: four GPUs behind one switch, four GPUs behind
separate switches, and four GPUs connected through host bridges.

Buckets with upper bounds from 32 through 64 KiB use `joint_latency`, 256 threads,
and a physical grid of 64 CTAs. Their runtime profiles include `payload_rsag`
for switched paths and `pull` for host-bridge paths. Every other bucket's
geometry follows a switched-first rule: keep both switched classes within 2% of
their respective best geometry where possible, otherwise use the smallest
shared penalty, then minimize the worst host-bridge relative latency.

Each bucket contains at most four runtime profiles. A menu covers every
topology class within 1% of that geometry's best profile when four profiles
suffice; otherwise the tolerance is 5%. The tolerance concerns profiles within
a fixed geometry, not the cost of sharing one geometry across topologies. The
JSON file selects geometry and runtime profiles only; payload-ownership
arithmetic belongs to the kernel implementation.

Not qualified: independent replay verification of the menus. Bucket interiors
and topologies outside the three classes have no measured performance bound.

## Column all-gather recommendations

The `all_gather` ladder contains 207 byte-size buckets, from
128 bytes through 8 MiB per rank, for cold-cache gathers on four SM120 GPUs
with an 8 MiB channel capacity. Bucket spacing is 16 intervals per doubling,
aligned to complete 128-byte rows.

| Per-rank input | Kernel | Threads | Physical CTAs |
|---|---|---:|---:|
| 128 B–1 KiB | Joint | 32 | 4 |
| >1–2 KiB | Joint | 256 | 8 |
| >2–4 KiB | Joint | 256 | 16 |
| >4–8 KiB | Joint | 256 | 32 |
| >8–16 KiB | Joint | 256 | Bucket maximum divided by 256, rounded up |
| >16 KiB–8 MiB | Barrier | 512 | 64 |

Each bucket contains at most five runtime profiles, covering protocol and active
CTA choices across row widths and topologies, including low-CTA barrier
profiles for wide rows on host-bridge paths. Equal-sized gathers use the same
menu; the captured runtime race evaluates it on the actual row geometry.
Word exchange uses one warp
per CTA. Wide CTAs prefetch the selector, packet counter, and input words with
predicated bounds checks before dispatch; this avoids reconvergence overhead
on the switched path. Barrier mode uses all threads.
Row widths are kernel addressing parameters, never ladder lookup keys.

Sharing one geometry per byte size costs latency against a kernel chosen for
one exact row layout and topology. For local row widths of 128 through 8192
bytes at 4 KiB through 1 MiB per rank, the menu's fastest profile is within
3.6% of the fastest configuration on switched paths. On host-bridge paths the
difference grows with size, from 3.3% at 4 KiB to 20.1% at 1 MiB with
512-byte rows, where the shared 512-thread geometry is slower than a
128-thread kernel.

The recommendations rest on component CUDA-graph timings. They do not establish
model throughput, guarantee that startup selects the fastest profile, or bound
bucket interiors, other row widths, or other GPU topologies.
