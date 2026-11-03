# Offline collective ladders

`--collective-tuning runtime --collective-ladder PATH` selects a joint PCIe
kernel's compiled geometry from a JSON file. Preparation compiles that geometry
without racing alternatives, regardless of `--autotune-effort`. Captured tuning
graphs select among the file's submode and active-CTA profiles when the package
loads. Serving replays the same compiled kernel with the selected profile.

From the repository root, add these options to `ignition export`:

```sh
--collective-tuning runtime \
--collective-ladder ignition/preparation/ladders/gb202_pcie_tp4.json
```

The option is required whenever the resolved collective policy is `runtime`,
including configuration-file and recipe defaults. Static policy rejects the
option. File contents travel with the preparation plan and package rebuild
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
An explicit `null` is valid for a reduction-only ladder. Version 1 files
must be updated to version 2 with an explicit `all_gather` field.

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
required for peer-visible writes. This uses existing kernel launch attributes
in execution ABI 5 and does not require cooperative launch.

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
`gb202_pcie_tp4.json` remains the ordinary catalog.

Select the catalog with both phased modes during export:

```sh
.venv/bin/ignition export deepseek-v4.1-flash --gpus 0,1,2,3 \
  --collective-tuning runtime \
  --collective-ladder ignition/preparation/ladders/gb202_pcie_tp4_clustered_full.json
```

Use `gb202_pcie_tp4_clustered.json` in the same flag to exclude the phased Full
choice at 80 KiB. `--collective-tuning static` uses preparation-time autotuning
and accepts no ladder; these catalogs require the explicit `runtime` policy.
Changing the catalog requires package export because cluster geometry is compiled
into its kernels and captured launch attributes. Existing packages retain their
prepared geometry and profiles.

## GB202 PCIe TP4 recommendations

`gb202_pcie_tp4.json` contains 210 buckets from 16 bytes through 8 MiB.
The 16–1008-byte bucket uses one active CTA of the 64-thread latency kernel
in `full` mode for readiness barriers and small validation reductions; this
profile has correctness coverage but no measured performance ranking. The
remaining 209 buckets cover 1 KiB through 8 MiB at 16 intervals per doubling.
The ladder targets out-of-place BF16 reductions on four SM120 GPUs. The
measurements cover a same-switch group (GPUs 0,1,2,3) and a
separate-switch group (GPUs 0,5,7,9) on Pulsar, plus same-switch GPUs 12,13,14,15
and host-bridge GPUs 0,4,8,12 on Frank2. Frank2's same-switch group is classified
by its physical wiring; its OS topology report does not describe that wiring
correctly. Timing protocols remain separate measurement cohorts, and relative
latencies are normalized within each report.

Buckets with upper bounds from 32 through 64 KiB use `joint_latency`, 256 threads,
and a physical grid of 64 CTAs. Their runtime profiles include `payload_rsag`
for switched paths and `pull` for host-bridge paths. Other buckets use the
combined reports' switched-first geometry choices: keep both switched cohorts
within 2% of their respective best where possible, otherwise use the smallest
shared penalty, then minimize the worst host-bridge relative latency.

Each bucket contains at most four runtime profiles. Menus cover every cohort
within 1% of that geometry's best profile when four profiles suffice; otherwise
the tolerance is 5%. At the measured 32, 41.5, and 64 KiB crossover points, menus
also retain an exact best profile from the Pulsar switched sweep using
partition-boundary comparisons for payload ownership. The 5% rule concerns
profiles within a fixed geometry, not the cost of sharing one geometry across
topologies.

With comparison-based ownership in the 256-thread latency kernel, the Pulsar
switched penalties against the best geometry using that ownership calculation
are 2.1%, 0.7%, and 3.1% at those three crossover points. These measurements
require that kernel implementation; the JSON file only selects geometry and
runtime profiles. Frank2 measurements use division-based ownership. The combined
menus have not had independent replay verification, and bucket interiors and
unmeasured topologies have no measured performance bound.

## Column all-gather recommendations

The `all_gather` ladder contains 207 byte-size buckets, from
128 bytes through 8 MiB per rank. The dense measurements use 128-byte local rows,
cold caches, an 8 MiB channel
capacity, and rank-maximum GPU timings on Pulsar: GPUs 0,1,2,4 share a switch;
GPUs 0,5,7,10 communicate across host bridges. Each measured candidate has at
least 102 timings. Bucket spacing is 16 intervals per doubling, aligned to
complete rows.

| Per-rank input | Kernel | Threads | Physical CTAs |
|---|---|---:|---:|
| 128 B–1 KiB | Joint | 32 | 4 |
| >1–2 KiB | Joint | 256 | 8 |
| >2–4 KiB | Joint | 256 | 16 |
| >4–8 KiB | Joint | 256 | 32 |
| >8–16 KiB | Joint | 256 | Bucket maximum divided by 256, rounded up |
| >16 KiB–8 MiB | Barrier | 512 | 64 |

Each bucket retains at most five runtime profiles, covering protocol and active
CTA choices across row widths and topologies. Equal-sized gathers use the same
menu; the captured runtime race evaluates it on the actual row geometry.
Word exchange uses one warp
per CTA. Wide CTAs prefetch the selector, packet counter, and input words with
predicated bounds checks before dispatch; this avoids reconvergence overhead
on the switched path. Barrier mode uses all threads.

For the dense 128-byte-row sweep, no switched bucket has a measured regression
against the separate-kernel references. At 896 bytes, repeated host
comparisons range from 5.49% (6.144 versus 5.824 microseconds) to 8.10%
(6.192 versus 5.728 microseconds). Above 1 KiB, the worst host regression is
5.79% near 1.48 MiB.
An equal-byte comparison covers local row widths of 128, 512, 2048, and 8192
bytes at 4, 8, 16, 32, 64, 256, and 1024 KiB per rank, omitting rows larger than
the payload. GPUs 0,1,2,4 share a switch; GPUs 0,5,8,13 occupy four separate
switches. All 54 cases use at least 102 timings per candidate. The installed
menus retain the 32/256/512-thread progression and add low-CTA barrier profiles
needed by wider rows on host bridges.

| Per-rank bytes | Worst switched regret | Worst host-bridge regret |
|---|---:|---:|
| 4 KiB | 0.84% | 3.33% |
| 8 KiB | 3.60% | 3.99% |
| 16 KiB | 1.17% | 4.16% |
| 32 KiB | 2.84% | 5.66% |
| 64 KiB | 2.17% | 6.43% |
| 256 KiB | 0.15% | 13.15% |
| 1 MiB | 0.00% | 20.11% |

Regret compares the fastest measured profile in the installed menu with the
fastest measured configuration for that exact layout and topology, including
static kernels. Independent switched repeats put the 8 KiB joint choice within
0.7–2.2% and the 32 KiB barrier choice within 0.5–3.3% of the best measured
configuration. The 1 MiB host penalty occurs with 512-byte rows: the shared
512-thread geometry is slower than the 128-thread alternative. Row widths
remain kernel addressing parameters, never ladder lookup keys.

These are component CUDA-graph measurements, not model throughput, guarantees
that startup selects the measured fastest profile, or bounds for bucket
interiors, unmeasured row widths, or other machines.
