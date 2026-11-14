# Tensor-parallel compressed scale loading

Flare accepts `csf_32x2_v1` and the slab encoding `csf` for compressed NVFP4
expert scales. Both store packed FP4 weights and calibration unchanged,
with a scale encoding independent of tensor-parallel size. The
[FlashInfer format specification](../../flashinfer/docs/csf-scale-blocks.md)
defines the shared NVFP4 and MXFP4 representation and native destinations.

In `csf_32x2_v1`, each scale matrix has five safetensors components: `csf32_bases` and
`csf32_codes` as uint8, `csf32_page_bases` as uint32, and `csf32_offsets` and
`csf32_exceptions` as uint16. Blocks cover 32 rows and two columns, grouped
into clipped 128-row by four-column tiles. Pages contain 32 whole groups.
NVFP4 uses four-bit palette offsets; MXFP4 uses one-bit offsets. Exceptions
preserve exact scale bytes. Checkpoints store no tile padding or TP directories.

## Loading

`DirectWeightLoader.load_nvfp4_experts` transfers packed rank-local weight
views and compressed scale components directly to CUDA through b12x. Component
transfers are queued in batches into contiguous staging allocations. Staging
is bounded to one layer. GPU kernels gather packed fields for logical partitions
and joined up/gate rows, scan exception counts, and write compact group indexes
and native-layout payloads. GPU validation checks palette bounds and exception
indexes before they are used as addresses. Only allocation sizes and validation
status return to the host; scale bytes are neither decoded nor regrouped there.

The slab encoding stores 16-row base/code streams and 32-bit global-position
exceptions. Loading converts these GPU components into the same compact runtime
payload. Both encodings use the same tuned expansion and ordinary compute kernels.
There is no prepared-scale disk cache.

Qwen's intermediate width 640 produces local widths 640, 320, and 160 at
TP1, TP2, and TP4. The loader regroups blocks across the stacked up/gate seam.
The TP4 down projection has ten logical scale columns. Native compute tails
occupy reusable expansion scratch; they do not extend the stored checkpoint.

The Qwen and GLM recipes use A4 expert execution by default. b12x expands their compressed
scales directly into the native F8_128x4 layout. Its shared format also supports
packed W4A16 and MXFP4 N64/N256 scale destinations; Qwen3.8 Flash prepared with
`--force-moe-decode-a16` uses packed W4A16 experts for decode and verification.
Ordinary MoE and GEMM kernels
consume native scales without inline compressed-scale decoding.

## Preparation and replay

`PreparedRoutedExperts` declares a scale-expansion plan for each route capacity
and makes the consuming MoE request depend on it. The scale-expansion and MoE
execution tuning sweeps both time CUDA graph
replay with live activation and routing producers; graph timing has separate
selection keys. Small calls expand routed experts, and large calls expand all
experts. The selected MoE binding retains expansion programs and expert-presence
scratch. Serialized layers share caller-owned scale buffers; overlapping expert
calls require disjoint buffers or explicit stream dependencies.

Checkpoint tests compare logical scale bytes and packed weights for Qwen and
GLM partitions. Qwen replay tests poison reusable scale scratch. Tuning
integration tests exercise decode and prefill capacities through 4096 tokens.
`scripts/benchmark_csf_conversion.py --snapshot PATH` measures one complete
Qwen expert layer's scale reads, regrouping, native preparation, and upload;
`--tp-size` and `--tp-rank` select a logical partition.
