# Native Windows serving

Flare builds on Windows 11 x64 and ARM64 through Rust's
`x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc` targets.
Copy an exported `.safetensors` package matching the serving GPU's SM120 or
SM121 architecture and its publisher profile to the Windows serving host.
Serving requires payload format version 2 packages with typed prepared
contracts and signed zstd-compressed executable frames.
Ignition can also prepare the single-GPU `qwen3.8-27b` recipe on Windows with
`--no-compile`.

The Windows runtime uses the NVIDIA driver directly through `nvcuda.dll`.
Python, the CUDA toolkit, WSL, and Wine are not runtime dependencies.
Install the NVIDIA driver and the Microsoft Visual C++ Redistributable for the
host architecture.

On Windows terminals, startup displays the banner's final frame once.
Loading progress continues below the static banner.

## Build from Linux

Install Rust, Clang/LLVM with `lld-link`, CMake, Ninja, NASM, and Protobuf's
`protoc`, then run from the repository root:

```sh
rustup target add x86_64-pc-windows-msvc
cargo install --locked cargo-xwin --version 0.23.1
cargo xwin build --locked --release --manifest-path flare/Cargo.toml \
  --target x86_64-pc-windows-msvc
```

The executable is `flare/target/x86_64-pc-windows-msvc/release/flare.exe`.
`cargo-xwin` downloads the Microsoft SDK and CRT; its Microsoft license terms
apply. Use the release profile: the default cross-toolchain cache does not
include the debug CRT requested by libjpeg-turbo's debug configuration.

## Build and test on Windows

Install Rust with the native MSVC toolchain, Visual Studio's C++ build tools and
Windows SDK, CMake, and `protoc`. x64 builds also require NASM. ARM64 builds
require the ARM64 C++ tools and Clang for the `ring` dependency. Use the ARM64
host tools through `vcvarsarm64.bat` on an ARM64 machine.

Run in a Visual Studio developer shell with Cargo, CMake, `protoc`, and the
architecture-specific Clang or NASM on `PATH`. Confirm `rustc -vV` reports the
intended host target. From the repository root:

```powershell
cargo build --locked --release --manifest-path flare/Cargo.toml
cargo test --locked --release --manifest-path helios/Cargo.toml --features runtime-package,server
cargo test --locked --release --manifest-path flare/Cargo.toml
```

The `Native serving` GitHub Actions workflow builds natively on Linux and Windows,
runs the native CPU tests, and exercises the Windows executable on Windows.
GPU tests require explicitly supplied packages and hardware and remain ignored
in the CPU suite.

## Serve a prepared package

```powershell
.\flare\target\release\flare.exe run C:\models\qwen38-tp1-mtp5.safetensors `
  --publisher-keys C:\models\publisher.json --gpus 0
```

Publisher acceptance, request sampling, graph-capacity selection, and HTTP APIs
follow the [CLI specification](cli-design.md). The default trust store is
`Flare\trust.toml` under the Windows Local App Data known folder. `--trust-store`
selects another path. Ctrl-C and Ctrl-Break drain native work before releasing
CUDA resources.

Disk embedding tables use Windows I/O rings with registered files and buffers.
All package shards share each worker's ring and staging allocation; file indices
keep equal offsets in separate shards distinct.
Each disk-reader worker refills completed slots during the gather; the default
four workers share 128 slots and 8 MiB of staging per table. Disk prefix caches
continue using I/O completion ports. Both paths use unbuffered overlapped I/O with
bounded queues. File offsets, lengths, and buffers retain the package's
4 KiB alignment; incompatible storage geometry is rejected. RAM embedding tables
and memory prefix caches retain their existing explicit storage selections.
Registered prefix-cache slabs can have different CPU and GPU addresses under
WDDM. Disk I/O uses the CPU address; batched CUDA transfers use the device alias
resolved at registration, with the same offsets and record layout. Treating the
CPU address as a GPU pointer can fail with `CUDA_ERROR_ILLEGAL_ADDRESS` on the
first checkpoint copy.

Qwen Flash TP1 on RTX Spark N1X has been exercised with disk-only and combined
RAM/disk prefix caching: concurrent requests, checkpoint writes, restoration
after restart, and clean shutdown. Other model cache layouts and tensor-parallel
Windows offload still require complete-model validation.

Retained package handles are also unbuffered and prevent replacement or writes
while the model is loaded. Buffered package handles are limited to startup:
retaining one alongside direct readers can severely reduce random-read throughput
even when the buffered handle is idle. Diagnose table reads separately from graph
replay using the `flare_engram_disk_{plan,submit,wait,complete}_nanoseconds_total`
counters; their help text describes platform-specific timing boundaries.

The `ignition debug-serve` development handoff for shared RAM embedding tables uses
Linux file descriptors. On Windows, export a package and load it with
`flare run --embedding-table-storage ram`; native resident-table serving uses
the Windows reader and does not require this development handoff.

## Hardware compatibility

Native ARM64 Flare serves `nvidia/Qwen3.8-27B-NVFP4` on an RTX Spark N1X
(SM121) under Windows 11 WDDM. Generation, prefix restoration, and resume have
been exercised with both Linux-prepared and Windows-prepared packages containing
only device allocations. A Linux-exported SM121 Engram fixture also passes native Windows
replay with mapped mailboxes, disk and RAM row delivery, inactive rows, and read
failure handling. Mailbox publication tests cover up to 65,536 IDs across repeated
allocation-free graph replay on Linux and Windows. These fixture tests do not
establish complete Qwen PLE model correctness or performance on Windows.
Concurrent 27B generation can emit EOS before completing a requested counting
sequence; the complete scheduler test suite does not pass. Linux-prepared device
reservations can also fail intermittently with an address collision before
weight loading. Successful single-request generation does not cover these cases.

On RTX Spark, CUDA's free and total memory include the dedicated GPU carveout
and shared system memory. Windows reports only CPU-visible RAM through its
physical-memory API; NVML reports only the dedicated carveout. GPU planning uses
CUDA's combined budget. Compiler and tuning worker budgets are also bounded by
available CPU-visible RAM.

WDDM charges every CUDA allocation, including the dedicated carveout, against the
system commit limit: CPU-visible RAM plus the page files. Windows grows a
system-managed page file on demand up to a maximum it chooses, about three times
RAM. Serving sizes caches from the smaller of CUDA's free memory and the commit the
process can obtain once every page file reaches its maximum size, then subtracts the
configured reserve, driver margin, and integrated-GPU host headroom, so the reserve
remains available to other processes as commit. Page-file capacity is not GPU
memory, but it bounds GPU allocations: when RAM plus the maximum page-file size is
smaller than the server's footprint plus other processes' commit, KV capacity
shrinks to fit. A custom page-file maximum raises that limit without consuming RAM.

The runtime preserves the package's recorded virtual addresses. Linux registers
host pages for mapped graph storage. Windows uses host-backed CUDA VMM with CPU
and GPU access to the same address. Linux host allocations are rounded and merged
to satisfy the Windows driver's allocation granularity without changing recorded
buffer addresses or graph parameters. This requires host-NUMA VMM support from
the Windows driver and GPU.

Ignition places rank-zero host storage at 12 TiB on Windows and on Linux when
all device rank ranges fit below it (up to four Linux ranks). Larger Linux
layouts use 32 TiB so their host and device ranges remain disjoint, including
sixteen-rank preparation. Heliopause selects the base and reports each rank's
host range to Ignition before allocation.
The package format and mailbox release/acquire protocol are shared. Packages
still require matching GPU architecture, available virtual addresses, and supported
CUDA operations. On RTX Spark N1X with driver 617.42, requesting the 16 TiB host
address used by earlier exports hangs inside `cuMemAddressReserve`; the 32 TiB
address also stalls, even for a 2 MiB reservation.
Packages using those host addresses need re-exporting before Windows use. For
a Linux TP1 package, rebuild Heliopause with the updated preparation code and
export to a new filename; its host storage will use 12 TiB. Other Windows
compatibility requirements still apply. The runtime does not relocate graph storage.

Validate fixed device mappings and graph replay using the exported runtime
fixtures on each target GPU. Packages with mapped host storage additionally
require mapped access, stream-memory waits, and external event ordering tests.
Speculation, disk/RAM table delivery, and tensor parallelism require separate
model tests, including CUDA access to peer memory on the intended GPU pair.

Multi-host RoCE serving is not supported on Windows.
Windows uses the single-process, per-rank-thread execution
path for single-host tensor parallelism, subject to driver and device capabilities.

## Prepare Qwen on Windows

The exercised preparation environment uses x64 CPython 3.12, PyTorch
2.13.0+cu130, CuTe DSL 4.8.0, triton-windows 3.7.1.post27, and the x64 CUDA
13.4.1 toolkit. On ARM64 Windows, Python and the preparation tools run under
x64 emulation; the serving executable remains native ARM64. Build Heliopause
for `x86_64-pc-windows-msvc` to match Python, using the MSVC ARM64-to-x64
developer shell. FlashInfer must include its Windows checkpoint reader and
MSVC JIT build support. The Linux dependency lock does not provision this
Windows environment.

Configure `[paths] cuda` with the toolkit root. Preparation requires NVCC,
NVVM, CCCL, the CUDA runtime, cuRAND headers, NVRTC, nvJitLink, and CUPTI.
The toolkit's PTX assembler must support SM121; Ignition selects it from the
configured toolkit. Use an explicit memory reserve for the OS and compiler
workers, such as `[memory] reserve = "8G"`.

From the repository root in that x64 preparation environment, export four-request
capacity to the model cache. Replace `CACHE_ENTRY` with the printed entry path when converting
to a standalone package:

```powershell
python -m ignition export qwen3.8-27b --gpus 0 --max-concurrency 4 `
  --no-compile --config C:\models\ignition.toml
python -m ignition convert CACHE_ENTRY --config C:\models\ignition.toml `
  --form packed --output C:\models\qwen38-27b.safetensors
```

`--no-compile` disables forward compilation. Kernel compilation, autotuning,
warmup, and CUDA graph capture still run before export. The checkpoint reader
uses unbuffered overlapped reads and bounded pinned transfer buffers; the
prepared model's storage remains device memory. Export rejects any reachable
library allocation below the restorable address floor. Development signing
keys use a protected Windows DACL granting access only to the owner and SYSTEM.

Forward compilation uses spawn-based worker pools on Windows. The GPU AOT tests
exercise parallel metadata-only compilation, saved-forward restoration, mutation
and alias preservation, and allocation-stable graph replay. Omit `--no-compile`
to enable compilation for a model export.
