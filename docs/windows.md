# Native Windows serving

Flare builds on Windows 11 x64 and ARM64 through Rust's
`x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc` targets.
Copy an exported `.safetensors` core matching the serving GPU's SM120 or
SM121 architecture and its publisher profile to the Windows serving host.
Serving loads payload format version 2 cores with typed prepared
contracts and signed zstd-compressed executable frames. The loader also accepts
signed format-4 and format-6 logical-resource containers on every platform
([runtime core design](runtime-core-design.md)). Not qualified: loading those
containers on Windows.
Ignition can also prepare the single-GPU `qwen3.8-27b` recipe on Windows with
`--no-compile`.

The Windows runtime uses the NVIDIA driver directly through `nvcuda.dll`.
Python, the CUDA toolkit, WSL, and Wine are not runtime dependencies.
Install the NVIDIA driver. CLI builds also require the Microsoft Visual C++
Redistributable for the host architecture. The desktop distribution includes
its runtime dependencies.

On Windows terminals, startup displays the banner's final frame once.
Loading progress continues below the static banner.

## Build from Linux

Install Rust, Clang/LLVM with `lld-link`, CMake, Ninja, NASM, and Protobuf's
`protoc`, then run from the repository root:

```sh
rustup target add x86_64-pc-windows-msvc
cargo install --locked cargo-xwin --version 0.23.1
cargo xwin build --locked --release --manifest-path flare-cli/Cargo.toml \
  --target x86_64-pc-windows-msvc
```

The executable is `flare-cli/target/x86_64-pc-windows-msvc/release/flare.exe`.
`cargo-xwin` downloads the Microsoft SDK and CRT; its Microsoft license terms
apply. Use the release profile: the default cross-toolchain cache does not
include the debug CRT requested by libjpeg-turbo's debug configuration.

## Build and test on Windows

From a clone, provision the native serving development environment with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

The script installs missing Visual Studio C++ tools, native Rust, CMake, and
Protobuf. ARM64 uses native MSVC and Clang; x64 also installs NASM. Visual Studio
installation may require administrator consent. Python 3.12 and an editable
Ignition test environment supply frontend fixtures; model preparation dependencies
and the CUDA toolkit are not installed. Add `-Test` to run the native test suites
after building and checking the CLI. Rerunning setup preserves an existing
`.local-dev/config.toml`. `-RustToolchain VERSION` installs and selects that
Rust toolchain in place of `stable`. `-BuildToolsOnly` stops after installing
the compilers and build tools: it creates no Python environment, fixtures, or
configuration and does not build the CLI.

Enter the environment and use prepared cores:

```powershell
& ./scripts/dev_windows.ps1
flare download PUBLISHER/PREPARED_MODEL --gpus 0
flare run PUBLISHER/PREPARED_MODEL --gpus 0
```

Replace `PUBLISHER/PREPARED_MODEL` with a repository publishing Flare runtime
cores. The shell's `flare` function supplies `.local-dev/config.toml`, which
defaults to localhost port 8000, one active request, and an 8 GiB reserve.
Publisher verification and first-use acceptance use the normal Flare prompts.
Raw checkpoints and GGUF files are not prepared Flare cores.

The scripts keep tools, Cargo build output, and model caches under
`%LOCALAPPDATA%\FlareDev`. Short physical paths prevent MSVC and CMake scratch
files from exceeding legacy Windows path limits in deeply nested clones.
CMake uses Ninja. Enabling Windows long paths alone does not cover applications
that have not opted in; see Microsoft's
[path-length documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).
The executable is `$env:LOCALAPPDATA\FlareDev\target\release\flare.exe`.
To keep those files in a different short directory, pass
`-DevRoot C:\Projects\.flare-dev` to setup. That choice is saved in
`.local-dev/dev-root.txt` and reused by the development shell. Move existing
tool caches and build files to the chosen directory before reusing them there.

Rebuild and run native tests from any configured PowerShell:

```powershell
& ./scripts/dev_windows.ps1 -Action Build
& ./scripts/dev_windows.ps1 -Action Test
& ./scripts/dev_windows.ps1 -Action Smoke
```

Setup supports ARM64 and x64 paths; validate each architecture on its own host.
It does not download model weights or establish complete-model GPU correctness.
Not qualified: the `selene` test suite on Windows. Its process nonce source
(`selene/src/shared/identity.rs`) opens `/dev/urandom`, which Windows does not
provide, and its cache read-ahead tests can stall on ARM64. Report those
failures separately from build and CLI check results when validating native
changes.

For a manually provisioned environment, follow these requirements:

Install Rust with the native MSVC toolchain, Visual Studio's C++ build tools and
Windows SDK, CMake, and `protoc`. x64 builds also require NASM. ARM64 builds
require the ARM64 C++ tools and Clang for the `ring` dependency. Use the ARM64
host tools through `vcvarsarm64.bat` on an ARM64 machine.

Run in a Visual Studio developer shell with Cargo, CMake, `protoc`, and the
architecture-specific Clang or NASM on `PATH`. Confirm `rustc -vV` reports the
intended host target. From the repository root:

```powershell
cargo build --locked --release --manifest-path flare-cli/Cargo.toml
cargo test --locked --release --manifest-path helios/Cargo.toml --features runtime-core,server
cargo test --locked --release --manifest-path flare-cli/Cargo.toml
```

The `Native serving` GitHub Actions workflow builds natively on Linux and Windows,
runs the native CPU tests, and exercises the Windows executable on Windows.
GPU tests require explicitly supplied cores and hardware and are ignored in
the CPU suite.

## Serve a prepared core

```powershell
.\flare-cli\target\release\flare.exe run C:\models\qwen38-tp1-mtp5.safetensors `
  --publisher-keys C:\models\publisher.json --gpus 0
```

`.\flare-cli\target\release\flare.exe` is the output of a build that leaves
`CARGO_TARGET_DIR` unset, such as the manually provisioned build above. A shell
configured by `scripts/dev_windows.ps1` builds
`$env:LOCALAPPDATA\FlareDev\target\release\flare.exe` (or the same path under the
`-DevRoot` directory), and its `flare` function runs that executable with
`.local-dev/config.toml`.

Publisher acceptance, request sampling, graph-capacity selection, and HTTP APIs
follow the [CLI specification](cli-design.md). The default trust store is
`Flare\trust.toml` under the Windows Local App Data known folder. `--trust-store`
selects another path. Ctrl-C and Ctrl-Break drain native work before releasing
CUDA resources.

Disk embedding tables use Windows I/O rings with registered files and buffers.
All core shards share each worker's ring and staging allocation; file indices
keep equal offsets in separate shards distinct.
Each disk-reader worker refills completed slots during the gather; the default
four workers share 128 slots and 8 MiB of staging per table. Disk prefix caches
use I/O completion ports. Both paths use unbuffered overlapped I/O with
bounded queues. File offsets, lengths, and buffers use the core's 4 KiB
alignment; incompatible storage geometry is rejected. The storage selections for
RAM embedding tables and memory prefix caches are independent of these disk
paths.
Registered prefix-cache slabs can have different CPU and GPU addresses under
WDDM. Disk I/O uses the CPU address; batched CUDA transfers use the device alias
resolved at registration, with the same offsets and record layout. Treating the
CPU address as a GPU pointer can fail with `CUDA_ERROR_ILLEGAL_ADDRESS` on the
first checkpoint copy.

Windows prefix-cache offload is qualified for Qwen Flash TP1 on RTX Spark N1X
with disk-only and combined RAM/disk caching: concurrent requests, checkpoint
writes, restoration after restart, and shutdown. Not qualified: other model
cache layouts and tensor-parallel offload on Windows.

Retained core handles are also unbuffered and prevent replacement or writes
while the model is loaded. Buffered core handles are limited to startup:
retaining one alongside direct readers can severely reduce random-read throughput
even when the buffered handle is idle. Diagnose table reads separately from graph
replay using the `flare_engram_disk_{plan,submit,wait,complete}_nanoseconds_total`
counters; their help text describes platform-specific timing boundaries.

The `ignition debug-serve` development handoff for shared RAM embedding tables uses
Linux file descriptors. On Windows, export a core and load it with
`flare run --embedding-table-storage ram`; native resident-table serving uses
the Windows reader and does not require this development handoff.

## Hardware compatibility

Native ARM64 serving of `nvidia/Qwen3.8-27B-NVFP4` is qualified on an RTX Spark
N1X (SM121) under Windows 11 WDDM for single-request generation, prefix
restoration, and resume, with Linux-prepared and Windows-prepared cores
containing only device allocations. Fixture tests on Windows cover native replay
of a Linux-exported SM121 Engram fixture with mapped mailboxes, disk and RAM row
delivery, inactive rows, and read failure handling. Mailbox publication tests
cover up to 65,536 IDs across repeated allocation-free graph replay on Linux and
Windows.

Not qualified on that configuration: complete Qwen PLE model correctness and
performance, which the fixture tests do not establish; concurrent generation,
where 27B requests can emit EOS before completing a requested counting sequence
and `tests/server/test_scheduler.py` does not pass in full; and repeatable
loading of Linux-prepared cores, whose device reservations can fail
intermittently with an address collision before weight loading.

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
is available to other processes as commit. Page-file capacity is not GPU
memory, but it bounds GPU allocations: when RAM plus the maximum page-file size is
smaller than the server's footprint plus other processes' commit, KV capacity
shrinks to fit. A custom page-file maximum raises that limit without consuming RAM.

The runtime preserves the core's recorded virtual addresses. Linux registers
host pages for mapped graph storage. Windows uses host-backed CUDA VMM with CPU
and GPU access to the same address. Linux host allocations are rounded and merged
to satisfy the Windows driver's allocation granularity without changing recorded
buffer addresses or graph parameters. This requires host-NUMA VMM support from
the Windows driver and GPU.

Ignition places rank-zero host storage at 12 TiB on Windows and on Linux when
all device rank ranges fit below it (up to four Linux ranks). Larger Linux
layouts use 32 TiB so their host and device ranges are disjoint, including
sixteen-rank preparation. Heliopause selects the base and reports each rank's
host range to Ignition before allocation.
Linux and Windows share the core format and the mailbox release/acquire
protocol. A core loads on Windows only when its GPU architecture matches, its
recorded virtual addresses are available, and its CUDA operations are supported.
Host storage recorded at 16 TiB or 32 TiB cannot be reserved under Windows on
RTX Spark N1X with NVIDIA driver 617.42: `cuMemAddressReserve` does not return
for a host address at either base, including a 2 MiB reservation. The runtime
does not relocate graph storage, so such a core must be re-exported before
Windows use. Re-export a Linux TP1 core to a different filename; its host
storage is placed at 12 TiB.

Validate fixed device mappings and graph replay using the exported runtime
fixtures on each target GPU. Cores with mapped host storage additionally
require mapped access, stream-memory waits, and external event ordering tests.
Speculation, disk/RAM table delivery, and tensor parallelism require separate
model tests, including CUDA access to peer memory on the intended GPU pair.

Multi-host RoCE serving is not supported on Windows.
Windows uses the single-process, per-rank-thread execution
path for single-host tensor parallelism, subject to driver and device capabilities.

## Native desktop client

WinUI 3 and C++/WinRT own presentation. The Rust `flare` library owns discovery,
acquisition, publisher review, local references, settings, and serving operations.
`flare-cli` produces the terminal `flare.exe`; `flare-winui` links the shared
library and Helios through a small C binding. The desktop does not launch the
CLI or install a serving service.

Build from the repository root after setting up the native development tools:

```powershell
.\scripts\build_winui.ps1 -Platform ARM64 -SingleFile
.\.local-dev\winui\ARM64\SingleFile\Flare.exe
```

The default launch reads the same user configuration and core cache as the
CLI: `~/.config/flare/config.toml` and, unless `paths.core_cache` is set,
`~/.cache/flare/cores`, where `~` is the Windows user profile folder. When
`cores` is absent and `~/.cache/flare/packages` exists, `packages` is used in
place. Pass `--config .\.local-dev\config.toml` only to use the separate
development cache configured by the setup script.

Use `-Platform x64` for an x64 distribution. The helper restores pinned Windows
App SDK and C++/WinRT packages, links the Rust static library, and bundles the
Release runtime into one desktop `Flare.exe`. Its launcher silently extracts
the runtime beneath the Windows Local App Data known folder at
`Flare/Desktop/<ARM64|x64>/<archive-SHA256>` on first launch, then starts the
native `flare-winui.exe`. Repeat launches reuse the extracted version; versions
extracted from other archives are not removed. Extraction does not change the
model cache, configuration, publisher trust, or desktop state. Arguments,
including `--config`, are forwarded with the caller's working directory
preserved.

To package an already-built Release folder without rebuilding C++, run
`./flare-winui/package-single-file.ps1 -Platform ARM64`. Add `-ArchiveOnly` to
create and validate the embedded ZIP without compiling the launcher. The ZIP
records file sizes and SHA-256 digests with fixed ordering and timestamps.
The launcher uses the static C runtime; build outputs are unsigned. An
installer, signing workflow, and automatic runtime-cache cleanup are not provided.
[Binary releases](release.md) publish `Flare.exe` with its license notices as
`flare-windows-amd64.zip` and `flare-windows-arm64.zip`.

Omit `-SingleFile` for the loose development workflow, launching
`.local-dev\winui\<ARM64|x64>\Release\flare-winui.exe` with its complete
adjacent runtime directory.
Serving requires a supported NVIDIA driver; Rust, Python, and the CUDA toolkit
are development dependencies. See the [desktop build reference](../flare-winui/README.md)
for output paths and archive contents.

The desktop accepts repository IDs and local prepared files or folders. Public
Hugging Face search uses tags, repository names, and publisher metadata as
discovery hints. Retained acquisition records associate discovery results with
cores already downloaded by either client. Those cards offer the local core
without another download; this association does not establish trust or claim
that the local version is the repository's latest revision. Publisher acceptance
is bound to the verified signing key and namespace. External core references
do not copy or delete their source files. `desktop-state.json`, beside the
selected configuration, records references, downloads, appearance, runtime
setting overrides, and the model directory. Core details distinguish
authenticated capabilities from unverified
metadata. External references can be relocated without copying model files.

Home presents installed cores and actions to find or import a core before a
model is loaded. The running model, switch/stop controls, endpoint, and runtime
dashboard occupy Home. The serving address and copy action share the row with
the switch and stop buttons. Chat is available in the sidebar. Automatic agent
installation and configuration are not implemented.

The dashboard uses Helios telemetry and the CLI overview's two-second rate
window with 120 seconds of history. It plots prefill, generation, verification,
and draft acceptance/depth, and exposes request/token totals, KV-cache capacity,
host prefix-cache activity, and NVML GPU memory. Acceptance and depth are counts
per request round. Weight bandwidth is an estimate, not measured throughput.
First-response latency is the median over a bounded 60-second window, measured
from HTTP admission until the server offers answer or reasoning text. Empty
windows show no value. Chat, the fourth navigation tab, streams replies from the
active local session and
retains successful exchanges as context for subsequent messages. Stop cancels
the current reply; New chat clears the conversation. Enter sends and Shift+Enter
adds a line break. Conversations remain in memory while navigating the app and
are cleared when the serving session ends; they are not saved to disk.

The desktop uses a fixed dark theme with NVIDIA green accents. Windows theme
changes and saved System or Light preferences do not change its appearance.
Settings offers Comfortable and Compact density choices.
Runtime settings include serving-memory and KV-capacity limits, concurrency,
listener address/port, weight verification, host prefix-cache retention, and
speculation/embedding overrides. Save persists desktop overrides without
changing an active session; Reload model applies them after draining requests.
Weight verification and host prefix caching to disk default off. An explicitly
configured prefix-cache path enables disk caching; saved desktop preferences
take precedence. Enabling disk caching uses the engine’s automatic disk budget
unless a limit is specified. See the
[desktop settings reference](../flare-winui/README.md) for units and precedence.

Serving uses one prepared model on one local GPU. Switching pauses admission,
finishes accepted responses, drains execution, and releases the prior session.
Closing cancels downloads cooperatively and drains serving. Unconfirmed GPU
drain requires restarting the application. Multi-GPU serving is available only
through the CLI.
Inference and switching require validation with a compatible prepared core on
each target GPU; UI compilation and application tests do not qualify those paths.

The ARM64 desktop is qualified on an RTX Spark N1X with a cached Qwen3.8-27B
core: loading, multi-turn Chat replies with runtime memory and concurrency
defaults, and cooperative close that releases the process and local listener.
Not qualified: x64 linking and inference (the x64 C++ shell compiles),
network-transfer interruption, switching under load, accessibility, and recovery
from sleep or GPU reset.

## Prepare Qwen on Windows

Preparing `qwen3.8-27b` on Windows is qualified with x64 CPython 3.12, PyTorch
2.13.0+cu130, CuTe DSL 4.8.0, triton-windows 3.7.1.post27, and the x64 CUDA
13.4.1 toolkit. On ARM64 Windows, Python and the preparation tools run under
x64 emulation; the serving executable is native ARM64. Build Heliopause
for `x86_64-pc-windows-msvc` to match Python, using the MSVC ARM64-to-x64
developer shell. On Windows, Heliopause compiles a bundled static OpenSSL, which
requires Perl on `PATH`; see
[environment setup](model-preparation.md#environment-setup). FlashInfer must
include its Windows checkpoint reader and
MSVC JIT build support. The Linux dependency lock does not provision this
Windows environment.

Configure `[paths] cuda` with the toolkit root. Preparation requires NVCC,
NVVM, CCCL, the CUDA runtime, cuRAND headers, NVRTC, nvJitLink, and CUPTI.
The toolkit's PTX assembler must support SM121; Ignition selects it from the
configured toolkit. Use an explicit memory reserve for the OS and compiler
workers, such as `[memory] reserve = "8G"`.

From the repository root in that x64 preparation environment, export four-request
capacity to the model cache. Replace `CACHE_ENTRY` with the printed entry path when converting
to a standalone core:

```powershell
python -m ignition export qwen3.8-27b --gpus 0 --max-concurrency 4 `
  --no-compile --config C:\models\ignition.toml
python -m ignition convert CACHE_ENTRY --config C:\models\ignition.toml `
  --form packed --output C:\models\qwen38-27b.safetensors
```

`--no-compile` disables forward compilation only: kernel compilation,
autotuning, warmup, and CUDA graph capture run before export. The checkpoint
reader uses unbuffered overlapped reads and bounded pinned transfer buffers; the
prepared model's storage is device memory. Export rejects any reachable
library allocation below the restorable address floor. Development signing
keys use a protected Windows DACL granting access only to the owner and SYSTEM.

Forward compilation uses spawn-based worker pools on Windows. The GPU tests in
`tests/preparation/test_aot.py` exercise parallel metadata-only compilation,
saved-forward restoration, mutation and alias preservation, and
allocation-stable graph replay. Omit `--no-compile` to enable compilation for a
model export.
