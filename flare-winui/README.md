# Native Windows desktop client

`flare-winui.exe` presents the shared Rust Flare application library through
Windows App SDK controls. C++ owns the window, navigation, native file pickers,
publisher consent dialogs, clipboard, taskbar progress, and bounded view state.
Flare cores are the prepared model artifacts loaded by Helios.
The statically linked Rust bridge owns core discovery, acquisition, library
state, configuration, publisher verification, and the Helios serving session.
No webview or managed UI runtime is used.

## Build

Run the repository's
[Windows setup](../docs/windows.md#build-and-test-on-windows) first. The desktop build
uses Visual Studio 2022 C++ tools, Windows SDK 10.0.26100.0, Rust, and pinned
Windows App SDK and C++/WinRT NuGet packages. It does not require the Visual
Studio UWP workload or a separate XAML compiler. The installed SDK's MakePri
tool merges the WinUI resource indexes for unpackaged lookup, and the application
delegates XAML type lookup to WinUI's metadata provider. From the repository root:

```powershell
./flare-winui/build.ps1 -Platform ARM64 -Configuration Release -SingleFile
```

Use `-Platform x64` for an x64 executable. Cross compilation installs the matching
Rust standard library target. The native build restores dependencies beneath
`.local-dev/native-packages` and writes the application beneath
`.local-dev/winui/ARM64/Release` or `.local-dev/winui/x64/Release`.

`-SingleFile` produces one distributable desktop executable at
`.local-dev/winui/ARM64/SingleFile/Flare.exe` (or `x64/SingleFile/Flare.exe`).
The Rust launcher embeds the native application and its runtime dependencies.
The native application and Rust bridge use the static C runtime consistently,
including tokenizer dependencies. The bridge's Cargo output uses the short
`winui-target` directory beneath the Flare development directory
(`%LOCALAPPDATA%\FlareDev`, or the `-DevRoot` directory chosen during setup); the
launcher uses `launcher-target` beneath the same directory.

To bundle an existing Release folder without rebuilding the C++ application:

```powershell
./flare-winui/package-single-file.ps1 -Platform ARM64
```

The packaging script creates `.local-dev/winui-bundle/ARM64/payload.zip`, then
builds only the launcher with locked Cargo dependencies and the static C runtime.
`-ArchiveOnly` validates and creates the ZIP without building the launcher.
`-InputDirectory PATH` selects another already-built Release runtime folder.
The ZIP contains a versioned `bundle.json` with each file's relative path, size,
and SHA-256. Entries use ordinal ordering and a fixed timestamp. Runtime DLLs,
resources, metadata, images, and SDK `RestartAgent.exe` are included; debugging
symbols, import libraries, and backup executables are excluded. No additional
archive tool is required.

Omit `-SingleFile` for the loose development workflow. The build copies the
selected Windows App SDK components and Visual C++ runtime files beside
`flare-winui.exe`; distribute that complete Release directory when using this
form. Both distribution forms are unsigned build outputs, without an MSIX or
installer. Neither form contains license notices. `packaging/archive.py
--winui-packages` packages the single-file `Flare.exe` with the notices of its
Rust and NuGet dependencies, as
[third-party notices](../THIRD_PARTY_NOTICES.md#updating-the-notices-and-creating-archives)
describes; [binary releases](../docs/release.md) publish that archive.
Windows App SDK's WebView metadata is required
to generate the WinUI projection, but no WebView loader is linked or copied.

## Use

Launch `Flare.exe` to use the user's Flare configuration:

```powershell
./.local-dev/winui/ARM64/SingleFile/Flare.exe
```

On first launch, the launcher silently extracts the embedded WinUI runtime to
`Flare/Desktop/<ARM64|x64>/<archive-SHA256>` beneath the Windows Local App Data
known folder, then starts `flare-winui.exe` from that directory. Later launches
reuse that runtime version. Different embedded archives have separate cache
versions; the launcher does not remove versions extracted from other archives.
Each launch checks the cached files against the embedded manifest. Missing or
corrupt files trigger extraction into a staging directory; the application starts
only after the complete replacement has passed verification. The launcher exits
after starting the native application. Taskbar relaunch metadata points to the
launcher executable and preserves an explicitly selected configuration.
The launcher `Flare.exe` is the entry point, so no adjacent DLL folder
is needed. This runtime cache is separate from model cores, configuration,
publisher trust, and desktop state. Packaging and extraction do not move or copy
model data.

The single-file `Flare.exe` is qualified on Windows 11 ARM64 when launched from
a folder containing no sidecar files, with spaces and Unicode in the executable
and configuration paths: first extraction, cache reuse, missing-resource repair,
and taskbar relaunch properties. On an RTX Spark N1X, the bundled application is
qualified to load a Qwen3.8-27B core and return a chat completion. Not
qualified: a machine without the development environment installed, and the x64
runtime.

Pass `--config PATH` to select a configuration explicitly. The launcher forwards
arguments and retains the caller's working directory for relative paths.
The loose `Release/flare-winui.exe` accepts the same arguments. Unless
`paths.core_cache` is configured, the shared library stores cores in
`~/.cache/flare/cores` beneath the user profile folder. When `cores` is absent
and `~/.cache/flare/packages` exists, it uses `packages` in place without copying
or moving downloaded cores.

Cores can be discovered on Hugging Face, downloaded by repository ID, or opened
from a prepared `.safetensors` file or core folder. Opening an external
core records a reference without copying its files. Source checkpoints must
first be prepared through Ignition. A discovery tag does not establish core
compatibility or publisher trust; the Rust application validates the core
before serving and supplies the exact publisher identity for consent.

Discovery cards recognize downloaded cores through retained acquisition records,
including CLI downloads, and offer Run model for that local version. The
association does not claim that the local version is the latest Hub revision.

Discovery cards show the repository author, published precision, task, license,
context, and hardware builds when that metadata is available. Missing fields
are omitted. Hub file sizes cover every file in one resolved revision, including
documentation and other deployments; they are not an exact download size or a
GPU memory requirement. Cards show the last update date of the resolved hardware
branch when Hugging Face provides it. Details identifies the branch, commit, and
last commit time in UTC. For downloaded discovery entries, this date describes
the Hub branch; the installed core may be from an earlier commit. Cached
cores show logical payload size as Core data, with context and hardware from
installed deployments. Logical payload size can differ from storage usage.
Discovery reads Hub metadata and small configuration files, without fetching
model weights. Repository claims do not establish core compatibility or trust.

Interface copy uses factual labels, status, and actions. Page headers omit
promotional subtitles; the empty Downloads view shows No downloads and Browse
models alongside access to core storage.

Home centers on running a local core and observing its runtime. Installed
cores have a Run model action. Cached models with one recorded Hugging Face
repository use its repository name on Home and in the library. Local exports
use the model name recorded in their core metadata. Add model opens a single
chooser for Hugging Face discovery, a repository link, or a local core file or
folder. Model details are available in Models. Loading shows runtime progress.
A running model exposes switch and stop controls alongside the serving address
and its copy action.
Chat remains available in the sidebar for testing the active session.
The dashboard uses the CLI overview's two-second rate window and 120-second
history for prefill, generation, verification, and draft acceptance/depth.
It also shows request counts, token totals, median first-response latency,
KV-cache capacity, host prefix-cache activity, and driver-reported GPU memory.
Accepted drafts and depth are counts per request round, not percentages.
Weight bandwidth is an estimate from verification rate and declared decode
weight bytes, not measured GPU throughput. A dash means a metric is unavailable;
zero is a measured value. The terminal scheduler trace is not exposed in this UI.
Models and downloads show
data from immutable Rust snapshots. The UI limits visible lists to 200 models
and 100 transfers. Transfer
controls appear only when the shared application supports that operation.

Downloads first read the primary core, then prepare the required companion-file
transfers. Preparation reports completed metadata checks with at most eight
requests in flight. Transfer speed uses the last two seconds of byte movement;
speed and ETA are hidden during preparation, validation, and installation.
Those phases use indeterminate progress, and only a completed install shows 100%.
Clear failed removes failed entries from download history. Completed, active,
paused, and cancelled entries remain, and downloaded data is kept for retries.

Model details inspect signed metadata without granting publisher consent or
loading the GPU. For this PC hides undownloaded repositories whose published
platform branches exclude the first visible GPU's platform, and local cores
whose inspected requirements are known to be incompatible. Missing build metadata
or an unknown GPU platform leaves a repository visible. Downloaded cores use
their inspected requirements even if the repository no longer publishes a
matching platform branch. A matching branch does not establish core compatibility
or available memory; loading performs those checks. Chat streams
multi-turn test conversations from the active Helios session.
Enter sends a message; Shift+Enter adds a line.
Stop preserves partial output, and New chat clears the conversation. Successful
completed exchanges supply subsequent context; cancelled or failed exchanges do
not. Changing the serving session clears the conversation.

Settings exposes desktop serving overrides with the same parsers and engine
options as the CLI. Controls use modes, switches, sliders, numeric controls,
and a folder picker; users do not need to enter CLI size or duration syntax.
Speculation and embedding controls are grouped under Advanced. Save persists
overrides in `desktop-state.json` next to the
selected configuration file; it preserves that TOML configuration, API key,
library, downloads, and appearance. Changes apply on the next model load.
The running listener and memory allocations remain unchanged until reload.
Reload model finishes accepted requests before releasing and loading the core.

| Setting | Runtime behavior |
|---|---|
| Serving memory limit | `30G` caps occupied memory, `-8G` reserves free memory, and `0.8` selects a total-memory fraction per GPU. Blank inherits the configured/core reserve. KV pages use the remainder after model and runtime allocations. |
| KV capacity limit | Available memory, one prepared context, or prepared context times concurrency; this does not change prepared context length. |
| Maximum concurrent requests | Blank inherits model/configuration/core defaults. An override must fit prepared capacity. |
| Listener | Localhost by default; all IPv4 interfaces (`0.0.0.0`) or a custom IP and port are selectable. Client URLs use loopback for wildcard listeners, while Settings shows the actual bind. |
| Verify weights on load | Off by default, matching the CLI. Enabling it scans stored weight ranges before GPU loading. Publisher/executable checks and download-integrity checks remain enabled. |
| Host prefix cache | Disk caching defaults off. An explicit `paths.prefix_cache` enables it; saved desktop preferences take precedence. When enabled without an explicit path, it uses `prefix-cache` beside the model storage directory. Automatic disk limit uses the engine’s 10% free-space budget. RAM budget, expiry, and bounded shutdown draining are configurable. Disk caching also reserves transfer-staging RAM, including when RAM retention is disabled. |
| Speculation | Draft count and tri-state history/adaptive-verification choices must fit prepared graphs and inputs. |
| Embedding tables | Default/disk/RAM storage and row-cache budget apply to auxiliary PLE/Engram tables. |

The default window is 1200 × 1000 logical pixels, constrained to the display’s
working area. The dashboard shows four plots in a two-by-two layout.

One built-in recommendation maps the exact prepared repository
`local-inference-lab/Qwen3.8-27B-FLARE` to the display identity
`nvidia/Qwen3.8-27B-NVFP4`. It is available offline on an empty Home screen and
in Models. Cached cores acquire that identity only through recorded acquisition
provenance. Download locations, opaque core identifiers, publisher review,
signatures, and licenses keep their actual values. The scoped mapping lives
in `flare/src/recommendations.rs` so it can be removed independently.

Conversations remain in memory across navigation and are never saved to disk.
Each chat accepts up to 32 exchanges within the Rust transcript limits. Rejected
submissions preserve the draft. Responses are selectable, with copy controls
and separately collapsible reasoning. Scrolling upward pauses automatic following
of streamed output.

Device memory is sampled through the Rust hardware monitor every two seconds
and includes other processes. First response shows the rolling server-side
median over 60 seconds, with its sample count; the retained window is capped at
4,096 requests. Flare uses a dark appearance regardless of the Windows theme or
a System or Light theme value saved in `desktop-state.json`. Display settings
offer comfortable or compact spacing, which
survives application restarts. Windows HighContrast colors apply when
HighContrast is enabled.

Switching a running model requires confirmation and drains accepted work before
releasing the active session. The endpoint is unavailable while a replacement
loads. Closing the window stops local serving and may interrupt active requests;
the UI remains responsive while the Rust session closes. Downloads and serving
run in this process and do not continue after the application exits.
