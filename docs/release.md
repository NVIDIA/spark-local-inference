# Binary releases

The GitHub Actions workflow `.github/workflows/release.yml` publishes a GitHub
Release when a version tag is pushed. A release holds one archive per platform:

| Asset | Runner | Executable |
|---|---|---|
| `flare-linux-amd64.tgz` | `ubuntu-24.04` | `flare/flare`, the CLI |
| `flare-linux-arm64.tgz` | `ubuntu-24.04-arm` | `flare/flare`, the CLI with the RoCE transport |
| `flare-windows-amd64.zip` | `windows-2025` | `flare/Flare.exe`, the [desktop client](../flare-winui/README.md) |
| `flare-windows-arm64.zip` | `windows-11-arm` | `flare/Flare.exe`, the desktop client |

Archive names carry no version, so
`releases/latest/download/flare-linux-arm64.tgz` always names the most recent
release. Each archive also contains `LICENSE`, `NOTICE`,
`THIRD_PARTY_NOTICES.md`, `licenses/`, and `third_party/`, as
[third-party notices](../THIRD_PARTY_NOTICES.md) describes.

A release contains no Windows CLI, Debian package, Selene manager, or
container image. Build those from source: [Windows serving](windows.md),
[Debian packages](debian.md), [the Selene manager](../selene-manager/README.md),
and [container](container.md). Ignition, Python, and model cores are separate
preparation inputs and are never part of a release.

## Host requirements

Every executable loads the NVIDIA driver at run time; no archive contains CUDA
libraries, and no build links the CUDA toolkit.

- The Linux executables are built on Ubuntu 24.04 and require glibc 2.39 or
  later.
- The arm64 Linux executable links `libibverbs` for multi-host RoCE serving.
  Install `libibverbs1` and `ibverbs-providers`.
- The Windows desktop requires Windows 11. It bundles its Windows App SDK and
  Visual C++ runtime files. `Flare.exe` is unsigned.

Release executables embed no publisher verification profiles; publisher
acceptance follows [signing](signing.md) as it does for a source build.

## Publish a release

1. Set the same `version` in `flare-cli/Cargo.toml`,
   `flare-winui/bridge/Cargo.toml`, and `flare-winui/launcher/Cargo.toml`,
   update the three `Cargo.lock` files, and merge the change to the default
   branch.
2. Tag that commit `vMAJOR.MINOR.PATCH` with the same version and push the tag.

The workflow publishes to the repository that received the tag. Running it
through `workflow_dispatch` performs every build and check and uploads the four
archives as workflow artifacts without creating a release.

Enable the repository's immutable-releases setting so that published assets
cannot be replaced and `gh release verify-asset` can check a downloaded file.
The workflow attaches every asset to a draft and publishes the draft last, the
order that setting requires.

## Builds and checks

All builds use Rust 1.94.1 and the locked Cargo manifests. Every job must pass
before anything is published; no target can be skipped.

- **Version.** A tag push requires the tagged commit to be an ancestor of the
  default branch. The three crate versions listed above must equal the tag's
  version. A dispatched run takes the version from `flare-cli/Cargo.toml`.
- **Linux.** The runner builds `flare` with
  `cargo build --locked --release --manifest-path flare-cli/Cargo.toml` and
  packages it with `packaging/archive.py`. The extracted archive then runs in
  an `ubuntu:24.04` container that holds only that image's libraries, plus the
  two RoCE packages on arm64: `flare --version` must print `flare VERSION`, and
  `flare run --help`, `flare agent --help`, and `flare orbit --help` must
  succeed. The job also requires the license and notice files in the archive.
- **Windows.** The runner installs tools with
  `scripts/setup_windows.ps1 -RustToolchain 1.94.1 -BuildToolsOnly`, builds
  with `scripts/build_winui.ps1 -Platform PLATFORM -SingleFile`, and packages
  with `packaging/archive.py --winui-packages`. `Flare.exe` is then started
  from a directory that contains no other file. The launcher must exit with
  code zero, and `flare-winui.exe` must open a window from the extracted
  runtime and still be running 15 seconds later.
- **Publication.** `gh release create --draft --generate-notes` attaches the
  four archives, then `gh release edit --draft=false` publishes the release.
  If publication fails after the draft exists, delete the draft before
  rerunning the job.

## Qualification

The runners have no NVIDIA GPU. A release establishes that each executable
builds from the locked manifests, starts on its operating system and
architecture, and ships with its notices. It does not establish model serving.
Serving qualification is stated per platform in [Windows serving](windows.md),
[the desktop client](../flare-winui/README.md), and
[model preparation and serving](model-preparation.md), and a release does not
extend it. The workflow runs no crate test suites; `Native serving` runs them
on x86_64 Linux and x86_64 Windows only.

Qualified on x86_64 Linux outside GitHub Actions: packaging the CLI with
`packaging/archive.py` and the workflow's clean-container check, including its
rejection of a version that differs from the expected one. The desktop archive
layout is qualified with a placeholder in place of `Flare.exe` and the pinned
NuGet packages downloaded from nuget.org.

Not qualified: any run of the workflow on GitHub-hosted runners, the arm64
Linux build and check, the Windows build and startup check on either
architecture, and publication.
