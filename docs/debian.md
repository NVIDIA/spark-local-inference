# Debian packages

`flare-inference` installs the native Flare executable at `/usr/bin/flare` on
Ubuntu 24.04, with separate `arm64` and `amd64` packages. APT selects the package
for the installed system's architecture. The ARM64 build includes RoCE support
for DGX Spark. The package contains serving, host-agent, and Orbit commands.
Ignition, Python, model weights,
and the CUDA toolkit are separate preparation inputs. Serving uses the NVIDIA
driver installed by DGX OS and a model core matching the GPU architecture and
tensor-parallel size.

The Debian package name distinguishes the inference server from Ubuntu's `flare`
game package. The command remains `flare`.

## Install

Build the package from source using the instructions below, then install the
file for the host architecture:

```sh
sudo apt install ./dist/debian/flare-inference_VERSION_ARCH.deb
flare --help
```

Replace `VERSION` and `ARCH` with the values in the generated filename.

Installation and upgrades do not create or restart services. To install an
agent, run `/usr/bin/flare agent install` as the user who owns its models. The
command prints a pairing code and enables a systemd user service; see
[agent setup](multi-host.md#one-time-setup-the-flare-agent). The installer copies
the executable to `~/.local/lib/flare-agent/flare`. Orbit uses that copy for both
the agent and default serving; `orbit run --executable PATH` selects an uploaded
override. After a Debian package upgrade, stop the groups that agent serves and
rerun `/usr/bin/flare agent install` to update the copy and restart the service.
Pass `--listen IP:PORT` again if the listener was customized. Existing identities
and caches remain. Restart saved groups explicitly; agent startup does not
restore them. Check `command -v flare` when another installed executable might
take precedence over `/usr/bin/flare`.

## Build

From a committed checkout, with Docker Buildx available:

```sh
./scripts/build_deb.sh --maintainer 'Release Maintainer <maintainer@example.org>'
```

The builder compiles the locked Cargo dependencies in Ubuntu 24.04 using Rust
1.94.1. It builds for Docker's native architecture and writes the `.deb` under
`dist/debian/`. `--output DIR` changes the destination; `--version VERSION`
sets an explicit Debian version. The default version contains the Cargo version,
UTC commit timestamp, and abbreviated commit ID. The installed
`/usr/share/doc/flare-inference/build-info` records the complete revision.
Commit native source changes before building; uncommitted changes cannot be
identified by that revision. Run the same build
on each target architecture.
Docker's `DOCKER_HOST=ssh://HOST` option can select a remote native builder while
returning the `.deb` to the local output directory.

`dpkg-shlibdeps` derives shared-library requirements from Ubuntu's symbol tables.
ARM64 additionally depends on the verbs providers needed for RoCE. Compilation
and dependency discovery take place inside the build image so an NVIDIA OFED
installation on the builder cannot raise the package's minimum library versions.
Root ownership is set in the archive without requiring a privileged build.
The package includes the project license and dependency license notices.
