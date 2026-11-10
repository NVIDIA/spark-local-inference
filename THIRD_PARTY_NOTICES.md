# Third-party notices

Flare's CLI binary consists of the native Rust executable and its linked
dependencies. The Windows desktop also includes the C++/WinRT shell, its Rust
launcher, and the bundled Windows App SDK runtime. Ignition, Heliopause, Python,
and Python dependencies are used only
for building, model preparation, or testing; they are not included in the binary
release. The public source repository includes the preparation tools and their
dependency declarations. Model weights, datasets, prepared model cores, and
container images are not included in the binary release.

## Native CLI release dependencies

[third_party/rust/README.md](third_party/rust/README.md) groups the native Cargo
dependencies by their declared license expression and links their complete license
and notice texts. [packages.json](third_party/rust/packages.json) records versions,
source locations, and runtime versus build/proc-macro roles for each target.
Build-tool notices are included conservatively; this does not mean the tools are
distributed. The inventory covers the default Flare build for Linux x86_64,
Linux aarch64, and Windows x86_64 MSVC. Optional `rdma` builds require their own
dependency review before release.

The Rust standard library's notices are copied from the build toolchain into
`third_party/rust-standard-library/` by the release packager. Their contents
depend on the compiler used to build the executable.

Crate-level license declarations do not replace notices for embedded native code.
The collected texts include libjpeg-turbo through `turbojpeg-sys`, Oniguruma
through `onig_sys`, zstd through `zstd-sys`, and BoringSSL-derived and fiat code
through `ring`. The permissive BSD license alternative is used for zstd; retaining
its upstream COPYING file does not select its alternative GPL license.

### MPL source availability

`option-ext` 0.2.0 is used unmodified under MPL-2.0. Its covered source is available
from the [exact crate source archive](https://crates.io/api/v1/crates/option-ext/0.2.0/download).
Its license text is included in the native dependency notices. This source
availability notice accompanies binary distributions containing that dependency.

### Code adapted into this repository

[NOTICE](NOTICE) identifies the adapted code and upstream copyright notices.
PyTorch-derived image resizing code is compiled into the native runtime; its
full license and copyrights are included in `licenses/PyTorch.txt` in binary
archives and in `helios/licenses/PyTorch.txt` in the source tree.

The Linux aarch64 runtime also compiles the RoCE proxy in
`helios/native/roce_proxy.c`, contributed by its original author from the pinned
FlashInfer/b12x source. Its upstream Apache-2.0 license
is retained in `helios/licenses/FlashInfer.txt` and included in binary archives.
This native C source is distinct from the separately installed Python package.

The vLLM-derived sampler and DeepSeek recipe assets belong to the source and
preparation tools. Their existing notices and license texts remain in the source
tree. They are not part of the native executable's dependency graph.

The BRRR-derived DeepSeek V4.1 CUDA decoder and Python weight-layout adaptation
are retained in `ignition/recipes/deepseek41/native_brrr`, with Fraser Price's
original NOTICE and Apache-2.0 license. Flare compiles the extension during model
preparation; it is not linked into the CLI binary. Prepared cores containing its
captured kernels retain the upstream licensing requirements. See the root
[NOTICE](NOTICE) for the source revision and adaptation boundary.

## Preparation and build dependencies

`pyproject.toml` and `uv.lock` describe the separately installed preparation and
test environment, including PyTorch, Transformers, FlashInfer/b12x, CUDA Python
bindings, and NVIDIA CUDA wheels. Heliopause is a native extension used by that
Python preparation environment, not a shipped runtime library. These packages
are installed from their own distributions and retain their own license terms.
They must be included in a source-repository review, but are not bundled into
Flare's native binary release. The copied native RoCE proxy described above is
compiled into the executable on Linux aarch64.

Upstream chat templates are fetched from the model repositories during model
preparation and annotated using the bundled insertion patches. They are not
bundled with the native release. The repository's custom Kimi template remains
part of Flare's source.

## External runtime libraries

The NVIDIA CUDA driver is loaded from the user's system. Linux aarch64 builds
also link the system libibverbs and pthread interfaces for RoCE. Platform C/C++
runtime libraries remain system dependencies for the CLI. The CLI release
archive packager copies no shared libraries or CUDA SDK components.

The Windows desktop instead bundles app-local Windows App SDK and Visual C++
runtime files. Its dependency versions are pinned in
`flare-winui/packages.config`, `flare-winui/bridge/Cargo.lock`, and
`flare-winui/launcher/Cargo.lock`. The desktop single-file packager does not yet
collect these dependencies' distribution notices; the CLI notice inventory
above does not cover the complete desktop distribution.

| System component | Terms and use |
| --- | --- |
| NVIDIA CUDA driver | Separately installed NVIDIA driver terms; loaded at runtime. No CUDA SDK or driver library is bundled. |
| libibverbs (rdma-core) | [OpenIB.org BSD alternative](https://github.com/linux-rdma/rdma-core/blob/v50.0/libibverbs/verbs.h) selected for the RoCE interface. The header copyright and license notice is reproduced in `helios/licenses/libibverbs.txt` and packaged under `licenses/`. |
| pthread (glibc on supported GNU/Linux targets) | [LGPL-2.1-or-later](https://github.com/bminor/glibc/blob/glibc-2.39/nptl/pthread_create.c); dynamically linked system implementation, not bundled. |

System-library versions depend on the build image and installed operating system,
not the Cargo lockfile. The source links above identify the license texts, not
a claim that those exact versions ship with Flare. Debian packaging derives
runtime version requirements with `dpkg-shlibdeps`.

## Source test fixtures

The source tree includes small test inputs. The legacy `.safetensors` fixtures
contain synthetic core data (byte patterns, text placeholders, and metadata),
not pretrained model checkpoints. `tests/fixtures/qwen38_mtp_feedback.pt`
contains five hidden-state activation rows and five token IDs for a numerical
test; that test loads its actual model weights from a separate checkpoint.
Image preprocessing fixtures are generated by the scripts in `tests/frontend/`.
These test inputs are not included in native binary release archives.

## Updating the notices and creating archives

Run from the repository root after changing the native dependency lockfile:

```sh
python3 packaging/licenses.py /tmp/flare-rust-notices
```

Use a fresh output directory, review it, and replace `third_party/rust/` with the
result. The collector fails if any dependency lacks license text. Files omitted
from published crates are supplied in `packaging/license-supplements/`, with
links to the exact upstream revisions from which they were copied.

Build with the locked manifest, then use the same toolchain to package that
executable. For example, on Linux x86_64:

```sh
cargo build --locked --release --manifest-path flare-cli/Cargo.toml
rustup component add rust-docs
python3 packaging/archive.py --binary flare-cli/target/release/flare \
  --target x86_64-unknown-linux-gnu --output flare-linux-x86_64.tgz
```

Use the matching target, executable path, and `.zip` output on Windows. The
packager includes LICENSE, NOTICE, this document, adapted-code licenses, native
dependency notices, and toolchain notices. It copies no Python packages,
Heliopause extension, model weights, or prepared model assets.
