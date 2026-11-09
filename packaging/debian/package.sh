#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${PACKAGE_VERSION:?package version is required}"
: "${SOURCE_COMMIT:?source commit is required}"
: "${SOURCE_DATE_EPOCH:?source timestamp is required}"
: "${PACKAGE_MAINTAINER:?maintainer name and email are required}"
dpkg --validate-version "$PACKAGE_VERSION"
arch=$(dpkg --print-architecture)
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
chmod 0755 "$stage"
install -Dm755 flare-cli/target/release/flare "$stage/usr/bin/flare"
docs="$stage/usr/share/doc/flare-inference"
install -Dm644 packaging/debian/copyright "$docs/copyright"
install -m644 LICENSE NOTICE THIRD_PARTY_NOTICES.md "$docs/"
install -d "$docs/licenses"
install -m644 helios/licenses/*.txt "$docs/licenses/"
install -m644 packaging/debian/README "$docs/README.Debian"
install -Dm644 packaging/debian/lintian-overrides "$stage/usr/share/lintian/overrides/flare-inference"
install -Dm644 packaging/debian/flare.1 "$stage/usr/share/man/man1/flare.1"
gzip -n -9 "$stage/usr/share/man/man1/flare.1"
python3 packaging/debian/licenses.py "$docs/third_party/rust"
printf 'Source-Commit: %s\nRust: %s\n' "$SOURCE_COMMIT" "$(rustc --version)" > "$docs/build-info"
date_text=$(date -u -d "@$SOURCE_DATE_EPOCH" -R)
cat > "$docs/changelog.Debian" <<EOF
flare-inference ($PACKAGE_VERSION) noble; urgency=medium

  * Native Flare serving, agent, and Orbit commands.
    Source revision: $SOURCE_COMMIT.

 -- $PACKAGE_MAINTAINER  $date_text
EOF
gzip -n -9 "$docs/changelog.Debian"

# dpkg-shlibdeps reads the Ubuntu build image's symbol versions, not the
# builder host's NVIDIA OFED packages.
mkdir -p debian
printf 'Source: flare-inference\n\nPackage: flare-inference\nArchitecture: any\n' > debian/control
dependencies=$(dpkg-shlibdeps -O -e"$stage/usr/bin/flare")
dependencies=${dependencies#shlibs:Depends=}
if [[ "$arch" == arm64 ]]; then
    dependencies+=", ibverbs-providers"
fi
installed_size=$(du -sk "$stage/usr" | cut -f1)
mkdir -p "$stage/DEBIAN" /out
cat > "$stage/DEBIAN/control" <<EOF
Package: flare-inference
Version: $PACKAGE_VERSION
Architecture: $arch
Maintainer: $PACKAGE_MAINTAINER
Installed-Size: $installed_size
Depends: $dependencies, ca-certificates
Section: utils
Priority: optional
Homepage: https://github.com/NVIDIA/spark-local-inference
Description: native inference server for prepared Flare model packages
 Runs prepared CUDA graphs on supported NVIDIA GPUs. Includes the Flare
 command, the host agent, and Orbit multi-host management. ARM64 builds
 include the RoCE transport for DGX Spark clusters.
 .
 Requires a compatible NVIDIA driver and a separately prepared model package.
EOF
(cd "$stage" && find usr -type f -print0 | sort -z | xargs -0 md5sum > DEBIAN/md5sums)
find "$stage" -exec touch --no-dereference -d "@$SOURCE_DATE_EPOCH" '{}' +
dpkg-deb --root-owner-group -Zxz --build "$stage" "/out/flare-inference_${PACKAGE_VERSION}_${arch}.deb"
