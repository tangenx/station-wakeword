#!/bin/sh
# Release artifacts:
#   dist/wakeword-<ver>.yampkg   YAM v2 package (namespace/name, tar.zst)
#   dist/wakeword-<ver>.tar.gz   install.sh bundle (no Go needed)
#
#   ./package.sh [version]
#
# The .yampkg is built and proven installable with the official `yam` binary
# (aarch64 static, downloaded into tools/yam/). The binary is a Linux ELF, so
# it runs natively on a Linux aarch64 host and through Docker (qemu-user-static)
# everywhere else; `yam install --root <tmp> && yam doctor` is the smoke test.
set -eu
root=$(CDPATH='' cd "$(dirname "$0")" && pwd)
cd "$root"
VERSION=${1:-dev}
YAM_VERSION=${YAM_VERSION:-0.2.1}
PLATFORM=${YAM_PLATFORM:-yandexmidi}

sh build.sh

yamdir="$root/tools/yam"
mkdir -p "$yamdir"
yambin="$yamdir/yam-$YAM_VERSION-aarch64-unknown-linux-musl"
if [ ! -s "$yambin" ]; then
    echo "downloading yam $YAM_VERSION..."
    curl -fsSL -o "$yambin" \
        "https://github.com/nyakokitsu/yam/releases/download/yam-v$YAM_VERSION/yam-aarch64-unknown-linux-musl"
    chmod +x "$yambin"
fi

# Platform profiles ship in the yam repo, not in the release asset.
profiles="$yamdir/platforms"
if [ ! -d "$profiles" ]; then
    echo "downloading platform profiles..."
    tmp=$(mktemp -d)
    curl -fsSL "https://github.com/nyakokitsu/yam/archive/refs/tags/yam-v$YAM_VERSION.tar.gz" | tar -xz -C "$tmp"
    cp -R "$tmp/yam-yam-v$YAM_VERSION/platforms" "$profiles"
    rm -rf "$tmp"
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/wakeword-pkg.XXXXXX")
# yam runs as root inside the container, so it may leave files the invoking
# user cannot delete; cleanup must never fail the build.
trap 'rm -rf "$work" 2>/dev/null || true' EXIT
mkdir -p "$work/sandbox"

# run_yam ARGS... — run yam with --root set to $work/sandbox. On a Linux
# aarch64 host yam runs directly; otherwise through Docker, which provides
# qemu. Inside the container the work dir is mounted at /work, so pass host
# paths that survive the mapping: yam only sees /work/... and --root is
# translated by the caller below via YAM_ROOT in the container namespace.
in_docker() {
    [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "aarch64" ] && return 1
    command -v docker >/dev/null 2>&1 && return 0
    echo "need docker (or a Linux aarch64 host) to run the aarch64 yam" >&2
    exit 1
}

run_yam() {
    # Sandbox root for install/doctor/list; `pack` ignores --root.
    if ! in_docker; then
        "$yambin" --platform "$PLATFORM" --profiles "$profiles" --root "$work/sandbox" "$@"
        return
    fi
    docker run --rm --platform linux/arm64 \
        -v "$yambin:/usr/local/bin/yam:ro" \
        -v "$profiles:/profiles:ro" \
        -v "$work:/work" \
        -w /work alpine:3.20 \
        yam --platform "$PLATFORM" --profiles /profiles --root /work/sandbox "$@"
}

# Packed source: yam.yaml with the release version, the built binary and data.
sed "s/^version: .*/version: $VERSION/" yam.yaml > "$work/yam.yaml"
cp -R dist "$work/dist"
mkdir -p "$work/station"
cp station/S97wakeword "$work/station/S97wakeword"

yampkg="$root/dist/wakeword-$VERSION.yampkg"
if ! in_docker; then
    "$yambin" --platform "$PLATFORM" --profiles "$profiles" --root "$work/sandbox" \
        pack "$work" --output "$work/wakeword-$VERSION.yampkg"
else
    # Inside the container the work dir is /work; yam writes the artifact there.
    run_yam pack /work --output "/work/wakeword-$VERSION.yampkg"
fi
cp "$work/wakeword-$VERSION.yampkg" "$yampkg"
echo "built $yampkg"

# --- install.sh bundle (binary + scripts; unchanged layout)
bundle="$work/bundle"
mkdir -p "$bundle/dist" "$bundle/station"
cp install.sh uninstall.sh lib.sh README.md "$bundle/"
cp station/S97wakeword "$bundle/station/"
cp dist/wakeword dist/negatives.bin "$bundle/dist/"
chmod 755 "$bundle/install.sh" "$bundle/uninstall.sh" "$bundle/station/S97wakeword" "$bundle/dist/wakeword"
COPYFILE_DISABLE=1 tar -czf "$root/dist/wakeword-$VERSION.tar.gz" -C "$work" bundle

# --- smoke test: install the packed artifact into the sandbox, then doctor it.
if ! in_docker; then
    run_yam install --yes "$work/wakeword-$VERSION.yampkg"
else
    run_yam install --yes "/work/wakeword-$VERSION.yampkg"
fi
run_yam doctor
run_yam list
echo "sandbox install: ok"

echo "$yampkg"
echo "$root/dist/wakeword-$VERSION.tar.gz"
