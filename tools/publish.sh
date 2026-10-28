#!/bin/sh
# Publish .yampkg artifacts to the YAM registry (https://repo.yse.wtf/).
#
#   tools/publish.sh [version] [platform ...]
#
# One artifact per target; name+version+target is immutable in the registry,
# so bump the version to fix anything. Environment (never commit or echo):
#   YAM_PUBLISH_TOKEN    publish token from the registry profile
#   YAM_AUTHOR_KEY_FILE  PEM PKCS#8 Ed25519 private key for the namespace
#
# The yam tools (yam, yam-publish) are static musl aarch64 binaries built on
# demand into tools/yam/; they run inside an alpine container here, the same
# environment the station has.
set -eu
root=$(CDPATH='' cd "$(dirname "$0")/.." && pwd)
cd "$root"
VERSION=${1:-dev}
shift 2>/dev/null || true
YAM_VERSION=${YAM_VERSION:-0.2.2}
TOOLS="$root/tools/yam"

[ -n "$YAM_PUBLISH_TOKEN" ] || { echo "YAM_PUBLISH_TOKEN is not set" >&2; exit 1; }
[ -n "$YAM_AUTHOR_KEY_FILE" ] && [ -f "$YAM_AUTHOR_KEY_FILE" ] || {
    echo "YAM_AUTHOR_KEY_FILE does not point at a key" >&2; exit 1
}

# aarch64 Buildroot/quasar stations. armv7 stations (Street, JBL) cannot run
# our binary, so they get no artifact.
PLATFORMS="$*"
[ -n "$PLATFORMS" ] || PLATFORMS="yandexmicro plum yandexmini yandexmini_2 bergamot pickle cucumber yandexmidi saturn yandexstation orion"

sh build.sh

publisher="$TOOLS/yam-publish-$YAM_VERSION-aarch64-unknown-linux-musl"
yam="$TOOLS/yam-$YAM_VERSION-aarch64-unknown-linux-musl"
profiles="$TOOLS/platforms"
if [ ! -d "$profiles" ]; then
    echo "platform profiles missing; run ./package.sh first" >&2
    exit 1
fi
if [ ! -x "$publisher" ] || [ ! -x "$yam" ]; then
    command -v docker >/dev/null 2>&1 || {
        echo "need docker to build the yam tools, or prepared $publisher and $yam" >&2
        exit 1
    }
    echo "building yam $YAM_VERSION tools (aarch64 musl)..."
    docker run --rm --platform linux/arm64 -v "$TOOLS:/out" -w /build rust:1-bookworm /bin/sh -c '
        set -e
        apt-get update -qq >/dev/null 2>&1
        apt-get install -y -qq musl-tools musl-dev >/dev/null 2>&1
        rustup target add aarch64-unknown-linux-musl >/dev/null 2>&1
        git clone -q --depth 1 https://github.com/nyakokitsu/yam yam
        cd yam
        git checkout -q "yam-$0" 2>/dev/null || true
        cargo build --release --locked --target aarch64-unknown-linux-musl --bin yam --bin yam-publish 2>&1 | tail -1
        cp target/aarch64-unknown-linux-musl/release/yam "/out/yam-$0-aarch64-unknown-linux-musl"
        cp target/aarch64-unknown-linux-musl/release/yam-publish "/out/yam-publish-$0-aarch64-unknown-linux-musl"
        chmod +x "/out/yam-$0-aarch64-unknown-linux-musl" "/out/yam-publish-$0-aarch64-unknown-linux-musl"
    ' "$YAM_VERSION"
fi

# Pack source with the release version.
work="$root/.packwork"
trap 'rm -rf "$work" 2>/dev/null || true' EXIT
rm -rf "$work"; mkdir -p "$work"
sed "s/^version: .*/version: $VERSION/" yam.yaml > "$work/yam.yaml"
cp -R dist "$work/dist"
mkdir -p "$work/station"
cp station/S97wakeword "$work/station/S97wakeword"

# run_yam BINARY ARGS... — the tools are Linux ELFs; run them in an alpine
# container that mounts the work dir and the tools at matching paths.
run_yam() {
    bin=$1; shift
    if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "aarch64" ]; then
        "$bin" "$@"
        return
    fi
    command -v docker >/dev/null 2>&1 || {
        echo "need docker to run the aarch64 yam tools" >&2; exit 1
    }
    # The host repo is mapped into the container at /host, so the binary's
    # host path still resolves inside.
    rel=${bin#"$root"/}
    docker run --rm --platform linux/arm64 \
        -v "$root:/host" -v "$work:/work" -v "$TOOLS/platforms:/profiles:ro" \
        -v "$YAM_AUTHOR_KEY_FILE:/key.pem:ro" \
        -w /work alpine:3.20 /bin/sh -c \
        'cp "/host/$1" /work/.tool && chmod +x /work/.tool && shift && exec /work/.tool "$@"' \
        sh "$rel" "$@"
}

for platform in $PLATFORMS; do
    echo "publishing tangenx/wakeword $VERSION for $platform ..."
    # Inside the container the work dir is /work and the key is /key.pem.
    # pack builds the .yampkg from the full yam.yaml. publish is called
    # separately (not via `release`) because yam-publish 0.2.2 parses the
    # source yam with a two-field manifest (deny_unknown_fields) and rejects
    # every documented key but name and version.
    artifact="/work/wakeword-$VERSION-$platform.yampkg"
    run_yam "$publisher" pack \
        --source /work \
        --platform "$platform" \
        --profiles /profiles \
        --output "$artifact"
    mkdir -p "$work/min"
    printf 'name: %s\nversion: %s\n' "tangenx/wakeword" "$VERSION" > "$work/min/yam.yaml"
    run_yam "$publisher" publish \
        --artifact "$artifact" \
        --source /work/min \
        --target "$platform" \
        --token "$YAM_PUBLISH_TOKEN" \
        --key /key.pem
    # Prove the artifact installs before calling it published.
    rm -rf "$work/smoke"
    run_yam "$yam" --platform "$platform" --profiles /profiles --root /work/smoke install --yes "$artifact"
    run_yam "$yam" --platform "$platform" --profiles /profiles --root /work/smoke doctor
    cp "$work/wakeword-$VERSION-$platform.yampkg" "$root/dist/"
    echo "ok: $root/dist/wakeword-$VERSION-$platform.yampkg (installed and doctor clean)"
done
