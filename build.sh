#!/bin/sh
# Build the static aarch64 station binary and fetch the bundled negative set.
#
#   ./build.sh    # -> dist/wakeword, dist/negatives.bin
set -eu
root=$(CDPATH='' cd "$(dirname "$0")" && pwd)
cd "$root"
mkdir -p dist
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -trimpath -ldflags="-s -w" -o dist/wakeword ./cmd/wakeword
echo "built dist/wakeword"

if [ ! -f dist/negatives.bin ]; then
    # ~17 MB of precomputed openWakeWord features (CC BY-NC-SA 4.0), fetched
    # by byte ranges; needs python3 with numpy.
    python3 tools/build_negatives.py dist/negatives.bin
fi
