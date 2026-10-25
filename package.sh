#!/bin/sh
# Release artifacts:
#   dist/wakeword-<ver>.yap      YAM package (binary service)
#   dist/wakeword-<ver>.tar.gz   install.sh bundle (no Go needed)
#
#   ./package.sh [version]
set -eu
root=$(CDPATH='' cd "$(dirname "$0")" && pwd)
cd "$root"
VERSION=${1:-dev}

sh build.sh

work=$(mktemp -d "${TMPDIR:-/tmp}/wakeword-pkg.XXXXXX")
trap 'rm -rf "$work"' EXIT

# --- YAM package: yamd runs it from the package dir, so the default
# -negatives (next to the binary) resolves; words live in /data/wakeword.
yap="$work/yap"
mkdir -p "$yap"
cp dist/wakeword dist/negatives.bin "$yap/"
chmod 755 "$yap/wakeword"
: > "$yap/launch.lua"
cat > "$yap/plugin.toml" <<EOF
id = "wakeword"
name = "Свои слова активации"
version = "$VERSION"
entry = "launch.lua"
description = "Обучите станцию откликаться на своё слово. Настройка: http://<станция>:8780"
permissions = ["exec"]

[launch]
kind = "binary"
mode = "service"
path = "wakeword"
args = ["run", "-data", "/data/wakeword", "-listen", ":8780"]

[compatibility]
features = ["arch.aarch64"]
EOF

# --- install.sh bundle
bundle="$work/wakeword"
mkdir -p "$bundle/dist" "$bundle/station"
cp install.sh uninstall.sh lib.sh README.md "$bundle/"
cp station/S97wakeword "$bundle/station/"
cp dist/wakeword dist/negatives.bin "$bundle/dist/"
chmod 755 "$bundle/install.sh" "$bundle/uninstall.sh" "$bundle/station/S97wakeword" "$bundle/dist/wakeword"

mkdir -p dist
# COPYFILE_DISABLE keeps macOS from adding ._ AppleDouble entries.
( cd "$yap" && COPYFILE_DISABLE=1 tar -czf "$root/dist/wakeword-$VERSION.yap" plugin.toml launch.lua wakeword negatives.bin )
COPYFILE_DISABLE=1 tar -czf "dist/wakeword-$VERSION.tar.gz" -C "$work" wakeword
echo "dist/wakeword-$VERSION.yap"
echo "dist/wakeword-$VERSION.tar.gz"
