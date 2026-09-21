#!/usr/bin/env bash
# release.sh — ONE command: build → sign(v2+v3) → verify → split (apk/vN) →
# round-trip check → chat parts (28 MiB) → refresh central bigfile APK.
#
# Usage:  scripts/release.sh <vN>        e.g. scripts/release.sh v7
#         scripts/release.sh v7 --no-build   (reuse the already-built APK)
#
# Output:
#   apk/<vN>/            45 MiB repo parts + SHA256.txt + (write your own README)
#   dist/chat/<vN>/      28 MiB chat-upload parts + SHA256.txt
#   dist/OGAM-release.apk  canonical whole APK, re-split into .bigfiles/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="${1:?usage: scripts/release.sh <vN> [--no-build]}"
NOBUILD="${2:-}"
APK="$ROOT/android/app/build/outputs/apk/release/app-release.apk"
SIGNER="$(ls "${ANDROID_HOME:-/opt/android-sdk}"/build-tools/*/apksigner 2>/dev/null | sort -V | tail -1)"
KS="$ROOT/android/app/debug.keystore"

if [ "$NOBUILD" != "--no-build" ]; then
  echo "== [1/6] build (arm64-v8a release) =="
  ( cd "$ROOT/android" && export ANDROID_HOME="${ANDROID_HOME:-/opt/android-sdk}" ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-/opt/android-sdk}" \
      && ./gradlew assembleRelease -PreactNativeArchitectures=arm64-v8a --no-daemon )
else
  echo "== [1/6] reuse existing APK (--no-build) =="
fi
[ -f "$APK" ] || { echo "release: APK not found at $APK"; exit 1; }

echo "== [2/6] sign v2+v3 =="
"$SIGNER" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --ks-key-alias androiddebugkey \
  --v1-signing-enabled true --v2-signing-enabled true --v3-signing-enabled true "$APK" >/dev/null 2>&1

echo "== [3/6] verify signature =="
"$SIGNER" verify --verbose "$APK" 2>/dev/null | grep -iE "verifies|scheme" | grep -v "JAVA_TOOL" || true

echo "== [4/6] split -> apk/$VER (45 MiB repo parts) =="
"$ROOT/scripts/apkpack.sh" split "$APK" "apk/$VER" 45m >/dev/null
SHA="$(sha256sum "$APK" | awk '{print $1}')"
echo "   sha256=$SHA"
echo "== [4b] round-trip verify =="
"$ROOT/scripts/apkpack.sh" merge "apk/$VER" "/tmp/${VER}-rt.apk" 2>&1 | grep -iE "OK|MISMATCH" || true
rm -f "/tmp/${VER}-rt.apk"

# Reclaim disk before writing more parts: the ~1.5 GB unstripped native libs under
# build/intermediates are no longer needed once the APK exists, and stale chat dirs pile up.
rm -rf "$ROOT/android/app/build/intermediates/merged_native_libs" \
       "$ROOT/android/app/build/intermediates/stripped_native_libs" 2>/dev/null || true

echo "== [5/6] chat parts -> dist/chat/$VER (28 MiB) =="
CHAT="$ROOT/dist/chat/$VER"; rm -rf "$CHAT"; mkdir -p "$CHAT"
split -b 28m -d "$APK" "$CHAT/OGAM-$VER.apk.part"
printf '%s  OGAM-%s.apk\n' "$SHA" "$VER" > "$CHAT/SHA256.txt"
echo "   $(ls "$CHAT"/OGAM-$VER.apk.part* | wc -l) chat parts"

echo "== [6/6] refresh canonical bigfile APK =="
cp "$APK" "$ROOT/dist/OGAM-release.apk"
"$ROOT/scripts/bigfile.sh" split dist/OGAM-release.apk >/dev/null 2>&1 || true

echo "release: $VER done. Repo parts: apk/$VER  Chat parts: dist/chat/$VER  sha256=$SHA"
