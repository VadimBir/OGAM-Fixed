#!/usr/bin/env bash
# release.sh — ONE command: build → sign(v2+v3) → verify → split (apk/vN) →
# round-trip check → chat parts (28 MiB) → refresh central bigfile APK.
#
# Usage:  scripts/release.sh <vN|next>        e.g. scripts/release.sh v7 | scripts/release.sh next
#         scripts/release.sh next --no-build   (reuse the already-built APK)
#         COMMIT_MSG="..." scripts/release.sh next   → also git add -A, commit, push (same call)
#
# `next` = highest existing apk/vN + 1. An existing apk/<vN> is NEVER overwritten.
#
# Output:
#   apk/<vN>/            45 MiB repo parts + SHA256.txt + (write your own README)
#   dist/chat/<vN>/      28 MiB chat-upload parts + SHA256.txt
#   dist/OGAM-release.apk  canonical whole APK, re-split into .bigfiles/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="${1:?usage: scripts/release.sh <vN|next> [--no-build]}"
NOBUILD="${2:-}"
if [ "$VER" = "next" ]; then
  last=$(ls -d "$ROOT"/apk/v* 2>/dev/null | sed 's#.*/v##' | grep -E '^[0-9]+$' | sort -n | tail -1)
  VER="v$(( ${last:-0} + 1 ))"
fi
[ -e "$ROOT/apk/$VER" ] && { echo "release: apk/$VER already exists — refusing to overwrite (use next)"; exit 1; }
cd "$ROOT"
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

# Reclaim disk BEFORE signing/splitting: app/build/intermediates (~3 GB, incl. ~1.5 GB unstripped
# native libs) is dead once the APK exists; a near-full disk made apksigner fail (v24 run).
# Old chat-part dirs (gitignored) are dropped too; only this version's are written below.
rm -rf "$ROOT/android/app/build/intermediates" "$ROOT/android/app/build/tmp" 2>/dev/null || true
find "$ROOT/dist/chat" -mindepth 1 -maxdepth 1 -type d ! -name "$VER" -exec rm -rf {} + 2>/dev/null || true
echo "   free: $(df -h "$ROOT" | awk 'NR==2{print $4}')"

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

echo "== [5/6] chat parts -> dist/chat/$VER (28 MiB) =="
CHAT="$ROOT/dist/chat/$VER"; rm -rf "$CHAT"; mkdir -p "$CHAT"
split -b 28m -d "$APK" "$CHAT/OGAM-$VER.apk.part"
printf '%s  OGAM-%s.apk\n' "$SHA" "$VER" > "$CHAT/SHA256.txt"
echo "   $(ls "$CHAT"/OGAM-$VER.apk.part* | wc -l) chat parts"

echo "== [6/6] refresh canonical bigfile APK =="
cp "$APK" "$ROOT/dist/OGAM-release.apk"
"$ROOT/scripts/bigfile.sh" split dist/OGAM-release.apk >/dev/null 2>&1 || true

# Free the session disk allowance: the remaining intermediates (~3 GB) are rebuilt on demand.
rm -rf "$ROOT/android/app/build/intermediates" 2>/dev/null || true

echo "release: $VER done. Repo parts: apk/$VER  Chat parts: dist/chat/$VER  sha256=$SHA"

if [ -n "${COMMIT_MSG:-}" ]; then
  BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
  echo "== [7/7] commit + push ($BRANCH) =="
  git add -A
  git commit -q -m "$COMMIT_MSG" -m "APK: apk/$VER sha256 $SHA"
  for d in 0 2 4 8 16; do
    sleep "$d"
    if git push -u origin "$BRANCH"; then
      echo "release: pushed $(git rev-parse --short HEAD) to $BRANCH"; exit 0
    fi
  done
  echo "release: push failed after retries"; exit 1
fi
