#!/usr/bin/env bash
#
# apkpack.sh — split a large APK into git-committable chunks, and merge them back.
#
# WHY: the release APK (arm64-v8a, JS-bundled) is ~164 MiB. GitHub hard-rejects any
# single blob >100 MiB on a normal push, so the binary cannot be committed whole.
# This packs it into <100 MiB parts (default 45 MiB, below GitHub's 50 MiB warning)
# that live under dist/apk/, and reassembles + checksum-verifies them on demand.
#
# USAGE:
#   scripts/apkpack.sh split [APK_PATH] [OUT_DIR] [CHUNK]
#       APK_PATH  default: android/app/build/outputs/apk/release/app-release.apk
#       OUT_DIR   default: apk/v1
#       CHUNK     default: 45m
#
#   scripts/apkpack.sh merge [PARTS_DIR] [OUT_APK]
#       PARTS_DIR default: apk/v1
#       OUT_APK   default: apk/app-release.apk
#
# The split writes app-release.apk.partNN + SHA256.txt (full-file digest).
# The merge concatenates part* in lexical order and verifies against SHA256.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_APK="$ROOT/android/app/build/outputs/apk/release/app-release.apk"
DEFAULT_DIR="$ROOT/apk/v1"
BASENAME="app-release.apk"

sha256() { sha256sum "$1" | awk '{print $1}'; }

cmd_split() {
  local apk="${1:-$DEFAULT_APK}" out="${2:-$DEFAULT_DIR}" chunk="${3:-45m}"
  [ -f "$apk" ] || { echo "apkpack: APK not found: $apk" >&2; exit 1; }
  mkdir -p "$out"
  rm -f "$out/${BASENAME}.part"* "$out/SHA256.txt"
  local digest size
  digest="$(sha256 "$apk")"
  size="$(wc -c < "$apk" | tr -d ' ')"
  split -b "$chunk" -d -a 2 "$apk" "$out/${BASENAME}.part"
  {
    echo "$digest  $BASENAME"
  } > "$out/SHA256.txt"
  echo "apkpack: split $apk ($size bytes) -> $out"
  ls -l "$out/${BASENAME}.part"* | awk '{print "  " $NF "  " $5 " bytes"}'
  echo "apkpack: full-file sha256 = $digest  (recorded in $out/SHA256.txt)"
  echo "apkpack: rebuild with -> scripts/apkpack.sh merge \"$out\""
}

cmd_merge() {
  local dir="${1:-$DEFAULT_DIR}" outapk="${2:-$ROOT/apk/$BASENAME}"
  local parts
  parts="$(ls "$dir/${BASENAME}.part"* 2>/dev/null | sort || true)"
  [ -n "$parts" ] || { echo "apkpack: no parts in $dir" >&2; exit 1; }
  mkdir -p "$(dirname "$outapk")"
  # shellcheck disable=SC2086
  cat $parts > "$outapk"
  echo "apkpack: merged $(echo "$parts" | wc -l | tr -d ' ') parts -> $outapk"
  if [ -f "$dir/SHA256.txt" ]; then
    local want got
    want="$(awk '{print $1}' "$dir/SHA256.txt")"
    got="$(sha256 "$outapk")"
    if [ "$want" = "$got" ]; then
      echo "apkpack: OK  sha256 matches ($got)"
    else
      echo "apkpack: FAIL  sha256 mismatch" >&2
      echo "  expected $want" >&2
      echo "  got      $got" >&2
      exit 2
    fi
  else
    echo "apkpack: WARNING no SHA256.txt in $dir — cannot verify integrity" >&2
  fi
}

case "${1:-}" in
  split) shift; cmd_split "$@" ;;
  merge) shift; cmd_merge "$@" ;;
  *) echo "usage: $0 {split|merge} [args]  (see header for details)" >&2; exit 64 ;;
esac
