#!/usr/bin/env bash
# bigfile — central large-file format for this repo (poor-man's git-LFS, no server).
#
# WHY: some artifacts we want in git exceed GitHub's 100 MiB blob hard limit
# (prebuilt native .so up to ~154 MiB, release APKs ~164 MiB). Instead of a real
# LFS server, we store each big file as fixed-size PARTS under .bigfiles/<slug>/
# plus one central manifest. Parts are < the GitHub limit, so a plain `git push`
# works. The real (whole) file is gitignored and reconstructed on demand.
#
# CENTRAL MANIFEST: .bigfiles/manifest.tsv, one line per tracked file:
#   <realpath>\t<chunk_bytes>\t<sha256-of-whole-file>
#
# LIFECYCLE (run via git hooks in .githooks/, installed with `bigfile install`):
#   pre-push       -> `bigfile split`  : (re)split any tracked file whose whole
#                                          copy changed, refresh manifest sha256.
#   post-merge     -> `bigfile merge`  : reassemble every tracked file, verify sha.
#   post-checkout  -> `bigfile merge`  : same, after checkout/clone-with-hooks.
#
# COMMANDS:
#   bigfile track  <path> [chunk=95m]  register + split + gitignore the whole file
#   bigfile untrack <path>             drop from manifest (leaves parts on disk)
#   bigfile split  [path...]           split tracked files into parts (all if none)
#   bigfile merge  [path...]           reassemble tracked files (all if none)
#   bigfile status                     show tracked files, present/missing, sha ok
#   bigfile install                    point git at .githooks and reassemble now
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORE="$ROOT/.bigfiles"
MANIFEST="$STORE/manifest.tsv"
DEFAULT_CHUNK="95m"

sha_of() { sha256sum "$1" | awk '{print $1}'; }
slug_of() { printf '%s' "$1" | sed 's#[^A-Za-z0-9._-]#_#g'; }

ensure_store() { mkdir -p "$STORE"; touch "$MANIFEST"; }

# read manifest -> arrays via callback: while_manifest <fn>; fn realpath chunk sha
while_manifest() {
  local fn="$1" rp ck sh
  [ -f "$MANIFEST" ] || return 0
  while IFS=$'\t' read -r rp ck sh; do
    [ -z "${rp:-}" ] && continue
    "$fn" "$rp" "${ck:-$DEFAULT_CHUNK}" "${sh:-}"
  done < "$MANIFEST"
}

gitignore_add() {
  local rel="$1" gi="$ROOT/.gitignore"
  grep -qxF "/$rel" "$gi" 2>/dev/null || printf '/%s\n' "$rel" >> "$gi"
}

manifest_set() { # realpath chunk sha  (upsert)
  local rp="$1" ck="$2" sh="$3" tmp
  tmp="$(mktemp)"
  awk -F'\t' -v p="$rp" '$1!=p' "$MANIFEST" > "$tmp" 2>/dev/null || true
  printf '%s\t%s\t%s\n' "$rp" "$ck" "$sh" >> "$tmp"
  sort -o "$tmp" "$tmp"
  mv "$tmp" "$MANIFEST"
}

split_one() { # realpath chunk
  local rp="${1:-}" ck="${2:-$DEFAULT_CHUNK}" whole dir sh
  [ -z "$rp" ] && return 0
  whole="$ROOT/$rp"
  if [ ! -f "$whole" ]; then echo "bigfile: split skip (no whole file): $rp" >&2; return 0; fi
  dir="$STORE/$(slug_of "$rp")"
  rm -rf "$dir"; mkdir -p "$dir"
  ( cd "$dir" && split -b "$ck" -d -a 3 "$whole" "part" )
  sh="$(sha_of "$whole")"
  printf '%s  %s\n' "$sh" "$(basename "$rp")" > "$dir/SHA256.txt"
  manifest_set "$rp" "$ck" "$sh"
  echo "bigfile: split $rp -> $(ls "$dir"/part* | wc -l) parts (chunk $ck, sha ${sh:0:12}…)"
}

merge_one() { # realpath chunk sha
  local rp="${1:-}" _ck="${2:-}" want="${3:-}" dir whole got
  [ -z "$rp" ] && return 0
  dir="$STORE/$(slug_of "$rp")"
  if ! ls "$dir"/part* >/dev/null 2>&1; then echo "bigfile: merge skip (no parts): $rp" >&2; return 0; fi
  whole="$ROOT/$rp"
  # up-to-date? skip re-merge
  if [ -f "$whole" ] && [ -n "$want" ] && [ "$(sha_of "$whole")" = "$want" ]; then
    echo "bigfile: ok (unchanged) $rp"; return 0
  fi
  mkdir -p "$(dirname "$whole")"
  cat "$dir"/part* > "$whole"
  got="$(sha_of "$whole")"
  if [ -n "$want" ] && [ "$got" != "$want" ]; then
    echo "bigfile: SHA MISMATCH $rp  want ${want:0:12}… got ${got:0:12}…" >&2; return 1
  fi
  echo "bigfile: merged $rp (sha ${got:0:12}…)"
}

status_one() { # realpath chunk sha
  local rp="${1:-}" _ck="${2:-}" want="${3:-}" whole state
  [ -z "$rp" ] && return 0
  whole="$ROOT/$rp"
  if [ -f "$whole" ]; then
    [ "$(sha_of "$whole")" = "$want" ] && state="present, sha OK" || state="present, SHA DRIFT"
  else state="absent (run: bigfile merge)"; fi
  printf '  %-70s %s\n' "$rp" "$state"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  track)
    ensure_store
    rp="${1:?usage: bigfile track <path> [chunk]}"; ck="${2:-$DEFAULT_CHUNK}"
    rp="${rp#"$ROOT"/}"; rp="${rp#./}"
    split_one "$rp" "$ck"; gitignore_add "$rp"
    echo "bigfile: tracked $rp — add .bigfiles/ + .gitignore to git" ;;
  untrack)
    ensure_store; rp="${1:?usage: bigfile untrack <path>}"; rp="${rp#./}"
    tmp="$(mktemp)"; awk -F'\t' -v p="$rp" '$1!=p' "$MANIFEST" > "$tmp"; mv "$tmp" "$MANIFEST"
    echo "bigfile: untracked $rp (parts left under .bigfiles/$(slug_of "$rp"))" ;;
  split)
    ensure_store
    if [ "$#" -gt 0 ]; then
      for p in "$@"; do p="${p#./}"; ck="$(awk -F'\t' -v x="$p" '$1==x{print $2}' "$MANIFEST")"; split_one "$p" "${ck:-$DEFAULT_CHUNK}"; done
    else while_manifest split_one; fi ;;
  merge)
    ensure_store
    if [ "$#" -gt 0 ]; then
      for p in "$@"; do p="${p#./}"; row="$(awk -F'\t' -v x="$p" '$1==x{print}' "$MANIFEST")"; ck="$(printf '%s' "$row"|cut -f2)"; sh="$(printf '%s' "$row"|cut -f3)"; merge_one "$p" "${ck:-$DEFAULT_CHUNK}" "$sh"; done
    else while_manifest merge_one; fi ;;
  status)
    ensure_store; echo "bigfile: tracked files (manifest: .bigfiles/manifest.tsv)"; while_manifest status_one ;;
  install)
    git -C "$ROOT" config core.hooksPath .githooks
    chmod +x "$ROOT"/.githooks/* 2>/dev/null || true
    echo "bigfile: git hooks active (core.hooksPath=.githooks). Reassembling…"
    ensure_store; while_manifest merge_one ;;
  *)
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed '1d'; exit 1 ;;
esac
