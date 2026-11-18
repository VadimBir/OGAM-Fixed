#!/usr/bin/env bash
# sdcpp-convert-gguf.sh — PC-side: quantize a Stable-Diffusion checkpoint (.safetensors / .ckpt)
# into a GGUF that stable-diffusion.cpp loads with no load-time conversion.
#
# Why: with weightType q8_0/q4_0 the app re-quantizes every Linear weight at EVERY load
# (per loader thread: one source tensor + one f32 copy of it as transient RAM) and the converted
# weights live in anonymous RAM. A GGUF already holding those types is memory-mapped instead
# (sd.cpp mmaps a tensor only when the file type equals the in-memory type).
# Same rules as the in-app conversion (ModelLoader::tensor_should_be_converted): Linear weights
# convert; Conv2d weights, biases, norms, embeddings and time/label embedders keep the source type.
# So an SD1.5 q4_0 GGUF is ~1.5 GB, not ~0.5 GB: 589 M of the UNet's 860 M params are convs.
#
# Usage:  scripts/sdcpp-convert-gguf.sh <in.safetensors|in.ckpt> [type] [out.gguf]
#         type: q8_0 | q5_0 | q4_0 (default q8_0) | f16
#         SDCPP_VIA_F16=1  first converts to f16 (use for fp32 checkpoints: convs/biases otherwise
#                          stay fp32 in the GGUF and get converted to f16 at every load).
# Builds the host sd-cli once from the SAME pinned commit as scripts/build-sdcpp.sh.
# The app routes .gguf files to the text-model importer today; see docs/notes/sdcpp-memory-audit.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDCPP_REPO="https://github.com/leejet/stable-diffusion.cpp"
SDCPP_COMMIT="$(sed -n 's/^SDCPP_COMMIT="\(.*\)"$/\1/p' "$ROOT/scripts/build-sdcpp.sh")"
SDCPP_SRC="${SDCPP_SRC:-${HOME}/.cache/offgrid/stable-diffusion.cpp}"
HOST_BUILD="${SDCPP_HOST_BUILD:-${HOME}/.cache/offgrid/sdcpp-host-build}"

IN="${1:?usage: sdcpp-convert-gguf.sh <in.safetensors|in.ckpt> [type] [out.gguf]}"
TYPE="${2:-q8_0}"
OUT="${3:-${IN%.*}.${TYPE}.gguf}"
case "$TYPE" in q8_0|q5_0|q4_0|f16) ;; *) echo "sdcpp-convert-gguf: unsupported type $TYPE"; exit 1 ;; esac
[ -f "$IN" ] || { echo "sdcpp-convert-gguf: input not found: $IN"; exit 1; }
[ -n "$SDCPP_COMMIT" ] || { echo "sdcpp-convert-gguf: SDCPP_COMMIT not found in build-sdcpp.sh"; exit 1; }

if [ ! -f "$SDCPP_SRC/include/stable-diffusion.h" ]; then
  echo "== clone sd.cpp @ $SDCPP_COMMIT =="
  mkdir -p "$(dirname "$SDCPP_SRC")"
  git clone --filter=blob:none "$SDCPP_REPO" "$SDCPP_SRC"
fi
if [ "$(git -C "$SDCPP_SRC" rev-parse HEAD)" != "$SDCPP_COMMIT" ]; then
  git -C "$SDCPP_SRC" fetch --depth 1 origin "$SDCPP_COMMIT"
  git -C "$SDCPP_SRC" checkout -q "$SDCPP_COMMIT"
fi
git -C "$SDCPP_SRC" submodule update --init --depth 1 ggml

CLI="$HOST_BUILD/bin/sd-cli"
if [ ! -x "$CLI" ]; then
  echo "== build host sd-cli (CPU) =="
  cmake -S "$SDCPP_SRC" -B "$HOST_BUILD" -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON \
    -DSD_WEBP=OFF -DSD_WEBM=OFF
  cmake --build "$HOST_BUILD" --target sd-cli -j"$(getconf _NPROCESSORS_ONLN)"
fi

SRC_MODEL="$IN"
if [ "${SDCPP_VIA_F16:-0}" = "1" ] && [ "$TYPE" != "f16" ]; then
  SRC_MODEL="${OUT%.gguf}.f16-tmp.gguf"
  echo "== pass 1: $IN -> $SRC_MODEL (f16) =="
  "$CLI" -M convert -m "$IN" -o "$SRC_MODEL" --type f16 -v
fi

echo "== convert: $SRC_MODEL -> $OUT ($TYPE) =="
"$CLI" -M convert -m "$SRC_MODEL" -o "$OUT" --type "$TYPE" -v
[ "$SRC_MODEL" = "$IN" ] || rm -f "$SRC_MODEL"
echo "sdcpp-convert-gguf: $OUT ($(du -h "$OUT" | cut -f1)) sd.cpp@$SDCPP_COMMIT"
