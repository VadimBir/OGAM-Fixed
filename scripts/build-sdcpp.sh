#!/usr/bin/env bash
# build-sdcpp.sh — build libsdcpp_jni.so (stable-diffusion.cpp + JNI bridge) for arm64-v8a and
# drop it into jniLibs as a prebuilt, like libstable_diffusion_core.so.
#
# Usage:  scripts/build-sdcpp.sh            (clones the pinned sd.cpp commit into $SDCPP_SRC)
#         SDCPP_SRC=/path/to/checkout scripts/build-sdcpp.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDCPP_REPO="https://github.com/leejet/stable-diffusion.cpp"
SDCPP_COMMIT="c92d73c408515c94beef32161bb5960764fde7a0"
SDCPP_SRC="${SDCPP_SRC:-${HOME}/.cache/offgrid/stable-diffusion.cpp}"
SDK="${ANDROID_HOME:-/opt/android-sdk}"
NDK_VER="$(sed -n 's/.*ndkVersion *= *"\(.*\)".*/\1/p' "$ROOT/android/build.gradle" | head -1)"
NDK="${ANDROID_NDK_HOME:-$SDK/ndk/$NDK_VER}"
BUILD="${SDCPP_BUILD:-${HOME}/.cache/offgrid/sdcpp-build}"
OUT="$ROOT/android/app/src/main/jniLibs/arm64-v8a/libsdcpp_jni.so"

[ -f "$NDK/build/cmake/android.toolchain.cmake" ] || { echo "build-sdcpp: NDK not found at $NDK"; exit 1; }

if [ ! -f "$SDCPP_SRC/include/stable-diffusion.h" ]; then
  echo "== clone sd.cpp @ $SDCPP_COMMIT =="
  mkdir -p "$(dirname "$SDCPP_SRC")"
  git clone --filter=blob:none "$SDCPP_REPO" "$SDCPP_SRC"
fi
git -C "$SDCPP_SRC" fetch --depth 1 origin "$SDCPP_COMMIT" 2>/dev/null || true
git -C "$SDCPP_SRC" checkout -q "$SDCPP_COMMIT"
git -C "$SDCPP_SRC" submodule update --init --depth 1 ggml

echo "== configure (NDK $NDK_VER, arm64-v8a, android-24) =="
cmake -S "$ROOT/android/app/src/main/cpp/sdcpp" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release -DSDCPP_SRC="$SDCPP_SRC"

echo "== build =="
cmake --build "$BUILD" --target sdcpp_jni -j"$(nproc)"

"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded \
  -o "$OUT" "$BUILD/libsdcpp_jni.so"
echo "build-sdcpp: $OUT ($(du -h "$OUT" | cut -f1)) sd.cpp@$SDCPP_COMMIT"
