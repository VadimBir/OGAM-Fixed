#!/usr/bin/env bash
# build-sdcpp.sh — build libsdcpp_jni.so (stable-diffusion.cpp + JNI bridge, Vulkan GPU backend) for
# arm64-v8a and drop it into jniLibs as a prebuilt, like libstable_diffusion_core.so.
#
# Works from Git Bash on Windows (host tag windows-x86_64) and from Linux (linux-x86_64).
#
# Usage:  scripts/build-sdcpp.sh
#         SDCPP_SRC=/path/to/checkout SDCPP_VULKAN_HEADERS_SRC=/path/to/Vulkan-Headers scripts/build-sdcpp.sh
#         SDCPP_VULKAN=OFF scripts/build-sdcpp.sh        (CPU-only legacy build; not the app policy)
#
# Host requirements (Vulkan build only): the ggml shader generator "vulkan-shaders-gen" is a HOST tool,
# built with a host C/C++ compiler found on PATH (gcc/clang; on Windows e.g. w64devkit or llvm-mingw).
#   SDCPP_HOST_COMPILER_BIN=<dir with gcc/g++ or clang/clang++>   is prepended to PATH for this script.
# glslc, the SPIR-V headers and libvulkan.so come from the NDK; vulkan.hpp from KhronosGroup/Vulkan-Headers
# (cloned at the pinned commit below). cmake/ninja: PATH, else $ANDROID_HOME/cmake/3.22.1/bin.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDCPP_REPO="https://github.com/leejet/stable-diffusion.cpp"
SDCPP_COMMIT="c92d73c408515c94beef32161bb5960764fde7a0"
VKH_REPO="https://github.com/KhronosGroup/Vulkan-Headers"
VKH_COMMIT="b26454f3ccd67e704117a39295fa13935fd3804c"
SDCPP_VULKAN="${SDCPP_VULKAN:-ON}"
API="${SDCPP_ANDROID_API:-26}"
SDCPP_SRC="${SDCPP_SRC:-${HOME}/.cache/offgrid/stable-diffusion.cpp}"
VKH_SRC="${SDCPP_VULKAN_HEADERS_SRC:-${HOME}/.cache/offgrid/Vulkan-Headers}"
SDK="${ANDROID_HOME:-/opt/android-sdk}"
NDK_VER="$(sed -n 's/.*ndkVersion *= *"\(.*\)".*/\1/p' "$ROOT/android/build.gradle" | head -1)"
NDK="${ANDROID_NDK_HOME:-$SDK/ndk/$NDK_VER}"
BUILD="${SDCPP_BUILD:-${HOME}/.cache/offgrid/sdcpp-build}"
OUT="$ROOT/android/app/src/main/jniLibs/arm64-v8a/libsdcpp_jni.so"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) HOST_TAG=windows-x86_64; EXE=.exe ;;
  Darwin) HOST_TAG=darwin-x86_64; EXE= ;;
  *) HOST_TAG=linux-x86_64; EXE= ;;
esac
# CMake on Windows needs mixed (C:/...) paths; elsewhere paths pass through unchanged.
winpath() { if command -v cygpath >/dev/null; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

[ -f "$NDK/build/cmake/android.toolchain.cmake" ] || { echo "build-sdcpp: NDK not found at $NDK"; exit 1; }
[ -z "${SDCPP_HOST_COMPILER_BIN:-}" ] || PATH="$SDCPP_HOST_COMPILER_BIN:$PATH"
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
  CMAKE_DIR="$(ls -d "$SDK"/cmake/3.* 2>/dev/null | sort -V | tail -1)"
  [ -n "$CMAKE_DIR" ] || { echo "build-sdcpp: cmake/ninja not on PATH and none under $SDK/cmake"; exit 1; }
  PATH="$CMAKE_DIR/bin:$PATH"
fi

if [ ! -f "$SDCPP_SRC/include/stable-diffusion.h" ]; then
  echo "== clone sd.cpp @ $SDCPP_COMMIT =="
  mkdir -p "$(dirname "$SDCPP_SRC")"
  git clone --filter=blob:none "$SDCPP_REPO" "$SDCPP_SRC"
fi
git -C "$SDCPP_SRC" checkout -q "$SDCPP_COMMIT"
git -C "$SDCPP_SRC" submodule update --init --depth 1 ggml

CMAKE_VK_ARGS=()
if [ "$SDCPP_VULKAN" = "ON" ]; then
  if [ ! -f "$VKH_SRC/include/vulkan/vulkan.hpp" ]; then
    echo "== clone Vulkan-Headers @ $VKH_COMMIT =="
    mkdir -p "$(dirname "$VKH_SRC")"
    git clone --filter=blob:none "$VKH_REPO" "$VKH_SRC"
  fi
  git -C "$VKH_SRC" checkout -q "$VKH_COMMIT"
  command -v g++ >/dev/null || command -v clang++ >/dev/null || {
    echo "build-sdcpp: no host C++ compiler on PATH for vulkan-shaders-gen; set SDCPP_HOST_COMPILER_BIN"; exit 1; }
  SYSROOT_LIB="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/sysroot/usr/lib/aarch64-linux-android/$API"
  CMAKE_VK_ARGS=(
    -DSDCPP_VULKAN=ON
    "-DSDCPP_VULKAN_HEADERS=$(winpath "$VKH_SRC")"
    "-DSDCPP_SPIRV_HEADERS_INCLUDE=$(winpath "$NDK/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include")"
    "-DSDCPP_GLSLC=$(winpath "$NDK/shader-tools/$HOST_TAG/glslc$EXE")"
    "-DSDCPP_LIBVULKAN=$(winpath "$SYSROOT_LIB/libvulkan.so")"
  )
else
  CMAKE_VK_ARGS=(-DSDCPP_VULKAN=OFF)
fi

echo "== configure (NDK $NDK_VER, arm64-v8a, android-$API, vulkan=$SDCPP_VULKAN) =="
cmake -S "$(winpath "$ROOT/android/app/src/main/cpp/sdcpp")" -B "$(winpath "$BUILD")" -G Ninja \
  "-DCMAKE_TOOLCHAIN_FILE=$(winpath "$NDK/build/cmake/android.toolchain.cmake")" \
  -DANDROID_ABI=arm64-v8a "-DANDROID_PLATFORM=android-$API" -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release "-DSDCPP_SRC=$(winpath "$SDCPP_SRC")" "${CMAKE_VK_ARGS[@]}"

echo "== build =="
cmake --build "$(winpath "$BUILD")" --target sdcpp_jni -j"$(nproc)"

"$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/llvm-strip$EXE" --strip-unneeded \
  -o "$OUT" "$BUILD/libsdcpp_jni.so"
echo "build-sdcpp: $OUT ($(du -h "$OUT" | cut -f1)) sd.cpp@$SDCPP_COMMIT vulkan=$SDCPP_VULKAN"
