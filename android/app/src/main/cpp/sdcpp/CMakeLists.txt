# libsdcpp_jni.so — JNI bridge over stable-diffusion.cpp (leejet, MIT) for SdCppImageModule.
# Runs raw Stable-Diffusion checkpoints (.safetensors / .ckpt / .gguf) directly, no conversion.
# Built out-of-gradle by scripts/build-sdcpp.sh (pinned sd.cpp commit) and committed as a
# prebuilt jniLib, same as libstable_diffusion_core.so — gradle builds stay unchanged.
cmake_minimum_required(VERSION 3.18)
project(sdcpp_jni C CXX)

set(SDCPP_SRC "" CACHE PATH "stable-diffusion.cpp source checkout")
if(NOT EXISTS "${SDCPP_SRC}/include/stable-diffusion.h")
  message(FATAL_ERROR "SDCPP_SRC must point at a stable-diffusion.cpp checkout (got '${SDCPP_SRC}')")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CPU backend only: portable across arm64 SoCs. No examples, no WebP/WebM I/O (PNG is written in Kotlin).
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory("${SDCPP_SRC}" sdcpp EXCLUDE_FROM_ALL)

add_library(sdcpp_jni SHARED sdcpp_jni.cpp)
target_include_directories(sdcpp_jni PRIVATE "${SDCPP_SRC}/include")
target_link_libraries(sdcpp_jni PRIVATE stable-diffusion android log)
target_compile_options(sdcpp_jni PRIVATE -fvisibility=hidden)
# 16 KB page alignment (Android 15+ devices) + drop unused code.
target_link_options(sdcpp_jni PRIVATE "-Wl,-z,max-page-size=16384" "-Wl,--gc-sections" "-Wl,--exclude-libs,ALL")
