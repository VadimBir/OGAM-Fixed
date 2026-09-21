# CP-09 — Safetensors image backend: research + recommendation (R2)

User decision: DON'T re-attempt safetensors→MNN conversion (failed repeatedly). Instead find
an existing open-source project that runs safetensors on-device and adapt/inherit its code.

## Key context
This app ALREADY embeds `xororz/local-dream` (Kotlin `LocalDreamModule` +
`libstable_diffusion_core.so`), but local-dream runs MNN (CPU/OpenCL) + QNN (NPU) model
formats — NOT raw `.safetensors`. So safetensors needs a SECOND image backend.

## Candidate projects (all run .safetensors on-device)
1. **stable-diffusion.cpp (leejet)** — MIT. Pure C/C++; loads ckpt/safetensors/diffusers
   directly; auto-quantizes (q8_0…q2_k); builds for Android with OpenCL/Vulkan via NDK.
   → THE backend to embed. https://github.com/leejet/stable-diffusion.cpp
2. **ShiftHackZ/Stable-Diffusion-KMP** — Kotlin Multiplatform Android/iOS; on-device
   SDXL via stable-diffusion.cpp; safetensors/ckpt; CPU/OpenCL/Vulkan; custom local paths.
   → BEST reference to "cut out and import": same language (Kotlin) as this app's native
   layer, shows the JNI/build wiring for sd.cpp. https://github.com/ShiftHackZ/Stable-Diffusion-KMP
3. **rmatif/Local-Diffusion** — Flutter; sd.cpp backend; SD1.x/2/3, SDXL, FLUX.1;
   safetensors/ckpt; auto-quantize. Good reference for model handling/UX, but Flutter (not
   RN) so less directly copyable. https://github.com/rmatif/Local-Diffusion
4. EdVince/Stable-Diffusion-NCNN — C++ ncnn; txt2img/img2img (SD1.5). Alternative engine.

## Recommended approach
Add a `SafetensorsDiffusionModule` (Kotlin) that JNIs into a `stable-diffusion.cpp` `.so`
built for arm64 (OpenCL/Vulkan). Model residency: user picks a local `.safetensors` (reuse
`importLocalModel` document-picker flow, extend to accept `.safetensors`). Wire it as a
third image backend behind the existing `localDreamGeneratorService` selection so chat/tool
image-gen is engine-agnostic. Decode path already fixed (CP-01) — sd.cpp returns raw RGB
which the raw path handles; keep the format-detecting decoder.

## MUST verify before importing code
- LICENSES: sd.cpp MIT (OK to embed with attribution). Check Stable-Diffusion-KMP license
  before copying its Kotlin/JNI code. local-dream is Apache-2.0.
- Build size: sd.cpp `.so` + OpenCL adds APK weight; gate behind download/opt-in.
- S26 target: build arm64-v8a with OpenCL; Adreno/Vulkan path for the Snapdragon in S26.

## Scope
Native + build-system work (CMake/NDK, JNI, model loader, backend selection UI). Multi-day.
Sequence after load-stability (R1) and CP-01 verification. Needs a design spike commit first.
