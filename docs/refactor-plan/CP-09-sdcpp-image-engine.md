# CP-09 — Second image engine: stable-diffusion.cpp (raw SD checkpoints, no conversion)

## What runs where
| Model format (picked by file content, not a toggle) | Engine | Native |
|---|---|---|
| QNN dir / MNN dir (zip import, HF downloads) | LocalDream (unchanged) | `LocalDreamModule` → `libstable_diffusion_core.so` |
| Core ML (iOS) | LocalDream service → `CoreMLDiffusionModule` (unchanged) | — |
| `.safetensors` / `.ckpt` single-file SD checkpoint | **stable-diffusion.cpp** | `SdCppImageModule` → `libsdcpp_jni.so` |

`libsdcpp_jni.so` = sd.cpp @ `c92d73c4` (MIT) + `android/app/src/main/cpp/sdcpp/sdcpp_jni.cpp`, CPU backend
(arm64 `armv8.2-a+dotprod+fp16`, 16 KB pages), prebuilt by `scripts/build-sdcpp.sh` and committed like the QNN core.

## Routing (the one seam)
Every invoke path — chat IMG button, auto-detect, `generate_image` tool call, failure-card retry — calls
`imageGenerationService.generateImage(GenerateImageParams)`. It resolves:
1. steps / guidance: `resolveMobileImageParameters` (unchanged, shared by both engines);
2. engine signature: `resolveEngineImageRequest(model, settings, params, squareSize)` (`imageEngineParams.ts`):
   - `sdcpp`: W×H free on a 64-px grid (64–1024, no 256 floor), `sampler`, `scheduler`, enabled LoRAs;
   - LocalDream: the existing square size; `sampler` only if `dpm` | `euler_a` (the core's set; default `dpm`).
3. `imageEngineRouter` (replaces the direct `localDreamGenerator` import in the service + activeModelService)
   sends load/generate/cancel/unload to the engine the loaded model's `backend` selected.
   `backend === 'sdcpp'` → sd.cpp; anything else → LocalDream with the same call shape as before.

## Format detection (`sdCppModelDetection.ts`)
`.safetensors` is dual-use, so the header's tensor keys decide (native reads only the header):
- `model.diffusion_model.*` → SD checkpoint; family from `cond_stage_model.transformer.*` (sd1),
  `cond_stage_model.model.*` (sd2), `conditioner.embedders.1.*`/`label_emb` (sdxl), `double_blocks` (flux), `joint_blocks` (sd3);
- `lora_unet_*` / `lora_te*_*` or `unet.`/`text_encoder.` + `lora_*` → SD LoRA (goes to `image_loras/`, listed in settings);
- `model.layers.*` / `embed_tokens` / `lm_head` → LLM → refused (text runs on GGUF);
- standalone VAE / single diffusers component / unknown → refused with the reason.
`.ckpt` = SD checkpoint by extension (pickle has no cheap key index).

## Settings surfaces
`ImageEngineSettings` (both the model-settings screen and the in-chat sheet): sd.cpp model → sampler,
scheduler, width, height, LoRA list (enable / weight −1..2 / remove); QNN/MNN model → sampler (DPM / Euler A).

## Limits
- CPU only (no Vulkan/OpenCL yet): SD1.5 512² ≈ tens of seconds to minutes per image depending on SoC/steps.
- RAM ≈ checkpoint size (f16 SD1.5 ≈ 2 GB, SDXL ≈ 7 GB) — goes through the residency memory gate.
- Flux/SD3 single files usually lack their text encoders (T5/CLIP-G) → sd.cpp load error is surfaced.
- No img2img / inpaint / previews on the sd.cpp engine yet (txt2img + LoRA).
