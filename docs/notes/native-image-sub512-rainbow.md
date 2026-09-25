# Native image engine (local-dream) — sub-512 "rainbow" output

Status: **OPEN** for CPU/GPU (MNN) models. NPU (QNN) path guarded since apk/v21.
Owner files: `android/app/src/main/java/ai/offgridmobile/localdream/LocalDreamModule.kt`,
`src/services/localDreamGenerator.ts`, `android/app/src/main/jniLibs/arm64-v8a/libstable_diffusion_core.so`.

## Symptom (device, user report)
Every requested size < 512 (256/320/384/448) produces a rainbow/diagonal-noise image on the
native engine, on CPU and on GPU (OpenCL) runtimes. 512x512 is clean. Device has no NPU.

## What the bundled binary is
`libstable_diffusion_core.so` = upstream xororz/local-dream `app/src/main/cpp/src/main.cpp`
between commits `c56cf6c` (2025-12-17, diffusion preview) and `0ac3607` (2026-04-18, SDXL):
- has: `show_diffusion_process`, `width`/`height` request keys (`4d400be` "more shapes"),
  `/upscale`, QNN runtime 2.39.
- lacks: `sdxl`, `lowram`, `karras`, `listen_all`, batch generation.
`main.cpp` is unchanged across that range, so `c56cf6c` is the reference source.

## Verified from that source
- `/generate` reads `width`/`height` (or `size` for both); latent = W/8 x H/8.
- Final image and previews are sent as base64 **raw RGB, HWC** (not JPEG). Our decoder
  (`saveServerImageToPng`) takes the `w*h*3` raw path — layout matches.
- **MNN (CPU/OpenCL)**: UNet and VAE sessions are created per request and `resizeTensor`'d to
  `{B,4,H/8,W/8}` → dynamic shapes. Upstream's own UI offers 128..512 (step 64, default **256**)
  for CPU models, using the same `xororz/sd-mnn` models this app downloads.
  ⇒ On paper sub-512 MNN should work; the rainbow is NOT explained by the engine source.
- **QNN (NPU)**: `unet.bin` / `vae_decoder.bin` are context binaries compiled for 512x512. No
  shape check exists: at W<512 `vae_dec_out_pixels` is allocated W*H*3 but the 512 graph writes
  512*512*3 (overflow), then the buffer is read with the wrong row stride → diagonal rainbow.
  Other sizes need a resolution patch (`--patch <W>.patch`), which this binary names only for 768
  / 1024. App fix (v21): NPU requests without a matching patch render at 512 (`[IMG-SIZE]` log).

## Open hypotheses for the MNN rainbow (not verified)
1. Something between JS size and server differs from upstream (e.g. a request key, an old OpenCL
   tuning cache `unet_cache.mnnc.<W>` / `vae_dec_cache.mnnc.<W>` from another shape — "Clear OpenCL
   cache" in the app, then retry on CPU).
2. The on-disk model is not the upstream `sd-mnn` export (static shapes baked in Reshape ops),
   e.g. an imported/converted model.
3. The binary is a fork/rebuild of `c56cf6c` with a different MNN build.

## Triage (one sub-512 run on device)
`adb logcat -s LocalDreamModule:D` and capture:
- `[IMG-SIZE] …` (size the app actually sent),
- `COMMAND: …` (server flags), server stdout line `… Size:… Img2Img:…`,
- `[IMG-DECODE] size=… wh=…x… wh3=… first8=…` (payload vs expected raw size).
Also: same model + 256x256 in the upstream Local Dream app. Clean there ⇒ app-side bug here;
rainbow there too ⇒ model/export.

## Rebuilding the native engine (not done — reasons)
Source is open (xororz/local-dream `app/src/main/cpp`). Needs: Android NDK (have), CMake, Rust
1.84 + aarch64-linux-android target (tokenizers-cpp), MNN built for Android, Qualcomm QNN SDK 2.39
(multi-GB). Does not fit the ~3.7 GB free per-session disk. A rebuild cannot make NPU models
render < 512 (the model files are fixed-shape); it only helps if hypothesis 3 holds.

## App-side option (not implemented)
Render at 512 and downscale the bitmap to the requested size in `buildFinalResult`: correct size,
no rainbow, no memory saving.
