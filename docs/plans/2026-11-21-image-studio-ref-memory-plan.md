# Plan: Image Studio, reference images, region edit, memory, newer/faster image models

Date 2026-11-21. Branch `app-refactor-fixes`. State words: PROVEN (file:line or URL), UNKNOWN (not measured).
Sources: three read-only audits (sd.cpp GPU, local-dream NPU, memory gate), 2026-11-21.

## User asks (verbatim intent)
1. A button/tab next to chat that only makes images.
2. A second tool `imgref()`: prior image or uploaded image used as a reference, with a "how much to keep" scale; checkbox "prior image = reference"; maybe a reference settable from the start.
3. Edit part of an image (split into ~8 segments, the LLM picks which to edit). "Was this ControlNet?"
4. Find memory leaks: text + image model should coexist, but a memory warning appears.
5. Newer image models than SD 1.5, same size class.
6. GPU image gen is far slower than NPU. Can models be converted to the NPU format?

## Findings that drive the plan
- **Img2img is plumbed but dead.** `initImage`/`denoiseStrength`/`maskImage` exist (`imageGenerationTypes.ts:58-63`), but no UI or tool sets them (0 hits in components/screens/hooks).
- **NPU img2img bug.** The app sends `init_img` (`LocalDreamModule.kt:1106`); the bundled core reads `image` and `mask` (strings in `libstable_diffusion_core.so`; upstream c56cf6c). So every img2img silently runs as txt2img. NPU img2img itself is real: QNN VAE encoder, with `vae_encoder.bin` in the sd-qnn zips (58.9 MB).
- **Stale comments.** "img2img only on CPU" at `LocalDreamModule.kt:1100-1102` and `types/index.ts:465-466` is wrong.
- **GPU (sd.cpp c92d73c) supports img2img, masks, ControlNet, IP-Adapter, `ref_images` and newer models.** The JNI passes none of them (`sdcpp_jni.cpp:487-491`), and the JS side throws (`sdCppGenerator.ts:297-298`).
- **Q3: segment editing is inpainting with a mask, not ControlNet.** ControlNet steers structure (pose, edges, depth). sd.cpp masks work with any checkpoint (`diffusion_engine.cpp:2590`); local-dream inpaints on the NPU.
- **Q4: the two models cannot coexist in Balanced.** Estimates text 6,045 MB + image 2,894 MB = 8,939 MB, above the 7,910 MB cap. Without the stacked 10% factors and the temporary VAE buffer, the real resident total is about 7,626 MB, which fits. Then a second, stricter text check (`llmSafetyChecks.ts:149-161`, 1.2x weights + 1,224 MB) runs AFTER the image was already evicted, and that check shows the alert.
- **Q4: no multi-GB native leak found.** sd.cpp free (`sdcpp_jni.cpp:438-457`) and llama.rn release (`llm.ts:187-190`) are correct.
  - Coexistence killer: "Load Anyway" forces one-model-at-a-time for the whole session (`modelResidency/index.ts:425-435`).
  - Churn: the text model is reloaded after every image, which evicts the image again (`activeModelService/index.ts:489-498`).
  - Conditional leak: a local-dream server that is alive but not ready is not stopped (`activeModelService/index.ts:389-409`).
- **Q5 on NPU.** Upstream local-dream runs SDXL on 8 Gen 3 or newer (`xororz/sdxl-qnn`, 46 models of about 3.5 GB, DMD2 variants). Z-Image Turbo and FLUX.2 Klein 4B run on SM8750 or newer (v3 alphas). The S26 Ultra is SM8850. The app reads only SD1.5 repos (`huggingFaceModelBrowser.ts:22-25`), and its bundled core is the old SD1.5-only CLI.
- **Q5 on GPU.** FLUX.2-klein-4B is about 5.3 GB on disk (Q4_0 + Qwen3-4B Q4 + VAE), runs in 4 steps and edits with references. Z-Image-Turbo is about 6.7 GB and runs in 8 steps. Both need multi-file loading; the JNI forces `diffusion_model_path`/`llm_path` empty (`sdcpp_jni.cpp:246,249`).
- **Q6 on GPU speed.** Unused speed knobs: TAESD tiny VAE (9.8 MB, `taesd_path` never sent), LCM-LoRA (2-8 steps, `lcm` sampler already listed), `diffusion_conv_direct`, step cache (`gp.cache`). Their speedup on Adreno is UNKNOWN until measured.
- **Q6 conversion.** NPU conversion is host-only: Linux/WSL, QNN SDK 2.28, about 20 GB RAM + swap, 30 GB disk, hours per resolution (ld-guide.chino.icu/conversion/sd15). Not possible on the phone. NPU LoRA is bake-in only.

## Hard limits (cannot change in app code)
- NPU SD1.5: 512 square, plus resolution patches only if present. The sd-qnn zips ship 512 only.
- NPU LoRA: merged before conversion only.
- NPU conversion of a Civitai checkpoint: on a PC only.
- 12 GB RAM: Klein-4B or Z-Image plus an 8B text model will not coexist. The expected outcome is one at a time (UNKNOWN until measured).

## Phases (order = recommended)

### P0 Bugs (small, proven)
- P0.1 `LocalDreamModule.kt:1106`: change `init_img` to `image`. Send the mask as the format the core decodes; the mask encoding is UNKNOWN, so verify it on device. Fix the 2 stale comments.
- P0.2 Memory gate:
  - charge already-resident models without the 10% factor and without temporary compute; apply 10% once, to the incoming peak only;
  - make the text check use the residency estimate (`memory.ts:107`), and run it BEFORE any eviction;
  - do not cap free RAM at the 70% budget (`memoryBudget.ts:113`);
  - mmap'd CPU weights: charge KV + compute against free RAM, and weights against the physical cap only;
  - "Load Anyway" applies to that load only, never session-wide;
  - skip the post-image text reload when the sum exceeds the cap;
  - stop the local-dream process whenever it is alive;
  - release the llama context before nulling it (`llm.ts:205`);
  - image unload waits for memory to return.
- P0.3 Delete the dead check code (`memory.ts:208-369`) and rename `footprintBytes` (`DeviceMemoryModule.kt:55`).

### P1 Image Studio tab (image only, no LLM)
- New bottom tab "Image" beside Chat. It holds:
  - prompt, negative prompt, size/steps/seed (reusing the shared settings components), engine/model picker, gallery;
  - a Reference slot: none / upload / pick from gallery / "use last result";
  - a "Keep" slider, 0 to 100% (denoise = 1 - keep);
  - a "Chain: each result becomes the next reference" checkbox.
- Calls `imageGenerationService` directly. No chat history, and the text model is not loaded (frees RAM).

### P2 Chat reference images: tool `imgref`
- New tool `imgref(prompt, source: "last" | "uploaded", keep 0..1)`. It is routed to img2img on the active image model.
- Settings (Image Generation):
  - "Uploaded image is the reference" (default on);
  - "Previous image is the reference" checkbox;
  - default Keep slider.
- Composer: an uploaded image becomes the reference for the next image request.

### P3 GPU img2img + mask (sd.cpp)
- JNI: pass RGB bytes into `gp.init_image` + `gp.strength`, and `gp.mask_image`. Remove the throws at `sdCppGenerator.ts:297-298`.
- Speed:
  - TAESD toggle (download 9.8 MB `taesd`), LCM-LoRA preset, `diffusion_conv_direct` toggle;
  - measure each on device with the `[SDCPP]` timings.

### P4 Region edit (your 8 segments)
- Grid mask (3x3 or 2x4) drawn over the image. Either you tap cells, or the LLM picks the cell numbers through `imgref(..., regions:[...])`.
- The mask goes to inpaint on either engine. ControlNet is NOT needed for this; it stays deferred.

### P5 Newer models
- P5a NPU (fastest path): upgrade the bundled local-dream core to the current upstream CLI. That adds:
  - SDXL (`xororz/sdxl-qnn`, `_8gen3` tier, DMD2 = few steps; 3.5 GB each);
  - and later Z-Image Turbo / Klein through its Hexagon path.
  - Needs a new core binary build and the CLI change at `LocalDreamModule.kt:257-266`. It is large.
- P5b GPU: multi-file sd.cpp loading (diffusion GGUF + LLM text encoder + VAE). First model: FLUX.2-klein-4B (4 steps, `ref_images` editing = the best `imgref` quality). Detection learns `.gguf` plus the new families.

### P6 Conversion
- Not doable on the phone. Ship a doc and a script wrapper for PC (WSL) conversion of a Civitai SD1.5 checkpoint into an sd-qnn zip, then import that zip in the app.

## Decisions needed from the user
- D1 Studio placement: bottom tab (recommended) or a button in the chat header.
- D2 Order: P0, then P1 + NPU img2img, then P2, then P3, then P4, then P5 (recommended); or another order.
- D3 First newer model: SDXL on the NPU (P5a) or Klein-4B on the GPU (P5b).
- D4 Default Keep value for `imgref`: 40% (denoise 0.6, the current default) or another value.

## Device tests needed (phone link required)
- NPU img2img after P0.1, mask format, both models coexisting after P0.2, GPU speed knobs (P3), Klein/SDXL RAM.
