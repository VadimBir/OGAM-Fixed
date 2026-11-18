# Changelog

Branch `app-refactor-fixes`. Newest first. Each entry gives the commit it ships in.
State words: DONE = finished and unit tested; WIP = an agent is still editing, so the code is not checked; UNKNOWN = not verified.

## 2026-11-18: full gate run (all agents merged)
- **jest:** 571 of 576 suites pass, 3 skipped. 6766 tests pass, 29 skipped.
  - Failing: the orphaned `imageGenerator.test.ts` (33 tests), waiting for the owner's OK to delete it.
  - `batch4-image-provider-remove` was fixed after this run: delete unloads the model only when it is the active or the loaded one. 2 new tests.
- **tsc:** 14 errors in `src/`, all in `src/services/sync/*`. They come from the rebuilt `@offgrid/sync` types; that folder is unchanged since 2cb50bff.
- **eslint:** 0 errors (11 warnings).
- **gradle `:app:testDebugUnitTest`:** BUILD SUCCESSFUL.
- **Release APK:** build started.

## 2026-11-18: agents Q, R, A3 reports + merge fixes

### DONE: sd.cpp on the GPU with Vulkan (agent A3)
- **New `libsdcpp_jni.so`:**
  - built with Vulkan, 57.9 MB (the old CPU-only file was 34.8 MB);
  - same pinned commit `c92d73c`;
  - exports all 6 old JNI functions plus `nativeLoadEx`.
- **No CPU fallback, per the "only GPU or NPU" rule:**
  - a Vulkan probe runs before ggml starts;
  - on a phone without a Vulkan GPU, loading fails with `ERR_NO_GPU` (`SdCppLoadError.kt`) instead of crashing;
  - needs Android 9 or later; on older phones the library does not load.
- **Build:** `scripts/build-sdcpp.sh` was rewritten for Windows, NDK 27 and Vulkan. The old CPU-only library is backed up in the scratchpad.
- **Tests:** 10 of 10 gradle tests pass (`SdCppLoadErrorTest`, `SdCppRunGateTest`). `CivitaiSection` jest passes 10 of 10.
- **UNKNOWN:**
  - the Adreno Vulkan driver, speed, and whether Vulkan flash attention and q4_0 actually run on the GPU;
  - the NDK's glslc is old, so newer shader variants were probably not built.

### DONE: RAM audit, unload during image generation, "selected, not loaded" (agent R)
- **Audit:**
  - A model loaded with mmap off was counted as clean (pageable) memory; now it is counted as dirty.
  - Whisper kept a phantom memory entry after delete. Fixed.
  - Unloading a text model asked the engine of the *selected* model, not the loaded one, so the loaded model could stay in RAM with nothing tracking it. Fixed with `loadedTextEngines()`.
  - A load that hit its timeout could still finish natively and leave a model in RAM with no owner. Fixed with `raceNativeLoad`.
  - Cleared, not bugs: base64 images in stores, the context clamp, and the sd.cpp free path.
- **Unload the LLM while an image generates:**
  - before generating, the app checks whether the image model fits next to the loaded text model (`canKeepOthersDuringGeneration`);
  - if not, it unloads the text model and keeps its selection;
  - after the image, it reloads the text model, except in conservative mode.
- **Image model after Unload:** stays selected and shows "Selected, not loaded"; tapping it loads it again.
- **P's follow-ups:** `setLoadedSettings` records nUbatch, mmapMode and cacheTypeV. The memory estimate passes cacheTypeV.
- **Tests:** new `m4Unload` (17), `generationGate` (8), `ImageTabSelectedNotLoaded` (3).
- **UNKNOWN:** all RAM numbers on the phone; whether Android's `availMem` counts cached file pages.

### DONE: test suite repair (agent Q)
- **Result:** 81 of 83 failing jest suites fixed, and gradle `testDebugUnitTest` is green (it had 3 failures before).
- **Why suites failed:**
  - 57 need the private `pro/` module, which is not in this repo; they are now listed in `jest.config.js`;
  - 5 need the mesh-sync source;
  - the rebuilt shared packages did not match what the app expects (`SHARED_RECONSTRUCTION_FIXES.md`);
  - some tests were stale after intended changes;
  - some failed only because this machine uses the ru-RU locale.
- **Real bugs fixed:**
  - `mmproj.ts`: a model's name included its repo folder, so the vision projector never matched;
  - Windows path handling in `LocalDreamModule.kt` and its tests.

### Merge fixes (main session)
- **Flash attention:**
  - `advancedSamplingParams.test.ts` now expects 'auto' for an f16 V cache on Android;
  - the Flash Attention toggle text was updated;
  - the shadowed `isAndroid` was renamed.
- **eslint:**
  - inline styles moved to `textGenAdvancedStyles.ts` (`multilineInput`, `multilineInputShort`);
  - `liveAvailableMB <= 0` in `llmHelpers.ts`;
  - unused helper in `generationService.queueNotReady.test.ts`;
  - unused `INFERENCE_BACKENDS` import in `loaders.ts` (done by R).
- **`imageProvider.ts`:** deleting an image model unloads the loaded one only if it is the active model. Before, deleting any image model dropped your selection.
- **`imageDownloadActions.ts`:** the zip and multi-file download paths now claim the finalization too, so one download no longer shows two alerts there either.
- **Open:** `__tests__/unit/services/imageGenerator.test.ts` tests a module that no longer exists (last seen in commit `bd6934bb`). Deleting it needs the owner's permission.

## 2026-11-18: agent A2 report (GPU quants and flash attention, DONE, not committed yet)
- **Proof base:**
  - llama.cpp b11010 @ `4bc272fd`, vendored inside llama.rn;
  - the shipped OpenCL `.so` was checked to match that commit.
- **Weight types that run on the GPU (OpenCL):**
  - F16, BF16, F32;
  - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0;
  - Q4_K, Q5_K, Q6_K;
  - IQ4_NL, MXFP4, Q1_0;
  - source: `ggml-opencl.cpp:8795-8852`.
- **Weight types that run on the NPU (HTP):** F16, F32, Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4, Q4_K and Q6_K. Source: `ggml-hexagon.cpp:4945-5011`.
- **CPU only:** Q2_K, Q3_K, IQ1, IQ2, IQ3, IQ4_XS and TQ.
- **Labels in the app:**
  - the old rule in `acceleration.ts` said only Q4_0 and Q8_0 run on the GPU, which was wrong;
  - each file now shows GPU+NPU, GPU, or CPU fallback;
  - downloading a quant with a CPU fallback asks for confirmation first;
  - new "GPU" and "Q4_0" filters in model search.
- **Flash attention on OpenCL:**
  - runs on the GPU for K and V both f16, both q8_0 or both q4_0;
  - mixed K and V types are slow;
  - Adreno 730/740/750 run it on the CPU.
- **Flash-attention diff applied in `llmHelpers.ts`:**
  - on Android an f16 V cache now sends `flash_attn_type` 'auto' instead of 'on', so llama.cpp keeps flash attention on wherever the GPU or NPU runs it (`llama-context.cpp:555-558`);
  - a quantized V cache still sends 'on';
  - the new `flashAttnWarningFor` function feeds `ModelLoadParams.flashAttnWarning`, which the model load logs (`llm.ts`).
- **Stale comments fixed:**
  - `useChatGenerationActions.ts`: "K-quants fall back to the CPU" was wrong;
  - `textGenAdvancedSections.tsx`: Mali is not supported;
  - `llmHelpers.ts`: HTP flash attention with a q4_0 KV cache runs on the CPU;
  - `useTextGenerationAdvanced.ts`.
- **Tests:**
  - `acceleration` 37 of 37, `ModelsScreen/utils` 71 of 71;
  - `llmHelpers.branches` 40 of 40, including new `flashAttnWarningFor` cases;
  - the 3 Android tests that expected 'on' for f16 now expect 'auto'.
- **UNKNOWN:**
  - the exact Adreno model in the S26 Ultra (on a Snapdragon, it is not 730/740/750 if it is SM8850; not checked on the device);
  - the layer mix of unsloth UD-* and *_XL files.

## 2026-11-18: agent P report (settings parameters and memory, DONE, not committed yet)
- **Sampling:** `samplingParams.ts` holds the defaults for:
  - top_k, min_p, typical_p;
  - penalty_last_n, presence and frequency penalty;
  - DRY, XTC, mirostat;
  - seed and stop strings.
  - Defaults follow llama.cpp `common.h:225-252`.
- **Every sampling field is sent on every call.** llama.rn reuses the previous call's value for any field left out (`JSIParams.cpp:438-458`).
- **Stop list bug fixed:** the shared `STOP_TOKENS` array grew on every completion. Each call now gets a new array.
- **Remote:**
  - your own server gets every field;
  - third-party hosts get OpenAI fields only, with at most 4 stop strings;
  - Ollama gets its own field names.
- **LiteRT:** new `liteRTTopK` setting.
- **Loading:**
  - n_ubatch, mmap mode (auto/on/off), and separate K and V cache types;
  - if loading fails with K≠V, it retries once with V=K (needed for MLA models).
- **Memory:** the KV-cache estimate sizes K and V separately. A RAM readout in the Advanced section refreshes every 3 s.
- **Image defaults:** a default negative prompt, a fixed seed, and clip skip (sd.cpp and remote only; LocalDream has no input for it).
- **Migration:** `migrateAdvancedSettings`.
- **Tests:** `advancedSamplingParams` (31), `appStoreMigrations.advanced` (3), `imageDefaults` (7). In `ModelSettingsScreen`, 97 of 97 pass.
- **Still open, handed to agent R:**
  - `loaders.ts`: `setLoadedSettings` must record nUbatch, mmapMode and cacheTypeV;
  - `memory.ts:104` must pass cacheTypeV;
  - `imageGenerationOutcome.test.ts` fails on R's new line at `imageGenerationService.ts:683`.
- **UNKNOWN:** whether llama-server's `/v1/chat/completions` and LM Studio accept the llama.cpp-only fields.

## 2026-11-18: work-in-progress snapshot 2 (47eb56c8, 13 files)

All of this is WIP: the agents were still editing, and nothing was tested. Agents are matched to files by who owns each file; they have not reported these edits yet.
- **Agent R (memory, unload the LLM during image generation):** `activeModelService/index.ts`, `activeModelService/loaders.ts`, `imageGenerationService.ts`.
- **Agent A2 (GPU quant labels):** `ModelCardContent.tsx`, `TextModelsTab.tsx`, `useTextModels.ts`, and `acceleration.test.ts`.
- **Agent A3 (sd.cpp Vulkan):** `scripts/build-sdcpp.sh`.
- **Agents P and Q (tests):**
  - `ModelSettingsScreen.test.tsx`;
  - `ToolsScreen.test.tsx`;
  - `ModelsScreen/utils.test.ts`;
  - `advancedSamplingParams.test.ts`;
  - `imageDefaults.test.ts`.

## 2026-11-18: work-in-progress snapshot (250130eb, all changes on disk)

This snapshot was taken while 5 agents were still running. Their edits so far are included unfinished. No full jest, tsc or gradle run has been done on the snapshot, and no APK was built from it.

### DONE: tool-call instructions (agent G)
- `tools/registry.ts`: each tool has a one-line "when to use" note and example arguments. `generate_image` takes prompt, width, height, steps, guidance, seed and negative_prompt.
- `TEXT_TOOL_CALL_FORMAT` gives the text-based tool-call format to models that have no native tool calls.
- `generationToolLoop.ts`, tool-call parser: accepts calls wrapped in code fences, Python-style dicts, kwargs, a stray trailing `}`, and `args`/`input` keys. Calls without tags are accepted only for tools offered in that turn.
- `generationToolLoop.ts`, recovery: empty replies are retried, and the repair hint names the wrong argument path. If the prompt is missing, the user's own message is used as the prompt.
- `imageTool.ts`: image sizes snap to multiples of 64, prompt aliases are accepted, and the tool returns a hint for the model's next step.
- `llmMessages.ts`: tool results are wrapped in `<tool_response>`, and the results of one round are merged into one message.
- Tests: `__tests__/unit/services/tools/toolInstructions.test.ts`.

### DONE: tool and chat bugs (agent T)
- The local llama model can now use tools. When a tool is not offered, the log says why.
- Tool calls replace the earlier list instead of being appended to it. Raw tool-call markup is hidden in chat.
- Thinking: the token budget is set automatically. Max tokens now defaults to 2048, applied by a migration.
- Quant sizes in bits per weight: Q4_0 4.5, Q6_K 6.5625, Q8_0 8.5.
- Generated images stay attached to the answer that made them.
- For remote models, messages are sent in the right order.

### DONE: image model search and Civitai (agent S)
- Search fixes:
  - typing a search now skips the ★ filter and the automatic NPU backend filter;
  - a downloaded match now shows "already downloaded";
  - an older request can no longer overwrite a newer result list.
- Download fixes:
  - one download no longer runs two finalizers or shows two alerts (`imageFinalizationClaims.ts`);
  - the same fix for the zip and multi-file download paths is NOT applied yet.
- Source fixes:
  - Hugging Face listings follow the `Link: rel=next` header, so all pages are read;
  - duplicate NPU (QNN) builds are removed;
  - when every CoreML repo fails, the error is shown instead of being cached.
- Civitai (`civitaiBrowser.ts`, `civitaiDownload.ts`, `civitaiApiKey.ts`, `CivitaiSection.tsx`):
  - search: safetensors only, for SD1.x, SD2.x and SDXL; NSFW is off by default; the API key is stored in Keychain;
  - download to the phone: checks free space, size and SHA-256, then registers the file as an sd.cpp model or LoRA.
- Native download (`WorkerDownload.kt`): `civitai.com` added to the allowed hosts, and the query string is no longer logged. This code is UNKNOWN: not compiled yet.
- Download on the server: BLOCKED, because the server has no endpoint for it. The spec is: `POST /v1/models/download` returns a `jobId`; `GET` and `DELETE /v1/models/download/{jobId}` report progress and cancel.

### WIP: settings parameters and memory (agent P)
- Missing settings being added: top_k, min_p, typical_p, penalties, DRY, XTC, mirostat, seed and stop strings. Also n_ubatch, separate K and V cache types, an mmap setting and image defaults.

### WIP: other agents
- **Agent Q:** fixes the failing jest suites.
- **Agent A3:** builds sd.cpp with Vulkan for the GPU. Build files are in `android/app/src/main/cpp/sdcpp/`.
- **Agent R:**
  - RAM audit;
  - unload the LLM while an image is generated;
  - keep the image model shown as "selected, not loaded".
- **Agent A2:** checks which quant types run on the GPU and whether flash attention works on OpenCL.

### Known open items
- Device-only checks: Vulkan on Adreno, RAM numbers, and whether the phone's chip is Snapdragon or Exynos.
- Not doable from this repo: mesh sync (no source code).

## 2cb50bff: vendor
- Rebuilt the `@offgrid` shared packages (83 KB) and added a setup note.

## 7c088a22: fixes
- Bugs fixed in memory, image generation, tool calls, chat, remote connection, native code and model handling.

## b02af3b5: release
- APK for the `generate_image` eviction fix. Removed `apk/v20-v23`.
