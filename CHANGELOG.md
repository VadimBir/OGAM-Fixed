# Changelog

Branch `app-refactor-fixes`. Newest first. Each entry gives the commit it ships in.
State words: DONE = finished and unit tested; WIP = an agent is still editing, so the code is not checked; UNKNOWN = not verified.

## 2026-11-18: work-in-progress snapshot 2 (13 files)

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
