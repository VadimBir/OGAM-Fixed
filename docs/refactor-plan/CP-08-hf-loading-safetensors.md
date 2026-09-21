# CP-08 — HF search (exists!), Android load-kill, safetensors image backend, file upload

New requirements from user (2026-09-20 round 2). RECORDED verbatim-intent to avoid loss.

## Finding: HuggingFace GGUF search ALREADY EXISTS (do not rebuild)
- `huggingface.ts:searchModels(query,{filter:'gguf', sort, pipelineTag})` (:30) hits the real
  HF API `/models?filter=gguf&search=...`.
- Wired into UI: `screens/ModelsScreen/useTextModels.ts` — `searchQuery` state (:100),
  debounced `runSearch` (:141), `huggingFaceService.searchModels(...)` (:155), results →
  `getModelFiles`/download (:197). `TextModelsTab.tsx` renders it.
- `getModelFiles`/`getModelDetails` resolve GGUF files per repo with byte sizes.
⇒ Free "download any GGUF from HF" is present. If the user can't use it, cause is
  discoverability, a search bug, or (most likely) the LOAD step below — NOT missing search.
  IMAGE side is different: `huggingFaceModelBrowser.ts` is curated to 2 repos
  (xororz/sd-mnn, sd-qnn) only — no free image search (see safetensors item).

## R1 — Android kills app when loading even small models despite free RAM (PRIORITY)
User hypothesis: no chunked loading → Android OS OOM-kills the app.
- Investigate: model load path `activeModelService/loaders.ts` + `llmHelpers.ts`
  `initContextWithFallback` (:178) + memory guard `activeModelService/memory.ts` +
  `services/memoryBudget`/`modelResidency/policy.ts`. Check `use_mmap`/`use_mlock`
  (`llmHelpers.ts:100-101` use_mlock:false, use_mmap gated by shouldDisableMmap).
- Likely suspects: (a) mmap disabled → full model read into RAM at once; (b) a too-strict
  pre-load memory guard rejecting/parallel-loading; (c) large single allocation vs Android's
  per-app cgroup limit → LMK kill. "Chunked loading" = rely on mmap page-in, not a full read.
- NEED device Debug Log: what model, size, RAM, and the LMK/`initContextWithFallback` lines.

## R2 — Image gen: second backend accepting SAFETENSORS (user has .safetensors, not ONNX/MNN)
- Today image = MNN/QNN (local-dream) + iOS CoreML; no safetensors runtime. Adding one is a
  large native effort (a safetensors SD runtime on-device). Scope/spike required. Options:
  convert safetensors→MNN off-device, or embed a runtime that reads safetensors. DESIGN NEEDED.

## R3 — Local file upload/import for BOTH text (GGUF) and image (safetensors)
- Text import exists partly: `modelManager/importLocalModel.ts`. Confirm it accepts a picked
  .gguf via document picker + registers it. Image import of safetensors depends on R2.

## R4 — SillyTavern import: support V2 (chara_card_v2) AND V3 (ccv3). (feeds CP-07)

## R5 — KV quant on GPU (user question): no upstream backend supports quantized KV on
  OpenCL/HTP today (needs flash-attn). A "third backend" won't change that. Revisit only if
  llama.cpp adds GPU quantized-KV. Keep CP-03 guidance.

## Order (user): R1 (HF/loading sanity) → CP-06. R2/R3/R4 slot in per design sign-off.
