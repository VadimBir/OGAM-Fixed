# 02 — Text-Gen Parameter Wiring (compute backend + KV cache quant)

Covers requirement item 3:
> (a) OpenCL vs CPU inference choice "does not mean anything" — changing it has no effect;
> (b) cannot use q4 or q8 KV cache — it always seems to be fp16 (a 1GB model uses far more
> memory than the context size should need).

## Chain of custody (UI → store → params → llama.rn)

UI control → `settings.<field>` (appStore) → `buildModelParams()`
(`src/services/llmHelpers.ts:77-121`) → `initContextWithFallback()`
(`llmHelpers.ts:178-225`) → `initLlama({...})` (llama.rn native).
The active-model load site that snapshots reload-settings is
`src/services/activeModelService/loaders.ts:179-251`.

## Compute backend (CPU / OpenCL / Metal / HTP-NPU / n_gpu_layers)

| Stage | Where | Evidence |
|---|---|---|
| UI writes it | `BackendSelector.onSelect` → `updateSettings({ inferenceBackend: id })` | `src/components/settings/textGenAdvancedSections.tsx:85` |
| UI writes layer count | `updateSettings({ gpuLayers: value })` | `textGenAdvancedSections.tsx:96` |
| Backend options | CPU / OPENCL(label "GPU") / HTP(NPU beta, only if `hasNPU && HTP_ENABLED`) | `textGenAdvancedSections.tsx:40-70` |
| Store field | `settings.inferenceBackend`, `settings.gpuLayers`, legacy `settings.enableGpu` | `appStore.ts` |
| Params derive gpuEnabled | `gpuEnabled = backend ? backend !== CPU : enableGpu !== false` | `llmHelpers.ts:91` |
| Params derive nGpuLayers | `nGpuLayers = gpuEnabled ? (gpuLayers ?? DEFAULT) : 0` | `llmHelpers.ts:92` |
| Re-resolve per backend | HTP→layers to NPU; OPENCL→check `getOpenCLCapability`, else 0; CPU→0 | `src/services/llm.ts:174-192` |
| RAM-cap the layers | `getGpuLayersForDevice(totalMemory, nGpuLayers)` may floor to 0 | `llm.ts:166-170` |
| Native call | `initLlama({ ...params, n_ctx, n_gpu_layers })` | `llmHelpers.ts:195` |

**Verdict: WIRED, but with two effect-erasing behaviors that explain "does not mean anything":**
1. **Silent CPU fallback (functional no-op).** On OpenCL, if `hardwareService.getOpenCLCapability()`
   reports unsupported, `safeGpuLayers = 0` (`llm.ts:182-186`) → identical to CPU. Also
   `getGpuLayersForDevice` (`llm.ts:166`) can cap layers to 0 on low-RAM tiers. And
   `initContextWithFallback` (`llmHelpers.ts:199-214`) retries on CPU (`n_gpu_layers=0`) whenever
   the GPU init throws/times out (25s Adreno, `llmHelpers.ts:132`). On any of these the selected
   "GPU/OpenCL" runs on CPU with no UI error — indistinguishable from CPU to the user.
2. **Requires a model reload.** Changing `inferenceBackend` only mutates the store; the loaded
   context is unchanged until reload. `loaders.ts:227-240` snapshots `inferenceBackend` into
   `loadedSettings` for the pending-settings banner, but nothing re-inits the context on toggle.
   If the user toggles without reloading, "nothing happens."

So the row is **WIRED to native**, but effect is **conditionally erased** (device fallback) and
**deferred** (reload required). No hardcode in the param path itself.

## KV cache quant (cache_type_k / cache_type_v)  ← the "always fp16" complaint

| Stage | Where | Evidence |
|---|---|---|
| UI writes it | `KvCacheTypeToggle` → `handleCacheTypeChange(ct)` → `updateSettings({ cacheType })` | `textGenAdvancedSections.tsx:149-169`, `hooks/useTextGenerationAdvanced.ts:53-60` |
| Options | f16 / q8_0 / q4_0 | `useTextGenerationAdvanced.ts:15` |
| Requested cache | `requestedCache = settings.cacheType || (flashAttn ? 'q8_0' : 'f16')` | `llmHelpers.ts:94` |
| **Coercion** | `cacheType = effectiveCacheType(backend, requestedCache)` | `llmHelpers.ts:97` |
| `effectiveCacheType` | returns `'f16'` when `backendForcesF16Cache`, else requested | `llmHelpers.ts:73-75` |
| `backendForcesF16Cache` | **true for OPENCL, and for HTP when HTP_ENABLED** | `llmHelpers.ts:68-70` |
| **Param emit** | `...(backend === OPENCL ? {} : { cache_type_k: cacheType, cache_type_v: cacheType })` | `llmHelpers.ts:114` |
| Load snapshot | loaders forces stored cacheType to `'f16'` for OpenCL | `activeModelService/loaders.ts:238` |
| UI lock | `cacheDisabled = gpuForcesF16` → toggle locked to f16 | `useTextGenerationAdvanced.ts:30-32`, `textGenAdvancedSections.tsx:160` |

**Verdict: HARDCODED-to-f16 for OpenCL/HTP by design.**
- On **OpenCL**: cache params are **omitted entirely** (`llmHelpers.ts:114`) → llama.cpp defaults
  to f16. q4/q8 is impossible.
- On **HTP**: `effectiveCacheType` coerces to `'f16'` (`llmHelpers.ts:69,74`).
- Only **CPU / Metal** actually pass the user's `cache_type_k/v`.
- Extra gate: quantized cache also requires **flash attention**; flash attn is forced `'off'` for
  OpenCL/HTP (`llmHelpers.ts:89-90`), and turning flash attn off resets cacheType to f16
  (`useTextGenerationAdvanced.ts:45-51`).

**This fully explains item 3b.** If the user selected the Android "GPU" backend (label for
OpenCL, `textGenAdvancedSections.tsx:48`), the KV cache is force-f16 regardless of the toggle,
and f16 KV is 2× the memory of q8_0 and 4× of q4_0 — matching "a 1GB model uses far more memory
than the context size should need." The behavior is intentional (comment `llmHelpers.ts:61-70,
95-96`: OpenCL/HTP llama.cpp paths don't support a quantized KV cache and can crash), but it is
NOT surfaced as "your backend forces f16" beyond disabling the toggle — so the user perceives it
as "q4/q8 never works."

## Ambiguity to flag
Item 3a says the CPU/OpenCL choice is inert. Confirmed the value propagates to `initLlama`, so
the likely real cause is (1) the device silently falling back to CPU, or (2) no reload after
toggling. Distinguishing them needs a device repro: check the `[WIRE-LLAMA-LOAD]` log
(`llmHelpers.ts:191`) and `[LLM] OpenCL … falling back to CPU` (`llm.ts:185`) after selecting
GPU and reloading. Record which one the user hits before "fixing" wiring that is already connected.

## Related settings that DO require reload (loaders.ts:227-239 snapshot)
`enableGpu, inferenceBackend, gpuLayers, nThreads, nBatch, contextLength, flashAttn,
speculativeDecoding, cacheType`. The pending-settings banner compares these.
