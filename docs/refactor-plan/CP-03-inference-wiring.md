# CP-03 — Text-gen inference wiring (OpenCL/CPU + KV cache quant)

Read first: `docs/refactor-map/02-inference-wiring.md`.

## Symptoms
1. OpenCL vs CPU choice "does not mean anything" (no runtime effect).
2. Cannot use q4/q8 KV cache — always fp16. Evidence: 1GB model uses far more RAM than
   the context size should need (fp16 KV is 2x q8, 4x q4).

## Trace targets (UI → store → provider → native llama.rn)
- Compute backend: `src/utils/acceleration.ts`, `src/services/activeModelService/loaders.ts`,
  `src/services/providers/localProvider.ts`, `src/services/llm.ts`, `llmHelpers.ts`.
  Check `n_gpu_layers` / GPU/OpenCL flag actually reaches `initLlama` in llama.rn.
- KV cache quant: grep `cache_type_k`/`cache_type_v`/`cacheType`. Files:
  `src/hooks/useTextGenerationAdvanced.ts`, `src/components/settings/textGenAdvancedSections.tsx`,
  `src/stores/appStore.ts`, `src/services/generationServiceHelpers.ts`,
  `src/services/activeModelService/loaders.ts`, `src/services/llmHelpers.ts`.
  Confirm the selected quant is passed to context init (llama.rn `cache_type_k/v` or
  `--cache-type-k`) and not hardcoded/omitted.

## VERIFIED FINDINGS (2026-09-20) — READ BEFORE "FIXING"
- KV cache f16 on OpenCL/HTP is INTENTIONAL and CORRECT, not a bug. `llmHelpers.ts`
  `backendForcesF16Cache` (:68), `effectiveCacheType` (:73), comments :61-70,87-97:
  OpenCL & HTP llama.cpp paths do NOT support a quantized KV cache, and quantized KV
  requires flash-attention which crashes those backends. So GPU/NPU ⇒ f16 by necessity.
  On CPU backend, `cache_type_k/v` ARE passed (:114) and q4/q8 works (default q8_0 :94).
  ⇒ User's "always fp16 / 1GB model uses too much RAM" = running on OpenCL/HTP. The lever
  is: use CPU backend for quantized KV, OR accept f16 on GPU/NPU. DO NOT remove the guard.
- "CPU vs OpenCL does nothing": value DOES reach initLlama (`:195` n_gpu_layers). But (a)
  OpenCL can silently fall back to CPU on timeout/unsupported (`llm.ts` fallback,
  `llmHelpers.ts:168-175,199-214`), and (b) the change requires a MODEL RELOAD — captured
  in `loaders.ts:227-240` (`setLoadedSettings`) with a mismatch banner. Perception of
  "no effect" is likely fallback + not reloading. NEEDS device `[WIRE-LLAMA-LOAD]` log to
  confirm; not resolvable headless.

## Actionable (UX, not ripping the guard)
- When backend = OpenCL/HTP, the KV-cache selector should show "f16 (forced by GPU/NPU)"
  / be disabled, so the UI never implies q4/q8 is active. Confirm current UI behavior
  (usesF16Cache is already the single source; check textGenAdvancedSections.tsx).
- Ensure a backend change surfaces the reload banner and the reload actually re-inits.

## Likely fix shape
- Add/repair the mapping from store setting → llama.rn init options. KV cache quant must
  be set at context creation (not per-request). Some builds require `flash_attn` for
  quantized KV — verify llama.rn version supports it; surface a note if not.

## Verification
- Log the resolved init options (a `[WIRE-*]` log like image path uses).
- On device: pick q4 KV, load ~1GB model, confirm RAM drops vs fp16 (Debug Logs / memory
  card). Toggle CPU vs OpenCL and confirm tokens/sec + backend log changes.
- Unit: extend `__tests__/unit/services/llmHelpers*.test.ts`, `activeModelService.loaders`.

## Hardware/config limits to surface to user
- llama.rn/ggml KV quant options + flash-attn requirement; OpenCL availability per SoC.
