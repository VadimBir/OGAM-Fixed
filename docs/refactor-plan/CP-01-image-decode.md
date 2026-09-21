# CP-01 — Image generation rainbow/noise (wrong decode)

Read first: `docs/refactor-map/01-image-gen.md`.

## Symptom
Generated image is rainbow noise, not the real image. User hypothesis: base64 output is
decoded AS raw pixels instead of the real decoded output.

## Suspect files
- `android/.../localdream/LocalDreamModule.kt` — `saveRgbToPng` (~L150), `buildFinalResult`
  (~L737), `handleProgressEvent`/`savePreviewImage` (~L664), `buildGenerationBody` (~L639).
- `ios/.../CoreMLDiffusionModule.swift` + `.m` — decode/write path.
- `src/services/localDreamGenerator.ts` — `buildResult` (L158), progress/preview.
- `src/services/imageGenerator.ts` — legacy path (see CP-02).
- Server contract: local-dream `libstable_diffusion_core.so` /generate SSE `complete`
  event `image` field — RAW RGB (w*h*3)? RGBA (w*h*4)? or base64 PNG/JPEG?

## Root-cause hypotheses to confirm (do NOT edit before confirming contract)
1. Server changed to return an ENCODED PNG/JPEG (base64) but native still unpacks it as
   raw w*h*3 RGB → every byte becomes a pixel channel → rainbow. Fix: detect/decode PNG
   via BitmapFactory instead of manual RGB packing.
2. Channel mismatch RGB vs RGBA (3 vs 4) → stride off-by-one drift → diagonal rainbow.
3. Preview payload differs from final payload; only final is correct (or vice versa).

## CONFIRMED against the shipped binary (2026-09-20)
`strings android/app/src/main/jniLibs/arm64-v8a/libstable_diffusion_core.so` contains:
`image/jpeg`, `JPEG encoding time: %d ms, size: %zu KB`, `JPEG encoding failed`,
`JPEG decoding failed`, `event: complete`, plus a Rust base64 decoder (base64-0.13.1).
⇒ The server JPEG-encodes the final image and base64s it into the SSE `complete.image`.
The old saveRgbToPng packed those JPEG bytes as raw RGB pixels = rainbow. The fix
(saveServerImageToPng, format-detect → BitmapFactory) is correct. Its encoded branch
does NOT use the passed width/height (decodes real dims), so it is also immune to any
width/height mis-wiring — only the raw fallback uses them. Root cause proven, not guessed.

## Verification
- Reproduce: generate on device (S26 Ultra), screenshot. Check `[WIRE-IMAGE]` log
  (localDreamGenerator L159) and native `saveRgbToPng` size-mismatch throw.
- Confirm bytes length vs w*h*3 in the size check (kt L153). If it does NOT throw, the
  payload IS w*h*3 → bug is elsewhere (channel order / preview). If it throws in the wild,
  payload is encoded → hypothesis 1.
- After fix: image renders correctly; add/adjust unit test in
  `__tests__/contracts/localDream.contract.test.ts` and `coreMLDiffusion.contract.test.ts`.

## Open question for user
Is rainbow on Android, iOS, or both? Which model(s)? (Determines platform + contract.)
