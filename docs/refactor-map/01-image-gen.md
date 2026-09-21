# 01 — Image Generation Pipeline & the Rainbow-Noise Bug

Covers requirement item 1 (rainbow decode bug) and the `useOpenCL` drop.

## User's verbatim hypothesis (item 1)
> Image gen bug: output is a rainbow (noise) instead of a real image. User's hypothesis:
> code decodes the base64 output AS raw pixels instead of the real (decoded) output — i.e.
> wrong image decode used during generation.

## Pipeline overview (both platforms)
1. UI/chat → `imageGenerationService.generateImage()` (`src/services/imageGenerationService.ts:330`).
   State machine idle→enhancing→loading→generating→completed/error.
2. Local path → `localDreamGeneratorService.generateImage()` (`src/services/localDreamGenerator.ts:177`),
   which `Platform.select`s the native module (`localDreamGenerator.ts:16-20`):
   - **Android** → `LocalDreamModule` (Kotlin).
   - **iOS** → `CoreMLDiffusionModule` (Swift).
3. JS→native params built by `buildNativeParams` (`localDreamGenerator.ts:142-156`).
4. Native returns `{ id, imagePath, width, height, seed }`; JS wraps in `buildResult` (`localDreamGenerator.ts:158-175`).

`imageGenerator.ts` (the legacy `ImageGeneratorModule` path) is DEAD — see `02`/CP-02.

## Android decode path (where the bug is) — `LocalDreamModule.kt`
Key file: `android/app/src/main/java/ai/offgridmobile/localdream/LocalDreamModule.kt`
— purpose: spawn local-dream server, POST /generate, parse SSE, write image file.

- `buildGenerationBody(params)` `:639-650` — build /generate JSON. Keys sent: prompt,
  negative_prompt, steps, cfg, seed, width, height, scheduler="dpm",
  show_diffusion_process=true, show_diffusion_stride. **Does NOT send `use_opencl`.**
- `openGenerationConnection()` `:652-662` — POST, Accept text/event-stream, readTimeout=0.
- `parseSseStream()` `:702-735` — reads `event: progress` / `event: complete`; a `complete`
  event's JSON is captured as `completeData`.
- `handleProgressEvent()` `:678-694` + `savePreviewImage()` `:664-676` — decode preview `image`
  field via `saveRgbToPng` using the REQUESTED body width/height.
- `buildFinalResult(completeData)` `:737-759` — reads `completeData.getString("image")` (base64),
  `width`, `height`, then **`saveRgbToPng(imageBase64, width, height, outputPath)`** `:749`.
- **`saveRgbToPng(base64Rgb, width, height, outputPath)`** `:150-176` — THE DECODER:
  - `:151` `Base64.decode(base64Rgb)`.
  - `:152-157` size guard: `if (rgbBytes.size != width*height*3) throw IllegalArgumentException`.
  - `:161-167` manual pixel packing, interleaved **RGB24**, `pixels[i] = 0xFF<<24 | r<<16 | g<<8 | b`.
  - `:172-174` `Bitmap.compress(PNG)` to disk.
  - Duplicate copy of the same logic in `saveRgbAsPng` (`:844-892`).

### Server contract — confirmed from the binary
`libstable_diffusion_core.so` (arm64) string table confirms the /generate SSE contract keys:
`/generate`, `/health`, `event: progress|complete|error`, `use_opencl`,
`show_diffusion_process`, `show_diffusion_stride`, `total_steps`, `generation_time_ms`,
`negative_prompt`, `scheduler`. **Critically it also contains a JPEG codec:**
`"JPEG encoding time: %d ms, size: %zu KB"`, `"JPEG encoding failed"`, `"JPEG decoding failed"`,
`"BMP JPEG/PNG"`, plus a Rust `base64` decoder. (Search: `strings … | grep -iE 'jpeg|base64|use_opencl|event: complete'`.)

### The mismatch that produces rainbow noise
The Android decoder assumes the `complete.image` payload is **raw interleaved RGB24
(`width*height*3` bytes)** and hand-packs each byte triple into a pixel
(`saveRgbToPng`, kt:150-167). But the server binary carries a full **JPEG encoder** and a
base64 encoder — strong evidence the payload the server actually emits for the final
image is a **base64-encoded JPEG** (encoded bytes), not raw RGB. Treating encoded/compressed
bytes as raw pixel channels = every byte becomes a color channel = **rainbow noise** — exactly
the user's hypothesis (wrong decode: encoded output read as raw pixels).

Correct decode for an encoded payload: `BitmapFactory.decodeByteArray(bytes, 0, bytes.size)`
(detect JPEG/PNG magic) instead of the manual `width*height*3` packing.

### Decisive test to confirm on-device (records the one ambiguity)
The size guard at `kt:153` gates the two runtime outcomes:
- If the payload is base64 JPEG, `rgbBytes.size != width*height*3` → the guard **throws**
  `IllegalArgumentException("RGB data size … doesn't match expected …")` → generation
  FAILS with a `GENERATION_ERROR`, not a rendered rainbow.
- If a rainbow image is actually rendered (no error), then the byte-length DID equal
  `width*height*3`, so the payload is raw-but-wrong-layout (BGR vs RGB, or planar
  CHW vs interleaved) rather than JPEG.
Either way the fix is the same class: **stop hand-packing raw RGB; decode the payload by
its real format.** To pin the exact sub-mechanism, check the device log for either the
`saveRgbToPng` size-mismatch throw or the `[WIRE-IMAGE]` result line
(`localDreamGenerator.ts:159`).

**OPEN QUESTION for the user (from CP-01):** Is the rainbow on Android, iOS, or both, and
which model(s)? iOS uses a genuine encoder (see below) and cannot produce this rainbow, so
if the user also sees it on iOS the cause is different there.

## iOS decode path — NOT affected
`ios/CoreMLDiffusionModule.swift`:
- `generateImage()` `:252-383` runs Apple's `StableDiffusionPipeline.generateImages()` →
  `[CGImage?]`.
- `:349-354` `UIImage(cgImage:).pngData()` → `pngData.write(to:)`. A real, correct PNG
  encode of a real `CGImage`. No manual byte packing → **no rainbow possible on iOS.**
- Returns `{ id, imagePath, width, height, seed }` `:364-370`.

## The `useOpenCL` drop (explicitly requested note)
- Setting exists: `settings.imageUseOpenCL` (`appStore`), read at
  `imageGenerationService.ts:440` (`useOpenCL: settings.imageUseOpenCL ?? true`) and threaded
  through `_runGenerationAndSave` → `generateImage`.
- JS→native: `buildNativeParams` sets `useOpenCL: params.useOpenCL ?? true`
  (`localDreamGenerator.ts:152`) and passes it in the native `generateImage` params.
- **DROP:** `LocalDreamModule.buildGenerationBody` (`kt:639-650`) never reads `useOpenCL` and
  never puts `use_opencl` into the /generate JSON body — even though `use_opencl` is a
  recognized server param (confirmed in the .so) and the code comment at `kt:100-102` claims
  "OpenCL GPU acceleration is requested per-request via `use_opencl`: true in the JSON body."
  Net effect: the image OpenCL toggle is inert on Android; the server runs its default path.

## Related known gaps (docs/GAPS_BACKLOG.md, corroborating)
- IM2 `localDreamGenerator.ts` + `loaders.ts:297`: `backend` is hardcoded `'auto'` at the
  image loader call site (`activeModelService/loaders.ts:297`) — the `'mnn'/'qnn'` branches are
  never reached from TS.
- IM4/IM5: `hasKernelCache` wraps `hasOpenCLCache` (name mismatch); `clearOpenCLCache`/
  `hasKernelCache` are silent no-ops on iOS.

## Key-file function index (Android)
`LocalDreamModule.kt`:
- `buildCommand` `:90-148` — assemble server argv (MNN `--cpu` vs QNN backend). vars: isCpu, embeddingSize.
- `loadModel` `:352-405` — resolve backend/dir, start server with fallback. vars: normalizedBackend, modelDir.
- `generateImage` `:789-838` — health-check, POST body, parse SSE, resolve final result.
- `saveRgbToPng` `:150-176` — **decoder (bug site)**. vars: rgbBytes, expectedSize, pixels.
- `buildFinalResult` `:737-759` — read `image`/`width`/`height`, call decoder, return map.
