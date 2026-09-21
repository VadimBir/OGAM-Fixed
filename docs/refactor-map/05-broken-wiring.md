# 05 — Consolidated Broken / Dropped / Hardcoded Wiring (items 3 & 5) + HomeScreen Pro (item 4)

Every dropped param, dead toggle, or hardcoded value found, with file:line evidence. See 01/02
for full traces. Marks: DROPPED = value sent by caller, never forwarded; HARDCODED = fixed in
source, ignores the setting; DEAD = whole path unreachable; BY-DESIGN = intentional coercion the
UI does not fully explain to the user.

## Consolidated wiring table

| # | Setting / value | UI (writes) | Where it dies | Mark | Effect |
|---|---|---|---|---|---|
| W1 | image `use_opencl` | `imageUseOpenCL` → `imageGenerationService.ts:440` → `localDreamGenerator.ts:152` (`useOpenCL`) | `LocalDreamModule.buildGenerationBody` never adds `use_opencl` to /generate body — `LocalDreamModule.kt:639-650` | DROPPED | image OpenCL toggle inert on Android; server runs default path (comment kt:100-102 wrongly claims it's sent) |
| W2 | KV cache q8_0/q4_0 on **OpenCL** | `KvCacheTypeToggle` → `useTextGenerationAdvanced.ts:53` → `cacheType` | cache params omitted for OpenCL: `llmHelpers.ts:114` (`backend===OPENCL ? {} : {cache_type_k/v}`) → llama.cpp defaults f16 | HARDCODED/BY-DESIGN | q4/q8 impossible on OpenCL; f16 KV uses 2–4× RAM (item 3b) |
| W3 | KV cache on **HTP/NPU** | same toggle | `effectiveCacheType` coerces to `'f16'` when `backendForcesF16Cache` — `llmHelpers.ts:69,74,97` | HARDCODED/BY-DESIGN | q4/q8 impossible on HTP |
| W4 | flash attention on OpenCL/HTP | `FlashAttentionToggle` `textGenAdvancedSections.tsx:132` | forced `'off'`: `llmHelpers.ts:89-90` (`gpuBackendIncompatible`) | HARDCODED/BY-DESIGN | flash-attn toggle inert on GPU/NPU; and quant KV needs flash-attn → compounds W2/W3 |
| W5 | CPU vs OpenCL backend | `BackendSelector` → `updateSettings({inferenceBackend})` `textGenAdvancedSections.tsx:85` | value reaches `initLlama` (`llmHelpers.ts:195`) but OpenCL silently falls to CPU when unsupported/capped/timeout — `llm.ts:182-192`, `166-170`; `llmHelpers.ts:199-214` | CONDITIONAL NO-OP | OpenCL selection often == CPU on-device with no visible error (item 3a) |
| W6 | backend change with no reload | same | only snapshotted for banner: `loaders.ts:227-240`; context not re-inited on toggle | DEFERRED | "changing it does nothing" until model reloaded (item 3a) |
| W7 | image `backend` mnn/qnn | n/a (no UI) | hardcoded `'auto'` at load site: `activeModelService/loaders.ts:297`; mnn/qnn branches unreachable from TS (GAPS IM2) | HARDCODED | backend arg is decorative from TS |
| W8 | `imageGenerator.ts` service | — | targets `NativeModules.ImageGeneratorModule`, which is NOT registered on either platform (no `ImageGeneratorModule.kt`/`.swift`); live path is `localDreamGenerator.ts` | DEAD | legacy duplicate (see 02/CP-02) |
| W9 | iOS image OpenCL cache | `hasKernelCache`/`clearOpenCLCache` | silent no-op on iOS: `localDreamGenerator.ts:274-282` (returns early); name mismatch `hasKernelCache`→`hasOpenCLCache` (GAPS IM4/IM5) | DEAD-on-iOS | harmless, but a silent platform no-op |
| W10 | runtime system-prompt augmentation | n/a | appended invisibly: `generationToolLoop.ts:966-1036` (tool guidance/date/ext hints); DebugSheet shows base only (`DebugSheet.tsx:106`) | HIDDEN | user cannot see/own the real prompt (item 6B) |

Notes:
- W2/W3/W4 are intentional (OpenCL/HTP llama.cpp paths lack quant-KV / crash with flash-attn —
  comments `llmHelpers.ts:61-70,87-96`). They are listed because from the user's seat they read as
  "q4/q8 KV never works / toggles do nothing," which is item 3b/3a. The fix is UX (explain/why-locked
  + let CPU/Metal expose them), not a native rewire.
- The image rainbow bug (item 1) is a decode mismatch, not a settings-wiring drop — it lives in
  `01-image-gen.md` (`LocalDreamModule.kt:150-176`, `:737-749`).

## Duplications (item 2) — cross-ref 02/CP-02
- **Image generators (2):** `src/services/imageGenerator.ts` (`ImageGeneratorModule`, DEAD — W8) vs
  `src/services/localDreamGenerator.ts` (`LocalDreamModule`/`CoreMLDiffusionModule`, LIVE — imported
  by `imageGenerationService.ts:1`, `activeModelService/loaders.ts:12`).
- **Default persona (historically 3 copies):** `appStore.ts:230-233`, `projectStore.ts:44-46`,
  now both point at `APP_CONFIG.defaultSystemPrompt` (`constants/index.ts:110`) — comment says this
  collapsed the third copy. Verify no stragglers before edits.
- **Advanced text-gen controls:** previously duplicated across Model Settings screen and in-chat
  modal; now single shared impl `components/settings/textGenAdvancedSections.tsx` (header comment
  `:1-11`).
- **`saveRgbToPng` vs `saveRgbAsPng`:** two identical RGB→PNG packers in `LocalDreamModule.kt`
  (`:150-176` and `:844-892`).

## Item 4 — Pro-related affordances on HomeScreen (`src/screens/HomeScreen/index.tsx`)
The user wants to remove the annoying "Off Grid Pro" tile. Every Pro affordance on Home, with lines:

1. **Crown button (top-right header)** — `index.tsx:190-198`. `onPress` → `navigation.navigate('ProDetail')`;
   `MaterialCommunityIcons "crown"`; `style={styles.crownButton}`; accessibilityLabel "Open Off Grid AI
   Pro". **Matches the user's own hint (~line 190 navigating to 'ProDetail').** It is a small header
   icon, not a card/"tile."
2. **SyncHomeCard (Pro-gated tile)** — `index.tsx:256-268`. A slot-injected card
   (`useSlot(SLOTS.homeSyncCard)`, `:61`), rendered only when the Pro submodule provides it. Its
   "Clipboard" action navigates to `'ProDetail'` when not unlocked (`:264`). This is the most
   "tile-like" Pro promo on Home.
3. **DesktopPromoCard** — `index.tsx:328` (`components/DesktopPromoCard.tsx`). "Off Grid AI Desktop"
   announcement card. It is a desktop promo, **not** a Pro upsell (owns its own dismiss state).

**Ambiguity to confirm with the user:** "Off Grid Pro tile" most likely means (1) the crown button
(their line hint) or (2) the SyncHomeCard (the actual Pro-gated card/tile). Removing the crown is a
delete of `:190-198` (and the unused `styles.crownButton`). Removing the SyncHomeCard means dropping
the slot render `:256-268` (or unregistering `SLOTS.homeSyncCard` in the Pro bootstrap). Ask which
one before deleting.
