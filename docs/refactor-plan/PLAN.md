# OGAM Refactor + Fix Plan (master)

Branch: `claude/app-refactor-img-gen-cards-pa8qrz`
Owner session: img-gen / cards / wiring refactor.

## Why this file
7 distinct asks spanning native (Android/iOS) + JS. Multi-session. Each checkpoint
below is self-contained so a fresh/post-compact session can resume without re-deriving
context. Read this file, then the relevant `CP-XX-*.md`.

## Source of truth for the codebase map
`docs/refactor-map/*.md` (produced by the mapping agent). Read the matching map file
before touching a checkpoint. Do NOT conclude "X does not exist" from a single keyword
search — names differ; try synonyms and read full files.

## The 7 asks (verbatim intent)
1. Image gen: rainbow/noise instead of real image. Hypothesis: base64 output decoded AS
   raw pixels instead of the real decoded output — wrong decode during gen. → CP-01
2. Dedup: same semantics in 2 places (e.g. two image-generator services). → CP-02
3. Text-gen wiring broken: (a) OpenCL vs CPU choice has no effect; (b) can't use q4/q8
   KV cache — always fp16 (1GB model uses far more RAM than context needs). → CP-03
4. Remove the "Off Grid Pro" tile from Home (annoying). → CP-04
5. ID ALL broken wiring (settings that don't reach native, dead toggles). → CP-05
6. Expand: (A) model tool-calls on demand; (B) open/viewable system prompt;
   (C) skills system + a "generate image" skill whose result shows in chat. → CP-06
7. SillyTavern/Oobabooga-style CHARACTER CARD system: create cards, avatars, import
   ST PNG cards (auto-fill desc), model calls img-gen, user controls prompts exactly,
   existing features (web search etc.) become editable skill cards; editing in settings
   or a new bottom tab. → CP-07

## Global constraints
- Keep each change in the smallest existing owner (rules.md). Paid code → `pro/` submodule.
- UI: use `@offgrid/design`. Verify changed journey on device; screenshot each change.
- Compile target: Samsung Galaxy S26 Ultra (Android). Commit + push after each checkpoint.
- Do NOT push to any branch but the designated one.

## Confirmed findings so far (pre-mapping-agent)
- Two image generators: `src/services/imageGenerator.ts` (ImageGeneratorModule) vs
  `src/services/localDreamGenerator.ts` (LocalDreamModule/CoreMLDiffusionModule). Likely
  one is legacy/dead. [CP-02]
- Android image wiring bug: `localDreamGenerator.buildNativeParams` sends `useOpenCL`,
  but `LocalDreamModule.buildGenerationBody` (kt ~L639) never reads it / never puts
  `use_opencl` in the JSON body → OpenCL toggle silently dropped. [CP-03/CP-05]
- Android native RGB→PNG decode (`saveRgbToPng`, kt ~L150; `buildFinalResult` ~L737)
  looks correct for a raw-RGB (w*h*3) contract. Rainbow bug therefore likely: (i) server
  now returns a different payload than raw RGB, or (ii) iOS/JS path, or (iii) preview vs
  final mismatch. CONFIRM via server contract before editing. [CP-01]

## Order of execution (recommended)
CP-04 (quick win) → CP-01 (highest user pain) → CP-03/CP-05 (wiring) → CP-02 (dedup) →
CP-06 (tools/skills/sysprompt) → CP-07 (card system). CP-06 and CP-07 are the large,
design-heavy items and need user sign-off on UX before build.

## Status
- [x] Repo oriented, mapping agent run, `docs/refactor-map/*` written
- [x] CP-01 image decode — Android format-detecting decoder + tests (pushed). Verify on S26.
- [ ] CP-02 dedup — THREE image services found (imageGenerator.ts dead, localDreamGenerator
      live-generate, onnxImageGeneratorService in loaders.ts). Needs care.
- [~] CP-03 inference wiring — analyzed: KV f16 on GPU/NPU is a correct hw guard (not a bug);
      forwarded dropped image `use_opencl`. Text OpenCL/CPU "no-op" needs device log. UX:
      annotate/disable KV selector on GPU/NPU. (pushed partial)
- [x] CP-04 remove Pro tile — ProUpsellBanner removed from Settings (pushed)
- [ ] CP-05 broken-wiring audit — table seeded in docs/refactor-map/05-broken-wiring.md
- [~] CP-06 tools/skills/sysprompt — #2 DONE: generate_image tool added (registry+handler,
      enabled by default, micro-prompt = tool description, routes to chat via conversationId,
      optional negative_prompt+steps; registry tests updated). REMAINING: 6B open/editable
      assembled system prompt (augmentSystemPromptForTools is hidden), 6C skill cards.
- [ ] CP-07 character card system — design pending user sign-off

## NEEDS USER DECISION before building CP-06/07 (large)
- Tool-calling: native only (user said native, with per-tool prompt → card).
- Settings "System" tab: System Prompt + Skill/Tool Cards + Characters + Persona.
- SillyTavern import spec version (V2 chara_card_v2?).
- CP-03: OK to add UI annotation "f16 forced by GPU/NPU" + keep guard? (recommended)
