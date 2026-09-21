# CP-02 — De-duplicate same-semantics code

Read first: `docs/refactor-map/00-OVERVIEW.md`, `02` where relevant.

## Known duplication
- Image generators:
  - `src/services/imageGenerator.ts` → `ImageGeneratorModule` (Android only, older API).
  - `src/services/localDreamGenerator.ts` → `LocalDreamModule`/`CoreMLDiffusionModule`
    (current, cross-platform, SSE + preview).
  Determine which is LIVE (grep importers). Likely delete/retire `imageGenerator.ts` +
  `ImageGeneratorModule` if unused. Confirm no native module still registered/needed.

## Approach
- For each duplicate cluster the mapping agent lists in `05-broken-wiring.md`/overview:
  1. Grep importers of each variant. 2. Keep the one that is wired + tested. 3. Remove the
  dead one (JS + native registration + tests) in a dedicated commit. 4. Run knip
  (`knip.json` present) to catch newly-dead exports.

## Verification
- `npx knip`, `npx tsc --noEmit`, jest for touched areas. App still builds + generates.

## Guardrail
Do not merge two LIVE paths blindly — confirm one is truly dead before deletion.
