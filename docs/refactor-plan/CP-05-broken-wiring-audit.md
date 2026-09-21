# CP-05 — Broken-wiring audit (all dropped/dead/hardcoded settings)

Read first: `docs/refactor-map/05-broken-wiring.md`.

## Goal
One authoritative table of every setting that does not reach its native/runtime effect.
Seeded findings (extend from mapping agent):

| # | Setting | UI | Store | Expected sink | Status | Evidence |
|---|---------|----|-------|---------------|--------|----------|
| 1 | Image OpenCL | image settings | useOpenCL | Android /generate `use_opencl` | DROPPED | localDreamGenerator.buildNativeParams sends it; LocalDreamModule.buildGenerationBody (kt ~L639) omits it |
| 2 | KV cache quant | text-gen advanced | (tbd) | llama.rn cache_type_k/v | SUSPECT fp16 | CP-03 |
| 3 | CPU vs OpenCL (text) | settings | (tbd) | n_gpu_layers/backend | SUSPECT no-op | CP-03 |

## Method
Grep each toggle → follow to the boundary (native module arg or llama.rn init). Mark
WIRED / DROPPED / HARDCODED with file:line. Add a `[WIRE-*]` log at each boundary so the
device transcript proves the value arrives.

## Deliverable
Fill this table; each DROPPED/HARDCODED row gets a fix (own commit) or a noted limitation.

## Verification
Per row: change setting, read the `[WIRE-*]` log on device, confirm the value propagated.
