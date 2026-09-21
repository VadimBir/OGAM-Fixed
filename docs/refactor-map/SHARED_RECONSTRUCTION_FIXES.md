# Reconstructed `@offgrid/*` — consumer-contract fixes

The mobile app bundles the out-of-root monorepo packages `@offgrid/{sync,models,speech,ui,rag}`
(mapped by `metro.config.js` to `../shared/packages/*/dist`). That `../shared` tree is a local
reconstruction (the private monorepo is not attachable to this session). Because the app was
written against the REAL package APIs, any shape/arity drift in the reconstruction ships straight
into the JS bundle. Two such drifts were shipped in `apk/v1` and are fixed in `apk/v2`.

## Fix 1 — `REASONING_BUDGET_OPTIONS` must be `number[]`, not `{value,label}[]`

Sole consumer `src/components/settings/textGenAdvancedSections.tsx:214`:

```ts
...REASONING_BUDGET_OPTIONS.filter(value => value < maxTokens),   // numeric compare
// and steps.indexOf(shownBudget) / onChange(steps[step])          // numeric index
```

Objects coerce to `NaN` under `< maxTokens`, so every option is filtered out and the
Thinking-Budget slider collapses to `[Auto, maxTokens]` — the "max thinking tokens is Auto and
cannot be changed" bug. `../shared/packages/models/dist/index.js` now exports:

```js
var REASONING_BUDGET_OPTIONS = [256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768];
function reasoningBudgetLabel(v){ return v==null||v===-1?'Auto':v<=0?'Off':String(v)+' tokens'; }
```

Note: the slider ladder is still bounded by `settings.maxTokens` (default 1024). Raising Max
Tokens exposes more stops; this is by design (you cannot think more tokens than you may emit).

## Fix 2 — thinking/reasoning payload builders take `(enableThinking, budget)`

Consumers pass TWO args and expect specific shapes:

| Builder | Call site | Required return |
|---|---|---|
| `thinkingBudgetPayload(enable, budget)` | `llmHelpers.ts:310` (local llama.rn) | `{ thinking_budget_tokens?: number }` |
| `reasoningBudgetPayload(enable, budget)` | `openAICompatibleProvider.ts:115` (remote-vision) | `{ thinking: { type, budget_tokens? } }` |
| `openRouterReasoningPayload(enable, budget)` | `openAICompatibleProvider.ts:119` (OpenRouter) | `{ reasoning: { max_tokens? } }` |

The v1 reconstruction took a SINGLE `budget` arg and returned the wrong keys. With Thinking on,
`buildThinkingCompletionParams` spread `{ thinking: { type:'enabled', budget_tokens: true } }`
(the boolean `true` mistaken for the budget) into the **llama.rn completion params**. A malformed
native completion param corrupts decoding — the intermittent multilingual "word-salad" the model
emitted during tool use. Fixed in `../shared/packages/models/dist/index.js` (see `_positiveBudget`
+ the three builders).

## Reproducing the corrected `../shared` in a fresh container

`../shared` is not tracked in this repo. If a new session rebuilds it, apply the two shapes above
to `packages/models/dist/index.js`. Verify:

```sh
node -e "const m=require('../shared/packages/models/dist/index.js');
 console.assert(Array.isArray(m.REASONING_BUDGET_OPTIONS) && typeof m.REASONING_BUDGET_OPTIONS[0]==='number');
 console.assert(JSON.stringify(m.thinkingBudgetPayload(true,8192))==='{\"thinking_budget_tokens\":8192}');
 console.log('ok');"
```

## Fix 3 — image parameter resolver arity (gigapixel std::bad_alloc crash)

Symptom (from the on-device error popup): server log `Size:1057820237x1057820237 ... std::bad_alloc`.
The server was asked to allocate a ~1e9 × 1e9 image and crashed.

Root cause: two more reconstructed `@offgrid/models` functions had wrong signatures vs their
consumer `src/services/imageParameterPolicy.ts`:

- `resolveImageParameters(model, store)` must return `{ steps, cfgScale, size }` from
  `store[model.id]` with SD1.5 defaults. The v1 reconstruction took `(store, overrides)` and
  returned the merged model object, so `resolved.size` was `undefined` →
  `Math.max(SWEET_SPOT_SIZE, undefined)` = `NaN` → marshalled to a native int as garbage
  (~1057820237) → gigapixel allocation → `std::bad_alloc`.
- `effectiveImageParameter(requestValue, fallback)` must return the request value when positive
  else the fallback. The v1 reconstruction took `(store, key, fallback)`.

Fixed in `../shared/packages/models/dist/index.js` (SD1.5 defaults: steps 20, cfg 7.5, size 512).
Defense-in-depth: `LocalDreamModule.clampDimension()` snaps width/height to 64..1024 (mult of 8)
and steps to 1..100, so no bad value can reach the server again.
