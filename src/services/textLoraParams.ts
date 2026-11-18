/**
 * Pure mapping from the text LoRA setting to llama.rn adapter params. No store / native imports, so
 * buildModelParams (llmHelpers) and the memory estimate can use it.
 *
 * llama.rn takes `lora_list: Array<{ path; scaled? }>` at init (node_modules/llama.rn/src/types.ts:149,
 * forwarded by initLlama, src/index.ts:1496-1500 / :1574) and the same shape for a live change
 * (applyLoraAdapters, src/index.ts:1003-1009). Native reads `scaled`, default 1.0
 * (cpp/jsi/JSIParams.cpp:126).
 */
import { TEXT_LORA_SCALE_MAX, TEXT_LORA_SCALE_MIN, type TextLora } from './textLoraTypes';

export interface LlamaLoraAdapter {
  path: string;
  scaled: number;
}

export const clampTextLoraScale = (scale: number): number =>
  Number.isFinite(scale) ? Math.min(TEXT_LORA_SCALE_MAX, Math.max(TEXT_LORA_SCALE_MIN, scale)) : 1;

/** Enabled adapters in list order. A scale of 0 is kept out: it loads the file for no effect. */
export function enabledLoraList(loras: TextLora[] | undefined): LlamaLoraAdapter[] {
  return (loras ?? [])
    .filter(l => l.enabled && typeof l.path === 'string' && l.path.length > 0)
    .map(l => ({ path: l.path, scaled: clampTextLoraScale(l.scale) }))
    .filter(l => l.scaled > 0);
}

/** Identity of an adapter set (paths + scales, in order): equal keys = nothing to apply. */
export function loraListKey(list: LlamaLoraAdapter[]): string {
  return list.map(l => `${l.path}@${l.scaled}`).join('|');
}
