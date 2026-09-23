/**
 * Format detection for the stable-diffusion.cpp image engine.
 *
 * `.safetensors` is dual-use (LLM weights, SD checkpoints, LoRAs, VAEs), so the extension alone
 * never routes a file. The tensor-key layout in the header does: an SD single-file checkpoint
 * carries the UNet under `model.diffusion_model.*`; an LLM carries `model.layers.*` /
 * `embed_tokens`; an SD LoRA carries `lora_unet_*` / `lora_te*_*` (kohya) or `unet.*`+`lora`
 * (diffusers/PEFT). Pure functions: the native side only returns the key list.
 */

export type SafetensorsKind =
  | 'sd-checkpoint'
  | 'sd-lora'
  | 'sd-vae'
  | 'sd-component'
  | 'llm'
  | 'unknown';

export type SdFamily = 'sd1' | 'sd2' | 'sdxl' | 'sd3' | 'flux' | 'unknown';

export interface SafetensorsClassification {
  kind: SafetensorsKind;
  family: SdFamily;
  /** Human-readable reason, logged and shown when a file is rejected. */
  reason: string;
}

/** File extensions the sd.cpp engine can open directly. */
export const SDCPP_CHECKPOINT_EXTENSIONS = ['.safetensors', '.ckpt'] as const;

export function isSdCppCandidateFileName(name: string): boolean {
  const lower = name.toLowerCase();
  return SDCPP_CHECKPOINT_EXTENSIONS.some(ext => lower.endsWith(ext));
}

const has = (keys: string[], pred: (k: string) => boolean): boolean => keys.some(pred);
const prefix = (keys: string[], p: string): boolean => has(keys, k => k.startsWith(p));

function unetFamily(keys: string[]): SdFamily {
  if (prefix(keys, 'model.diffusion_model.double_blocks.')) return 'flux';
  if (prefix(keys, 'model.diffusion_model.joint_blocks.')) return 'sd3';
  if (
    prefix(keys, 'conditioner.embedders.1.') ||
    prefix(keys, 'model.diffusion_model.label_emb.')
  ) {
    return 'sdxl';
  }
  if (prefix(keys, 'cond_stage_model.model.')) return 'sd2';
  if (prefix(keys, 'cond_stage_model.transformer.')) return 'sd1';
  return 'unknown';
}

const isKohyaSdLoraKey = (k: string): boolean =>
  /^lora_(unet|te\d?)_/.test(k);
const isLoraTensorKey = (k: string): boolean =>
  /\.(lora_down|lora_up|lora_A|lora_B|alpha|hada_w1_a|lokr_w1)(\.|$)/.test(k);
const isLlmKey = (k: string): boolean =>
  /(^|\.)(model\.layers\.\d+|embed_tokens|lm_head|transformer\.h\.\d+|gpt_neox\.)/.test(k);

export function classifySafetensorsKeys(keys: string[]): SafetensorsClassification {
  if (keys.length === 0) {
    return { kind: 'unknown', family: 'unknown', reason: 'empty safetensors header' };
  }
  if (prefix(keys, 'model.diffusion_model.')) {
    const family = unetFamily(keys);
    return { kind: 'sd-checkpoint', family, reason: `UNet/DiT under model.diffusion_model.* (${family})` };
  }
  if (has(keys, isKohyaSdLoraKey)) {
    return { kind: 'sd-lora', family: 'unknown', reason: 'kohya LoRA keys (lora_unet_/lora_te_)' };
  }
  const loraKeys = has(keys, isLoraTensorKey);
  const sdLoraTarget =
    prefix(keys, 'unet.') ||
    prefix(keys, 'text_encoder') ||
    (prefix(keys, 'transformer.') && !has(keys, isLlmKey));
  if (loraKeys && sdLoraTarget) {
    return { kind: 'sd-lora', family: 'unknown', reason: 'diffusers/PEFT LoRA keys (unet./text_encoder.)' };
  }
  if (has(keys, isLlmKey)) {
    return {
      kind: 'llm',
      family: 'unknown',
      reason: loraKeys ? 'LLM LoRA adapter (model.layers.* + lora)' : 'LLM weights (model.layers.* / embed_tokens / lm_head)',
    };
  }
  if (prefix(keys, 'encoder.down.') && prefix(keys, 'decoder.up.')) {
    return { kind: 'sd-vae', family: 'unknown', reason: 'standalone VAE (encoder./decoder. only)' };
  }
  if (prefix(keys, 'down_blocks.') || prefix(keys, 'text_model.')) {
    return {
      kind: 'sd-component',
      family: 'unknown',
      reason: 'single diffusers component (UNet or text encoder) - needs a full single-file checkpoint',
    };
  }
  return { kind: 'unknown', family: 'unknown', reason: 'no known SD or LLM tensor layout' };
}
