/**
 * Format detection for the stable-diffusion.cpp image engine.
 *
 * `.safetensors` is dual-use (LLM weights, SD checkpoints, LoRAs, VAEs), so the extension alone
 * never routes a file. The tensor-key layout in the header does: an SD single-file checkpoint
 * carries the UNet under `model.diffusion_model.*`; an LLM carries `model.layers.*` /
 * `embed_tokens`; an SD LoRA carries `lora_unet_*` / `lora_te*_*` (kohya) or `unet.*`+`lora`
 * (diffusers/PEFT). Pure functions: the native side only returns the key list and metadata.
 */

export type SafetensorsKind =
  | 'sd-checkpoint'
  | 'sd-lora'
  /** SD LoRA keys, but no tensor type sd.cpp applies (OFT / IA3 / GLoRA ...): it would do nothing. */
  | 'sd-lora-unsupported'
  | 'sd-vae'
  | 'sd-component'
  | 'video'
  | 'llm'
  | 'unknown';

export type SdFamily = 'sd1' | 'sd2' | 'sdxl' | 'sd3' | 'flux' | 'unknown';

/**
 * Base model a LoRA was trained for. 'sd1-or-sd2': the keys only say "SD 1.x or 2.x" (same UNet
 * block names, no text-encoder layers and no metadata to tell them apart).
 */
export type LoraBase = 'sd1' | 'sd2' | 'sd1-or-sd2' | 'sdxl' | 'sd3' | 'flux' | 'unknown';

export interface SafetensorsClassification {
  kind: SafetensorsKind;
  family: SdFamily;
  /** Human-readable reason, logged and shown when a file is rejected. */
  reason: string;
  /** sd-lora only: the base model the LoRA targets. */
  loraBase?: LoraBase;
  /** sd-lora only: an applies-differently caveat to show the user (DoRA). */
  loraWarning?: string;
}

/** Safetensors `__metadata__` (string values; SdCppImageModule.readSafetensorsHeader). */
export type SafetensorsMetadata = Record<string, string | undefined>;

/** File extensions the sd.cpp engine can open directly. */
export const SDCPP_CHECKPOINT_EXTENSIONS = ['.safetensors', '.ckpt'] as const;

export function isSdCppCandidateFileName(name: string): boolean {
  const lower = name.toLowerCase();
  return SDCPP_CHECKPOINT_EXTENSIONS.some(ext => lower.endsWith(ext));
}

const has = (keys: string[], pred: (k: string) => boolean): boolean => keys.some(pred);
const prefix = (keys: string[], p: string): boolean => has(keys, k => k.startsWith(p));

/**
 * Video diffusion models, by the tensor names stable-diffusion.cpp itself uses to detect them
 * (ModelLoader::get_sd_version, src/model_loader.cpp @ c92d73c). This app makes still images only.
 */
const VIDEO_MARKERS: ReadonlyArray<readonly [string, string]> = [
  ['model.diffusion_model.blocks.0.cross_attn.norm_k.', 'Wan'], // model_loader.cpp:536
  ['model.diffusion_model.txt_in.individual_token_refiner.', 'HunyuanVideo'], // :499
  ['model.diffusion_model.adaln_single.', 'LTX-Video'], // :529
  ['model.diffusion_model.patch_embedder.', 'Lingbot video'], // :544
  ['model.diffusion_model.video_patch_proj.', 'MiniMax video'], // :532
  ['model.diffusion_model.input_blocks.8.0.time_mixer.', 'Stable Video Diffusion'], // :568
];

function videoModelName(keys: string[]): string | null {
  for (const [marker, name] of VIDEO_MARKERS) {
    if (prefix(keys, marker)) return name;
  }
  return null;
}

function unetFamily(keys: string[]): SdFamily {
  // sd.cpp: double_blocks or single_transformer_blocks under model.diffusion_model = Flux (model_loader.cpp:452).
  if (
    prefix(keys, 'model.diffusion_model.double_blocks.') ||
    prefix(keys, 'model.diffusion_model.single_transformer_blocks.')
  ) {
    return 'flux';
  }
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

// ---- LoRA base model -------------------------------------------------------------------------
// Structural facts used below (UNet layouts, not naming conventions):
//  - SD 1.x / 2.x UNets have ONE transformer block per attention layer (transformer_blocks.0 only);
//    SDXL has 2 or 10, so a UNet LoRA key with transformer_blocks.1+ is SDXL.
//  - SD 1.x / 2.x put attention in the first down block (diffusers down_blocks.0.attentions,
//    LDM input_blocks.1.1 / 2.1); SDXL's first down block is ResBlocks only.
//  - SDXL has two text encoders (kohya lora_te1_/lora_te2_, diffusers text_encoder_2.); SD 1.x / 2.x
//    one (kohya lora_te_). Flux kohya LoRAs also use lora_te1_ (CLIP-L), so te1 alone decides nothing.
//  - SD 1.x text encoder = CLIP-L, 12 layers (0-11); SD 2.x = OpenCLIP-H (kohya: up to layer 22).
const UNET_KEY = /^(lora_unet_|lora_unet\.|unet\.|diffusion_model\.|model\.diffusion_model\.)/;
const FLUX_LORA_KEY = /(^|[._])(double_blocks|single_blocks|single_transformer_blocks)[._]/;
const SD3_LORA_KEY = /(^|[._])joint_blocks[._]/;
const SDXL_TE2_KEY = /^(lora_te2_|text_encoder_2\.)/;
const SDXL_DEPTH_KEY = /transformer_blocks[._][1-9]/;
const SD12_UNET_KEY = /(down_blocks[._]0[._]attentions[._]|input_blocks[._][12][._]1[._])/;
const SD12_TE_KEY = /^lora_te_/;
const TE_LAYER_KEY = /^(?:lora_te_|text_encoder\.).*text_model[._]encoder[._]layers[._](\d+)[._]/;

/** Training-tool metadata (kohya sd-scripts `ss_base_model_version`, SAI `modelspec.architecture`). */
function loraBaseFromMetadata(meta: SafetensorsMetadata | undefined): LoraBase | null {
  if (!meta) return null;
  for (const raw of [meta['modelspec.architecture'], meta.ss_base_model_version]) {
    const v = (raw ?? '').trim().toLowerCase();
    if (!v) continue;
    if (v.startsWith('stable-diffusion-xl') || v.startsWith('sdxl')) return 'sdxl';
    if (v.startsWith('stable-diffusion-v1') || v.startsWith('sd_v1')) return 'sd1';
    if (v.startsWith('stable-diffusion-v2') || v.startsWith('sd_v2')) return 'sd2';
    if (v.startsWith('stable-diffusion-3') || v.startsWith('sd3')) return 'sd3';
    if (v.startsWith('flux')) return 'flux';
  }
  if ((meta.ss_v2 ?? '').toLowerCase() === 'true') return 'sd2';
  return null;
}

/**
 * The base model an SD LoRA targets. Tensor keys first (structure cannot be stale); the training
 * metadata only fills in what the keys cannot tell (no strong marker, or SD 1.x vs 2.x).
 */
export function detectLoraBase(keys: string[], metadata?: SafetensorsMetadata): LoraBase {
  const meta = loraBaseFromMetadata(metadata);
  if (has(keys, k => FLUX_LORA_KEY.test(k))) return 'flux';
  if (has(keys, k => SD3_LORA_KEY.test(k))) return 'sd3';
  if (has(keys, k => SDXL_TE2_KEY.test(k) || (UNET_KEY.test(k) && SDXL_DEPTH_KEY.test(k)))) return 'sdxl';
  if (has(keys, k => SD12_TE_KEY.test(k) || (UNET_KEY.test(k) && SD12_UNET_KEY.test(k)))) {
    let maxLayer = -1;
    for (const k of keys) {
      const m = TE_LAYER_KEY.exec(k);
      if (m) maxLayer = Math.max(maxLayer, Number(m[1]));
    }
    if (maxLayer >= 12) return 'sd2';
    if (maxLayer >= 0) return 'sd1';
    return meta === 'sd1' || meta === 'sd2' ? meta : 'sd1-or-sd2';
  }
  return meta ?? 'unknown';
}

const KNOWN_MODEL_FAMILIES = new Set<string>(['sd1', 'sd2', 'sdxl', 'sd3', 'flux']);

/**
 * Whether a LoRA fits the loaded checkpoint. A mismatched LoRA is not an error inside sd.cpp: it
 * skips every tensor whose name or shape does not match (lora.hpp get_weight_diff / get_out_diff
 * "skip incompatible LoRA tensor", model_manager.cpp prepare_lora_sources) and the image renders
 * without it. 'unknown' = cannot be proven either way (no base detected, or a .ckpt checkpoint).
 */
export function loraCompatibility(
  base: LoraBase | undefined,
  modelFamily: string | undefined,
): 'ok' | 'mismatch' | 'unknown' {
  if (!base || base === 'unknown' || !modelFamily || !KNOWN_MODEL_FAMILIES.has(modelFamily)) return 'unknown';
  if (base === 'sd1-or-sd2') {
    // SD 1.x and 2.x share block names but differ in cross-attention width (768 vs 1024).
    return modelFamily === 'sd1' || modelFamily === 'sd2' ? 'unknown' : 'mismatch';
  }
  return base === modelFamily ? 'ok' : 'mismatch';
}

const FAMILY_LABELS: Record<string, string> = {
  sd1: 'SD 1.x',
  sd2: 'SD 2.x',
  'sd1-or-sd2': 'SD 1.x/2.x',
  sdxl: 'SDXL',
  sd3: 'SD3',
  flux: 'Flux',
};

/** Display label of a checkpoint family or LoRA base; 'unknown' / unset -> 'unknown base'. */
export function sdFamilyLabel(family: string | undefined): string {
  return (family && FAMILY_LABELS[family]) || 'unknown base';
}

export function classifySafetensorsKeys(
  keys: string[],
  metadata?: SafetensorsMetadata,
): SafetensorsClassification {
  if (keys.length === 0) {
    return { kind: 'unknown', family: 'unknown', reason: 'empty safetensors header' };
  }
  const video = videoModelName(keys);
  if (video) {
    return { kind: 'video', family: 'unknown', reason: `${video} video model` };
  }
  if (prefix(keys, 'model.diffusion_model.')) {
    const family = unetFamily(keys);
    return { kind: 'sd-checkpoint', family, reason: `UNet/DiT under model.diffusion_model.* (${family})` };
  }
  if (has(keys, isKohyaSdLoraKey)) {
    return sdLoraClassification(keys, metadata, 'kohya LoRA keys (lora_unet_/lora_te_)');
  }
  const loraKeys = has(keys, isLoraTensorKey);
  const sdLoraTarget =
    prefix(keys, 'unet.') ||
    prefix(keys, 'text_encoder') ||
    (prefix(keys, 'transformer.') && !has(keys, isLlmKey));
  if (loraKeys && sdLoraTarget) {
    return sdLoraClassification(keys, metadata, 'diffusers/PEFT LoRA keys (unet./text_encoder./transformer.)');
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
  // A diffusion model saved WITHOUT the single-file `model.diffusion_model.` prefix (ComfyUI
  // models/unet or diffusion_models, BFL flux1-*.safetensors): UNet/DiT only, no text encoder or VAE.
  // sd.cpp loads such a file only as a separate --diffusion-model; this app loads one single file.
  // `!loraKeys`: an unprefixed Flux LoRA (double_blocks.*.lora_down) is not a diffusion model.
  const diffusionOnly = loraKeys ? null : diffusionOnlyName(keys);
  if (diffusionOnly) {
    return {
      kind: 'sd-component',
      family: 'unknown',
      reason: `${diffusionOnly} diffusion model only (no text encoder or VAE inside) - needs a full single-file checkpoint`,
    };
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

/** Family of an unprefixed UNet / DiT file, or null when the keys are not one (same markers as unetFamily). */
function diffusionOnlyName(keys: string[]): string | null {
  if (prefix(keys, 'double_blocks.') || prefix(keys, 'single_transformer_blocks.')) return 'Flux';
  if (prefix(keys, 'joint_blocks.')) return 'SD3';
  if (prefix(keys, 'input_blocks.') && prefix(keys, 'output_blocks.')) {
    return prefix(keys, 'label_emb.') ? 'SDXL UNet' : 'SD UNet';
  }
  return null;
}

function unetFamilyOfLoraBase(base: LoraBase): SdFamily {
  return base === 'sd1-or-sd2' ? 'unknown' : base;
}

/**
 * Tensor suffixes sd.cpp @ c92d73c applies (name_conversion.cpp convert_tensor_name lora_suffix_map;
 * lora.hpp get_lora_weight_diff / get_raw_weight_diff / get_loha_weight_diff / get_lokr_weight_diff):
 * kohya/LoCon lora_down/lora_up(/lora_mid), diffusers/PEFT lora_A/lora_B, LyCORIS LoHa hada_*,
 * LoKr lokr_*, full diff. A LoRA with none of them (OFT, IA3, GLoRA, ...) would load and change nothing.
 */
const isAppliedLoraTensorKey = (k: string): boolean =>
  /[._](lora_down|lora_up|lora_A|lora_B|hada_w[12]_[ab]|lokr_w[12](_[ab])?|diff(_b)?)([._]|$)/.test(k) ||
  /\.lora\.(down|up)\.weight$/.test(k);

function sdLoraClassification(
  keys: string[],
  metadata: SafetensorsMetadata | undefined,
  what: string,
): SafetensorsClassification {
  if (!has(keys, isAppliedLoraTensorKey)) {
    return {
      kind: 'sd-lora-unsupported',
      family: 'unknown',
      reason: `${what} without lora_down/lora_up, lora_A/lora_B, LoHa (hada_*), LoKr (lokr_*) or diff tensors`,
    };
  }
  const loraBase = detectLoraBase(keys, metadata);
  // lora.hpp:370 "TODO: .dora_scale?": DoRA magnitude vectors are never read, only the low-rank part applies.
  const dora = has(keys, k => /[._]dora_scale$/.test(k));
  return {
    kind: 'sd-lora',
    family: unetFamilyOfLoraBase(loraBase),
    loraBase,
    reason: `${what}, base ${sdFamilyLabel(loraBase)}${dora ? ', DoRA (magnitude ignored)' : ''}`,
    ...(dora ? { loraWarning: 'DoRA LoRA: stable-diffusion.cpp applies only its low-rank part (dora_scale is ignored), so it looks weaker or different than in other apps.' } : {}),
  };
}
