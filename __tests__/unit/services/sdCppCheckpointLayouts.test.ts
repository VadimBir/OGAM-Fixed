/**
 * The user's own image checkpoint must import: real single-file layouts (SD 1.5 with EMA, SDXL /
 * Pony / Illustrious, a checkpoint without VAE or CLIP, SD3.5 with T5, an AIO whose text encoder is
 * an LLM) classify as sd-checkpoint, never 'llm' (text refusal) or a rejected kind. A UNet / DiT file
 * without the single-file prefix is named as diffusion-only instead of "no known layout". A file whose
 * name lacks .safetensors is recognised by its header bytes.
 */
import { classifySafetensorsKeys } from '../../../src/services/sdCppModelDetection';
import { isSafetensorsPrefix } from '../../../src/services/textLoraGguf';
import { buildSafetensorsHeader } from '../../utils/ggufFixture';

const SD15_EMA = [
  'alphas_cumprod',
  'model_ema.decay',
  'model_ema.diffusion_modelinput_blocks00weight',
  'model.diffusion_model.input_blocks.0.0.weight',
  'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight',
  'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight',
  'first_stage_model.decoder.conv_in.weight',
];
const SDXL_PONY = [
  'conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight',
  'conditioner.embedders.1.model.token_embedding.weight',
  'denoiser.sigmas',
  'first_stage_model.encoder.down.0.block.0.conv1.weight',
  'model.diffusion_model.input_blocks.0.0.weight',
  'model.diffusion_model.label_emb.0.0.weight',
];
const NO_VAE_NO_CLIP = ['model.diffusion_model.input_blocks.0.0.weight', 'model.diffusion_model.out.2.weight'];
const SD35_T5 = [
  'model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight',
  'text_encoders.clip_l.transformer.text_model.embeddings.token_embedding.weight',
  'text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight',
];
const AIO_LLM_TEXT_ENCODER = [
  'model.diffusion_model.layers.0.attention.qkv.weight',
  'text_encoders.qwen3_4b.transformer.model.embed_tokens.weight',
  'text_encoders.qwen3_4b.transformer.model.layers.0.mlp.up_proj.weight',
  'vae.decoder.conv_in.weight',
];

describe('single-file checkpoints import as sd-checkpoint', () => {
  it.each([
    ['SD 1.5 with EMA', SD15_EMA, 'sd1'],
    ['SDXL / Pony / Illustrious', SDXL_PONY, 'sdxl'],
    ['UNet only under model.diffusion_model (no VAE, no CLIP)', NO_VAE_NO_CLIP, 'unknown'],
    ['SD3.5 with T5', SD35_T5, 'sd3'],
    ['AIO with an LLM text encoder (model.layers.*, embed_tokens)', AIO_LLM_TEXT_ENCODER, 'unknown'],
  ])('%s', (_label, keys, family) => {
    expect(classifySafetensorsKeys(keys)).toMatchObject({ kind: 'sd-checkpoint', family });
  });
});

describe('diffusion-model-only files (no model.diffusion_model. prefix)', () => {
  it.each([
    ['SDXL UNet', ['input_blocks.0.0.weight', 'label_emb.0.0.weight', 'middle_block.0.in_layers.0.weight', 'output_blocks.0.0.in_layers.0.weight']],
    ['SD UNet', ['input_blocks.0.0.weight', 'middle_block.0.in_layers.0.weight', 'output_blocks.0.0.in_layers.0.weight']],
    ['Flux', ['double_blocks.0.img_attn.qkv.weight', 'single_blocks.0.linear1.weight', 'img_in.weight']],
    ['SD3', ['joint_blocks.0.x_block.attn.qkv.weight', 'x_embedder.proj.weight']],
  ])('%s is named as a diffusion model without text encoder / VAE', (name, keys) => {
    const result = classifySafetensorsKeys(keys);
    expect(result.kind).toBe('sd-component');
    expect(result.reason).toContain(`${name} diffusion model only`);
  });

  it('an unprefixed Flux LoRA is not called a diffusion model', () => {
    const result = classifySafetensorsKeys(['double_blocks.0.img_attn.proj.lora_down.weight', 'double_blocks.0.img_attn.proj.lora_up.weight']);
    expect(result.kind).not.toBe('sd-component');
  });
});

describe('isSafetensorsPrefix', () => {
  it('accepts a safetensors file start', () => {
    expect(isSafetensorsPrefix(buildSafetensorsHeader(SDXL_PONY).subarray(0, 9))).toBe(true);
  });

  it('rejects GGUF, zip, short and absurd-length starts', () => {
    expect(isSafetensorsPrefix(new Uint8Array(Buffer.from('GGUF\u0003\u0000\u0000\u0000\u0000', 'latin1')))).toBe(false);
    expect(isSafetensorsPrefix(new Uint8Array(Buffer.from('PK\u0003\u0004abcde', 'latin1')))).toBe(false);
    expect(isSafetensorsPrefix(new Uint8Array([2, 0, 0, 0, 0, 0, 0, 0]))).toBe(false);
    const huge = new Uint8Array(9);
    huge[5] = 1; // 2^40 bytes of header
    huge[8] = 0x7b;
    expect(isSafetensorsPrefix(huge)).toBe(false);
  });
});
