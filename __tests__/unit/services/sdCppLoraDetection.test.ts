/**
 * Image LoRA / checkpoint detection for stable-diffusion.cpp: video refusal, Flux markers, LoRA base
 * model (keys first, training metadata second) and the LoRA-vs-checkpoint compatibility verdict.
 *
 * Priority: P0 - sd.cpp silently skips a mismatched LoRA (lora.hpp "skip incompatible LoRA tensor"),
 * so a wrong verdict here either blocks a valid LoRA or lets an image render without the LoRA.
 */
import {
  classifySafetensorsKeys,
  detectLoraBase,
  loraCompatibility,
  sdFamilyLabel,
} from '../../../src/services/sdCppModelDetection';

describe('checkpoint detection (sd.cpp model_loader.cpp get_sd_version markers)', () => {
  it.each([
    ['model.diffusion_model.blocks.0.cross_attn.norm_k.weight', 'Wan'],
    ['model.diffusion_model.txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.weight', 'HunyuanVideo'],
    ['model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias', 'LTX-Video'],
    ['model.diffusion_model.input_blocks.8.0.time_mixer.mix_factor', 'Stable Video Diffusion'],
  ])('refuses a video model: %s', (key, name) => {
    const c = classifySafetensorsKeys([key, 'model.diffusion_model.double_blocks.0.img_attn.qkv.weight']);
    expect(c.kind).toBe('video');
    expect(c.reason).toContain(name);
  });

  it('Flux by single_transformer_blocks too (sd.cpp model_loader.cpp:452)', () => {
    expect(classifySafetensorsKeys(['model.diffusion_model.single_transformer_blocks.0.attn.to_q.weight']))
      .toMatchObject({ kind: 'sd-checkpoint', family: 'flux' });
  });

  it('SD1 / SDXL checkpoints are unchanged', () => {
    expect(classifySafetensorsKeys(['model.diffusion_model.input_blocks.0.0.weight', 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight']))
      .toMatchObject({ kind: 'sd-checkpoint', family: 'sd1' });
    expect(classifySafetensorsKeys(['model.diffusion_model.label_emb.0.0.weight', 'conditioner.embedders.1.model.ln_final.weight']))
      .toMatchObject({ kind: 'sd-checkpoint', family: 'sdxl' });
  });
});

describe('detectLoraBase', () => {
  const sd1Kohya = [
    'lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight',
    'lora_te_text_model_encoder_layers_11_mlp_fc1.lora_down.weight',
  ];

  it('SD 1.x kohya: first-block attention + text encoder layers 0-11', () => {
    expect(detectLoraBase(sd1Kohya)).toBe('sd1');
  });

  it('SD 2.x kohya: text encoder layer >= 12 (OpenCLIP-H)', () => {
    expect(detectLoraBase([...sd1Kohya, 'lora_te_text_model_encoder_layers_22_mlp_fc1.lora_down.weight'])).toBe('sd2');
  });

  it('UNet-only SD 1.x/2.x LoRA: keys cannot tell; metadata can', () => {
    const unetOnly = ['lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight'];
    expect(detectLoraBase(unetOnly)).toBe('sd1-or-sd2');
    expect(detectLoraBase(unetOnly, { ss_base_model_version: 'sd_v1' })).toBe('sd1');
    expect(detectLoraBase(unetOnly, { ss_v2: 'True' })).toBe('sd2');
  });

  it('SDXL: kohya te2, sgm transformer depth >= 1, diffusers text_encoder_2', () => {
    expect(detectLoraBase(['lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight'])).toBe('sdxl');
    expect(detectLoraBase(['lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.lora_down.weight'])).toBe('sdxl');
    expect(detectLoraBase(['unet.down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_k.lora_A.weight'])).toBe('sdxl');
    expect(detectLoraBase(['text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight'])).toBe('sdxl');
  });

  it('Flux and SD3 win over their CLIP-L te1 keys', () => {
    expect(detectLoraBase(['lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight', 'lora_unet_double_blocks_0_img_attn_proj.lora_down.weight'])).toBe('flux');
    expect(detectLoraBase(['transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight'])).toBe('flux');
    expect(detectLoraBase(['lora_unet_joint_blocks_0_x_block_attn_qkv.lora_down.weight'])).toBe('sd3');
  });

  it('keys outrank stale metadata; metadata alone decides when keys say nothing', () => {
    expect(detectLoraBase(['lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight'], { ss_base_model_version: 'sd_v1' })).toBe('sdxl');
    expect(detectLoraBase(['lora_unet_foo.lora_down.weight'], { 'modelspec.architecture': 'stable-diffusion-xl-v1-base/lora' })).toBe('sdxl');
    expect(detectLoraBase(['lora_unet_foo.lora_down.weight'])).toBe('unknown');
  });

  it('classifySafetensorsKeys carries the base on sd-lora results', () => {
    expect(classifySafetensorsKeys(sd1Kohya)).toMatchObject({ kind: 'sd-lora', loraBase: 'sd1', family: 'sd1' });
    expect(classifySafetensorsKeys(['lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight']))
      .toMatchObject({ kind: 'sd-lora', loraBase: 'sd1-or-sd2', family: 'unknown' });
    // LLM LoRAs still never reach the image engine.
    expect(classifySafetensorsKeys(['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight']).kind).toBe('llm');
  });
});

describe('LoRA formats sd.cpp applies (name_conversion.cpp lora_suffix_map, lora.hpp)', () => {
  it.each([
    ['kohya LoRA / LoCon', 'lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight'],
    ['diffusers/PEFT', 'unet.down_blocks.0.attentions.0.proj_in.lora_A.default.weight'],
    ['LyCORIS LoHa', 'lora_unet_down_blocks_0_attentions_0_proj_in.hada_w1_a'],
    ['LyCORIS LoKr', 'lora_unet_down_blocks_0_attentions_0_proj_in.lokr_w1'],
    ['full diff', 'lora_unet_down_blocks_0_attentions_0_proj_in.diff'],
  ])('%s is an sd-lora', (_name, key) => {
    expect(classifySafetensorsKeys([key]).kind).toBe('sd-lora');
  });

  it('a LoRA with no applicable tensor type (e.g. OFT) is refused, not silently added', () => {
    const c = classifySafetensorsKeys(['lora_unet_down_blocks_0_attentions_0_proj_in.oft_blocks', 'lora_unet_down_blocks_0_attentions_0_proj_in.alpha']);
    expect(c.kind).toBe('sd-lora-unsupported');
  });

  it('DoRA is added with a warning (dora_scale unread, lora.hpp:370)', () => {
    const c = classifySafetensorsKeys(['lora_unet_x.lora_down.weight', 'lora_unet_x.dora_scale']);
    expect(c.kind).toBe('sd-lora');
    expect(c.loraWarning).toMatch(/dora_scale is ignored/);
  });
});

describe('loraCompatibility', () => {
  it('ok / mismatch / unknown', () => {
    expect(loraCompatibility('sd1', 'sd1')).toBe('ok');
    expect(loraCompatibility('sd1', 'sdxl')).toBe('mismatch');
    expect(loraCompatibility('sdxl', 'sd1')).toBe('mismatch');
    expect(loraCompatibility('sd2', 'sd1')).toBe('mismatch');
    expect(loraCompatibility('sd1-or-sd2', 'sdxl')).toBe('mismatch');
    expect(loraCompatibility('sd1-or-sd2', 'sd1')).toBe('unknown');
    expect(loraCompatibility('unknown', 'sd1')).toBe('unknown');
    expect(loraCompatibility('sdxl', 'unknown')).toBe('unknown');
    expect(loraCompatibility('sdxl', undefined)).toBe('unknown');
  });

  it('labels', () => {
    expect(sdFamilyLabel('sd1')).toBe('SD 1.x');
    expect(sdFamilyLabel('sdxl')).toBe('SDXL');
    expect(sdFamilyLabel(undefined)).toBe('unknown base');
  });
});
