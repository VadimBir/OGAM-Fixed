/**
 * stable-diffusion.cpp engine: format detection, per-engine param signature, router selection.
 *
 * Priority: P0 - a wrong route sends an LLM file to the image engine, or QNN params to sd.cpp.
 */
import {
  classifySafetensorsKeys,
  isSdCppCandidateFileName,
} from '../../../src/services/sdCppModelDetection';
import {
  resolveEngineImageRequest,
  snapSdCppDimension,
} from '../../../src/services/imageEngineParams';

const mockLocalDream = {
  isAvailable: jest.fn(() => true),
  isModelLoaded: jest.fn(async () => true),
  getLoadedModelPath: jest.fn(async () => '/qnn/model'),
  getLoadedThreads: jest.fn(() => 4),
  loadModel: jest.fn(async () => true),
  unloadModel: jest.fn(async () => true),
  generateImage: jest.fn(async () => ({ id: 'ld' })),
  cancelGeneration: jest.fn(async () => true),
  isGenerating: jest.fn(async () => false),
  hasKernelCache: jest.fn(async () => false),
  getGeneratedImages: jest.fn(async () => []),
  deleteGeneratedImage: jest.fn(async () => true),
  clearOpenCLCache: jest.fn(async () => 0),
  getConstants: jest.fn(() => ({})),
};
const mockSdCpp = {
  isAvailable: jest.fn(() => true),
  isModelLoaded: jest.fn(async () => true),
  getLoadedModelPath: jest.fn(async () => '/sd/model.safetensors'),
  getLoadedThreads: jest.fn(() => 6),
  loadModel: jest.fn(async () => true),
  unloadModel: jest.fn(async () => true),
  generateImage: jest.fn(async () => ({ id: 'sd' })),
  cancelGeneration: jest.fn(async () => true),
  isGenerating: jest.fn(async () => false),
};
jest.mock('../../../src/services/localDreamGenerator', () => ({ localDreamGeneratorService: mockLocalDream }));
jest.mock('../../../src/services/sdCppGenerator', () => ({ sdCppGeneratorService: mockSdCpp }));

const SD1 = [
  'model.diffusion_model.input_blocks.0.0.weight',
  'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight',
  'first_stage_model.decoder.conv_in.weight',
];

describe('classifySafetensorsKeys', () => {
  it('detects SD1 / SD2 / SDXL / Flux / SD3 single-file checkpoints', () => {
    expect(classifySafetensorsKeys(SD1)).toMatchObject({ kind: 'sd-checkpoint', family: 'sd1' });
    expect(classifySafetensorsKeys(['model.diffusion_model.out.2.weight', 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight']))
      .toMatchObject({ kind: 'sd-checkpoint', family: 'sd2' });
    expect(classifySafetensorsKeys(['model.diffusion_model.label_emb.0.0.weight', 'conditioner.embedders.1.model.ln_final.weight']))
      .toMatchObject({ kind: 'sd-checkpoint', family: 'sdxl' });
    expect(classifySafetensorsKeys(['model.diffusion_model.double_blocks.0.img_attn.qkv.weight']))
      .toMatchObject({ kind: 'sd-checkpoint', family: 'flux' });
    expect(classifySafetensorsKeys(['model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight']))
      .toMatchObject({ kind: 'sd-checkpoint', family: 'sd3' });
  });

  it('never routes LLM weights or LLM LoRAs to the image engine', () => {
    expect(classifySafetensorsKeys(['model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight', 'lm_head.weight']).kind).toBe('llm');
    expect(classifySafetensorsKeys(['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight']).kind).toBe('llm');
  });

  it('detects kohya and diffusers SD LoRAs', () => {
    expect(classifySafetensorsKeys(['lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight', 'lora_te_text_model_encoder_layers_0_mlp_fc1.alpha']).kind).toBe('sd-lora');
    expect(classifySafetensorsKeys(['unet.down_blocks.0.attentions.0.proj_in.lora_A.weight']).kind).toBe('sd-lora');
  });

  it('rejects standalone VAE / diffusers components / unknown layouts', () => {
    expect(classifySafetensorsKeys(['encoder.down.0.block.0.conv1.weight', 'decoder.up.0.block.0.conv1.weight']).kind).toBe('sd-vae');
    expect(classifySafetensorsKeys(['down_blocks.0.resnets.0.conv1.weight']).kind).toBe('sd-component');
    expect(classifySafetensorsKeys(['foo.bar']).kind).toBe('unknown');
    expect(classifySafetensorsKeys([]).kind).toBe('unknown');
  });

  it('only .safetensors/.ckpt are sd.cpp import candidates', () => {
    expect(isSdCppCandidateFileName('dreamshaper_8.SAFETENSORS')).toBe(true);
    expect(isSdCppCandidateFileName('v1-5.ckpt')).toBe(true);
    expect(isSdCppCandidateFileName('llama.gguf')).toBe(false);
  });
});

describe('resolveEngineImageRequest (param signature per engine)', () => {
  const loras = [
    { id: 'a', name: 'a', path: '/l/a.safetensors', weight: 0.8, enabled: true },
    { id: 'b', name: 'b', path: '/l/b.safetensors', weight: 1, enabled: false },
    { id: 'c', name: 'c', path: '/l/c.safetensors', weight: 0, enabled: true },
  ];

  it('sdcpp: free W×H on the 64 grid, no 256 floor, sampler/scheduler/enabled LoRAs', () => {
    const r = resolveEngineImageRequest(
      { backend: 'sdcpp' },
      { imageWidth: 128, imageHeight: 768, imageSampler: 'dpm++2m', imageScheduler: 'karras', imageLoras: loras },
      {},
      512,
    );
    expect(r).toEqual({
      width: 128,
      height: 768,
      engineOptions: { sampler: 'dpm++2m', scheduler: 'karras', loras: [{ path: '/l/a.safetensors', weight: 0.8 }], vaeTiling: true },
    });
  });

  it('sdcpp: per-request overrides win; bad sizes snap/clamp', () => {
    const r = resolveEngineImageRequest(
      { backend: 'sdcpp' },
      { imageWidth: 512, imageHeight: 512, imageSampler: 'euler' },
      { sampler: 'euler_a', scheduler: 'ays', width: 300, height: 5000 },
      512,
    );
    expect(r.width).toBe(320);
    expect(r.height).toBe(1024);
    expect(r.engineOptions).toMatchObject({ sampler: 'euler_a', scheduler: 'ays' });
    expect(snapSdCppDimension(NaN)).toBe(512);
    expect(snapSdCppDimension(10)).toBe(64);
  });

  it('LocalDream: keeps the square policy size, never sends scheduler/LoRA, sampler only dpm|euler_a', () => {
    const settings = { imageWidth: 128, imageHeight: 768, imageSampler: 'euler_a', imageScheduler: 'karras', imageLoras: loras };
    expect(resolveEngineImageRequest({ backend: 'qnn' }, settings, {}, 256)).toEqual({
      width: 256,
      height: 256,
      engineOptions: { sampler: 'euler_a' },
    });
    expect(resolveEngineImageRequest({ backend: 'mnn' }, { ...settings, imageSampler: 'dpm++2m' }, {}, 512))
      .toEqual({ width: 512, height: 512, engineOptions: {} });
  });
});

describe('imageEngineRouter', () => {
  let router: typeof import('../../../src/services/imageEngineRouter').imageEngineRouter;
  beforeEach(() => {
    jest.clearAllMocks();
    jest.isolateModules(() => {
      router = require('../../../src/services/imageEngineRouter').imageEngineRouter;
    });
  });

  it('non-sdcpp load goes to LocalDream with the unchanged option shape', async () => {
    await router.loadModel('/qnn/model', 4, { backend: 'auto', cpuOnly: false });
    expect(mockLocalDream.loadModel).toHaveBeenCalledWith('/qnn/model', 4, { backend: 'auto', cpuOnly: false });
    expect(mockSdCpp.loadModel).not.toHaveBeenCalled();
    expect(mockSdCpp.unloadModel).not.toHaveBeenCalled();
    expect(router.getActiveEngine()).toBe('localdream');
    await router.generateImage({ prompt: 'x' });
    expect(mockLocalDream.generateImage).toHaveBeenCalled();
  });

  it('sdcpp load frees LocalDream, then every call goes to sd.cpp; switching back frees sd.cpp', async () => {
    await router.loadModel('/sd/model.safetensors', 6, { backend: 'sdcpp', sdcpp: { weightType: 'q4_0', flashAttn: true } });
    expect(mockLocalDream.unloadModel).toHaveBeenCalledTimes(1);
    expect(mockSdCpp.loadModel).toHaveBeenCalledWith('/sd/model.safetensors', 6, { weightType: 'q4_0', flashAttn: true });
    expect(await router.getLoadedModelPath()).toBe('/sd/model.safetensors');
    expect(router.getLoadedThreads()).toBe(6);
    expect(await router.hasKernelCache('/sd/model.safetensors')).toBe(true);
    const params = { prompt: 'x', sampler: 'euler_a', scheduler: 'karras', loras: [{ path: '/l', weight: 1 }] };
    await router.generateImage(params);
    expect(mockSdCpp.generateImage).toHaveBeenCalledWith(params, undefined);
    await router.cancelGeneration();
    expect(mockSdCpp.cancelGeneration).toHaveBeenCalled();

    await router.loadModel('/qnn/model', 4, { backend: 'auto' });
    expect(mockSdCpp.unloadModel).toHaveBeenCalledTimes(1);
    expect(router.getActiveEngine()).toBe('localdream');
  });

  it('gallery store stays LocalDream (shared generated_images dir)', async () => {
    await router.loadModel('/sd/model.safetensors', 6, { backend: 'sdcpp' });
    await router.deleteGeneratedImage('id');
    expect(mockLocalDream.deleteGeneratedImage).toHaveBeenCalledWith('id', undefined);
  });
});
