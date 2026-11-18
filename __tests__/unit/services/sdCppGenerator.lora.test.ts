/**
 * sdCppGeneratorService LoRA path: enabled LoRAs reach native as {path, weight}; a LoRA whose base
 * model provably differs from the loaded checkpoint is refused BEFORE the native call (sd.cpp would
 * skip it silently and return an image without it); an undetected base is not refused.
 */
const mockHeaders: Record<string, { keys: string[]; metadata?: Record<string, string> }> = {};
const mockGenerate = jest.fn(async (_p: Record<string, unknown>) => ({ id: 'i', imagePath: '/o.png', width: 512, height: 512, seed: 1 }));
const mockReadHeader = jest.fn(async (path: string) => {
  const h = mockHeaders[path];
  if (!h) throw new Error('no such file');
  return h;
});
let mockLoadedPath: string | null = '/m/sd15.safetensors';

jest.mock('react-native', () => ({
  Platform: { OS: 'android', select: (o: any) => o.android ?? o.default },
  NativeEventEmitter: jest.fn().mockImplementation(() => ({ addListener: () => ({ remove: jest.fn() }) })),
  NativeModules: {
    SdCppImageModule: {
      readSafetensorsHeader: (p: string) => mockReadHeader(p),
      getLoadedModelPath: async () => mockLoadedPath,
      generateImage: (p: Record<string, unknown>) => mockGenerate(p),
      addListener: jest.fn(),
      removeListeners: jest.fn(),
    },
  },
}));

import { sdCppGeneratorService, SdCppLoraMismatchError } from '../../../src/services/sdCppGenerator';

const SD15 = ['model.diffusion_model.input_blocks.0.0.weight', 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'];
const SD15_LORA = ['lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight', 'lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight'];
const SDXL_LORA = ['lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight'];

beforeEach(() => {
  jest.clearAllMocks();
  for (const k of Object.keys(mockHeaders)) delete mockHeaders[k];
  mockHeaders['/m/sd15.safetensors'] = { keys: SD15 };
  mockLoadedPath = '/m/sd15.safetensors';
  for (const p of ['/m/sd15.safetensors', '/l/detail.safetensors', '/l/xl_style.safetensors', '/l/mystery.safetensors']) {
    sdCppGeneratorService.forgetFile(p);
  }
});

const base = { prompt: 'a cat', width: 512, height: 512, steps: 4 } as any;

describe('sd.cpp LoRA generation path', () => {
  it('matching LoRA: native receives path + weight unchanged', async () => {
    mockHeaders['/l/detail.safetensors'] = { keys: SD15_LORA };
    await sdCppGeneratorService.generateImage({ ...base, loras: [{ path: '/l/detail.safetensors', weight: 0.7 }] });
    expect(mockGenerate).toHaveBeenCalledTimes(1);
    expect(mockGenerate.mock.calls[0][0].loras).toEqual([{ path: '/l/detail.safetensors', weight: 0.7 }]);
  });

  it('SDXL LoRA on an SD 1.x checkpoint: typed refusal, no native call, names the LoRA', async () => {
    mockHeaders['/l/xl_style.safetensors'] = { keys: SDXL_LORA };
    const run = sdCppGeneratorService.generateImage({ ...base, loras: [{ path: '/l/xl_style.safetensors', weight: 1 }] });
    await expect(run).rejects.toBeInstanceOf(SdCppLoraMismatchError);
    await expect(sdCppGeneratorService.generateImage({ ...base, loras: [{ path: '/l/xl_style.safetensors', weight: 1 }] }))
      .rejects.toMatchObject({ code: 'ERR_SDCPP_LORA_MISMATCH', message: expect.stringMatching(/"xl_style" is for SDXL.*SD 1\.x/) });
    expect(mockGenerate).not.toHaveBeenCalled();
    // The refusal released the run: a following request is not "already in progress".
    expect(await sdCppGeneratorService.isGenerating()).toBe(false);
  });

  it('undetected LoRA base or unreadable header is not refused (native decides)', async () => {
    mockHeaders['/l/mystery.safetensors'] = { keys: ['lora_unet_foo.lora_down.weight'] };
    await sdCppGeneratorService.generateImage({ ...base, loras: [{ path: '/l/mystery.safetensors', weight: 1 }, { path: '/l/missing.safetensors', weight: 1 }] });
    expect(mockGenerate).toHaveBeenCalledTimes(1);
  });

  it('a .ckpt checkpoint (no header) never blocks a LoRA', async () => {
    mockLoadedPath = '/m/old.ckpt';
    mockHeaders['/l/xl_style.safetensors'] = { keys: SDXL_LORA };
    await sdCppGeneratorService.generateImage({ ...base, loras: [{ path: '/l/xl_style.safetensors', weight: 1 }] });
    expect(mockGenerate).toHaveBeenCalledTimes(1);
    expect(mockReadHeader).not.toHaveBeenCalledWith('/m/old.ckpt');
  });

  it('no LoRA: no header read at all', async () => {
    await sdCppGeneratorService.generateImage({ ...base });
    expect(mockReadHeader).not.toHaveBeenCalled();
    expect(mockGenerate.mock.calls[0][0].loras).toEqual([]);
  });

  it('header family is cached per path until forgotten', async () => {
    mockHeaders['/l/detail.safetensors'] = { keys: SD15_LORA };
    await sdCppGeneratorService.fileFamily('/l/detail.safetensors');
    await sdCppGeneratorService.fileFamily('/l/detail.safetensors');
    expect(mockReadHeader).toHaveBeenCalledTimes(1);
    sdCppGeneratorService.forgetFile('/l/detail.safetensors');
    mockHeaders['/l/detail.safetensors'] = { keys: SDXL_LORA };
    expect((await sdCppGeneratorService.fileFamily('/l/detail.safetensors')).loraBase).toBe('sdxl');
  });
});
