/**
 * TAESD plumbing: settings.imageSdTaesd -> sdCppLoadConfigFromSettings -> loadModel sends
 * taesdPath (read by SdCppImageModule.loadModel -> nativeLoadEx -> sd_ctx_params.taesd_path),
 * picked by the checkpoint's header family; no file / no family = no field (full VAE).
 * loadConfigDiffers sees the switch, so a toggle reloads the model.
 */
import RNFS from 'react-native-fs';

const mockLoadModel = jest.fn(async (_params: Record<string, unknown>) => true);
const mockHeader = jest.fn(async (_path: string) => ({ keys: [] as string[], metadata: {} }));

jest.mock('react-native', () => ({
  Platform: { OS: 'android', select: (o: any) => o.android ?? o.default },
  NativeEventEmitter: jest.fn().mockImplementation(() => ({ addListener: () => ({ remove: jest.fn() }) })),
  NativeModules: {
    SdCppImageModule: {
      loadModel: (p: Record<string, unknown>) => mockLoadModel(p),
      readSafetensorsHeader: (p: string) => mockHeader(p),
      addListener: jest.fn(),
      removeListeners: jest.fn(),
    },
  },
}));

import { sdCppGeneratorService, sdCppLoadConfigFromSettings } from '../../../src/services/sdCppGenerator';

const SDXL_KEYS = ['model.diffusion_model.label_emb.0.0.weight', 'model.diffusion_model.input_blocks.0.0.weight'];
const SD1_KEYS = ['model.diffusion_model.input_blocks.0.0.weight', 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'];
const SD3_KEYS = ['model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight'];

beforeEach(() => {
  mockLoadModel.mockClear();
  mockHeader.mockReset();
  (RNFS.exists as jest.Mock).mockReset();
  (RNFS.exists as jest.Mock).mockResolvedValue(true);
});

describe('TAESD load param', () => {
  it('settings switch maps into the load config (default off)', () => {
    expect(sdCppLoadConfigFromSettings({})).toMatchObject({ taesd: false });
    expect(sdCppLoadConfigFromSettings({ imageSdTaesd: true })).toMatchObject({ taesd: true });
  });

  it('SDXL checkpoint + TAESD on + file present -> taesdPath = image_vae/taesdxl.safetensors', async () => {
    mockHeader.mockResolvedValue({ keys: SDXL_KEYS, metadata: {} });
    await sdCppGeneratorService.loadModel('/models/xl.safetensors', 4, { weightType: 'auto', flashAttn: true, taesd: true });
    expect(mockLoadModel.mock.calls[0][0]).toMatchObject({ taesdPath: '/mock/documents/image_vae/taesdxl.safetensors' });
    expect(RNFS.exists).toHaveBeenCalledWith('/mock/documents/image_vae/taesdxl.safetensors');
  });

  it('SD 1.x checkpoint -> taesd_sd1.safetensors', async () => {
    mockHeader.mockResolvedValue({ keys: SD1_KEYS, metadata: {} });
    await sdCppGeneratorService.loadModel('/models/sd15.safetensors', 4, { weightType: 'auto', flashAttn: true, taesd: true });
    expect(mockLoadModel.mock.calls[0][0]).toMatchObject({ taesdPath: '/mock/documents/image_vae/taesd_sd1.safetensors' });
  });

  it('TAESD file missing -> no taesdPath (full VAE)', async () => {
    mockHeader.mockResolvedValue({ keys: SD1_KEYS, metadata: {} });
    (RNFS.exists as jest.Mock).mockResolvedValue(false);
    await sdCppGeneratorService.loadModel('/models/sd15b.safetensors', 4, { weightType: 'auto', flashAttn: true, taesd: true });
    expect(mockLoadModel.mock.calls[0][0]).not.toHaveProperty('taesdPath');
  });

  it('family without TAESD (SD3) -> no taesdPath', async () => {
    mockHeader.mockResolvedValue({ keys: SD3_KEYS, metadata: {} });
    await sdCppGeneratorService.loadModel('/models/sd3.safetensors', 4, { weightType: 'auto', flashAttn: true, taesd: true });
    expect(mockLoadModel.mock.calls[0][0]).not.toHaveProperty('taesdPath');
  });

  it('TAESD off -> no header read, no taesdPath', async () => {
    await sdCppGeneratorService.loadModel('/models/other.safetensors', 4, { weightType: 'auto', flashAttn: true });
    expect(mockHeader).not.toHaveBeenCalled();
    expect(mockLoadModel.mock.calls[0][0]).not.toHaveProperty('taesdPath');
  });

  it('loadConfigDiffers: toggling TAESD needs a reload; unset equals off', async () => {
    await sdCppGeneratorService.loadModel('/models/other.safetensors', 4, { weightType: 'auto', flashAttn: true });
    expect(sdCppGeneratorService.loadConfigDiffers({ weightType: 'auto', flashAttn: true, taesd: false })).toBe(false);
    expect(sdCppGeneratorService.loadConfigDiffers({ weightType: 'auto', flashAttn: true, taesd: true })).toBe(true);
  });
});
