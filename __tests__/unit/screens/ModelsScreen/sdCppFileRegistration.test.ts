import RNFS from 'react-native-fs';
import { resumeImageDownload } from '../../../../src/screens/ModelsScreen/imageDownloadResume';
import { registerSdFileInPlace } from '../../../../src/services/sdCppFileRegistration';

const mockFinalize = jest.fn();
const mockGetDownloaded = jest.fn();
const mockAddDownloaded = jest.fn();
const mockClassify = jest.fn();
const mockRemove = jest.fn();
const mockSetStatus = jest.fn();
const mockUpdateSettings = jest.fn();
let mockImageModels: Array<{ id: string; sdFamily?: string }> = [];

jest.mock('../../../../src/services/civitaiDownload', () => ({
  finalizeSdCppFileDownload: (...a: any[]) => mockFinalize(...a),
}));

jest.mock('../../../../src/services', () => ({
  modelManager: {
    getImageModelsDirectory: () => '/docs/image_models',
    getDownloadedImageModels: (...a: any[]) => mockGetDownloaded(...a),
    addDownloadedImageModel: (...a: any[]) => mockAddDownloaded(...a),
  },
  backgroundDownloadService: { moveCompletedDownload: jest.fn() },
}));

jest.mock('../../../../src/services/modelManager', () => ({
  modelManager: { addDownloadedImageModel: (...a: any[]) => mockAddDownloaded(...a) },
}));

jest.mock('../../../../src/services/sdCppGenerator', () => ({
  sdCppGeneratorService: { classifySafetensors: (...a: any[]) => mockClassify(...a), isAvailable: () => true, forgetFile: jest.fn() },
}));

jest.mock('../../../../src/stores/downloadStore', () => ({
  useDownloadStore: { getState: () => ({ remove: mockRemove, setStatus: mockSetStatus }) },
}));

jest.mock('../../../../src/stores', () => ({
  useAppStore: { getState: () => ({ settings: { imageLoras: [] }, updateSettings: mockUpdateSettings, downloadedImageModels: mockImageModels }) },
}));

jest.mock('../../../../src/services/imageDownloadActions', () => ({
  registerAndNotify: jest.fn(),
  proceedWithDownload: jest.fn(),
}));

const deps = () => ({
  addDownloadedImageModel: jest.fn(),
  activeImageModelId: null as string | null,
  setActiveImageModelId: jest.fn(),
  setAlertState: jest.fn(),
  triedImageGen: false,
});

beforeEach(() => {
  jest.clearAllMocks();
  mockGetDownloaded.mockResolvedValue([]);
  mockFinalize.mockResolvedValue(undefined);
  (RNFS.exists as jest.Mock).mockResolvedValue(false);
  (RNFS.mkdir as jest.Mock).mockResolvedValue(undefined);
  (RNFS.moveFile as jest.Mock).mockResolvedValue(undefined);
  (RNFS.unlink as jest.Mock).mockResolvedValue(undefined);
  (RNFS.stat as jest.Mock).mockResolvedValue({ size: 2_000_000_000 });
  mockImageModels = [];
});

const entry = {
  modelKey: 'image:civitai_101_11', downloadId: 'native-1', modelId: 'image:civitai_101_11',
  fileName: 'dream.safetensors', quantization: '', modelType: 'image' as const, status: 'processing' as const,
  bytesDownloaded: 1, totalBytes: 1, combinedTotalBytes: 1, progress: 1, createdAt: 1,
  metadataJson: JSON.stringify({ imageDownloadType: 'sdcpp-file', imageModelName: 'DreamShaper v8', imageModelFileName: 'dream.safetensors' }),
};

describe('resumeImageDownload sdcpp-file', () => {
  it('finalizes a Civitai single-file download after a restart', async () => {
    const d = deps();
    await resumeImageDownload(entry, d);
    expect(mockFinalize).toHaveBeenCalledWith({
      downloadId: 'native-1', modelId: 'civitai_101_11', fileName: 'dream.safetensors', displayName: 'DreamShaper v8', deps: d,
    });
  });

  it('only drops the stale row when the model is already registered', async () => {
    mockGetDownloaded.mockResolvedValue([{ id: 'civitai_101_11' }]);
    await resumeImageDownload(entry, deps());
    expect(mockFinalize).not.toHaveBeenCalled();
    expect(mockRemove).toHaveBeenCalledWith('image:civitai_101_11');
  });

  it('marks the row failed when finalization throws', async () => {
    mockFinalize.mockRejectedValue(new Error('move failed'));
    await resumeImageDownload(entry, deps());
    expect(mockSetStatus).toHaveBeenCalledWith('native-1', 'failed', { message: 'move failed' });
  });
});

describe('registerSdFileInPlace', () => {
  const opts = { path: '/d/m/f.safetensors', modelDir: '/d/m', fileName: 'f.safetensors', modelId: 'm', verb: 'downloaded' as const };

  it('registers a checkpoint as an sd.cpp model and selects it when none is active', async () => {
    mockClassify.mockResolvedValue({ kind: 'sd-checkpoint', family: 'sd1', reason: 'x' });
    const d = deps();
    const out = await registerSdFileInPlace({ ...opts, displayName: 'Dream', source: 'From Civitai' }, d);
    expect(out.kind).toBe('checkpoint');
    const model = mockAddDownloaded.mock.calls[0][0];
    expect(model).toMatchObject({ id: 'm', name: 'Dream', backend: 'sdcpp', sdFamily: 'sd1', modelPath: '/d/m/f.safetensors', size: 2_000_000_000 });
    expect(model.description).toMatch(/^From Civitai\./);
    expect(d.addDownloadedImageModel).toHaveBeenCalledWith(model);
    expect(d.setActiveImageModelId).toHaveBeenCalledWith('m');
    expect(d.setAlertState.mock.calls[0][0].message).toMatch(/downloaded/);
  });

  it('keeps the current selection when one exists', async () => {
    mockClassify.mockResolvedValue({ kind: 'sd-checkpoint', family: 'sdxl', reason: 'x' });
    const d = { ...deps(), activeImageModelId: 'other' };
    await registerSdFileInPlace(opts, d);
    expect(d.setActiveImageModelId).not.toHaveBeenCalled();
  });

  it('adds a LoRA to the LoRA list instead of the model list', async () => {
    mockClassify.mockResolvedValue({ kind: 'sd-lora', family: 'unknown', reason: 'kohya' });
    const out = await registerSdFileInPlace(opts, deps());
    expect(out.kind).toBe('lora');
    expect(RNFS.moveFile).toHaveBeenCalledWith('/d/m/f.safetensors', expect.stringMatching(/image_loras\/f\.safetensors$/));
    expect(mockUpdateSettings).toHaveBeenCalledWith({ imageLoras: [expect.objectContaining({ name: 'f', enabled: true })] });
    expect(mockAddDownloaded).not.toHaveBeenCalled();
  });

  it('removes and explains a file that is not an SD checkpoint', async () => {
    mockClassify.mockResolvedValue({ kind: 'llm', family: 'unknown', reason: 'llama keys' });
    const d = deps();
    const out = await registerSdFileInPlace(opts, d);
    expect(out.kind).toBe('rejected');
    expect(RNFS.unlink).toHaveBeenCalledWith('/d/m');
    expect(d.setAlertState.mock.calls[0][0].title).toBe('Not a Stable Diffusion checkpoint');
  });

  it('refuses a video model with its own title and a still-image hint', async () => {
    mockClassify.mockResolvedValue({ kind: 'video', family: 'unknown', reason: 'Wan video model' });
    const d = deps();
    const out = await registerSdFileInPlace(opts, d);
    expect(out).toEqual({ kind: 'rejected', reason: 'Wan video model' });
    expect(RNFS.unlink).toHaveBeenCalledWith('/d/m');
    expect(mockAddDownloaded).not.toHaveBeenCalled();
    expect(d.setAlertState.mock.calls[0][0].title).toBe('Video models are not supported');
    expect(d.setAlertState.mock.calls[0][0].message).toMatch(/still images/);
  });

  it('an LLM LoRA gets the GGUF-adapter hint, not the GGUF-weights hint', async () => {
    mockClassify.mockResolvedValue({ kind: 'llm', family: 'unknown', reason: 'LLM LoRA adapter (model.layers.* + lora)' });
    const d = deps();
    await registerSdFileInPlace(opts, d);
    expect(d.setAlertState.mock.calls[0][0].message).toMatch(/text-model \(LLM\) LoRA/);
    expect(d.setAlertState.mock.calls[0][0].message).toMatch(/convert_lora_to_gguf/);
  });

  it('a LoRA for another base than the selected checkpoint is added DISABLED and says why', async () => {
    mockClassify.mockResolvedValue({ kind: 'sd-lora', family: 'sdxl', loraBase: 'sdxl', reason: 'kohya' });
    mockImageModels = [{ id: 'sd15', sdFamily: 'sd1' }];
    const d = { ...deps(), activeImageModelId: 'sd15' };
    await registerSdFileInPlace(opts, d);
    expect(mockUpdateSettings).toHaveBeenCalledWith({ imageLoras: [expect.objectContaining({ enabled: false })] });
    expect(d.setAlertState.mock.calls[0][0].message).toMatch(/SDXL LoRA.*SD 1\.x.*added disabled/);
  });

  it('a matching LoRA is enabled and its base is named; an undetected base is flagged', async () => {
    mockClassify.mockResolvedValue({ kind: 'sd-lora', family: 'sd1', loraBase: 'sd1', reason: 'kohya' });
    mockImageModels = [{ id: 'sd15', sdFamily: 'sd1' }];
    const d = { ...deps(), activeImageModelId: 'sd15' };
    await registerSdFileInPlace(opts, d);
    expect(mockUpdateSettings).toHaveBeenCalledWith({ imageLoras: [expect.objectContaining({ enabled: true })] });
    expect(d.setAlertState.mock.calls[0][0].message).toMatch(/\(SD 1\.x\) is enabled/);

    mockClassify.mockResolvedValue({ kind: 'sd-lora', family: 'unknown', loraBase: 'unknown', reason: 'kohya' });
    const d2 = deps();
    await registerSdFileInPlace(opts, d2);
    expect(d2.setAlertState.mock.calls[0][0].message).toMatch(/could not be read from the file/);
  });

  it('an unsupported LoRA type is removed with its own title; a DoRA warning reaches the alert', async () => {
    mockClassify.mockResolvedValue({ kind: 'sd-lora-unsupported', family: 'unknown', reason: 'no applicable tensors' });
    const d = deps();
    expect((await registerSdFileInPlace(opts, d)).kind).toBe('rejected');
    expect(RNFS.unlink).toHaveBeenCalledWith('/d/m');
    expect(mockUpdateSettings).not.toHaveBeenCalled();
    expect(d.setAlertState.mock.calls[0][0].title).toBe('LoRA type not supported');

    mockClassify.mockResolvedValue({ kind: 'sd-lora', family: 'sd1', loraBase: 'sd1', reason: 'k', loraWarning: 'DoRA LoRA: x' });
    const d2 = deps();
    await registerSdFileInPlace(opts, d2);
    expect(d2.setAlertState.mock.calls[0][0].message).toMatch(/\n\nDoRA LoRA: x$/);
  });

  it('a checkpoint alert names the GPU engine (GPU-only libsdcpp_jni build)', async () => {
    mockClassify.mockResolvedValue({ kind: 'sd-checkpoint', family: 'sd1', reason: 'x' });
    const d = deps();
    await registerSdFileInPlace(opts, d);
    expect(d.setAlertState.mock.calls[0][0].message).toMatch(/\(GPU\)/);
    expect(d.setAlertState.mock.calls[0][0].message).not.toMatch(/CPU/);
  });
});
