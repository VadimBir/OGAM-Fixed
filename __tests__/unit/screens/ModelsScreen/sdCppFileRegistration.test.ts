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
  sdCppGeneratorService: { classifySafetensors: (...a: any[]) => mockClassify(...a), isAvailable: () => true },
}));

jest.mock('../../../../src/stores/downloadStore', () => ({
  useDownloadStore: { getState: () => ({ remove: mockRemove, setStatus: mockSetStatus }) },
}));

jest.mock('../../../../src/stores', () => ({
  useAppStore: { getState: () => ({ settings: { imageLoras: [] }, updateSettings: mockUpdateSettings }) },
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
});
