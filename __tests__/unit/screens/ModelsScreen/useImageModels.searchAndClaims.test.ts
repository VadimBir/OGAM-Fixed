import { renderHook, act, waitFor } from '@testing-library/react-native';
import { Platform } from 'react-native';
import { useImageModels } from '../../../../src/screens/ModelsScreen/useImageModels';
import {
  claimImageFinalization,
  releaseImageFinalization,
  isImageFinalizationClaimed,
} from '../../../../src/services/imageFinalizationClaims';

const mockUseAppStore = jest.fn();
const mockUseDownloadStore = jest.fn();
const mockGetDownloadedImageModels = jest.fn();
const mockResumeImageDownload = jest.fn();
const mockFetchAvailableModels = jest.fn();

jest.mock('../../../../src/stores', () => ({
  useAppStore: (selector?: any) => mockUseAppStore(selector),
}));

jest.mock('../../../../src/stores/downloadStore', () => {
  const useDownloadStore = (selector?: any) => mockUseDownloadStore(selector);
  (useDownloadStore as any).getState = () => (mockUseDownloadStore as any).state;
  return { useDownloadStore };
});

jest.mock('../../../../src/services', () => ({
  modelManager: { getDownloadedImageModels: (...a: any[]) => mockGetDownloadedImageModels(...a) },
  hardwareService: {
    getSoCInfo: jest.fn().mockResolvedValue({ hasNPU: true }),
    supportsSdxlNpu: jest.fn().mockResolvedValue(false),
    getImageModelRecommendation: jest.fn().mockResolvedValue({
      recommendedBackend: 'qnn', qnnVariant: '8gen2', bannerText: 'ok', compatibleBackends: ['qnn', 'mnn'],
    }),
  },
  backgroundDownloadService: { cancelDownload: jest.fn().mockResolvedValue(undefined) },
}));

jest.mock('../../../../src/screens/ModelsScreen/imageDownloadResume', () => ({
  resumeImageDownload: (...a: any[]) => mockResumeImageDownload(...a),
}));

jest.mock('../../../../src/services/huggingFaceModelBrowser', () => ({
  fetchAvailableModels: (...a: any[]) => mockFetchAvailableModels(...a),
  guessStyle: jest.fn(() => 'anime'),
}));

jest.mock('../../../../src/services/coreMLModelBrowser', () => ({
  fetchAvailableCoreMLModels: jest.fn().mockResolvedValue([]),
}));

const gpu = { id: 'chilloutmix_cpu', name: 'ChilloutMix', displayName: 'Chillout Mix (GPU)', backend: 'mnn', fileName: 'ChilloutMix.zip', downloadUrl: 'u1', size: 1, repo: 'xororz/sd-mnn' };
const npu = { id: 'anythingv5_npu_8gen2', name: 'AnythingV5', displayName: 'Anything V5 (NPU 8gen2)', backend: 'qnn', variant: '8gen2', fileName: 'a.zip', downloadUrl: 'u2', size: 1, repo: 'xororz/sd-qnn' };

const originalOS = Platform.OS;
beforeAll(() => { (Platform as any).OS = 'android'; });
afterAll(() => { (Platform as any).OS = originalOS; });

let appState: any;
function setStore(downloads: Record<string, any>) {
  const storeState = { downloads, remove: jest.fn() };
  (mockUseDownloadStore as any).state = storeState;
  mockUseDownloadStore.mockImplementation((selector?: any) => (selector ? selector(storeState) : storeState));
}

beforeEach(() => {
  jest.clearAllMocks();
  mockGetDownloadedImageModels.mockResolvedValue([]);
  mockResumeImageDownload.mockResolvedValue(undefined);
  mockFetchAvailableModels.mockResolvedValue([gpu, npu]);
  appState = {
    downloadedImageModels: [],
    setDownloadedImageModels: jest.fn(),
    addDownloadedImageModel: jest.fn(),
    activeImageModelId: null,
    setActiveImageModelId: jest.fn(),
    onboardingChecklist: { triedImageGen: false },
  };
  mockUseAppStore.mockImplementation((selector?: any) => (selector ? selector(appState) : appState));
  setStore({});
});

describe('useImageModels search', () => {
  it('finds a GPU model by name although the recommended-only star and the auto NPU filter hide it', async () => {
    const { result } = renderHook(() => useImageModels(jest.fn()));
    await act(async () => { await result.current.loadHFModels(); });
    await waitFor(() => expect(result.current.backendFilter).toBe('qnn'));
    expect(result.current.filteredHFModels.map(m => m.id)).toEqual(['anythingv5_npu_8gen2']);

    act(() => result.current.setImageSearchQuery('chillout'));
    expect(result.current.filteredHFModels.map(m => m.id)).toEqual(['chilloutmix_cpu']);
  });

  it('still honours a backend the user picked while searching', async () => {
    const { result } = renderHook(() => useImageModels(jest.fn()));
    await act(async () => { await result.current.loadHFModels(); });
    act(() => { result.current.setBackendFilter('qnn'); result.current.setUserChangedBackendFilter(true); });
    act(() => result.current.setImageSearchQuery('chillout'));
    expect(result.current.filteredHFModels).toEqual([]);
  });

  it('counts search matches that are already downloaded', async () => {
    appState.downloadedImageModels = [{ id: 'chilloutmix_cpu' }];
    const { result } = renderHook(() => useImageModels(jest.fn()));
    await act(async () => { await result.current.loadHFModels(); });
    act(() => result.current.setImageSearchQuery('chillout'));
    expect(result.current.filteredHFModels).toEqual([]);
    expect(result.current.downloadedSearchMatches).toBe(1);
  });

  it('lets only the newest catalog request write the list', async () => {
    let resolveFirst!: (v: any) => void;
    mockFetchAvailableModels
      .mockImplementationOnce(() => new Promise(r => { resolveFirst = r; }))
      .mockResolvedValueOnce([gpu]);
    const { result } = renderHook(() => useImageModels(jest.fn()));
    let first!: Promise<void>;
    act(() => { first = result.current.loadHFModels(); });
    await act(async () => { await result.current.loadHFModels(true); });
    await act(async () => { resolveFirst([gpu, npu]); await first; });
    expect(result.current.availableHFModels.map(m => m.id)).toEqual(['chilloutmix_cpu']);
    expect(result.current.hfModelsLoading).toBe(false);
  });
});

describe('useImageModels processing resume vs live finalizer', () => {
  const entry = {
    modelKey: 'image:m1', downloadId: 'dl-1', modelId: 'image:m1', fileName: 'm1.zip', quantization: '',
    modelType: 'image', status: 'processing', bytesDownloaded: 1, totalBytes: 1, combinedTotalBytes: 1,
    progress: 1, createdAt: 1,
  };

  it('skips a download a live completion listener is finalizing', async () => {
    expect(claimImageFinalization('m1')).toBe(true);
    try {
      setStore({ [entry.modelKey]: entry });
      renderHook(() => useImageModels(jest.fn()));
      await waitFor(() => expect(mockGetDownloadedImageModels).toHaveBeenCalledTimes(2));
      await act(async () => { await Promise.resolve(); });
      expect(mockResumeImageDownload).not.toHaveBeenCalled();
    } finally {
      releaseImageFinalization('m1');
    }
  });

  it('claims an unowned download while resuming it and releases it after', async () => {
    let finish!: () => void;
    mockResumeImageDownload.mockImplementation(() => new Promise<void>(r => { finish = r; }));
    setStore({ [entry.modelKey]: entry });
    renderHook(() => useImageModels(jest.fn()));
    await waitFor(() => expect(mockResumeImageDownload).toHaveBeenCalledTimes(1));
    expect(isImageFinalizationClaimed('m1')).toBe(true);
    await act(async () => { finish(); await Promise.resolve(); });
    await waitFor(() => expect(isImageFinalizationClaimed('m1')).toBe(false));
  });
});
