/**
 * activeModelService/memory.checkMemoryForModel: the pre-load warning reads the SAME live free-RAM
 * function as the real gate (memoryBudget.effectiveAvailableMB minus preloadSafetyMarginMB), not a
 * total-RAM budget. Falsify: revert to modelMemoryBudgetMB(total) and the low-live-RAM case reads "safe".
 */
const GB = 1024 ** 3;

const mockGetDeviceInfo = jest.fn();
const mockGetAppMemoryUsage = jest.fn();
const mockEstimateImage = jest.fn();

jest.mock('llama.rn', () => ({ loadLlamaModelInfo: jest.fn() }));
jest.mock('../../../src/services/llm', () => ({ llmService: { isModelLoaded: () => false } }));
jest.mock('../../../src/services/litert', () => ({ liteRTService: { isModelLoaded: () => false } }));
jest.mock('../../../src/services/hardware', () => ({
  hardwareService: {
    getDeviceInfo: (...a: any[]) => mockGetDeviceInfo(...a),
    getAppMemoryUsage: (...a: any[]) => mockGetAppMemoryUsage(...a),
    estimateImageModelRam: (...a: any[]) => mockEstimateImage(...a),
    estimateModelRam: jest.fn(() => 0),
  },
}));

import { Platform } from 'react-native';
import { checkMemoryForModel } from '../../../src/services/activeModelService/memory';
import { createONNXImageModel } from '../../utils/factories';

const model = createONNXImageModel({ id: 'sd', name: 'SD', size: 1 * GB, backend: 'mnn' });
const check = () =>
  checkMemoryForModel({
    modelId: 'sd',
    modelType: 'image',
    ids: { loadedTextModelId: null, loadedImageModelId: null },
    lists: { downloadedModels: [], downloadedImageModels: [model] },
    policy: 'balanced',
  });

describe('checkMemoryForModel uses live free RAM', () => {
  let originalOS: string;
  beforeEach(() => {
    originalOS = Platform.OS;
    Object.defineProperty(Platform, 'OS', { get: () => 'android', configurable: true });
    mockGetDeviceInfo.mockResolvedValue({ totalMemory: 12 * GB });
    mockEstimateImage.mockReturnValue(1.5 * GB);
  });
  afterEach(() => {
    Object.defineProperty(Platform, 'OS', { get: () => originalOS, configurable: true });
  });

  it('plenty of live RAM -> safe', async () => {
    mockGetAppMemoryUsage.mockResolvedValue({ used: 4 * GB, available: 8 * GB, total: 12 * GB });
    expect((await check()).severity).toBe('safe');
  });

  it('little live RAM (2 GB free, 1 GB margin) -> a 1.5 GB image model is blocked, though total RAM is 12 GB', async () => {
    mockGetAppMemoryUsage.mockResolvedValue({ used: 10 * GB, available: 2 * GB, total: 12 * GB });
    const r = await check();
    expect(r.severity).toBe('critical');
    expect(r.canLoad).toBe(false);
  });

  it('no live reading -> falls back to the total-RAM budget', async () => {
    mockGetAppMemoryUsage.mockRejectedValue(new Error('native unavailable'));
    expect((await check()).severity).toBe('safe');
  });
});
