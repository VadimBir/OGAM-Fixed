/**
 * activeModelService/memory.estimateTextModelMemory: the ONE text estimate the residency gate, the
 * register and llm.ts share. Peak (charged to a load) = resident x 1.1; resident (charged once loaded)
 * has no factor; reclaimable mmapped weights are left out of the live free-RAM charge only.
 * (The old pre-load checkMemoryForModel / checkMemoryForDualModel had no callers and were removed.)
 */
const GB = 1024 ** 3;
const MiB = 1024 * 1024;

jest.mock('llama.rn', () => ({ loadLlamaModelInfo: jest.fn() }));
jest.mock('../../../src/services/hardware', () => ({
  hardwareService: {
    getAvailableMemoryGB: jest.fn(() => 4),
    estimateModelRam: jest.fn((m: { fileSize?: number }, mult: number) => (m.fileSize ?? 0) * mult),
  },
}));

import { Platform } from 'react-native';

describe('estimateTextModelMemory: resident vs peak, live vs physical', () => {
  const { loadLlamaModelInfo } = require('llama.rn');
  const memEstimate = require('../../../src/services/llamaMemoryEstimate');
  const { estimateTextModelMemory, textFootprintOf } = require('../../../src/services/activeModelService/memory');
  const { useAppStore } = require('../../../src/stores');
  const { createDownloadedModel } = require('../../utils/factories');
  let spy: jest.SpyInstance;
  let originalOS: string;
  beforeEach(() => {
    originalOS = Platform.OS;
    Object.defineProperty(Platform, 'OS', { get: () => 'android', configurable: true });
    (loadLlamaModelInfo as jest.Mock).mockResolvedValue({});
    // KV 600 MiB + compute 400 MiB; weights come from the file size.
    spy = jest.spyOn(memEstimate, 'estimateLlamaMemory').mockReturnValue({ totalBytes: 0, kvBytes: 600 * MiB, computeBytes: 400 * MiB, context: 4096 });
  });
  afterEach(() => {
    spy.mockRestore();
    Object.defineProperty(Platform, 'OS', { get: () => originalOS, configurable: true });
  });

  it('peak = (weights + mmproj + KV + compute) x 1.1; resident = the same without the factor', async () => {
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, inferenceBackend: 'cpu', mmapMode: 'on' } });
    const est = await estimateTextModelMemory(createDownloadedModel({ engine: 'llama', filePath: '/m.gguf', fileSize: 4 * GB, mmProjFileSize: 200 * MiB }));
    const f = textFootprintOf(est, false);
    expect(f.residentMB).toBe(4096 + 200 + 600 + 400);
    expect(f.peakMB).toBe(Math.ceil((4096 + 200 + 600 + 400) * 1.1));
  });

  it('Android CPU backend + mmap: weights reclaimable → live charge = mmproj + KV + compute only', async () => {
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, inferenceBackend: 'cpu', mmapMode: 'on' } });
    const est = await estimateTextModelMemory(createDownloadedModel({ engine: 'llama', filePath: '/m.gguf', fileSize: 4 * GB, mmProjFileSize: 200 * MiB }));
    expect(est.predictedReclaimable).toBe(true);
    const f = textFootprintOf(est, est.predictedReclaimable);
    expect(f.liveResidentMB).toBe(200 + 600 + 400);
    expect(f.livePeakMB).toBe(Math.ceil(1200 * 1.1));
    expect(f.peakMB).toBe(Math.ceil(5296 * 1.1)); // physical charge keeps the weights
  });

  it('Android OpenCL backend or mmap off: weights are anonymous → charged in full against free RAM', async () => {
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, inferenceBackend: 'opencl', mmapMode: 'on' } });
    const gpu = await estimateTextModelMemory(createDownloadedModel({ engine: 'llama', filePath: '/m.gguf', fileSize: 4 * GB }));
    expect(gpu.predictedReclaimable).toBe(false);
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, inferenceBackend: 'cpu', mmapMode: 'off' } });
    const noMmap = await estimateTextModelMemory(createDownloadedModel({ engine: 'llama', filePath: '/m.gguf', fileSize: 4 * GB }));
    expect(noMmap.predictedReclaimable).toBe(false);
    expect(textFootprintOf(noMmap, false).livePeakMB).toBe(Math.ceil((4096 + 1000) * 1.1));
  });

  it('unreadable header: the file-size heuristic is both the resident and the peak charge', async () => {
    spy.mockReturnValue(null);
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, inferenceBackend: 'cpu' } });
    const est = await estimateTextModelMemory(createDownloadedModel({ engine: 'llama', filePath: '/m.gguf', fileSize: GB }));
    expect(est.parts).toBeNull();
    expect(textFootprintOf(est, true)).toEqual({ residentMB: 1536, peakMB: 1536, liveResidentMB: 1536, livePeakMB: 1536 });
  });
});

// The pre-load estimate must size the KV cache at the SAME clamped context the loader uses
// (buildModelParams min(requested, trained)), not the raw global setting.
describe('estimateTextModelMemoryMB uses the clamped context', () => {
  const { loadLlamaModelInfo } = require('llama.rn');
  const memEstimate = require('../../../src/services/llamaMemoryEstimate');
  const { estimateTextModelMemoryMB } = require('../../../src/services/activeModelService/memory');
  const { useAppStore } = require('../../../src/stores');
  const { createDownloadedModel } = require('../../utils/factories');
  let spy: jest.SpyInstance;
  beforeEach(() => {
    spy = jest.spyOn(memEstimate, 'estimateLlamaMemory').mockReturnValue({ totalBytes: 1024 * 1024 * 1024, kvBytes: 0, computeBytes: 0, context: 0 });
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, contextLength: 32768 } });
  });
  afterEach(() => spy.mockRestore());

  it('clamps to the trained context from the header', async () => {
    (loadLlamaModelInfo as jest.Mock).mockResolvedValueOnce({ 'general.architecture': 'gemma3', 'gemma3.context_length': 8192 });
    const mb = await estimateTextModelMemoryMB(createDownloadedModel({ engine: 'llama', filePath: '/m.gguf', fileSize: GB }));
    expect(spy.mock.calls[0][1].ctxLen).toBe(8192);
    // Peak = (1 GiB weights + 0 KV + 0 compute) x 1.1.
    expect(mb).toBe(Math.ceil(1024 * 1.1));
  });

  // V is sized with ITS type (buildModelParams resolves cacheTypeV), not the K type.
  // Falsify: drop cacheTypeV from the estimate call and the split case reads undefined.
  it('passes the resolved V cache type (split K/V and "same")', async () => {
    (loadLlamaModelInfo as jest.Mock).mockResolvedValue({});
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, inferenceBackend: 'cpu', cacheType: 'q8_0', cacheTypeV: 'f16' } });
    await estimateTextModelMemoryMB(createDownloadedModel({ engine: 'llama', filePath: '/m.gguf', fileSize: GB }));
    expect(spy.mock.calls[0][1]).toMatchObject({ cacheType: 'q8_0', cacheTypeV: 'f16' });
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, cacheTypeV: 'same' } });
    await estimateTextModelMemoryMB(createDownloadedModel({ engine: 'llama', filePath: '/m.gguf', fileSize: GB }));
    expect(spy.mock.calls[1][1]).toMatchObject({ cacheType: 'q8_0', cacheTypeV: 'q8_0' });
  });

  it('keeps the requested context when the header has no trained context', async () => {
    (loadLlamaModelInfo as jest.Mock).mockResolvedValueOnce({});
    await estimateTextModelMemoryMB(createDownloadedModel({ engine: 'llama', filePath: '/m.gguf', fileSize: GB }));
    expect(spy.mock.calls[0][1].ctxLen).toBe(32768);
  });
});
