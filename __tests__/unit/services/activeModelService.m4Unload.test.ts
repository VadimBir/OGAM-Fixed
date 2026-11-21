/**
 * activeModelService — RAM fixes (2026-11-18):
 *  1. Text unload releases the engine that HOLDS a model, not the selected model's engine
 *     (engines.loadedTextEngines). Falsify: revert to getActiveEngineService() and the llama model
 *     stays loaded after a LiteRT model was selected.
 *  2. M4: suspendTextForImageGeneration unloads the resident text model (keeping its selection) when
 *     the budget cannot hold it next to the image generation; restoreTextAfterImageGeneration reloads
 *     it afterwards except in 'conservative' mode.
 *  3. H1-B3: unloadImageModel(true) frees the image model but keeps its selection.
 *  4. raceNativeLoad: a load that finishes after its timeout releases the orphaned model.
 * The REAL modelResidencyManager decides; only the native engines and the RAM sensor are stubbed.
 */
jest.mock('../../../src/services/llm', () => ({
  llmService: {
    isModelLoaded: jest.fn(() => false),
    unloadModel: jest.fn(async () => {}),
    loadModel: jest.fn(async () => {}),
    getLoadedModelPath: jest.fn(() => null),
    getMultimodalSupport: jest.fn(() => null),
    isCurrentlyGenerating: jest.fn(() => false),
    weightsAreMmapped: jest.fn(() => true),
  },
}));
jest.mock('../../../src/services/litert', () => ({
  liteRTService: {
    isModelLoaded: jest.fn(() => false),
    unloadModel: jest.fn(async () => {}),
    supportsAudio: jest.fn(() => false),
  },
}));
jest.mock('../../../src/services/imageEngineRouter', () => ({
  imageEngineRouter: {
    isModelLoaded: jest.fn(async () => false),
    unloadModel: jest.fn(async () => true),
    getLoadedModelPath: jest.fn(async () => null),
    sdCppConfigDiffers: jest.fn(() => false),
    isGenerating: jest.fn(async () => false),
  },
}));
jest.mock('../../../src/services/remoteServerManager', () => ({
  remoteServerManager: { clearActiveRemoteModel: jest.fn() },
}));

import { activeModelService } from '../../../src/services/activeModelService';
import { raceNativeLoad } from '../../../src/services/activeModelService/loaders';
import { modelResidencyManager } from '../../../src/services/modelResidency';
import { llmService } from '../../../src/services/llm';
import { liteRTService } from '../../../src/services/litert';
import { imageEngineRouter } from '../../../src/services/imageEngineRouter';
import { estimateImageModelRamBytes } from '../../../src/services/imageModelMemory';
import { useAppStore } from '../../../src/stores';
import { setDeviceMemory, resetDeviceMemory } from '../../harness/deviceMemory';
import { createDownloadedModel, createONNXImageModel } from '../../utils/factories';
import { hardwareService } from '../../../src/services/hardware';
import { estimateTextModelMemory, textFootprintOf } from '../../../src/services/activeModelService/memory';
import { OverridableMemoryError } from '../../../src/services/modelLoadErrors';

const GB = 1024 ** 3;
const llm = llmService as jest.Mocked<typeof llmService>;
const litert = liteRTService as jest.Mocked<typeof liteRTService>;
const router = imageEngineRouter as jest.Mocked<typeof imageEngineRouter>;
const ams = activeModelService as any; // white-box: seed the loaded ids the loaders would set

const gguf = createDownloadedModel({ id: 'gguf', engine: 'llama', fileSize: 4 * GB });
const tflite = createDownloadedModel({ id: 'lite', engine: 'litert' as any, fileSize: 2 * GB });
const sd = { ...createONNXImageModel({ id: 'sd', backend: 'sdcpp' as any, size: 2 * GB }), sdFamily: 'sd1' } as any;
const sdMB = () => Math.round(estimateImageModelRamBytes(sd) / (1024 * 1024));

/** Seed: text `gguf` loaded + resident, image `sd` loaded + resident (as the loaders leave it). */
function seedBothResident(): void {
  llm.isModelLoaded.mockReturnValue(true);
  ams.loadedTextModelId = 'gguf';
  ams.loadedImageModelId = 'sd';
  ams.imageWeightsMaterialized = false;
  useAppStore.setState({ activeModelId: 'gguf', activeImageModelId: 'sd', textModelEvicted: false } as any);
  modelResidencyManager.register({ key: 'text', type: 'text', modelId: 'gguf', sizeMB: 5000 }, () => ams.doUnloadTextModelLocked(true));
  modelResidencyManager.register({ key: 'image', type: 'image', sizeMB: sdMB(), dirtyMemory: true }, () => ams.doUnloadImageModelLocked(true));
}

beforeEach(() => {
  jest.clearAllMocks();
  llm.isModelLoaded.mockReturnValue(false);
  litert.isModelLoaded.mockReturnValue(false);
  router.isModelLoaded.mockResolvedValue(false);
  ams.loadedTextModelId = null;
  ams.loadedImageModelId = null;
  ams.imageWeightsMaterialized = false;
  useAppStore.setState({
    downloadedModels: [gguf, tflite],
    downloadedImageModels: [sd],
    activeModelId: null,
    activeImageModelId: null,
  } as any);
  setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 8 });
});
afterEach(() => resetDeviceMemory());

describe('text unload releases the LOADED engine, not the selected one', () => {
  it('a GGUF is loaded, a LiteRT model is selected: unload still releases llama and the residency slot', async () => {
    llm.isModelLoaded.mockReturnValue(true);
    ams.loadedTextModelId = 'gguf';
    modelResidencyManager.register({ key: 'text', type: 'text', modelId: 'gguf', sizeMB: 5000 }, async () => {});
    activeModelService.selectTextModel('lite'); // selection now names the LiteRT engine

    await activeModelService.unloadTextModel(true);

    expect(llm.unloadModel).toHaveBeenCalledTimes(1);
    expect(litert.unloadModel).not.toHaveBeenCalled(); // nothing loaded there
    expect(modelResidencyManager.isResident('text')).toBe(false);
    expect(activeModelService.getLoadedModelIds().textModelId).toBeNull();
  });

  it('a selection that resolves to no downloaded model still releases the loaded model', async () => {
    llm.isModelLoaded.mockReturnValue(true);
    ams.loadedTextModelId = 'gguf';
    useAppStore.setState({ activeModelId: 'gone' } as any);
    await activeModelService.unloadTextModel(true);
    expect(llm.unloadModel).toHaveBeenCalledTimes(1);
  });

  it('a rejected native unload propagates and keeps the residency slot (no phantom-freed RAM)', async () => {
    llm.isModelLoaded.mockReturnValue(true);
    llm.unloadModel.mockRejectedValueOnce(new Error('native busy'));
    ams.loadedTextModelId = 'gguf';
    modelResidencyManager.register({ key: 'text', type: 'text', modelId: 'gguf', sizeMB: 5000 }, async () => {});
    await expect(activeModelService.unloadTextModel(true)).rejects.toThrow('native busy');
    expect(modelResidencyManager.isResident('text')).toBe(true);
  });
});

describe('M4 — the LLM yields RAM to a local image generation', () => {
  it('enough live RAM for the image generation: the text model stays', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 8 }); // 8192 - 1024 >= sd estimate
    seedBothResident();
    await expect(activeModelService.suspendTextForImageGeneration()).resolves.toBe(false);
    expect(llm.unloadModel).not.toHaveBeenCalled();
    expect(modelResidencyManager.isResident('text')).toBe(true);
  });

  it('too little live RAM: the text model is unloaded, its selection kept and flagged evicted', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 2.5 }); // 2560 - 1024 < sd estimate
    seedBothResident();
    expect(sdMB()).toBeGreaterThan(2560 - 1024); // the premise of this case
    await expect(activeModelService.suspendTextForImageGeneration()).resolves.toBe(true);
    expect(llm.unloadModel).toHaveBeenCalledTimes(1);
    expect(modelResidencyManager.isResident('text')).toBe(false);
    expect(modelResidencyManager.isResident('image')).toBe(true);
    expect(useAppStore.getState().activeModelId).toBe('gguf');
    expect(useAppStore.getState().textModelEvicted).toBe(true);
  });

  it('sd.cpp after its first generation: only the compute part is still to come, so the text stays', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 2.5 });
    seedBothResident();
    activeModelService.noteImageGenerationCompleted(); // weights now resident
    await expect(activeModelService.suspendTextForImageGeneration()).resolves.toBe(false);
    expect(llm.unloadModel).not.toHaveBeenCalled();
  });

  it('no text model loaded: nothing to suspend', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 1 });
    ams.loadedImageModelId = 'sd';
    await expect(activeModelService.suspendTextForImageGeneration()).resolves.toBe(false);
  });

  it('restore reloads the text model that was loaded before (balanced)', async () => {
    const load = jest.spyOn(activeModelService, 'loadTextModel').mockResolvedValue();
    useAppStore.setState({ activeModelId: 'gguf' } as any);
    await activeModelService.restoreTextAfterImageGeneration('gguf');
    expect(load).toHaveBeenCalledWith('gguf');
  });

  it('restore keeps one model at a time in conservative mode (no reload)', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 8, policy: 'conservative' });
    const load = jest.spyOn(activeModelService, 'loadTextModel').mockResolvedValue();
    useAppStore.setState({ activeModelId: 'gguf' } as any);
    await activeModelService.restoreTextAfterImageGeneration('gguf');
    expect(load).not.toHaveBeenCalled();
  });

  it('restore skips when the user changed the text selection, when text is loaded, or when there was none', async () => {
    const load = jest.spyOn(activeModelService, 'loadTextModel').mockResolvedValue();
    useAppStore.setState({ activeModelId: 'lite' } as any);
    await activeModelService.restoreTextAfterImageGeneration('gguf');
    ams.loadedTextModelId = 'gguf';
    useAppStore.setState({ activeModelId: 'gguf' } as any);
    await activeModelService.restoreTextAfterImageGeneration('gguf');
    ams.loadedTextModelId = null;
    await activeModelService.restoreTextAfterImageGeneration(null);
    expect(load).not.toHaveBeenCalled();
  });

  it('a failed reload is swallowed (never breaks the image result)', async () => {
    jest.spyOn(activeModelService, 'loadTextModel').mockRejectedValue(new Error('no room'));
    useAppStore.setState({ activeModelId: 'gguf' } as any);
    await expect(activeModelService.restoreTextAfterImageGeneration('gguf')).resolves.toBeUndefined();
  });
});

describe('H1-B3 — Unload keeps the image selection', () => {
  it('unloadImageModel(true) frees native + residency, keeps activeImageModelId, reports not loaded', async () => {
    router.isModelLoaded.mockResolvedValue(true);
    ams.loadedImageModelId = 'sd';
    useAppStore.setState({ activeImageModelId: 'sd' } as any);
    modelResidencyManager.register({ key: 'image', type: 'image', sizeMB: 3000, dirtyMemory: true }, async () => {});

    await activeModelService.unloadImageModel(true);

    expect(router.unloadModel).toHaveBeenCalledTimes(1);
    expect(modelResidencyManager.isResident('image')).toBe(false);
    expect(useAppStore.getState().activeImageModelId).toBe('sd');
    const info = activeModelService.getActiveModels();
    expect(info.image.model?.id).toBe('sd');
    expect(info.image.isLoaded).toBe(false);
  });

  it('unloadImageModel() (default) still clears the selection (remote switch / delete paths)', async () => {
    router.isModelLoaded.mockResolvedValue(true);
    ams.loadedImageModelId = 'sd';
    useAppStore.setState({ activeImageModelId: 'sd' } as any);
    await activeModelService.unloadImageModel();
    expect(useAppStore.getState().activeImageModelId).toBeNull();
  });
});

describe('raceNativeLoad — a load that finishes after its timeout is released', () => {
  const flush = () => new Promise(r => setTimeout(r, 0));

  it('timeout rejects; the late success runs releaseOrphan under the residency lock', async () => {
    let finish: () => void = () => {};
    const load = new Promise<void>(r => { finish = r; });
    const releaseOrphan = jest.fn(async () => {});
    await expect(raceNativeLoad(load, { timeoutMs: 5, message: 'timed out', label: 't', releaseOrphan }))
      .rejects.toThrow('timed out');
    expect(releaseOrphan).not.toHaveBeenCalled();
    finish();
    await flush(); await flush();
    expect(releaseOrphan).toHaveBeenCalledTimes(1);
  });

  it('in time: resolves the load value, no orphan release', async () => {
    const releaseOrphan = jest.fn(async () => {});
    await expect(raceNativeLoad(Promise.resolve(42), { timeoutMs: 1000, message: 'x', label: 't', releaseOrphan }))
      .resolves.toBe(42);
    await flush();
    expect(releaseOrphan).not.toHaveBeenCalled();
  });

  it('a load that FAILS (not a timeout) does not release anything', async () => {
    const releaseOrphan = jest.fn(async () => {});
    await expect(raceNativeLoad(Promise.reject(new Error('bad file')), { timeoutMs: 1000, message: 'x', label: 't', releaseOrphan }))
      .rejects.toThrow('bad file');
    await flush();
    expect(releaseOrphan).not.toHaveBeenCalled();
  });

  it('a timed-out load that later fails releases nothing', async () => {
    let fail: (e: Error) => void = () => {};
    const load = new Promise<void>((_, rej) => { fail = rej; });
    const releaseOrphan = jest.fn(async () => {});
    await expect(raceNativeLoad(load, { timeoutMs: 5, message: 'timed out', label: 't', releaseOrphan })).rejects.toThrow();
    fail(new Error('late failure'));
    await flush(); await flush();
    expect(releaseOrphan).not.toHaveBeenCalled();
  });
});

describe('memory gate: the free-RAM decision happens before any eviction (never evict-then-refuse)', () => {
  const { loadLlamaModelInfo } = require('llama.rn');
  const llamaMeta = {
    'general.architecture': 'llama', 'llama.block_count': 32, 'llama.attention.head_count': 32,
    'llama.attention.head_count_kv': 8, 'llama.embedding_length': 4096, 'llama.vocab_size': 32000,
  };
  beforeEach(() => (loadLlamaModelInfo as jest.Mock).mockResolvedValue(llamaMeta));
  afterEach(() => (loadLlamaModelInfo as jest.Mock).mockResolvedValue({}));

  function seedImageResident(sizeMB: number): jest.Mock {
    ams.loadedImageModelId = 'sd';
    useAppStore.setState({ activeImageModelId: 'sd' } as any);
    const unload = jest.fn(async () => { await ams.doUnloadImageModelLocked(true); });
    modelResidencyManager.register({ key: 'image', type: 'image', modelId: 'sd', sizeMB, dirtyMemory: true }, unload, 1);
    return unload;
  }

  it('a text load the gate refuses evicts nothing and never reaches llm.ts', async () => {
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, inferenceBackend: 'cpu', mmapMode: 'off' } } as any);
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 2 });
    const unloadImage = seedImageResident(600);
    // Weights in anonymous RAM (mmap off): ~4.4GB live, more than 2048 - 1024 + 600 even after evicting the image.
    await expect(activeModelService.loadTextModel('gguf')).rejects.toBeInstanceOf(OverridableMemoryError);
    expect(unloadImage).not.toHaveBeenCalled();
    expect(router.unloadModel).not.toHaveBeenCalled();
    expect(llm.loadModel).not.toHaveBeenCalled();
    expect(modelResidencyManager.isResident('image')).toBe(true);
  });

  it('a text load the gate approves reaches llm.ts with memoryGateApproved, and is registered at its resident bytes', async () => {
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, inferenceBackend: 'cpu', mmapMode: 'on' } } as any);
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 6 });
    seedImageResident(600);
    llm.loadModel.mockImplementation(async () => { llm.isModelLoaded.mockReturnValue(true); });

    await activeModelService.loadTextModel('gguf');

    expect(llm.loadModel).toHaveBeenCalledWith(gguf.filePath, undefined, { override: false, memoryGateApproved: true });
    // Both fit: the image stays resident next to the text model.
    expect(modelResidencyManager.isResident('image')).toBe(true);
    const est = await estimateTextModelMemory(gguf);
    const resident = modelResidencyManager.getResidents().find(r => r.key === 'text');
    expect(resident?.sizeMB).toBe(textFootprintOf(est, false).residentMB);
    expect(resident!.sizeMB).toBeLessThan(textFootprintOf(est, false).peakMB);
    llm.loadModel.mockImplementation(async () => {});
  });
});

describe('image unload: local-dream process and memory return', () => {
  it('stops a local-dream server that is alive but not ready (isModelLoaded=false, model path set)', async () => {
    router.isModelLoaded.mockResolvedValue(false);
    router.getLoadedModelPath.mockResolvedValueOnce('/models/sd');
    await activeModelService.unloadImageModel(true);
    expect(router.unloadModel).toHaveBeenCalledTimes(1);
  });

  it('nothing alive and nothing selected: no native call', async () => {
    router.isModelLoaded.mockResolvedValue(false);
    router.getLoadedModelPath.mockResolvedValueOnce(null);
    await activeModelService.unloadImageModel(true);
    expect(router.unloadModel).not.toHaveBeenCalled();
  });

  it('reads the memory baseline BEFORE the unload and returns once the drop is seen', async () => {
    router.isModelLoaded.mockResolvedValue(true);
    ams.loadedImageModelId = 'sd';
    modelResidencyManager.register({ key: 'image', type: 'image', sizeMB: 2000, dirtyMemory: true }, async () => {});
    const readings = [{ availableMB: 4000, footprintMB: 6000, limitMB: 10000 }, { availableMB: 7000, footprintMB: 3000, limitMB: 10000 }];
    let i = 0;
    const getProcessMemory = jest.spyOn(hardwareService, 'getProcessMemory').mockImplementation(async () => readings[Math.min(i++, 1)]);

    await activeModelService.unloadImageModel(true);

    expect(getProcessMemory).toHaveBeenCalledTimes(2);
    expect(getProcessMemory.mock.invocationCallOrder[0]).toBeLessThan(router.unloadModel.mock.invocationCallOrder[0]);
    expect(getProcessMemory.mock.invocationCallOrder[1]).toBeGreaterThan(router.unloadModel.mock.invocationCallOrder[0]);
  });
});

