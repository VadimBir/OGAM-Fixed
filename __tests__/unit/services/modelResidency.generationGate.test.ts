/**
 * 1. modelResidencyManager.canKeepOthersDuringGeneration (M4 gate): live RAM must hold what the
 *    generating model still has to allocate (minus the live headroom), and all residents must fit
 *    the physical ceiling. Real manager + budget; only the RAM sensor is stubbed.
 * 2. whisperStore: every native whisper unload also drops the 'whisper' resident. A phantom resident
 *    inflates liveBudgetMB (it credits resident MB as reclaimable) by memory that is already free.
 *    Falsify: remove the release('whisper') calls and the "after delete" assertions go red.
 */
jest.mock('../../../src/services/whisperService', () => ({
  WHISPER_MODELS: [{ id: 'base', size: 142 }],
  whisperService: {
    unloadModel: jest.fn(async () => {}),
    deleteModel: jest.fn(async () => {}),
    getModelPath: jest.fn((id: string) => `/mock/whisper/${id}.bin`),
    getLoadedModelPath: jest.fn(() => null),
    listDownloadedModels: jest.fn(async () => []),
  },
}));

import { modelResidencyManager } from '../../../src/services/modelResidency';
import { useWhisperStore } from '../../../src/stores/whisperStore';
import { setDeviceMemory, resetDeviceMemory, makeResident } from '../../harness/deviceMemory';

afterEach(() => resetDeviceMemory());

describe('canKeepOthersDuringGeneration (M4)', () => {
  it('true when the generating model is the only resident (nothing competes)', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 0.5 });
    makeResident({ key: 'image', type: 'image', sizeMB: 3000, dirtyMemory: true });
    await expect(modelResidencyManager.canKeepOthersDuringGeneration('image', 3000)).resolves.toBe(true);
  });

  it('true when live RAM minus the 1024 MB headroom holds the pending generation', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 5 }); // 5120 - 1024 = 4096
    makeResident({ key: 'text', type: 'text', sizeMB: 4000 });
    makeResident({ key: 'image', type: 'image', sizeMB: 3000, dirtyMemory: true });
    await expect(modelResidencyManager.canKeepOthersDuringGeneration('image', 4000)).resolves.toBe(true);
  });

  it('false when the pending generation exceeds live RAM minus the headroom', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 4 }); // 4096 - 1024 = 3072
    makeResident({ key: 'text', type: 'text', sizeMB: 4000 });
    makeResident({ key: 'image', type: 'image', sizeMB: 3500, dirtyMemory: true });
    await expect(modelResidencyManager.canKeepOthersDuringGeneration('image', 3500)).resolves.toBe(false);
  });

  it('aggressive keeps the smaller 512 MB headroom', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 4, policy: 'aggressive' }); // 4096 - 512
    makeResident({ key: 'text', type: 'text', sizeMB: 4000 });
    makeResident({ key: 'image', type: 'image', sizeMB: 3500, dirtyMemory: true });
    await expect(modelResidencyManager.canKeepOthersDuringGeneration('image', 3500)).resolves.toBe(true);
  });

  it('false when all residents together exceed the physical ceiling, even with live RAM to spare', async () => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 11 }); // balanced 12 GB cap ~8602 MB
    makeResident({ key: 'text', type: 'text', sizeMB: 6000 });
    makeResident({ key: 'image', type: 'image', sizeMB: 3000, dirtyMemory: true });
    await expect(modelResidencyManager.canKeepOthersDuringGeneration('image', 100)).resolves.toBe(false);
  });
});

describe('whisperStore releases the whisper resident on every native unload', () => {
  beforeEach(() => {
    setDeviceMemory({ platform: 'android', totalGB: 12, availGB: 8 });
    useWhisperStore.setState({ downloadedModelId: 'base', isModelLoaded: true, presentModelIds: ['base'] } as any);
  });

  it('unloadModel', async () => {
    makeResident({ key: 'whisper', type: 'whisper', sizeMB: 142 });
    await useWhisperStore.getState().unloadModel();
    expect(modelResidencyManager.isResident('whisper')).toBe(false);
  });

  it('deleteModel', async () => {
    makeResident({ key: 'whisper', type: 'whisper', sizeMB: 142 });
    await useWhisperStore.getState().deleteModel();
    expect(modelResidencyManager.isResident('whisper')).toBe(false);
  });

  it('deleteModelById of the selected model; another model id leaves the resident alone', async () => {
    makeResident({ key: 'whisper', type: 'whisper', sizeMB: 142 });
    await useWhisperStore.getState().deleteModelById('tiny');
    expect(modelResidencyManager.isResident('whisper')).toBe(true);
    await useWhisperStore.getState().deleteModelById('base');
    expect(modelResidencyManager.isResident('whisper')).toBe(false);
  });
});
