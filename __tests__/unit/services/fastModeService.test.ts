/**
 * Fast mode actions: a preset LoRA already on the phone applies at once; otherwise one tap starts
 * a background download (store row, exact size + sha256), the finished file is registered through
 * the Civitai finalize path (-> LoRA list) and the preset is applied, unless the user chose Off
 * meanwhile. TAESD downloads into image_vae/ and turns the switch on. NPU presets set
 * steps/cfg/sampler only.
 */
import RNFS from 'react-native-fs';
import { useAppStore } from '../../../src/stores/appStore';
import { useDownloadStore } from '../../../src/stores/downloadStore';
import { resetStores } from '../../utils/testHelpers';

const mockStartDownload = jest.fn();
const mockMove = jest.fn();
const mockCancel = jest.fn();
const completeListeners: Record<string, () => Promise<void> | void> = {};
const mockFinalize = jest.fn();
const mockMissingFreeSpace = jest.fn();

jest.mock('../../../src/services', () => ({
  backgroundDownloadService: {
    isAvailable: () => true,
    startDownload: (...a: any[]) => mockStartDownload(...a),
    moveCompletedDownload: (...a: any[]) => mockMove(...a),
    cancelDownload: (...a: any[]) => mockCancel(...a),
    onComplete: (id: string, cb: any) => { completeListeners[id] = cb; return () => { delete completeListeners[id]; }; },
    onError: () => () => {},
    startProgressPolling: jest.fn(),
  },
}));
jest.mock('../../../src/services/sdCppGenerator', () => ({
  sdCppGeneratorService: { isAvailable: () => true },
}));
jest.mock('../../../src/services/civitaiDownload', () => ({
  finalizeSdCppFileDownload: (...a: any[]) => mockFinalize(...a),
}));
jest.mock('../../../src/services/sdCppFileRegistration', () => ({
  missingFreeSpace: (...a: any[]) => mockMissingFreeSpace(...a),
  formatGB: (b: number) => `${(b / 1024 ** 3).toFixed(1)} GB`,
  safeSdFileId: (n: string) => n.replaceAll(/[^a-zA-Z0-9_-]/g, '_'),
}));

import {
  selectSdCppFastMode,
  selectNpuFastMode,
  setTaesdEnabled,
  turnOffFastMode,
  fastLoraDownloadId,
  cancelFastDownload,
} from '../../../src/services/fastModeService';
import { FAST_LORA_PRESETS, npuFastPreset } from '../../../src/services/fastModePresets';

const deps = () => ({
  addDownloadedImageModel: jest.fn(),
  activeImageModelId: 'm',
  setActiveImageModelId: jest.fn(),
  setAlertState: jest.fn(),
});

const settings = () => useAppStore.getState().settings;
const userValues = { imageSteps: 20, imageGuidanceScale: 7.5, imageSampler: 'euler_a', imageScheduler: 'karras' };

beforeEach(() => {
  resetStores();
  jest.clearAllMocks();
  useDownloadStore.setState({ downloads: {}, downloadIdIndex: {} } as any);
  useAppStore.getState().updateSettings({ ...userValues, imageLoras: [], imageFastMode: null, imageSdTaesd: false });
  mockMissingFreeSpace.mockResolvedValue(null);
  mockStartDownload.mockResolvedValue({ downloadId: 'native-1' });
  mockMove.mockResolvedValue('/moved');
  mockCancel.mockResolvedValue(undefined);
  (RNFS.exists as jest.Mock).mockReset();
  (RNFS.exists as jest.Mock).mockResolvedValue(false);
  (RNFS.mkdir as jest.Mock).mockResolvedValue(undefined);
  for (const k of Object.keys(completeListeners)) delete completeListeners[k];
  // The real finalize registers the LoRA (registerSdFileInPlace -> settings.imageLoras).
  mockFinalize.mockImplementation(async ({ fileName }: { fileName: string }) => {
    const { settings: s, updateSettings } = useAppStore.getState();
    updateSettings({ imageLoras: [...(s.imageLoras ?? []), {
      id: `lora_${fileName}`, name: fileName.replace('.safetensors', ''), path: `/mock/documents/image_loras/${fileName}`, weight: 1, enabled: true,
    }] });
  });
});

describe('GPU: LoRA already on the phone', () => {
  it('applies at once, no download', async () => {
    useAppStore.getState().updateSettings({ imageLoras: [
      { id: 'x', name: 'Hyper-SD15-2steps-lora', path: '/mock/documents/image_loras/Hyper-SD15-2steps-lora.safetensors', weight: 0.5, enabled: false },
    ] });
    (RNFS.exists as jest.Mock).mockResolvedValue(true);
    await expect(selectSdCppFastMode('sd1', 2, deps())).resolves.toBe('applied');
    expect(mockStartDownload).not.toHaveBeenCalled();
    expect(settings()).toMatchObject({ imageSteps: 2, imageGuidanceScale: 1, imageSampler: 'ddim_trailing', imageScheduler: 'simple' });
    expect(settings().imageLoras![0]).toMatchObject({ enabled: true, weight: 1 });
  });
});

describe('GPU: download -> register -> apply', () => {
  it('downloads the preset file with exact size + sha, registers it, applies the preset', async () => {
    const d = deps();
    const preset = FAST_LORA_PRESETS.sdxl[4];
    await expect(selectSdCppFastMode('sdxl', 4, d)).resolves.toBe('started');
    expect(mockMissingFreeSpace).toHaveBeenCalledWith(preset.sizeBytes);
    const params = mockStartDownload.mock.calls[0][0];
    expect(params).toMatchObject({
      url: 'https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors',
      fileName: 'sdxl_lightning_4step_lora.safetensors',
      modelId: 'image:fastmode_sdxl_lightning_4step_lora',
      totalBytes: 393_854_592,
      sha256: preset.sha256,
      modelType: 'image',
    });
    expect(JSON.parse(params.metadataJson)).toMatchObject({ imageDownloadType: 'sdcpp-file', imageModelFileName: preset.fileName });
    const row = useDownloadStore.getState().downloads['image:fastmode_sdxl_lightning_4step_lora' as any];
    expect(row?.downloadId).toBe('native-1');
    // Nothing applied until the file is registered.
    expect(settings().imageSteps).toBe(20);

    await completeListeners['native-1']();
    expect(mockFinalize).toHaveBeenCalledWith(expect.objectContaining({
      downloadId: 'native-1', modelId: 'fastmode_sdxl_lightning_4step_lora', fileName: preset.fileName,
    }));
    expect(settings()).toMatchObject({ imageSteps: 4, imageGuidanceScale: 1, imageSampler: 'euler', imageScheduler: 'simple' });
    expect(settings().imageLoras![0]).toMatchObject({ path: '/mock/documents/image_loras/sdxl_lightning_4step_lora.safetensors', enabled: true, weight: 1 });
    expect(settings().imageFastMode).toMatchObject({ steps: 4, engine: 'sdcpp', family: 'sdxl', previous: userValues });
    expect(d.setAlertState).toHaveBeenLastCalledWith(expect.objectContaining({ title: 'Fast mode on' }));

    // Off restores the user's own values and disables the LoRA.
    turnOffFastMode();
    expect(settings()).toMatchObject({ ...userValues, imageFastMode: null });
    expect(settings().imageLoras![0].enabled).toBe(false);
  });

  it('Off before the download lands: LoRA registered, preset not applied', async () => {
    await selectSdCppFastMode('sd1', 1, deps());
    turnOffFastMode();
    await completeListeners['native-1']();
    expect(mockFinalize).toHaveBeenCalled();
    expect(settings()).toMatchObject({ ...userValues, imageFastMode: null });
  });

  it('registration rejected (no LoRA in the list): settings unchanged', async () => {
    mockFinalize.mockResolvedValue(undefined);
    await selectSdCppFastMode('sd1', 4, deps());
    await completeListeners['native-1']();
    expect(settings()).toMatchObject({ ...userValues, imageFastMode: null });
  });

  it('not enough space: blocked with an alert, no download', async () => {
    mockMissingFreeSpace.mockResolvedValue(1024 ** 3);
    const d = deps();
    await expect(selectSdCppFastMode('sd1', 4, d)).resolves.toBe('blocked');
    expect(mockStartDownload).not.toHaveBeenCalled();
    expect(d.setAlertState).toHaveBeenCalledWith(expect.objectContaining({ title: 'Not Enough Storage' }));
  });

  it('second tap while downloading does not start a second download; cancel removes the row', async () => {
    await selectSdCppFastMode('sd1', 2, deps());
    await expect(selectSdCppFastMode('sd1', 2, deps())).resolves.toBe('busy');
    expect(mockStartDownload).toHaveBeenCalledTimes(1);
    const id = fastLoraDownloadId(FAST_LORA_PRESETS.sd1[2]);
    await cancelFastDownload(id);
    expect(mockCancel).toHaveBeenCalledWith('native-1');
    expect(useDownloadStore.getState().downloads[`image:${id}` as any]).toBeUndefined();
  });

  it('listed LoRA whose file is gone: stale entry dropped, downloaded again', async () => {
    useAppStore.getState().updateSettings({ imageLoras: [
      { id: 'gone', name: 'Hyper-SD15-4steps-lora', path: '/x/Hyper-SD15-4steps-lora.safetensors', weight: 1, enabled: true },
    ] });
    await expect(selectSdCppFastMode('sd1', 4, deps())).resolves.toBe('started');
    expect(settings().imageLoras).toEqual([]);
  });
});

describe('TAESD', () => {
  it('missing file: downloads into image_vae/ and turns the switch on when it lands', async () => {
    await expect(setTaesdEnabled(true, 'sd1', deps())).resolves.toBe('started');
    expect(mockStartDownload.mock.calls[0][0]).toMatchObject({
      url: 'https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors',
      totalBytes: 9_793_292,
      fileName: 'taesd_sd1.safetensors',
    });
    expect(settings().imageSdTaesd).toBe(false);
    await completeListeners['native-1']();
    expect(RNFS.mkdir).toHaveBeenCalledWith('/mock/documents/image_vae');
    expect(mockMove).toHaveBeenCalledWith('native-1', '/mock/documents/image_vae/taesd_sd1.safetensors');
    expect(settings().imageSdTaesd).toBe(true);
    expect(useDownloadStore.getState().downloads['image:fastmode_taesd_sd1' as any]).toBeUndefined();
  });

  it('file present: on at once; off cancels nothing and turns it off', async () => {
    (RNFS.exists as jest.Mock).mockResolvedValue(true);
    await expect(setTaesdEnabled(true, 'sdxl', deps())).resolves.toBe('applied');
    expect(settings().imageSdTaesd).toBe(true);
    await setTaesdEnabled(false, 'sdxl', deps());
    expect(settings().imageSdTaesd).toBe(false);
    expect(mockStartDownload).not.toHaveBeenCalled();
  });
});

describe('NPU', () => {
  it('few-step model: steps/cfg/sampler set, Off restores', () => {
    selectNpuFastMode(npuFastPreset('DMD2 SDXL')!, 4);
    expect(settings()).toMatchObject({ imageSteps: 4, imageGuidanceScale: 1, imageSampler: 'lcm' });
    turnOffFastMode();
    expect(settings()).toMatchObject({ ...userValues, imageFastMode: null });
  });
});
