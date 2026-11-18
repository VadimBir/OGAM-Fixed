import RNFS from 'react-native-fs';
import { useDownloadStore } from '../../../src/stores/downloadStore';
import {
  startCivitaiPhoneDownload,
  civitaiImageModelId,
  civitaiLocalFileName,
  finalizeSdCppFileDownload,
} from '../../../src/services/civitaiDownload';
import { claimImageFinalization, releaseImageFinalization, isImageFinalizationClaimed } from '../../../src/services/imageFinalizationClaims';
import type { CivitaiResult } from '../../../src/services/civitaiBrowser';

const mockStartDownload = jest.fn();
const mockMove = jest.fn();
const completeListeners: Record<string, () => Promise<void> | void> = {};
const errorListeners: Record<string, (e: any) => void> = {};
const mockRegister = jest.fn();
const mockMissingFreeSpace = jest.fn();
const mockCancelDownload = jest.fn();
let mockSdCppAvailable = true;

jest.mock('../../../src/services', () => ({
  modelManager: { getImageModelsDirectory: () => '/docs/image_models' },
  backgroundDownloadService: {
    isAvailable: () => true,
    startDownload: (...a: any[]) => mockStartDownload(...a),
    moveCompletedDownload: (...a: any[]) => mockMove(...a),
    onComplete: (id: string, cb: any) => { completeListeners[id] = cb; return () => { delete completeListeners[id]; }; },
    onError: (id: string, cb: any) => { errorListeners[id] = cb; return () => { delete errorListeners[id]; }; },
    startProgressPolling: jest.fn(),
    cancelDownload: (...a: any[]) => mockCancelDownload(...a),
  },
}));

jest.mock('../../../src/services/sdCppGenerator', () => ({
  sdCppGeneratorService: { isAvailable: () => mockSdCppAvailable },
}));

jest.mock('../../../src/services/sdCppFileRegistration', () => ({
  missingFreeSpace: (...a: any[]) => mockMissingFreeSpace(...a),
  formatGB: (b: number) => `${(b / 1024 ** 3).toFixed(1)} GB`,
  registerSdFileInPlace: (...a: any[]) => mockRegister(...a),
}));

const result: CivitaiResult = {
  key: '1:101:11', modelId: 1, versionId: 101, name: 'DreamShaper', versionName: 'v8', type: 'Checkpoint',
  baseModel: 'SD 1.5', family: 'sd1', nsfw: false,
  file: {
    id: 11, name: 'dream shaper(v8).safetensors', sizeBytes: 2_000_000_000,
    downloadUrl: 'https://civitai.com/api/download/models/101', sha256: 'AB'.repeat(32), primary: true,
  },
};

const deps = () => ({
  addDownloadedImageModel: jest.fn(),
  activeImageModelId: null,
  setActiveImageModelId: jest.fn(),
  setAlertState: jest.fn(),
});

const flush = () => new Promise(r => setImmediate(r));

beforeEach(() => {
  jest.clearAllMocks();
  mockSdCppAvailable = true;
  useDownloadStore.setState({ downloads: {}, downloadIdIndex: {} } as any);
  mockMissingFreeSpace.mockResolvedValue(null);
  mockStartDownload.mockResolvedValue({ downloadId: 'native-1' });
  mockMove.mockResolvedValue('/moved');
  mockCancelDownload.mockResolvedValue(undefined);
  mockRegister.mockResolvedValue({ kind: 'checkpoint', model: { id: 'civitai_101_11' } });
  (RNFS.exists as jest.Mock).mockResolvedValue(false);
  (RNFS.mkdir as jest.Mock).mockResolvedValue(undefined);
  (RNFS.unlink as jest.Mock).mockResolvedValue(undefined);
  for (const k of Object.keys(completeListeners)) delete completeListeners[k];
});

describe('civitaiDownload naming', () => {
  it('derives a stable id and a safe .safetensors file name', () => {
    expect(civitaiImageModelId(result)).toBe('civitai_101_11');
    expect(civitaiLocalFileName(result)).toBe('dream_shaper_v8_.safetensors');
  });
});

describe('startCivitaiPhoneDownload', () => {
  it('starts a background download with token only when a key is set, exact size, lowercase SHA, no URL in metadata', async () => {
    await startCivitaiPhoneDownload(result, deps(), 'my-key');
    const params = mockStartDownload.mock.calls[0][0];
    expect(params.url).toBe('https://civitai.com/api/download/models/101?token=my-key');
    expect(params.totalBytes).toBe(2_000_000_000);
    expect(params.sha256).toBe('ab'.repeat(32));
    expect(params.modelType).toBe('image');
    expect(params.modelKey).toBe('image:civitai_101_11');
    expect(params.metadataJson).not.toContain('civitai.com');
    expect(params.metadataJson).not.toContain('my-key');
    expect(JSON.parse(params.metadataJson).imageDownloadType).toBe('sdcpp-file');
    const entry = useDownloadStore.getState().downloads['image:civitai_101_11'];
    expect(entry.downloadId).toBe('native-1');
    expect(entry.status).toBe('pending');

    await startCivitaiPhoneDownload({ ...result, file: { ...result.file, id: 12 } }, deps(), null);
    expect(mockStartDownload.mock.calls[1][0].url).toBe('https://civitai.com/api/download/models/101');
  });

  it('refuses when the phone lacks the space, without starting a download', async () => {
    mockMissingFreeSpace.mockResolvedValue(3 * 1024 ** 3);
    const d = deps();
    await startCivitaiPhoneDownload(result, d);
    expect(mockStartDownload).not.toHaveBeenCalled();
    expect(d.setAlertState.mock.calls[0][0].title).toBe('Not Enough Storage');
  });

  it('refuses where stable-diffusion.cpp is unavailable', async () => {
    mockSdCppAvailable = false;
    const d = deps();
    await startCivitaiPhoneDownload(result, d);
    expect(mockStartDownload).not.toHaveBeenCalled();
    expect(d.setAlertState.mock.calls[0][0].title).toBe('Not Supported');
  });

  it('ignores a second tap while the first download is active', async () => {
    await startCivitaiPhoneDownload(result, deps());
    await startCivitaiPhoneDownload(result, deps());
    expect(mockStartDownload).toHaveBeenCalledTimes(1);
  });

  it('on completion claims, moves into image_models/<id>/ and registers', async () => {
    const d = deps();
    await startCivitaiPhoneDownload(result, d);
    const done = completeListeners['native-1']();
    expect(isImageFinalizationClaimed('civitai_101_11')).toBe(true);
    await done;
    expect(mockMove).toHaveBeenCalledWith('native-1', '/docs/image_models/civitai_101_11/dream_shaper_v8_.safetensors');
    expect(mockRegister).toHaveBeenCalledWith(
      expect.objectContaining({ modelId: 'civitai_101_11', verb: 'downloaded', displayName: 'DreamShaper v8' }),
      d,
    );
    expect(useDownloadStore.getState().downloads['image:civitai_101_11']).toBeUndefined();
    expect(isImageFinalizationClaimed('civitai_101_11')).toBe(false);
  });

  it('does not finalize when another finalizer holds the claim', async () => {
    await startCivitaiPhoneDownload(result, deps());
    claimImageFinalization('civitai_101_11');
    try {
      await completeListeners['native-1']();
      expect(mockMove).not.toHaveBeenCalled();
    } finally {
      releaseImageFinalization('civitai_101_11');
    }
  });

  it('marks the row failed and cleans up when install fails', async () => {
    mockRegister.mockRejectedValue(new Error('disk full'));
    const d = deps();
    await startCivitaiPhoneDownload(result, d);
    await completeListeners['native-1']();
    await flush();
    expect(RNFS.unlink).toHaveBeenCalledWith('/docs/image_models/civitai_101_11');
    expect(useDownloadStore.getState().downloads['image:civitai_101_11'].status).toBe('failed');
    expect(d.setAlertState).toHaveBeenCalledWith(expect.objectContaining({ title: 'Download Failed' }));
  });

  it.each([
    ['civitai_auth_required', 'Civitai API key needed', /^Civitai needs your API key\. Add it in Models > Image > Civitai/],
    ['civitai_key_rejected', 'Civitai API key needed', /^Civitai rejected your API key/],
    ['civitai_forbidden', 'Download Failed', /^Civitai does not permit your account/],
  ])('a native %s failure shows the key-specific message, not the generic one', async (code, title, text) => {
    const d = deps();
    await startCivitaiPhoneDownload(result, d, code === 'civitai_auth_required' ? null : 'k');
    errorListeners['native-1']({ downloadId: 'native-1', reason: 'The download server rejected access to this file.', reasonCode: code });
    const alert = d.setAlertState.mock.calls[0][0];
    expect(alert.title).toBe(title);
    expect(alert.message).toMatch(text);
    expect(mockRegister).not.toHaveBeenCalled();
    expect(mockMove).not.toHaveBeenCalled();
  });

  it('a retry after adding the key cancels the failed native row and starts a new one with the key', async () => {
    await startCivitaiPhoneDownload(result, deps(), null);
    expect(mockStartDownload.mock.calls[0][0].url).toBe('https://civitai.com/api/download/models/101');
    useDownloadStore.getState().setStatus('native-1', 'failed', { message: 'x', code: 'civitai_auth_required' });
    mockStartDownload.mockResolvedValue({ downloadId: 'native-2' });
    await startCivitaiPhoneDownload(result, deps(), 'new-key');
    expect(mockCancelDownload).toHaveBeenCalledWith('native-1');
    expect(mockStartDownload.mock.calls[1][0].url).toBe('https://civitai.com/api/download/models/101?token=new-key');
    expect(useDownloadStore.getState().downloads['image:civitai_101_11'].downloadId).toBe('native-2');
  });

  it('drops the placeholder quietly when cancelled while queued', async () => {
    mockStartDownload.mockRejectedValue(Object.assign(new Error('cancelled'), { cancelled: true }));
    const d = deps();
    await startCivitaiPhoneDownload(result, d);
    expect(useDownloadStore.getState().downloads['image:civitai_101_11']).toBeUndefined();
    expect(d.setAlertState).not.toHaveBeenCalled();
  });
});

describe('finalizeSdCppFileDownload', () => {
  it('skips the move when the file is already in place (restart after the move)', async () => {
    (RNFS.exists as jest.Mock).mockResolvedValue(true);
    await finalizeSdCppFileDownload({ downloadId: 'd', modelId: 'm', fileName: 'f.safetensors', displayName: 'F', deps: deps() });
    expect(mockMove).not.toHaveBeenCalled();
    expect(mockRegister).toHaveBeenCalled();
  });
});
