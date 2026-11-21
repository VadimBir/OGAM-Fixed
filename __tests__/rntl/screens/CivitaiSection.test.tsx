/**
 * Civitai search UI: search box, results (preview, base model, size), Download to phone states,
 * the disabled server target, the NSFW toggle (default off) and the Keychain-backed API key row.
 * The Civitai service and the download starter are mocked boundaries; the stores are real.
 */
import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { Linking } from 'react-native';
import {
  CivitaiSection, SERVER_DOWNLOAD_UNAVAILABLE, CIVITAI_ACCOUNT_URL, CIVITAI_KEY_MISSING_HINT,
} from '../../../src/screens/ModelsScreen/CivitaiSection';
import { useAppStore } from '../../../src/stores';
import { useDownloadStore } from '../../../src/stores/downloadStore';
import type { CivitaiResult } from '../../../src/services/civitaiBrowser';

const mockSearch = jest.fn();
const mockStart = jest.fn();
const mockGetKey = jest.fn();
const mockSetKey = jest.fn();
const mockRemoveKey = jest.fn();
const mockCancel = jest.fn();

jest.mock('../../../src/services/civitaiBrowser', () => ({
  searchCivitaiModels: (...a: any[]) => mockSearch(...a),
}));

jest.mock('../../../src/services/civitaiApiKey', () => ({
  getCivitaiApiKey: (...a: any[]) => mockGetKey(...a),
  setCivitaiApiKey: (...a: any[]) => mockSetKey(...a),
  removeCivitaiApiKey: (...a: any[]) => mockRemoveKey(...a),
}));

jest.mock('../../../src/services/civitaiDownload', () => {
  const actual = jest.requireActual('../../../src/services/civitaiDownload');
  return {
    civitaiImageModelId: actual.civitaiImageModelId,
    civitaiLocalFileName: actual.civitaiLocalFileName,
    startCivitaiPhoneDownload: (...a: any[]) => mockStart(...a),
  };
});

jest.mock('../../../src/services', () => ({
  hardwareService: { getTotalMemoryGB: () => 12 },
  backgroundDownloadService: { cancelDownload: (...a: any[]) => mockCancel(...a) },
  modelManager: { getImageModelsDirectory: () => '/x' },
}));

function result(over: Partial<CivitaiResult> = {}): CivitaiResult {
  return {
    key: '1:101:11', modelId: 1, versionId: 101, name: 'DreamShaper', versionName: 'v8', type: 'Checkpoint',
    baseModel: 'SD 1.5', family: 'sd1', nsfw: false, creator: 'lykon',
    previewUrl: 'https://image.civitai.com/a.jpeg',
    file: { id: 11, name: 'dream.safetensors', sizeBytes: 2 * 1024 ** 3, downloadUrl: 'https://civitai.com/api/download/models/101', primary: true, fp: 'fp16' },
    ...over,
  };
}

const page = (results: CivitaiResult[], extra: Record<string, any> = {}) => ({ results, nextCursor: null, skipped: 0, ...extra });

beforeEach(() => {
  jest.clearAllMocks();
  mockGetKey.mockResolvedValue(null);
  mockSetKey.mockResolvedValue(undefined);
  mockRemoveKey.mockResolvedValue(undefined);
  mockSearch.mockResolvedValue(page([result()]));
  mockStart.mockResolvedValue(undefined);
  mockCancel.mockResolvedValue(undefined);
  useDownloadStore.setState({ downloads: {}, downloadIdIndex: {} } as any);
  useAppStore.setState(s => ({
    downloadedImageModels: [],
    settings: { ...s.settings, imageLoras: [] },
  }));
});

describe('CivitaiSection', () => {
  it('searches SD 1.5 checkpoints on open with no NSFW filter and renders a result card', async () => {
    const { getByText, getByTestId, queryByTestId } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(getByTestId('civitai-result-0')).toBeTruthy());
    expect(mockSearch).toHaveBeenCalledWith(expect.objectContaining({
      query: '', type: 'Checkpoint', family: 'sd1', cursor: null,
    }));
    expect(mockSearch.mock.calls[0][0]).not.toHaveProperty('showNsfw');
    expect(getByText(/DreamShaper/)).toBeTruthy();
    expect(getByText(/SD 1\.5 · 2\.0 GB · Checkpoint · fp16/)).toBeTruthy();
    expect(getByTestId('civitai-thumb-0').props.source).toEqual({ uri: 'https://image.civitai.com/a.jpeg' });
    expect(getByText(/on the GPU \(Vulkan\) in this build/)).toBeTruthy();
    expect(queryByTestId('civitai-nsfw-switch')).toBeNull();
  });

  it('runs a typed search on submit', async () => {
    const { getByTestId } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
    fireEvent.changeText(getByTestId('civitai-search-input'), 'anime');
    fireEvent(getByTestId('civitai-search-input'), 'submitEditing');
    await waitFor(() => expect(mockSearch).toHaveBeenLastCalledWith(expect.objectContaining({ query: 'anime' })));
  });

  it('re-searches when the type or base model chip changes', async () => {
    const { getByTestId } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
    fireEvent.press(getByTestId('civitai-type-LORA'));
    await waitFor(() => expect(mockSearch).toHaveBeenLastCalledWith(expect.objectContaining({ type: 'LORA' })));
    fireEvent.press(getByTestId('civitai-family-sdxl'));
    await waitFor(() => expect(mockSearch).toHaveBeenLastCalledWith(expect.objectContaining({ family: 'sdxl' })));
  });

  it('starts a phone download from the Download to phone button', async () => {
    const setAlertState = jest.fn();
    const { getByTestId } = render(<CivitaiSection setAlertState={setAlertState} />);
    await waitFor(() => expect(getByTestId('civitai-download-phone-0')).toBeTruthy());
    fireEvent.press(getByTestId('civitai-download-phone-0'));
    expect(mockStart).toHaveBeenCalledWith(
      expect.objectContaining({ key: '1:101:11' }),
      expect.objectContaining({ setAlertState }),
      null,
    );
  });

  it('shows progress with Cancel while downloading, and On phone once registered', async () => {
    const { getByTestId, queryByTestId, getByText } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(getByTestId('civitai-result-0')).toBeTruthy());
    act(() => {
      useDownloadStore.getState().add({
        modelKey: 'image:civitai_101_11', downloadId: 'native-1', modelId: 'image:civitai_101_11',
        fileName: 'dream.safetensors', quantization: '', modelType: 'image', status: 'running',
        bytesDownloaded: 1, totalBytes: 4, combinedTotalBytes: 4, progress: 0.25, createdAt: 1,
      } as any);
    });
    expect(getByText('Downloading 25%')).toBeTruthy();
    expect(queryByTestId('civitai-download-phone-0')).toBeNull();
    fireEvent.press(getByTestId('civitai-cancel-0'));
    await waitFor(() => expect(mockCancel).toHaveBeenCalledWith('native-1'));
    expect(useDownloadStore.getState().downloads['image:civitai_101_11']).toBeUndefined();

    act(() => {
      useAppStore.setState({ downloadedImageModels: [{ id: 'civitai_101_11' } as any] });
    });
    expect(getByTestId('civitai-onphone-0')).toBeTruthy();
  });

  it('shows the server target disabled with its reason', async () => {
    const { getByTestId, getAllByText } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(getByTestId('civitai-download-server-0')).toBeTruthy());
    const server = getByTestId('civitai-download-server-0');
    expect(server.props.accessibilityState).toEqual({ disabled: true });
    expect(server.props.onPress).toBeUndefined();
    expect(getAllByText(SERVER_DOWNLOAD_UNAVAILABLE).length).toBe(1);
  });

  it('hides checkpoints too large for the RAM budget and says how many were hidden', async () => {
    mockSearch.mockResolvedValue(page([
      result(),
      result({ key: 'big', versionId: 9, family: 'sdxl', file: { ...result().file, id: 99, sizeBytes: 13.9 * 1024 ** 3 } }),
    ], { skipped: 2 }));
    const { getByTestId, queryByTestId, getByText } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(getByTestId('civitai-result-0')).toBeTruthy());
    expect(queryByTestId('civitai-result-1')).toBeNull();
    expect(getByText(/^3 hidden/)).toBeTruthy();
  });

  it('shows an error with Retry, and Load more when there is a next page', async () => {
    mockSearch.mockRejectedValueOnce(new Error('Civitai search failed (HTTP 500).'));
    const { getByText, getByTestId } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(getByText('Civitai search failed (HTTP 500).')).toBeTruthy());
    mockSearch.mockResolvedValueOnce(page([result()], { nextCursor: 'c2' }));
    fireEvent.press(getByTestId('civitai-retry'));
    await waitFor(() => expect(getByTestId('civitai-load-more')).toBeTruthy());
    mockSearch.mockResolvedValueOnce(page([result({ key: '2:2:2', modelId: 2, versionId: 2, name: 'Second' })]));
    fireEvent.press(getByTestId('civitai-load-more'));
    await waitFor(() => expect(getByTestId('civitai-result-1')).toBeTruthy());
    expect(mockSearch).toHaveBeenLastCalledWith(expect.objectContaining({ cursor: 'c2' }));
  });

  it('saves the API key to the Keychain and uses it for the next search and download', async () => {
    const { getByTestId, getByText, queryByTestId } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(getByText('Civitai API key: missing')).toBeTruthy());
    expect(getByTestId('civitai-apikey-hint').props.children).toBe(CIVITAI_KEY_MISSING_HINT);
    fireEvent.press(getByTestId('civitai-apikey-edit'));
    fireEvent.changeText(getByTestId('civitai-apikey-input'), ' key-1 ');
    await act(async () => { fireEvent.press(getByTestId('civitai-apikey-save')); });
    expect(mockSetKey).toHaveBeenCalledWith(' key-1 ');
    await waitFor(() => expect(getByText('Civitai API key: saved (used for downloads)')).toBeTruthy());
    expect(queryByTestId('civitai-apikey-hint')).toBeNull();
    fireEvent(getByTestId('civitai-search-input'), 'submitEditing');
    await waitFor(() => expect(mockSearch).toHaveBeenLastCalledWith(expect.objectContaining({ apiKey: 'key-1' })));
    fireEvent.press(getByTestId('civitai-download-phone-0'));
    expect(mockStart).toHaveBeenLastCalledWith(expect.objectContaining({ key: '1:101:11' }), expect.anything(), 'key-1');
  });

  it('Get key opens the Civitai account page, with no key and while pasting one', async () => {
    const open = jest.spyOn(Linking, 'openURL').mockResolvedValue(true);
    const { getByTestId } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(getByTestId('civitai-apikey-get')).toBeTruthy());
    fireEvent.press(getByTestId('civitai-apikey-get'));
    expect(open).toHaveBeenLastCalledWith(CIVITAI_ACCOUNT_URL);
    expect(CIVITAI_ACCOUNT_URL).toBe('https://civitai.com/user/account');
    fireEvent.press(getByTestId('civitai-apikey-edit'));
    fireEvent.press(getByTestId('civitai-apikey-get'));
    expect(open).toHaveBeenCalledTimes(2);
    open.mockRestore();
  });

  it('Cancel leaves the key unchanged; an empty draft cannot be saved', async () => {
    const { getByTestId, getByText } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(getByText('Civitai API key: missing')).toBeTruthy());
    fireEvent.press(getByTestId('civitai-apikey-edit'));
    expect(getByTestId('civitai-apikey-save').props.accessibilityState).toEqual(expect.objectContaining({ disabled: true }));
    fireEvent.changeText(getByTestId('civitai-apikey-input'), 'abc');
    fireEvent.press(getByTestId('civitai-apikey-cancel'));
    expect(mockSetKey).not.toHaveBeenCalled();
    expect(getByText('Civitai API key: missing')).toBeTruthy();
  });

  it('Clear removes a saved key and shows the missing state again', async () => {
    mockGetKey.mockResolvedValue('stored-key');
    const { getByTestId, getByText } = render(<CivitaiSection setAlertState={jest.fn()} />);
    await waitFor(() => expect(getByText('Civitai API key: saved (used for downloads)')).toBeTruthy());
    await act(async () => { fireEvent.press(getByTestId('civitai-apikey-remove')); });
    expect(mockRemoveKey).toHaveBeenCalled();
    await waitFor(() => expect(getByText('Civitai API key: missing')).toBeTruthy());
  });
});
