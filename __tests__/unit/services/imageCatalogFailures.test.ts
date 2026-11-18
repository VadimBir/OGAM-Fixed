/**
 * Catalog fetch edge cases: HF tree paging, duplicate QNN SDK builds, and Core ML repo failures.
 */
declare const global: any;
const mockFetch = jest.fn();
global.fetch = mockFetch as any;

import { fetchAvailableModels } from '../../../src/services/huggingFaceModelBrowser';

function res(body: unknown, link?: string) {
  return {
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
    headers: { get: (h: string) => (h.toLowerCase() === 'link' ? link ?? null : null) },
  };
}
const fileEntry = (path: string, size = 100) => ({ type: 'file', path, size });

describe('huggingFaceModelBrowser paging + dedupe', () => {
  beforeEach(() => mockFetch.mockReset());

  it('follows the HF Link rel="next" page so models past page one are listed', async () => {
    mockFetch
      .mockResolvedValueOnce(res([fileEntry('AnythingV5.zip')], '<https://huggingface.co/api/models/xororz/sd-mnn/tree/main?cursor=abc>; rel="next"'))
      .mockResolvedValueOnce(res([fileEntry('ChilloutMix.zip')]));
    const models = await fetchAvailableModels(true, { skipQnn: true });
    expect(models.map(m => m.name)).toEqual(['AnythingV5', 'ChilloutMix']);
    expect(mockFetch.mock.calls[1][0]).toBe('https://huggingface.co/api/models/xororz/sd-mnn/tree/main?cursor=abc');
  });

  it('never follows a next link to another host', async () => {
    mockFetch.mockResolvedValueOnce(res([fileEntry('AnythingV5.zip')], '<https://evil.io/x>; rel="next"'));
    const models = await fetchAvailableModels(true, { skipQnn: true });
    expect(models).toHaveLength(1);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('rejects a non-array tree body as a repo failure', async () => {
    mockFetch.mockResolvedValueOnce(res({ error: 'x' }));
    await expect(fetchAvailableModels(true, { skipQnn: true })).rejects.toThrow(/unexpected response/);
  });

  it('keeps one card per model+chip, the newest QNN SDK build', async () => {
    mockFetch
      .mockResolvedValueOnce(res([]))
      .mockResolvedValueOnce(res([
        fileEntry('AnythingV5_qnn2.9_8gen2.zip', 1),
        fileEntry('AnythingV5_qnn2.28_8gen2.zip', 2),
        fileEntry('AnythingV5_qnn2.10_8gen2.zip', 3),
        fileEntry('AnythingV5_qnn2.28_min.zip', 4),
      ]));
    const models = await fetchAvailableModels(true);
    const ids = models.map(m => m.id);
    expect(new Set(ids).size).toBe(ids.length);
    const gen2 = models.find(m => m.id === 'anythingv5_npu_8gen2');
    expect(gen2?.fileName).toBe('AnythingV5_qnn2.28_8gen2.zip');
    expect(gen2?.size).toBe(2);
    expect(models.find(m => m.id === 'anythingv5_npu_min')).toBeDefined();
    expect((gen2 as any).qnnVersion).toBeUndefined();
  });
});

describe('coreMLModelBrowser failures', () => {
  beforeEach(() => mockFetch.mockReset());

  it('throws (screen shows Retry) when every repo fails, and caches nothing', async () => {
    const { fetchAvailableCoreMLModels } = require('../../../src/services/coreMLModelBrowser');
    mockFetch.mockResolvedValue({ ok: false, status: 503, json: () => Promise.resolve(null) });
    await expect(fetchAvailableCoreMLModels(true)).rejects.toBeDefined();
    // A later open retries the network instead of answering a cached blank list.
    mockFetch.mockClear();
    await expect(fetchAvailableCoreMLModels(false)).rejects.toBeDefined();
    expect(mockFetch).toHaveBeenCalled();
  });
});
