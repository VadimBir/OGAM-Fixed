import {
  buildCivitaiSearchUrl,
  civitaiAuthHeaders,
  civitaiBaseModelFamily,
  isCivitaiUrl,
  pickRunnableFile,
  redactCivitaiToken,
  searchCivitaiModels,
  withCivitaiDownloadToken,
} from '../../../src/services/civitaiBrowser';

const mockFetch = jest.fn();
(globalThis as any).fetch = mockFetch;

function file(over: Record<string, any> = {}) {
  return {
    id: 11,
    name: 'dream.safetensors',
    sizeKB: 2_000_000,
    type: 'Model',
    primary: true,
    metadata: { format: 'SafeTensor', fp: 'fp16', size: 'pruned' },
    hashes: { SHA256: 'A'.repeat(64) },
    downloadUrl: 'https://civitai.com/api/download/models/101',
    ...over,
  };
}

function item(over: Record<string, any> = {}, versionOver: Record<string, any> = {}) {
  return {
    id: 1,
    name: 'DreamShaper',
    type: 'Checkpoint',
    nsfw: false,
    creator: { username: 'lykon' },
    stats: { downloadCount: 1234 },
    modelVersions: [{
      id: 101,
      name: 'v8',
      baseModel: 'SD 1.5',
      files: [file()],
      images: [{ url: 'https://image.civitai.com/x/1.jpeg', nsfwLevel: 1, type: 'image' }],
      ...versionOver,
    }],
    ...over,
  };
}

function respond(body: unknown, init: { ok?: boolean; status?: number; badJson?: boolean } = {}) {
  mockFetch.mockResolvedValueOnce({
    ok: init.ok ?? true,
    status: init.status ?? 200,
    json: init.badJson ? () => Promise.reject(new SyntaxError('bad json')) : () => Promise.resolve(body),
  });
}

const base = { type: 'Checkpoint' as const, family: 'sd1' as const };

describe('civitaiBrowser', () => {
  beforeEach(() => mockFetch.mockReset());

  describe('host-scoped auth', () => {
    it('recognises only https civitai.com and its subdomains', () => {
      expect(isCivitaiUrl('https://civitai.com/api/v1/models')).toBe(true);
      expect(isCivitaiUrl('https://image.civitai.com/a.jpeg')).toBe(true);
      expect(isCivitaiUrl('http://civitai.com/api')).toBe(false);
      expect(isCivitaiUrl('https://notcivitai.com/api')).toBe(false);
      expect(isCivitaiUrl('https://civitai.com.evil.io/api')).toBe(false);
      expect(isCivitaiUrl('https://evil.io/?u=https://civitai.com')).toBe(false);
    });

    it('sends the Bearer key only to civitai.com', () => {
      expect(civitaiAuthHeaders('https://civitai.com/api/v1/models', 'k1')).toEqual({ Authorization: 'Bearer k1' });
      expect(civitaiAuthHeaders('https://huggingface.co/api', 'k1')).toEqual({});
      expect(civitaiAuthHeaders('https://civitai.com/api', '  ')).toEqual({});
      expect(civitaiAuthHeaders('https://civitai.com/api', null)).toEqual({});
    });

    it('adds the download token only to civitai.com URLs, and redacts it', () => {
      const url = withCivitaiDownloadToken('https://civitai.com/api/download/models/1?type=Model', 'a b');
      expect(url).toBe('https://civitai.com/api/download/models/1?type=Model&token=a%20b');
      expect(withCivitaiDownloadToken('https://civitai.com/api/download/models/1', 'k')).toBe('https://civitai.com/api/download/models/1?token=k');
      expect(withCivitaiDownloadToken('https://cdn.other.com/f', 'k')).toBe('https://cdn.other.com/f');
      expect(withCivitaiDownloadToken('https://civitai.com/x', undefined)).toBe('https://civitai.com/x');
      expect(redactCivitaiToken(url)).toBe('https://civitai.com/api/download/models/1?type=Model&token=***');
    });

    it('puts the Authorization header on the search request when a key is set', async () => {
      respond({ items: [], metadata: {} });
      await searchCivitaiModels({ ...base, apiKey: 'secret' });
      const [url, init] = mockFetch.mock.calls[0];
      expect(isCivitaiUrl(url)).toBe(true);
      expect(init.headers.Authorization).toBe('Bearer secret');
    });

    it('sends no Authorization header without a key', async () => {
      respond({ items: [], metadata: {} });
      await searchCivitaiModels(base);
      expect(mockFetch.mock.calls[0][1].headers.Authorization).toBeUndefined();
    });
  });

  describe('search URL', () => {
    it('carries query, type, sort, family base models and cursor, and never an nsfw filter', () => {
      const url = buildCivitaiSearchUrl({ ...base, query: ' dream shaper ', cursor: 'c1' });
      expect(url.startsWith('https://civitai.com/api/v1/models?')).toBe(true);
      expect(url).toContain('types=Checkpoint');
      expect(url).toContain('query=dream%20shaper');
      expect(url).toContain('sort=Most%20Downloaded');
      expect(url).not.toContain('nsfw=');
      expect(url).toContain('baseModels=SD%201.5');
      expect(url).toContain('cursor=c1');
      expect(url).toContain('limit=20');
    });

    it('omits baseModels on request', () => {
      const url = buildCivitaiSearchUrl({ ...base, type: 'LORA', family: 'sdxl' }, false);
      expect(url).not.toContain('nsfw=');
      expect(url).not.toContain('baseModels');
      expect(url).toContain('types=LORA');
      expect(url).not.toContain('query=');
    });
  });

  describe('base model families', () => {
    it.each([
      ['SD 1.5', 'sd1'], ['SD 1.4', 'sd1'], ['SD 1.5 LCM', 'sd1'], ['SD 2.1', 'sd2'], ['SD 2.0 768', 'sd2'],
      ['SDXL 1.0', 'sdxl'], ['Pony', 'sdxl'], ['Illustrious', 'sdxl'],
      ['Flux.1 D', null], ['SD 3', null], ['SD 3.5', null], ['Hunyuan Video', null], [undefined, null],
    ])('%s -> %s', (label, family) => {
      expect(civitaiBaseModelFamily(label)).toBe(family);
    });
  });

  describe('runnable file choice', () => {
    it('never offers PickleTensor, GGUF, VAE, or non-civitai URLs', () => {
      expect(pickRunnableFile([file({ name: 'm.ckpt', metadata: { format: 'PickleTensor' } })])).toBeNull();
      expect(pickRunnableFile([file({ name: 'm.safetensors', metadata: { format: 'PickleTensor' } })])).toBeNull();
      expect(pickRunnableFile([file({ name: 'm.gguf', metadata: { format: 'GGUF' } })])).toBeNull();
      expect(pickRunnableFile([file({ type: 'VAE' })])).toBeNull();
      expect(pickRunnableFile([file({ downloadUrl: 'https://evil.io/m.safetensors' })])).toBeNull();
      expect(pickRunnableFile([file({ sizeKB: 0 })])).toBeNull();
      expect(pickRunnableFile('nope')).toBeNull();
    });

    it('prefers the primary SafeTensor, else the smallest, and converts sizeKB to bytes', () => {
      const chosen = pickRunnableFile([
        file({ id: 1, primary: false, sizeKB: 4_000_000 }),
        file({ id: 2, primary: false, sizeKB: 2_000_000, type: 'Pruned Model' }),
        file({ id: 3, primary: true, name: 'p.ckpt', metadata: { format: 'PickleTensor' } }),
      ]);
      expect(chosen?.id).toBe(2);
      expect(chosen?.sizeBytes).toBe(2_000_000 * 1024);
      expect(pickRunnableFile([file({ id: 5, primary: false }), file({ id: 6, primary: true })])?.id).toBe(6);
    });

    it('keeps a valid SHA-256 and drops a malformed one', () => {
      expect(pickRunnableFile([file()])?.sha256).toBe('A'.repeat(64));
      expect(pickRunnableFile([file({ hashes: { SHA256: 'xyz' } })])?.sha256).toBeUndefined();
    });
  });

  describe('searchCivitaiModels', () => {
    it('parses runnable results with preview, size, base model and creator', async () => {
      respond({ items: [item()], metadata: { nextCursor: 'next-1' } });
      const page = await searchCivitaiModels(base);
      expect(page.nextCursor).toBe('next-1');
      expect(page.skipped).toBe(0);
      expect(page.results).toHaveLength(1);
      expect(page.results[0]).toMatchObject({
        key: '1:101:11', modelId: 1, versionId: 101, name: 'DreamShaper', versionName: 'v8',
        type: 'Checkpoint', baseModel: 'SD 1.5', family: 'sd1', creator: 'lykon', downloadCount: 1234,
        previewUrl: 'https://image.civitai.com/x/1.jpeg',
      });
    });

    it('skips other types, other families and items without a runnable file, keeps NSFW models', async () => {
      respond({
        items: [
          item({ id: 2, type: 'LORA' }),
          item({ id: 3 }, { baseModel: 'SDXL 1.0' }),
          item({ id: 4, nsfw: true }),
          item({ id: 5 }, { files: [file({ name: 'x.ckpt', metadata: { format: 'PickleTensor' } })] }),
          { garbage: true },
          null,
          item({ id: 6 }),
        ],
        metadata: {},
      });
      const page = await searchCivitaiModels(base);
      expect(page.results.map(r => r.modelId)).toEqual([4, 6]);
      expect(page.results[0].nsfw).toBe(true);
      expect(page.skipped).toBe(5);
      expect(page.nextCursor).toBeNull();
    });

    it('uses the first version of the requested family', async () => {
      respond({
        items: [item({}, {})].map(i => ({
          ...i,
          modelVersions: [
            { id: 200, name: 'XL', baseModel: 'SDXL 1.0', files: [file({ id: 20 })], images: [] },
            { id: 100, name: '1.5', baseModel: 'SD 1.5', files: [file({ id: 10 })], images: [] },
          ],
        })),
      });
      const page = await searchCivitaiModels(base);
      expect(page.results[0].versionId).toBe(100);
      expect(page.results[0].previewUrl).toBeUndefined();
    });

    it('shows the first still preview whatever its rating (no content filter)', async () => {
      const images = [
        { url: 'https://image.civitai.com/r.jpeg', nsfwLevel: 4 },
        { url: 'https://image.civitai.com/u.jpeg' },
        { url: 'https://image.civitai.com/v.mp4', nsfwLevel: 1, type: 'video' },
        { url: 'https://image.civitai.com/s.jpeg', nsfw: 'None' },
      ];
      respond({ items: [item({ nsfw: true }, { images })] });
      const page = await searchCivitaiModels(base);
      expect(page.results[0].previewUrl).toBe('https://image.civitai.com/r.jpeg');
      expect(page.results[0].nsfw).toBe(true);
    });

    it('reads the cursor from metadata.nextPage when nextCursor is absent', async () => {
      respond({ items: [], metadata: { nextPage: 'https://civitai.com/api/v1/models?limit=20&cursor=abc%7C1' } });
      expect((await searchCivitaiModels(base)).nextCursor).toBe('abc|1');
    });

    it('passes the cursor of a follow-up page', async () => {
      respond({ items: [] });
      await searchCivitaiModels({ ...base, cursor: 'xyz' });
      expect(mockFetch.mock.calls[0][0]).toContain('cursor=xyz');
    });

    it('retries once without baseModels when the API answers 400', async () => {
      respond(null, { ok: false, status: 400 });
      respond({ items: [item()] });
      const page = await searchCivitaiModels(base);
      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(mockFetch.mock.calls[0][0]).toContain('baseModels=');
      expect(mockFetch.mock.calls[1][0]).not.toContain('baseModels=');
      expect(page.results).toHaveLength(1);
    });

    it('explains a rejected key, rate limiting and other HTTP errors', async () => {
      respond(null, { ok: false, status: 401 });
      await expect(searchCivitaiModels(base)).rejects.toThrow(/API key/);
      respond(null, { ok: false, status: 429 });
      await expect(searchCivitaiModels(base)).rejects.toThrow(/rate-limiting/);
      respond(null, { ok: false, status: 500 });
      await expect(searchCivitaiModels(base)).rejects.toThrow(/HTTP 500/);
    });

    it('rejects unreadable JSON and a body without items', async () => {
      respond(null, { badJson: true });
      await expect(searchCivitaiModels(base)).rejects.toThrow(/unreadable/);
      respond({ results: [] });
      await expect(searchCivitaiModels(base)).rejects.toThrow(/unexpected/);
    });

    it('reports a timeout as a connection problem', async () => {
      mockFetch.mockRejectedValueOnce(Object.assign(new Error('aborted'), { name: 'AbortError' }));
      await expect(searchCivitaiModels(base)).rejects.toThrow(/did not answer in time/);
    });
  });
});
