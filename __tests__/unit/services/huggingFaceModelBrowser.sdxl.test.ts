/**
 * SDXL NPU catalog entries (local-dream v2.8.1 core, `--type sdxl`): the six SDXL repos are fetched
 * only when the device tier allows it, their zips parse to NPU SDXL models, and only zips under
 * 4 GB are listed.
 */
import {
  fetchAvailableModels,
  getVariantLabel,
  SDXL_MAX_ZIP_BYTES,
  SDXL_REPOS,
} from '../../../src/services/huggingFaceModelBrowser';
import { isSdxlNpuSoc } from '../../../src/services/hardware';
import { matchesSdVersionFilter } from '../../../src/screens/ModelsScreen/utils';

const GB = 1024 * 1024 * 1024;
const mockFetch = jest.fn();
(globalThis as any).fetch = mockFetch;

function lfsFile(path: string, size: number) {
  return { type: 'file', path, size: 135, lfs: { oid: 'x', size, pointerSize: 135 } };
}

/** Answer each repo's tree by URL, so the result does not depend on the fetch call order. */
function serveRepos(trees: Record<string, unknown[]>) {
  mockFetch.mockImplementation((url: string) => {
    const repo = Object.keys(trees).find(r => url === `https://huggingface.co/api/models/${r}/tree/main`);
    if (!repo) return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) });
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(trees[repo]) });
  });
}

const fetchedUrls = () => mockFetch.mock.calls.map(c => c[0] as string);

describe('huggingFaceModelBrowser — SDXL NPU repos', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('lists the six SDXL repos named for the catalog', () => {
    expect([...SDXL_REPOS]).toEqual([
      'xororz/sdxl-qnn',
      'YuuiKurata/lustify_qnn2.28',
      'YuuiKurata/bigLove_qnn2.28',
      'YuuiKurata/bigLust_qnn2.28',
      'YuuiKurata/ponyRealism_qnn2.28',
      'YuuiKurata/cyberrealisticPony_qnn2.28',
    ]);
  });

  it('fetches no SDXL repo unless includeSdxl is set (tier filter)', async () => {
    serveRepos({ 'xororz/sd-mnn': [], 'xororz/sd-qnn': [] });
    await fetchAvailableModels(true, { skipQnn: false });
    expect(fetchedUrls().some(u => SDXL_REPOS.some(r => u.includes(r)))).toBe(false);
  });

  it('never fetches SDXL repos on a device without an NPU, even if asked', async () => {
    serveRepos({ 'xororz/sd-mnn': [] });
    await fetchAvailableModels(true, { skipQnn: true, includeSdxl: true });
    expect(fetchedUrls()).toEqual(['https://huggingface.co/api/models/xororz/sd-mnn/tree/main']);
  });

  it('parses SDXL zips as NPU SDXL models with the 8gen3 variant and the repo download URL', async () => {
    serveRepos({
      'xororz/sd-mnn': [],
      'xororz/sd-qnn': [],
      'xororz/sdxl-qnn': [lfsFile('illustrious_v16_qnn2.28_8gen3.zip', 3.6 * GB), lfsFile('README.md', 10)],
      'YuuiKurata/lustify_qnn2.28': [lfsFile('lustify_qnn2.28_8gen3.zip', 3.5 * GB)],
    });
    const models = await fetchAvailableModels(true, { includeSdxl: true });
    const sdxl = models.filter(m => m.kind === 'sdxl');
    expect(sdxl).toHaveLength(2);
    expect(sdxl.find(m => m.repo === 'xororz/sdxl-qnn')).toMatchObject({
      id: 'illustrious_v16_sdxl_npu_8gen3',
      displayName: 'illustrious v16 (SDXL NPU)',
      backend: 'qnn',
      kind: 'sdxl',
      variant: '8gen3',
      downloadUrl: 'https://huggingface.co/xororz/sdxl-qnn/resolve/main/illustrious_v16_qnn2.28_8gen3.zip',
    });
    expect(sdxl.find(m => m.repo === 'YuuiKurata/lustify_qnn2.28')?.downloadUrl)
      .toBe('https://huggingface.co/YuuiKurata/lustify_qnn2.28/resolve/main/lustify_qnn2.28_8gen3.zip');
  });

  it('lists only SDXL zips under 4 GB', async () => {
    serveRepos({
      'xororz/sd-mnn': [],
      'xororz/sd-qnn': [],
      'xororz/sdxl-qnn': [
        lfsFile('small_qnn2.28_8gen3.zip', 3.79 * GB),
        lfsFile('big_qnn2.28_8gen3.zip', SDXL_MAX_ZIP_BYTES),
      ],
    });
    const ids = (await fetchAvailableModels(true, { includeSdxl: true })).map(m => m.id);
    expect(ids).toContain('small_sdxl_npu_8gen3');
    expect(ids).not.toContain('big_sdxl_npu_8gen3');
  });

  it('sorts SDXL after the SD1.5 NPU models and tags SD1.5 entries sd15', async () => {
    serveRepos({
      'xororz/sd-mnn': [lfsFile('AnythingV5.zip', 1 * GB)],
      'xororz/sd-qnn': [lfsFile('AnythingV5_qnn2.28_8gen2.zip', 1 * GB)],
      'xororz/sdxl-qnn': [lfsFile('aaa_qnn2.28_8gen3.zip', 3.5 * GB)],
    });
    const models = await fetchAvailableModels(true, { includeSdxl: true });
    expect(models.map(m => m.kind)).toEqual(['sd15', 'sd15', 'sdxl']);
  });

  it('keeps the other repos when one SDXL repo fails', async () => {
    serveRepos({ 'xororz/sd-mnn': [], 'xororz/sd-qnn': [], 'xororz/sdxl-qnn': [lfsFile('a_qnn2.28_8gen3.zip', GB)] });
    const models = await fetchAvailableModels(true, { includeSdxl: true });
    expect(models.map(m => m.id)).toEqual(['a_sdxl_npu_8gen3']);
  });

  it('labels the 8gen3 variant as SDXL on the NPU', () => {
    expect(getVariantLabel('8gen3')).toContain('SDXL');
    expect(getVariantLabel('8gen3')).toContain('NPU');
  });

  it('the SD-version filter finds SDXL entries by kind, not by file name', () => {
    expect(matchesSdVersionFilter('illustrious_v16', 'sdxl', 'sdxl')).toBe(true);
    expect(matchesSdVersionFilter('illustrious_v16', 'sd15', 'sdxl')).toBe(false);
    expect(matchesSdVersionFilter('illustrious_v16', 'sdxl', 'sd15')).toBe(false);
  });
});

describe('isSdxlNpuSoc (SDXL tier filter, upstream Model.kt:545)', () => {
  it.each(['SM8650', 'SM8750', 'SM8750P', 'SM8850', 'SM8850P', 'SM8845', 'SM8850-AC', 'sm8850'])('%s runs SDXL', soc => {
    expect(isSdxlNpuSoc(soc)).toBe(true);
  });

  it.each(['SM8550', 'SM8475', 'SM7675', '', 'Exynos2400'])('%s does not', soc => {
    expect(isSdxlNpuSoc(soc)).toBe(false);
  });
});
