/**
 * modelManager/localModelFile: the on-disk name of a downloaded text model and whose file it is.
 */
import {
  diskFileNameOf,
  repoIdOfModel,
  modelRowMatchesCatalogFile,
  candidateLocalFileName,
  localFileNameFromMetadataJson,
  resolveLocalModelFile,
} from '../../../../src/services/modelManager/localModelFile';
import type { DownloadedModel } from '../../../../src/types';
import type { DownloadEntry } from '../../../../src/utils/downloadStatus';

const DIR = '/models';
const row = (over: Partial<DownloadedModel> & { id: string; fileName: string }): DownloadedModel => ({
  name: 'n', author: 'a', filePath: `${DIR}/${over.fileName.split('/').pop()}`, fileSize: 100,
  quantization: 'Q4_K_M', downloadedAt: '', engine: 'llama', ...over,
} as DownloadedModel);
const dl = (over: Partial<DownloadEntry>): DownloadEntry => ({
  modelKey: 'k', downloadId: 'd', modelId: 'x/y', fileName: 'f.gguf', quantization: '', modelType: 'text',
  status: 'running', bytesDownloaded: 0, totalBytes: 0, combinedTotalBytes: 0, progress: 0, ...over,
} as DownloadEntry);
/** Probe over a fake directory: name -> size. */
const probe = (files: Record<string, number>) => ({
  stat: jest.fn(async (name: string) => (name in files ? { exists: true, size: files[name] } : { exists: false })),
});

describe('identity helpers', () => {
  it('diskFileNameOf: base of filePath, else fileName', () => {
    expect(diskFileNameOf({ filePath: '/a/b/x--o-123456.gguf', fileName: 'x.gguf' })).toBe('x--o-123456.gguf');
    expect(diskFileNameOf({ filePath: '', fileName: 'x.gguf' })).toBe('x.gguf');
  });

  it('repoIdOfModel: origin, then id minus fileName (folder names included), never local_import', () => {
    expect(repoIdOfModel({ id: 'o/r/f.gguf', fileName: 'f.gguf', origin: { repoId: 'z/w' } as any })).toBe('z/w');
    expect(repoIdOfModel({ id: 'o/r/f.gguf', fileName: 'f.gguf' })).toBe('o/r');
    expect(repoIdOfModel({ id: 'o/r/Q4/f.gguf', fileName: 'Q4/f.gguf' })).toBe('o/r');
    expect(repoIdOfModel({ id: 'local_import/f.gguf', fileName: 'f.gguf' })).toBeUndefined();
    expect(repoIdOfModel({ id: 'recovered_f.gguf', fileName: 'f.gguf' })).toBeUndefined();
  });

  it('modelRowMatchesCatalogFile: exact id, or unattributed same name; never another repo', () => {
    expect(modelRowMatchesCatalogFile({ id: 'o/r/f.gguf', fileName: 'f.gguf' }, 'o/r', 'f.gguf')).toBe(true);
    expect(modelRowMatchesCatalogFile({ id: 'recovered_f.gguf', fileName: 'f.gguf' }, 'o/r', 'f.gguf')).toBe(true);
    expect(modelRowMatchesCatalogFile({ id: 'p/q/f.gguf', fileName: 'f.gguf' }, 'o/r', 'f.gguf')).toBe(false);
    expect(modelRowMatchesCatalogFile({ id: 'o/r/g.gguf', fileName: 'g.gguf' }, 'o/r', 'f.gguf')).toBe(false);
  });

  it('candidateLocalFileName: plain base, then owner+hash qualified, then numbered; deterministic', () => {
    expect(candidateLocalFileName('o/r', 'Q4/f.gguf', 0)).toBe('f.gguf');
    const q1 = candidateLocalFileName('un-sloth/Qwen3', 'f.gguf', 1);
    expect(q1).toMatch(/^f--unsloth-[0-9a-z]{6}\.gguf$/);
    expect(candidateLocalFileName('un-sloth/Qwen3', 'f.gguf', 1)).toBe(q1);
    expect(candidateLocalFileName('un-sloth/Qwen3', 'f.gguf', 2)).toBe(q1.replace('.gguf', '-2.gguf'));
    expect(candidateLocalFileName('other/Qwen3', 'f.gguf', 1)).not.toBe(q1);
  });

  it('localFileNameFromMetadataJson: string field only; bad JSON -> undefined', () => {
    expect(localFileNameFromMetadataJson(JSON.stringify({ localFileName: 'a.gguf' }))).toBe('a.gguf');
    expect(localFileNameFromMetadataJson(JSON.stringify({ localFileName: '' }))).toBeUndefined();
    expect(localFileNameFromMetadataJson(JSON.stringify({ localFileName: 3 }))).toBeUndefined();
    expect(localFileNameFromMetadataJson('{not json')).toBeUndefined();
    expect(localFileNameFromMetadataJson(undefined)).toBeUndefined();
  });
});

describe('resolveLocalModelFile', () => {
  const base = { modelId: 'o/r', catalogFileName: 'f.gguf', catalogSize: 100, registry: [] as DownloadedModel[], downloads: [] as DownloadEntry[] };

  it('free name -> plain, not complete', async () => {
    expect(await resolveLocalModelFile({ ...base, probe: probe({}) })).toEqual({ name: 'f.gguf', complete: false, exists: false });
  });

  it('unregistered file of the catalog size -> complete (never re-fetched)', async () => {
    expect(await resolveLocalModelFile({ ...base, probe: probe({ 'f.gguf': 100 }) }))
      .toEqual({ name: 'f.gguf', complete: true, exists: true });
  });

  it('unregistered file of another size -> same name, fetched again, exists flagged', async () => {
    expect(await resolveLocalModelFile({ ...base, probe: probe({ 'f.gguf': 7 }) }))
      .toEqual({ name: 'f.gguf', complete: false, exists: true });
  });

  it('own registered row -> complete when the file is there (registered size wins)', async () => {
    const registry = [row({ id: 'o/r/f.gguf', fileName: 'f.gguf' })];
    expect(await resolveLocalModelFile({ ...base, registry, probe: probe({ 'f.gguf': 55 }) }))
      .toEqual({ name: 'f.gguf', complete: true, exists: true });
  });

  it('another repo owns the plain name -> repo-qualified name; its file is untouched', async () => {
    const registry = [row({ id: 'p/q/f.gguf', fileName: 'f.gguf' })];
    const p = probe({ 'f.gguf': 100 });
    const r = await resolveLocalModelFile({ ...base, registry, probe: p });
    expect(r.name).toBe(candidateLocalFileName('o/r', 'f.gguf', 1));
    expect(r.complete).toBe(false);
  });

  it('unattributed registered row of other bytes (local import) -> qualified name', async () => {
    const registry = [row({ id: 'local_import/f.gguf', fileName: 'f.gguf' })];
    const r = await resolveLocalModelFile({ ...base, registry, probe: probe({ 'f.gguf': 7 }) });
    expect(r.name).toBe(candidateLocalFileName('o/r', 'f.gguf', 1));
  });

  it('another repo in flight on the plain name -> qualified; this repo\'s own retry row does not block', async () => {
    const other = dl({ modelId: 'p/q', fileName: 'f.gguf' });
    expect((await resolveLocalModelFile({ ...base, downloads: [other], probe: probe({}) })).name)
      .toBe(candidateLocalFileName('o/r', 'f.gguf', 1));
    const own = dl({ modelId: 'o/r', fileName: 'f.gguf' });
    expect((await resolveLocalModelFile({ ...base, downloads: [own], probe: probe({}) })).name).toBe('f.gguf');
    const done = dl({ modelId: 'p/q', fileName: 'f.gguf', status: 'completed' });
    expect((await resolveLocalModelFile({ ...base, downloads: [done], probe: probe({}) })).name).toBe('f.gguf');
  });

  it('another download\'s metadata localFileName claims the qualified name -> next number', async () => {
    const q1 = candidateLocalFileName('o/r', 'f.gguf', 1);
    const registry = [row({ id: 'p/q/f.gguf', fileName: 'f.gguf' })];
    const other = dl({ modelId: 'z/w', fileName: 'f.gguf', metadataJson: JSON.stringify({ localFileName: q1 }) });
    const r = await resolveLocalModelFile({ ...base, registry, downloads: [other], probe: probe({}) });
    expect(r.name).toBe(candidateLocalFileName('o/r', 'f.gguf', 2));
  });

  it('same repo, same base name in two folders: the other folder\'s install is not "ours"', async () => {
    const registry = [row({ id: 'o/r/v1/f.gguf', fileName: 'v1/f.gguf' })];
    const r = await resolveLocalModelFile({ ...base, catalogFileName: 'v2/f.gguf', registry, probe: probe({ 'f.gguf': 100 }) });
    expect(r.name).toBe(candidateLocalFileName('o/r', 'v2/f.gguf', 1));
    expect(r.complete).toBe(false);
  });

  it('same repo, same folder file: own row is recognised (also a legacy bare-name row)', async () => {
    const own = [row({ id: 'o/r/v1/f.gguf', fileName: 'v1/f.gguf' })];
    expect((await resolveLocalModelFile({ ...base, catalogFileName: 'v1/f.gguf', registry: own, probe: probe({ 'f.gguf': 1 }) })).complete).toBe(true);
    const legacy = [row({ id: 'o/r/f.gguf', fileName: 'f.gguf' })];
    expect((await resolveLocalModelFile({ ...base, catalogFileName: 'v1/f.gguf', registry: legacy, probe: probe({ 'f.gguf': 1 }) })).complete).toBe(true);
  });

  it('same repo, another folder\'s in-flight download -> qualified name', async () => {
    const other = dl({ modelId: 'o/r', fileName: 'v1/f.gguf' });
    const r = await resolveLocalModelFile({ ...base, catalogFileName: 'v2/f.gguf', downloads: [other], probe: probe({}) });
    expect(r.name).toBe(candidateLocalFileName('o/r', 'v2/f.gguf', 1));
  });

  it('throws when every candidate is taken', async () => {
    const registry = Array.from({ length: 6 }, (_, i) =>
      row({ id: `p${i}/q/f.gguf`, fileName: 'f.gguf', filePath: `${DIR}/${candidateLocalFileName('o/r', 'f.gguf', i)}` }));
    await expect(resolveLocalModelFile({ ...base, registry, probe: probe({}) })).rejects.toThrow('No free file name');
  });
});
