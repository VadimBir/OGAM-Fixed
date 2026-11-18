/**
 * modelManager/registryLock: every registry load-modify-save runs in order; and the registry
 * self-heal a READER triggers (storage.loadDownloadedModels / loadDownloadedImageModels, run by the
 * launch scan outside the lock) is persisted under the lock from a fresh read, so it cannot
 * overwrite a row another writer saved meanwhile.
 */
import RNFS from 'react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { withModelRegistryLock } from '../../../../src/services/modelManager/registryLock';
import {
  loadDownloadedModels,
  loadDownloadedImageModels,
  persistDownloadedModel,
} from '../../../../src/services/modelManager/storage';

const mockedRNFS = RNFS as jest.Mocked<typeof RNFS>;
const mockedAsyncStorage = AsyncStorage as jest.Mocked<typeof AsyncStorage>;
const flush = () => new Promise(resolve => setImmediate(resolve));

describe('withModelRegistryLock', () => {
  it('runs writers one after another, each seeing the previous result', async () => {
    const order: string[] = [];
    let value = 0;
    const slow = (name: string) => async () => {
      const seen = value;
      order.push(`${name}:start`);
      await flush();
      value = seen + 1;
      order.push(`${name}:end`);
      return value;
    };
    const results = await Promise.all([withModelRegistryLock(slow('a')), withModelRegistryLock(slow('b')), withModelRegistryLock(slow('c'))]);
    expect(results).toEqual([1, 2, 3]);
    expect(order).toEqual(['a:start', 'a:end', 'b:start', 'b:end', 'c:start', 'c:end']);
  });

  it('a failing writer rejects to its caller only; the next writer still runs', async () => {
    const failed = withModelRegistryLock(async () => { throw new Error('boom'); });
    const next = withModelRegistryLock(async () => 'ok');
    await expect(failed).rejects.toThrow('boom');
    await expect(next).resolves.toBe('ok');
  });
});

describe('registry self-heal runs under the lock', () => {
  const DIR = '/models';
  const TEXT_KEY = '@local_llm/downloaded_models';
  const IMAGE_KEY = '@local_llm/downloaded_image_models';
  let store: Record<string, string>;
  const row = (name: string) => ({ id: `o/r/${name}`, name, author: 'o', fileName: name, filePath: `${DIR}/${name}`, fileSize: 1, quantization: 'Q4_K_M', downloadedAt: '', engine: 'llama' });

  beforeEach(() => {
    store = {};
    mockedAsyncStorage.getItem.mockImplementation(async (k: string) => store[k] ?? null);
    mockedAsyncStorage.setItem.mockImplementation(async (k: string, v: string) => { store[k] = v; });
  });

  it('a reader\'s heal does not overwrite a row a locked writer saved between its read and its heal', async () => {
    store[TEXT_KEY] = JSON.stringify([row('gone.gguf'), row('present.gguf')]);
    // The reader's first probe of the missing file is slow; the finalize runs meanwhile.
    let releaseSlowProbe!: () => void;
    let firstGoneProbe = true;
    mockedRNFS.exists.mockImplementation(async (p: string) => {
      if (p.endsWith('gone.gguf') && firstGoneProbe) {
        firstGoneProbe = false;
        await new Promise<void>(resolve => { releaseSlowProbe = resolve; });
        return false;
      }
      return !p.endsWith('gone.gguf');
    });
    mockedRNFS.readDir.mockResolvedValue([]);

    const reader = loadDownloadedModels(DIR); // unlocked, as the launch scan's getDownloadedModels
    await flush();
    await persistDownloadedModel(row('new.gguf') as any, DIR); // locked finalize
    releaseSlowProbe();
    const seen = await reader;
    await flush();

    expect(seen.map(m => m.fileName)).toEqual(['present.gguf']);
    const saved = JSON.parse(store[TEXT_KEY]).map((m: { fileName: string }) => m.fileName);
    expect(saved).toEqual(['present.gguf', 'new.gguf']); // the finalized row survives
  });

  it('a heal with nothing to fix writes nothing', async () => {
    store[TEXT_KEY] = JSON.stringify([row('present.gguf')]);
    mockedRNFS.exists.mockResolvedValue(true);
    await loadDownloadedModels(DIR);
    await flush();
    expect(mockedAsyncStorage.setItem).not.toHaveBeenCalled();
  });

  it('a heal needed at read time is persisted from the current stored list', async () => {
    store[TEXT_KEY] = JSON.stringify([row('gone.gguf'), row('present.gguf')]);
    mockedRNFS.exists.mockImplementation(async (p: string) => !p.endsWith('gone.gguf'));
    await loadDownloadedModels(DIR);
    await flush();
    expect(JSON.parse(store[TEXT_KEY]).map((m: { fileName: string }) => m.fileName)).toEqual(['present.gguf']);
  });

  it('the image registry heal is persisted the same way', async () => {
    const img = (id: string) => ({ id, name: id, description: '', modelPath: `/img/${id}`, size: 1, downloadedAt: '', backend: 'mnn' });
    store[IMAGE_KEY] = JSON.stringify([img('gone'), img('kept')]);
    mockedRNFS.exists.mockImplementation(async (p: string) => !p.endsWith('/gone'));
    const seen = await loadDownloadedImageModels('/img');
    expect(seen.map(m => m.id)).toEqual(['kept']);
    await flush();
    expect(JSON.parse(store[IMAGE_KEY]).map((m: { id: string }) => m.id)).toEqual(['kept']);
  });

  it('a heal triggered by a reader INSIDE the lock does not deadlock', async () => {
    store[TEXT_KEY] = JSON.stringify([row('gone.gguf'), row('present.gguf')]);
    mockedRNFS.exists.mockImplementation(async (p: string) => !p.endsWith('gone.gguf'));
    const inside = await withModelRegistryLock(() => loadDownloadedModels(DIR));
    expect(inside.map(m => m.fileName)).toEqual(['present.gguf']);
    await flush();
    expect(JSON.parse(store[TEXT_KEY]).map((m: { fileName: string }) => m.fileName)).toEqual(['present.gguf']);
  });
});
