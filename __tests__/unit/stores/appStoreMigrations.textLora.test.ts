/**
 * settings.textLoras / settings.remoteTextLoras migration: old installs get empty values, malformed
 * rows are dropped, scales are clamped to 0-2, iOS container paths are rebased. Idempotent.
 */
import { migratePersistedState, migrateTextLoras } from '../../../src/stores/appStoreMigrations';
import { DEFAULT_SETTINGS } from '../../../src/stores/appStore';

const migrate = (persistedSettings: Record<string, unknown>, documentsPath?: string) =>
  migratePersistedState<any>(
    { settings: persistedSettings },
    { settings: { ...DEFAULT_SETTINGS } },
    { defaultSettings: DEFAULT_SETTINGS as any, documentsPath },
  ).settings;

describe('appStoreMigrations: text LoRA', () => {
  it('an install from before text LoRA gets empty lists', () => {
    const s = migrate({ cacheType: 'q8_0', inferenceBackend: 'cpu' });
    expect(s.textLoras).toEqual([]);
    expect(s.remoteTextLoras).toEqual({});
  });

  it('keeps valid rows, drops malformed ones, clamps scale', () => {
    const s = migrate({
      cacheType: 'q8_0', inferenceBackend: 'cpu',
      textLoras: [
        { id: 'a', name: 'a', path: '/d/text_loras/a.gguf', scale: 0.5, enabled: true },
        { id: 'b', name: 'b', path: '/d/text_loras/b.gguf', scale: 9, enabled: false },
        { id: 'c', name: 'c', path: '', scale: 1, enabled: true },
        { name: 'no-id', path: '/x', scale: 1, enabled: true },
        'junk',
      ],
      remoteTextLoras: { 'http://pc:8080': [{ id: 0, scale: 3 }, { id: 'x', scale: 1 }], bad: 'nope' },
    });
    expect(s.textLoras).toEqual([
      { id: 'a', name: 'a', path: '/d/text_loras/a.gguf', scale: 0.5, enabled: true },
      { id: 'b', name: 'b', path: '/d/text_loras/b.gguf', scale: 2, enabled: false },
    ]);
    expect(s.remoteTextLoras).toEqual({ 'http://pc:8080': [{ id: 0, scale: 2 }] });
  });

  it('rebases an iOS container path onto the current Documents folder', () => {
    const s = migrate({
      cacheType: 'q8_0', inferenceBackend: 'metal',
      textLoras: [{ id: 'a', name: 'a', path: '/var/mobile/Containers/Data/Application/OLD-UUID/Documents/text_loras/a.gguf', scale: 1, enabled: true }],
    }, '/var/mobile/Containers/Data/Application/NEW-UUID/Documents');
    expect(s.textLoras[0].path).toBe('/var/mobile/Containers/Data/Application/NEW-UUID/Documents/text_loras/a.gguf');
  });

  it('is idempotent', () => {
    const merged = { settings: { textLoras: [{ id: 'a', name: 'a', path: '/p', scale: 1.25, enabled: true }], remoteTextLoras: {} } };
    migrateTextLoras(merged);
    const once = JSON.stringify(merged);
    migrateTextLoras(merged);
    expect(JSON.stringify(merged)).toBe(once);
  });
});
