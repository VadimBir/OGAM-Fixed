/**
 * Old installs get the Advanced sampling / load / image-default fields at their defaults, and a value
 * of the wrong shape is replaced by the default (migrateAdvancedSettings). A valid user value stays.
 */
import { migratePersistedState } from '../../../src/stores/appStoreMigrations';
import { DEFAULT_SETTINGS } from '../../../src/stores/appStore';

const migrate = (persistedSettings: Record<string, unknown>) =>
  migratePersistedState<any>(
    { settings: persistedSettings },
    { settings: { ...DEFAULT_SETTINGS } },
    { defaultSettings: DEFAULT_SETTINGS as any },
  ).settings;

describe('appStoreMigrations: advanced settings', () => {
  it('an install from before the advanced fields gets every default (behaviour unchanged)', () => {
    const s = migrate({ temperature: 0.5, cacheType: 'q4_0', inferenceBackend: 'cpu', nBatch: 256 });
    expect(s).toMatchObject({
      temperature: 0.5, cacheType: 'q4_0', nBatch: 256,
      topK: 40, minP: 0.05, typicalP: 1, penaltyLastN: 64, presencePenalty: 0, frequencyPenalty: 0,
      dryMultiplier: 0, dryBase: 1.75, dryAllowedLength: 2, dryPenaltyLastN: 64,
      xtcProbability: 0, xtcThreshold: 0.1, mirostat: 0, mirostatTau: 5, mirostatEta: 0.1, seed: -1,
      customStopStrings: [], liteRTTopK: 40,
      nUbatch: 0, mmapMode: 'auto', cacheTypeV: 'same',
      imageNegativePrompt: '', imageSeed: -1, imageClipSkip: 0,
    });
  });

  it('keeps valid user values', () => {
    const s = migrate({
      cacheType: 'q8_0', inferenceBackend: 'cpu',
      topK: 10, mirostat: 2, seed: 99, customStopStrings: ['x'], nUbatch: 128, mmapMode: 'off',
      cacheTypeV: 'f16', imageNegativePrompt: 'blurry', imageSeed: 5, imageClipSkip: 2,
    });
    expect(s).toMatchObject({
      topK: 10, mirostat: 2, seed: 99, customStopStrings: ['x'], nUbatch: 128, mmapMode: 'off',
      cacheTypeV: 'f16', imageNegativePrompt: 'blurry', imageSeed: 5, imageClipSkip: 2,
    });
  });

  it('replaces wrong-shaped values with the default', () => {
    const s = migrate({
      cacheType: 'q8_0', inferenceBackend: 'cpu',
      topK: 'forty', minP: null, mirostat: 5, seed: 1.5, customStopStrings: 'x', nUbatch: -3,
      mmapMode: 'sometimes', cacheTypeV: 'q2_k', imageNegativePrompt: 3, imageSeed: null, imageClipSkip: -1,
    });
    expect(s).toMatchObject({
      topK: 40, minP: 0.05, mirostat: 0, seed: -1, customStopStrings: [], nUbatch: 0,
      mmapMode: 'auto', cacheTypeV: 'same', imageNegativePrompt: '', imageSeed: -1, imageClipSkip: 0,
    });
  });
});
