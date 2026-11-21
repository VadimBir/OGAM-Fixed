/**
 * imageDenoiseStrength: the old saved default 0.6 (no screen ever set it) moves to 0.4 = Image
 * Studio "Keep 60%" once; any other value is the user's own and stays. Flag makes it one-time.
 */
import { migratePersistedState } from '../../../src/stores/appStoreMigrations';
import { DEFAULT_SETTINGS } from '../../../src/stores/appStore';

const migrate = (persistedSettings: Record<string, unknown>) =>
  migratePersistedState<any>(
    { settings: persistedSettings },
    { settings: { ...DEFAULT_SETTINGS } },
    { defaultSettings: DEFAULT_SETTINGS as any },
  ).settings;

describe('appStoreMigrations: image Keep 60% default', () => {
  it('fresh installs default to denoise 0.4 (Keep 60%)', () => {
    expect(DEFAULT_SETTINGS.imageDenoiseStrength).toBe(0.4);
  });
  it('moves the old saved default 0.6 to 0.4 and sets the flag', () => {
    const s = migrate({ imageDenoiseStrength: 0.6 });
    expect(s.imageDenoiseStrength).toBe(0.4);
    expect(s.imageKeepDefault60).toBe(true);
  });
  it('keeps any other saved value', () => {
    expect(migrate({ imageDenoiseStrength: 0.25 }).imageDenoiseStrength).toBe(0.25);
  });
  it('runs once: after the flag a later 0.6 is the user\'s choice and stays', () => {
    expect(migrate({ imageDenoiseStrength: 0.6, imageKeepDefault60: true }).imageDenoiseStrength).toBe(0.6);
  });
});
