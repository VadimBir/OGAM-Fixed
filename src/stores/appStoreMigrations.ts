// One-time migrations for the persisted app store.
//
// Every function here repairs state written by an OLDER build, and each is idempotent: once a value
// no longer matches the shape it is looking for, it does nothing. `DEFAULT_SETTINGS` is passed IN
// rather than imported, so the store owns its defaults and this module stays a pure transform of
// whatever it is handed. It names no store type either, so nothing here imports the store back.

/** Only the settings fields these repairs actually read. The store's own type is a superset. */
interface PersistedSettings {
  contextLength: number;
  maxTokens: number;
  liteRTMaxTokens: number;
  [key: string]: unknown;
}

interface PersistedStateMigrationContext {
  defaultSettings: PersistedSettings;
  documentsPath?: string;
}
import { Platform } from 'react-native';
import { INFERENCE_BACKENDS } from '../types';

function migrateEnabledTools(merged: any): void {
  if (
    merged.settings?.enabledTools &&
    !merged.settings.enabledTools.includes('search_knowledge_base')
  ) {
    merged.settings = {
      ...merged.settings,
      enabledTools: [...merged.settings.enabledTools, 'search_knowledge_base'],
    };
  }
}

// The removed MCP context auto-boost pinned context to 32768 (and maxTokens to 8192 /
// liteRTMaxTokens to 32768) on MCP enable and never restored it, causing OOM crashes
// and tanked tok/s on flagship devices. Reset anyone left at the boost ceiling back to
// the device-safe defaults. Idempotent: once reset, the values no longer match.
const MCP_BOOST_CTX_CEILING = 32768;
const MCP_BOOST_MAX_OUTPUT_TOKENS = 8192;
function migrateBoostedContext(
  merged: any,
  DEFAULT_SETTINGS: PersistedSettings,
): void {
  const s = merged.settings;
  if (!s) return;
  // Match the EXACT values the boost wrote, not `>=`. The boost set these to
  // precise constants; a `>=` test also clobbers a user who legitimately chose a
  // large context/maxTokens above the default, which this one-time migration must
  // not touch.
  if (s.contextLength === MCP_BOOST_CTX_CEILING) {
    s.contextLength = DEFAULT_SETTINGS.contextLength;
    // maxTokens was raised alongside contextLength by the boost; only reset it when the
    // boost's exact value is present, so a legitimately-large user maxTokens isn't clobbered.
    if (s.maxTokens === MCP_BOOST_MAX_OUTPUT_TOKENS)
      s.maxTokens = DEFAULT_SETTINGS.maxTokens;
  }
  if (s.liteRTMaxTokens === MCP_BOOST_CTX_CEILING) {
    s.liteRTMaxTokens = DEFAULT_SETTINGS.liteRTMaxTokens;
  }
}
export function migratePersistedState<TState>(
  persistedState: any,
  currentState: TState,
  context: PersistedStateMigrationContext,
): TState {
  const { defaultSettings, documentsPath } = context;
  const merged = {
    ...currentState,
    ...persistedState,
    settings: { ...defaultSettings, ...persistedState?.settings },
  };
  // Drop legacy download tracking fields. The unified downloadStore (backed
  // by the native Room DB) is now the source of truth. Persisted entries
  // from old versions are silently ignored on rehydrate.
  delete merged.downloadProgress;
  delete merged.activeBackgroundDownloads;
  delete merged.imageModelDownloading;
  delete merged.imageModelDownloadIds;
  delete merged.imageModelDownloadId;
  // modelLoadingStrategy was removed (the residency manager owns swapping now).
  if (merged.settings?.modelLoadingStrategy !== undefined) {
    delete merged.settings.modelLoadingStrategy;
  }
  if (persistedState?.settings && !persistedState.settings.cacheType) {
    merged.settings = {
      ...merged.settings,
      cacheType: 'q8_0',
      flashAttn: true,
    };
  }
  if (persistedState?.settings && !persistedState.settings.inferenceBackend) {
    merged.settings = {
      ...merged.settings,
      inferenceBackend:
        Platform.OS === 'ios'
          ? INFERENCE_BACKENDS.METAL
          : INFERENCE_BACKENDS.CPU,
    };
  }

  if (
    merged.checklistDismissed &&
    merged.onboardingChecklist &&
    !Object.values(merged.onboardingChecklist).every(Boolean)
  )
    merged.checklistDismissed = false;
  migrateReasoningBudgetScale(merged, persistedState);
  migrateMaxTokensDefault(merged, persistedState);
  migrateImageKeepDefault(merged, persistedState);
  migrateEnabledTools(merged);
  migrateBoostedContext(merged, defaultSettings);
  migrateAdvancedSettings(merged, defaultSettings);
  migrateGeneratedImageTimestamps(merged);
  migrateGeneratedImagePaths(merged, documentsPath);
  migrateTextLoras(merged, documentsPath);
  return merged as TState;
}

const TEXT_LORA_SCALE_MAX = 2;
/**
 * settings.textLoras: an install from before text LoRA has none (the defaults spread gives []); a
 * malformed entry (hand-edited / partially synced store) is dropped, a scale outside 0-2 is clamped,
 * and an iOS container path is rebased onto the current Documents folder (the UUID changes on
 * reinstall, like generated images). settings.remoteTextLoras keeps only {id: integer, scale} rows.
 * Idempotent.
 */
export function migrateTextLoras(merged: any, documentsPath?: string): void {
  const s = merged.settings;
  if (!s) return;
  const list = Array.isArray(s.textLoras) ? s.textLoras : [];
  s.textLoras = list
    .filter((l: any) =>
      l && typeof l.id === 'string' && typeof l.name === 'string' && typeof l.path === 'string' && l.path.length > 0)
    .map((l: any) => {
      const scale = isFiniteNumber(l.scale) ? Math.min(TEXT_LORA_SCALE_MAX, Math.max(0, l.scale)) : 1;
      const rebased = documentsPath
        ? /\/Containers\/Data\/Application\/[^/]+\/Documents\/(text_loras\/.+)$/.exec(l.path)
        : null;
      return {
        id: l.id, name: l.name,
        path: rebased ? `${documentsPath}/${rebased[1]}` : l.path,
        scale, enabled: l.enabled === true,
      };
    });
  const remote = s.remoteTextLoras && typeof s.remoteTextLoras === 'object' && !Array.isArray(s.remoteTextLoras)
    ? s.remoteTextLoras : {};
  const cleaned: Record<string, Array<{ id: number; scale: number }>> = {};
  for (const [key, rows] of Object.entries(remote)) {
    if (!Array.isArray(rows)) continue;
    cleaned[key] = rows
      .filter((r: any) => r && Number.isInteger(r.id) && isFiniteNumber(r.scale))
      .map((r: any) => ({ id: r.id, scale: Math.min(TEXT_LORA_SCALE_MAX, Math.max(0, r.scale)) }));
  }
  s.remoteTextLoras = cleaned;
}

const isFiniteNumber = (v: unknown): boolean => typeof v === 'number' && Number.isFinite(v);
const CACHE_TYPES = ['f16', 'q8_0', 'q4_0'];
/** Settings added with the Advanced sampling / load / image-default fields, each with the shape
 *  its readers expect. An install from before them has none: the defaults spread above fills
 *  them; this also replaces a value of the wrong shape (a hand-edited or partially synced store)
 *  with the default, so no reader meets NaN / a non-array. Idempotent. */
const ADVANCED_SETTING_SHAPES: Record<string, (v: unknown) => boolean> = {
  topK: isFiniteNumber, minP: isFiniteNumber, typicalP: isFiniteNumber,
  penaltyLastN: isFiniteNumber, presencePenalty: isFiniteNumber, frequencyPenalty: isFiniteNumber,
  dryMultiplier: isFiniteNumber, dryBase: isFiniteNumber, dryAllowedLength: isFiniteNumber,
  dryPenaltyLastN: isFiniteNumber, xtcProbability: isFiniteNumber, xtcThreshold: isFiniteNumber,
  mirostat: v => v === 0 || v === 1 || v === 2,
  mirostatTau: isFiniteNumber, mirostatEta: isFiniteNumber,
  seed: v => Number.isInteger(v),
  customStopStrings: v => Array.isArray(v) && v.every(s => typeof s === 'string'),
  liteRTTopK: isFiniteNumber,
  nUbatch: v => Number.isInteger(v) && (v as number) >= 0,
  mmapMode: v => v === 'auto' || v === 'on' || v === 'off',
  cacheTypeV: v => v === 'same' || CACHE_TYPES.includes(v as string),
  imageNegativePrompt: v => typeof v === 'string',
  imageSeed: v => Number.isInteger(v),
  imageClipSkip: v => Number.isInteger(v) && (v as number) >= 0,
};
export function migrateAdvancedSettings(merged: any, defaultSettings: PersistedSettings): void {
  const s = merged.settings;
  if (!s) return;
  for (const [key, valid] of Object.entries(ADVANCED_SETTING_SHAPES)) {
    if (!valid(s[key]) && key in defaultSettings) s[key] = defaultSettings[key];
  }
}

/** Older builds stored reasoningBudget 0 for Auto; 0 now means "no thinking". Map once. */
function migrateReasoningBudgetScale(merged: any, persistedState: any): void {
  const persisted = persistedState?.settings;
  if (!persisted || persisted.reasoningBudgetZeroIsOff) return;
  const legacyAuto = persisted.reasoningBudget == null || persisted.reasoningBudget <= 0;
  merged.settings = {
    ...merged.settings,
    ...(legacyAuto ? { reasoningBudget: -1 } : {}),
    reasoningBudgetZeroIsOff: true,
  };
}

/**
 * Old default maxTokens 1024 -> 2048, once. maxTokens is n_predict (output length) on the local path:
 * KV/RAM is sized by contextLength, so this changes no memory. Only an install still on the exact old
 * default is raised; any other value is the user's choice and is kept. The flag makes it one-time.
 */
const OLD_DEFAULT_MAX_TOKENS = 1024;
const NEW_DEFAULT_MAX_TOKENS = 2048;
function migrateMaxTokensDefault(merged: any, persistedState: any): void {
  const persisted = persistedState?.settings;
  if (!persisted || persisted.maxTokensDefault2048) return;
  merged.settings = {
    ...merged.settings,
    ...(persisted.maxTokens === OLD_DEFAULT_MAX_TOKENS ? { maxTokens: NEW_DEFAULT_MAX_TOKENS } : {}),
    maxTokensDefault2048: true,
  };
}

const OLD_DEFAULT_DENOISE = 0.6;
const NEW_DEFAULT_DENOISE = 0.4; // Image Studio / imgref "Keep 60%" (user decision 2026-11-21)

/** One-time: the saved 0.6 denoise was the old default (no screen ever set it), so move it to 0.4. */
function migrateImageKeepDefault(merged: any, persistedState: any): void {
  const persisted = persistedState?.settings;
  if (!persisted || persisted.imageKeepDefault60) return;
  merged.settings = {
    ...merged.settings,
    ...(persisted.imageDenoiseStrength === OLD_DEFAULT_DENOISE ? { imageDenoiseStrength: NEW_DEFAULT_DENOISE } : {}),
    imageKeepDefault60: true,
  };
}

/** Keep generated-image paths valid when iOS gives the app a new container UUID on install. */
function migrateGeneratedImagePaths(merged: any, documentsPath?: string): void {
  if (!documentsPath || !Array.isArray(merged.generatedImages)) return;
  merged.generatedImages = merged.generatedImages.map((image: any) => {
    if (typeof image?.imagePath !== 'string') return image;
    const match =
      /\/Containers\/Data\/Application\/[^/]+\/Documents\/(generated_images\/.+)$/.exec(
        image.imagePath,
      );
    return match
      ? { ...image, imagePath: `${documentsPath}/${match[1]}` }
      : image;
  });
}

/**
 * Generated images stamped with epoch milliseconds as text, put right.
 *
 * The Android image module wrote `"1786317315833"`, which satisfies `createdAt: string` and is a date
 * to nobody: the gallery showed it as invalid, and every sync peer refused the image because
 * `Date.parse` answers NaN. The producers now write ISO-8601, but a phone that has generated images
 * already holds the old value, and nothing else would ever rewrite it.
 */
function migrateGeneratedImageTimestamps(merged: any): void {
  if (!Array.isArray(merged.generatedImages)) return;
  merged.generatedImages = merged.generatedImages.map((image: any) => {
    const epochMs = Number(image?.createdAt);
    return typeof image?.createdAt === 'string' &&
      /^\d+$/.test(image.createdAt) &&
      Number.isFinite(epochMs)
      ? { ...image, createdAt: new Date(epochMs).toISOString() }
      : image;
  });
}
