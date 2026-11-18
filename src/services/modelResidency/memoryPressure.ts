/**
 * Android memory-pressure signal (onTrimMemory / onLowMemory) → residency release tiers.
 *
 * React Native's AppState 'memoryWarning' event is iOS-only, so on Android the residency
 * manager never heard the OS asking for memory back. DeviceMemoryModule (native) now emits
 * TRIM_EVENT on every ComponentCallbacks2 callback; this file owns the pure mapping from
 * that payload to a severity and from a severity to the resident types to release, plus the
 * subscription. Pure parts are unit-tested without a device.
 */
import { NativeEventEmitter, NativeModules, Platform } from 'react-native';
import type { ResidentType } from './policy';

/** Event name emitted by android/.../devicememory/DeviceMemoryModule.kt. */
export const TRIM_EVENT = 'DeviceMemoryTrim';

/** android.content.ComponentCallbacks2 levels (values are API constants). */
export const TRIM_LEVEL = {
  RUNNING_MODERATE: 5,
  RUNNING_LOW: 10,
  RUNNING_CRITICAL: 15,
  UI_HIDDEN: 20,
  BACKGROUND: 40,
  MODERATE: 60,
  COMPLETE: 80,
} as const;

export interface TrimPayload {
  /** ComponentCallbacks2 level; -1 for onLowMemory. */
  level: number;
  /** 'trim' = onTrimMemory, 'lowMemory' = onLowMemory. */
  source?: string;
  /** ActivityManager.MemoryInfo.lowMemory at the time of the callback. */
  lowMemory?: boolean;
  /** ActivityManager.MemoryInfo.availMem in MB at the time of the callback. */
  availMB?: number;
}

export type TrimSeverity = 'none' | 'moderate' | 'critical';

/**
 * Severity of one trim callback.
 *  - critical: onLowMemory, RUNNING_CRITICAL, COMPLETE, or the system reports lowMemory.
 *  - moderate: RUNNING_MODERATE, RUNNING_LOW, BACKGROUND, MODERATE.
 *  - none: UI_HIDDEN alone (the UI went away; not a memory request) or an unknown level.
 */
export function classifyTrim(p: TrimPayload): TrimSeverity {
  if (p.source === 'lowMemory' || p.lowMemory === true) return 'critical';
  if (p.level === TRIM_LEVEL.RUNNING_CRITICAL || p.level >= TRIM_LEVEL.COMPLETE) return 'critical';
  if (
    p.level === TRIM_LEVEL.RUNNING_MODERATE ||
    p.level === TRIM_LEVEL.RUNNING_LOW ||
    p.level === TRIM_LEVEL.BACKGROUND ||
    p.level === TRIM_LEVEL.MODERATE
  ) {
    return 'moderate';
  }
  return 'none';
}

const SIDECAR_TIER: ResidentType[] = ['whisper', 'tts', 'embedding'];

/**
 * Release order, cheapest-to-reload first: sidecars, then the idle image model, then
 * (critical only) the idle text model. Pinned residents are never in a tier.
 */
export function trimReleaseTiers(severity: TrimSeverity): ResidentType[][] {
  if (severity === 'none') return [];
  const tiers: ResidentType[][] = [SIDECAR_TIER, ['image']];
  if (severity === 'critical') tiers.push(['text']);
  return tiers;
}

/** Subscribe to the native trim event. No-op (returns a no-op) off Android or without the module. */
export function subscribeToAndroidTrim(onTrim: (p: TrimPayload) => void): () => void {
  if (Platform.OS !== 'android') return () => {};
  const mod = NativeModules.DeviceMemoryModule;
  if (!mod) return () => {};
  const sub = new NativeEventEmitter(mod).addListener(TRIM_EVENT, (raw: unknown) => {
    if (!raw || typeof raw !== 'object') return;
    const p = raw as Partial<TrimPayload>;
    if (typeof p.level !== 'number') return;
    onTrim({ level: p.level, source: p.source, lowMemory: p.lowMemory, availMB: p.availMB });
  });
  return () => sub.remove();
}
