/**
 * Memory budget — shared physical caps and model-load policies by device/platform.
 *
 * Both consumers use these physical caps; dirty models (and every model on Android)
 * additionally require a live-available-RAM check:
 *  - the residency manager (capacity planning + eviction), and
 *  - the pre-load memory check (checkMemoryForModel).
 *
 * The safe fraction is device + platform aware (NOT flat) because the fixed OS+app
 * baseline is a big slice of 4GB but a small slice of 12GB, and on iOS we hold
 * com.apple.developer.kernel.increased-memory-limit which raises the per-process
 * cap well above the default:
 *  - ≤4GB: 0.50 (≈2GB on a 4GB device — safe in practice; the dynamic real-free-RAM
 *    guard tightens further under real pressure),
 *  - 6-8GB: 0.60,
 *  - 12GB+: 0.78 iOS / 0.70 Android (so a 12GB iPhone runs a 7GB model a flat 60%
 *    wrongly rejected).
 * This fraction is the absolute PHYSICAL ceiling; dirty model loads (and, on Android,
 * every model load) also respect live available RAM so the residency manager does not
 * load into swap.
 *
 * Previously two places computed this independently (a 0.6 in the residency policy
 * AND a separate device-tiered fraction in the model-load check); unifying it here
 * is the fix.
 *
 * ── Load policy ────────────────────────────────────────────────────────────────
 * The SAME functions are parameterised by a `LoadPolicy` so "aggressive mode" is a
 * single data-driven knob, NOT an `if (aggressive)` scattered through the load path:
 *  - 'balanced' (default): the conservative behaviour above. Every existing caller
 *    that omits the policy gets exactly this, so the default is behaviour-neutral.
 *  - 'aggressive': commit a larger fraction of RAM and hold a smaller OS reserve,
 *    so big models (e.g. a 21GB MoE on a 24GB phone, a 3GB LiteRT model whose dirty
 *    footprint the balanced dynamic guard rejects on a 12GB phone) are allowed to
 *    load. The reserve floor is reduced but NEVER removed — that is the "lenient
 *    safeguard": we still keep the OS + app baseline alive rather than guaranteeing
 *    an instant jetsam.
 *
 * "Load Anyway" (the per-load override that forces past the fit gate) is a separate,
 * ALWAYS-available escape hatch — not gated by this policy. Aggressive mode changes
 * the default budget so fewer loads need forcing; the override remains the explicit,
 * per-load, user-confirmed way to push past whatever budget is in effect.
 */
import { Platform } from 'react-native';

/**
 * How the residency manager handles multiple models:
 *  - 'conservative': ONE model at a time — loading any model evicts every other
 *    (no co-residency), the safest on tight devices.
 *  - 'balanced' (default): co-reside models while the RAM budget holds; evict
 *    lowest-priority/LRU when a new model doesn't fit.
 *  - 'aggressive': balanced co-residency but commits a larger share of RAM (smaller
 *    OS reserve) so bigger models load before the gate refuses.
 * (A per-load "Load Anyway" override is separate and works in every mode.)
 */
export type LoadPolicy = 'conservative' | 'balanced' | 'aggressive';

/** Never commit the last ~1.5GB — OS + app baseline must always have headroom. */
export const MEMORY_RESERVE_MB = 1500;
/** Aggressive mode still keeps a floor alive (lenient, not absent). */
export const AGGRESSIVE_RESERVE_MB = 800;
// (Removed the override survival-floor cluster — OVERRIDE_SURVIVAL_FLOOR_MB /
// ANDROID_OVERRIDE_SURVIVAL_FLOOR_MB / overrideSurvivalFloorMB.) It was defined but never wired to
// any caller, so it enforced nothing, and the product decision is that "Load Anyway" gives the user
// FULL control — they accept the OOM risk, the app does not impose a hard floor they can't cross.

/**
 * Live-RAM safety margin (MB): the part of the LIVE available reading (Android
 * ActivityManager.availMem, iOS os_proc_available_memory) a model load must leave untouched
 * for the OS, the low-memory killer and the app's own JS/UI heap. The residency manager's
 * dirty-load headroom (modelResidency/residents.ts) aliases these, so there is one owner.
 */
export const LIVE_RAM_MARGIN_MB = 1024;
/** Aggressive mode keeps a smaller live margin - lenient, never zero. */
export const AGGRESSIVE_LIVE_RAM_MARGIN_MB = 512;

type Plat = 'ios' | 'android' | string;

/** Live-RAM margin (MB) the model-load gates keep free, by policy. */
export function liveRamMarginMB(policy: LoadPolicy = 'balanced'): number {
  return policy === 'aggressive' ? AGGRESSIVE_LIVE_RAM_MARGIN_MB : LIVE_RAM_MARGIN_MB;
}

/**
 * Available RAM (MB) a model load may count on RIGHT NOW: the live OS reading, never capped.
 *
 * Android previously returned max(live, physical budget) - a "reclaim credit" that assumed the
 * low-memory killer would free background apps fast enough; the load filled RAM faster than the
 * system reclaimed it and the app was killed. It was then capped at the physical budget (70% of
 * RAM on 12GB), which under-reported free RAM on a roomy phone and refused loads that fit. The
 * live reading is now used as is; the caller keeps the live safety margin (liveRamMarginMB) free,
 * and the physical cap is checked separately by the residency gate.
 */
export function effectiveAvailableMB(realAvailMB: number): number {
  return Math.max(0, realAvailMB);
}

/**
 * Factor applied ONCE, to the peak of the model being loaded. Models already resident are charged
 * at their resident bytes (no factor, no transient compute), so two co-resident models are never
 * padded twice.
 */
export const INCOMING_PEAK_FACTOR = 1.1;

/**
 * A GGUF text model's memory, split by what the OS can reclaim:
 *  - weightsBytes: the GGUF tensors;
 *  - anonBytes: mmproj + text LoRA + KV cache + compute buffers (anonymous RAM, never reclaimable).
 * Both come from estimateLlamaMemory (KV, compute) and the file sizes.
 */
export interface TextMemoryParts {
  weightsBytes: number;
  anonBytes: number;
}

/** MB charged for a text model: physical (all bytes) and live (bytes that consume free RAM). */
export interface TextMemoryFootprint {
  /** Resident bytes, no factor: what an already-loaded model holds. */
  residentMB: number;
  /** residentMB x INCOMING_PEAK_FACTOR: what a load of this model is charged. */
  peakMB: number;
  /** Part of residentMB that consumes live free RAM. */
  liveResidentMB: number;
  /** Part of peakMB that consumes live free RAM. */
  livePeakMB: number;
}

/**
 * Whether a loaded GGUF's weights are reclaimable file-backed pages. They are when llama.cpp maps
 * the file (use_mmap) and keeps the tensors there: on Android only with no layer offloaded (the
 * CPU backend; OpenCL/HTP copy offloaded tensors into anonymous buffers); on iOS Metal maps the
 * same pages (see modelResidency getBudgetMB).
 */
export function textWeightsReclaimable(opts: { useMmap: boolean; gpuLayers: number; platform?: Plat }): boolean {
  const { useMmap, gpuLayers, platform = Platform.OS } = opts;
  return useMmap && (platform !== 'android' || gpuLayers === 0);
}

/**
 * The one text-model memory formula used by the residency gate, the register and the llm.ts check.
 * Reclaimable (mmapped) weights are left out of the live charge; they still count physically.
 */
export function textMemoryFootprint(parts: TextMemoryParts, weightsReclaimable: boolean): TextMemoryFootprint {
  const MiB = 1024 * 1024;
  const resident = parts.weightsBytes + parts.anonBytes;
  const live = weightsReclaimable ? parts.anonBytes : resident;
  return {
    residentMB: Math.ceil(resident / MiB),
    peakMB: Math.ceil((resident * INCOMING_PEAK_FACTOR) / MiB),
    liveResidentMB: Math.ceil(live / MiB),
    livePeakMB: Math.ceil((live * INCOMING_PEAK_FACTOR) / MiB),
  };
}

/** OS/app reserve (MB) that is never committed to models, by policy. */
export function memoryReserveMB(policy: LoadPolicy = 'balanced'): number {
  return policy === 'aggressive' ? AGGRESSIVE_RESERVE_MB : MEMORY_RESERVE_MB;
}

/** Safe fraction of total RAM this process may commit to models, by device tier. */
export function modelBudgetFraction(
  totalRamGB: number,
  platform: Plat = Platform.OS,
  policy: LoadPolicy = 'balanced',
): number {
  if (policy === 'aggressive') {
    // Lenient: use more of RAM. Low-RAM tiers stay comparatively cautious (a 60%
    // slice of 4GB is already close to what the OS will tolerate), high-RAM/entitled
    // tiers push near the physical ceiling so a 21GB model fits a 24GB phone.
    if (totalRamGB <= 4) return 0.60;
    if (totalRamGB <= 8) return 0.75;
    return platform === 'ios' ? 0.92 : 0.88;
  }
  if (totalRamGB <= 4) return 0.50; // ~2GB on 4GB — safe; dynamic guard tightens under pressure
  if (totalRamGB <= 8) return 0.60; // 6-8GB
  return platform === 'ios' ? 0.78 : 0.70; // 12GB+: iOS holds the increased-memory entitlement
}

/** Fraction at which we WARN (load allowed, perf may suffer). Below the budget. */
function modelWarningFraction(totalRamGB: number, platform: Plat = Platform.OS): number {
  if (totalRamGB <= 4) return 0.40;
  if (totalRamGB <= 8) return 0.50;
  return platform === 'ios' ? 0.66 : 0.60;
}

/** Hard budget in MB: the smaller of the fraction-of-RAM and (RAM minus reserve). */
export function modelMemoryBudgetMB(
  totalRamMB: number,
  platform: Plat = Platform.OS,
  policy: LoadPolicy = 'balanced',
): number {
  const totalRamGB = totalRamMB / 1024;
  const byFraction = totalRamMB * modelBudgetFraction(totalRamGB, platform, policy);
  const byReserve = totalRamMB - memoryReserveMB(policy);
  return Math.max(0, Math.min(byFraction, byReserve));
}

/** Warning threshold in MB (always ≤ the hard budget). */
export function modelWarningThresholdMB(totalRamMB: number, platform: Plat = Platform.OS): number {
  const totalRamGB = totalRamMB / 1024;
  const byFraction = totalRamMB * modelWarningFraction(totalRamGB, platform);
  return Math.min(byFraction, modelMemoryBudgetMB(totalRamMB, platform));
}

const BYTES_PER_GB = 1024 ** 3;

/**
 * Does a model file's on-disk size exceed this device's safe RAM budget
 * (`totalRamGB * modelBudgetFraction`)? The SINGLE device-aware fit primitive that
 * every download-warning decision and the detail-list compatibility filter compute
 * from — a static per-model flag was device-blind and fired on every device. Zero-IO
 * so it is unit-testable; callers pass the RAM read from the device boundary
 * (hardwareService.getTotalMemoryGB).
 */
export function fileExceedsBudget(sizeBytes: number, ramGB: number): boolean {
  return sizeBytes / BYTES_PER_GB >= ramGB * modelBudgetFraction(ramGB);
}

/**
 * Block until a just-released native model's memory is reclaimed (the memory reading drops by
 * ~minDropMB: iOS process footprint, Android system-wide used RAM) or a bounded timeout — so the NEXT model load never allocates on top of the outgoing
 * model. The native context.release() returns before the OS frees the weights + GPU/HTP buffers;
 * without this barrier a reload stacked BOTH models' memory at once and OOM'd under pressure
 * (device 2026-07-14: a reload with ~2GB free ground to 0 tok/s then the process was killed).
 * Zero-IO over an injected memory reader so it is unit-testable; best-effort — if the RAM sensor is
 * unavailable or the footprint never settles, wait a short fixed beat and proceed.
 */
export async function awaitMemoryReclaim(
  getProcessMemory: () => Promise<{ footprintMB: number } | null>,
  opts: {
    timeoutMs?: number;
    minDropMB?: number;
    intervalMs?: number;
    sleep?: (ms: number) => Promise<void>;
    /** Reading taken BEFORE the release started. Needed when the release call itself returns only
     *  after the memory is back (local-dream waits for its server process to exit): a baseline read
     *  after it would never see the drop. Absent = read now. */
    baseline?: { footprintMB: number } | null;
  } = {},
): Promise<void> {
  const { timeoutMs = 2500, minDropMB = 200, intervalMs = 120 } = opts;
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const before = opts.baseline !== undefined ? opts.baseline : await getProcessMemory();
  if (!before) { await sleep(250); return; }
  if (opts.baseline !== undefined) {
    // Baseline from before the release: the memory may already be back (no wait needed).
    const now = await getProcessMemory();
    if (!now || before.footprintMB - now.footprintMB >= minDropMB) return;
  }
  let waited = 0;
  while (waited < timeoutMs) {
    await sleep(intervalMs);
    waited += intervalMs;
    const now = await getProcessMemory();
    if (!now) return;
    if (before.footprintMB - now.footprintMB >= minDropMB) return;
  }
}
