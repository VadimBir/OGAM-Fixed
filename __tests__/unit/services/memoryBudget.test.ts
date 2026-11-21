/**
 * Unit tests for the single device + platform aware memory budget owner.
 * The headline regression: a 12GB iPhone must NOT be capped like a 6GB one, so a
 * ~7GB model fits (the gemma-4-E4B case that wrongly failed under a flat 60%).
 */
import {
  modelBudgetFraction,
  modelMemoryBudgetMB,
  modelWarningThresholdMB,
  memoryReserveMB,
  effectiveAvailableMB,
  liveRamMarginMB,
  LIVE_RAM_MARGIN_MB,
  AGGRESSIVE_LIVE_RAM_MARGIN_MB,
  MEMORY_RESERVE_MB,
  AGGRESSIVE_RESERVE_MB,
  awaitMemoryReclaim,
  textMemoryFootprint,
  textWeightsReclaimable,
  INCOMING_PEAK_FACTOR,
} from '../../../src/services/memoryBudget';

const GB = 1024;

describe('effectiveAvailableMB — live availability (single owner)', () => {
  it('Android 12GB with 3GB free: the gate sees the LIVE 3072MB, not a 70%-of-RAM credit', () => {
    // The owner's device: 12GB, ~75% used by other apps. The old reclaim credit returned
    // max(3072, 0.70*12288=8602) = 8602MB and admitted loads the phone could not hold.
    expect(effectiveAvailableMB(3 * GB)).toBe(3 * GB);
  });

  it('never caps free RAM at the 70% physical budget (a roomy phone reports its real free RAM)', () => {
    // Before: min(11000, 0.70 x 12288 = 8602) = 8602. The reserve/margin protects the system.
    expect(effectiveAvailableMB(11000)).toBe(11000);
    expect(effectiveAvailableMB(11000)).toBeGreaterThan(modelMemoryBudgetMB(12 * GB, 'android'));
  });

  it('never negative', () => {
    expect(effectiveAvailableMB(-5)).toBe(0);
  });
});

describe('live safety margin', () => {
  it('the live gates keep 1024MB (balanced) / 512MB (aggressive) of live RAM free', () => {
    expect(LIVE_RAM_MARGIN_MB).toBe(1024);
    expect(AGGRESSIVE_LIVE_RAM_MARGIN_MB).toBe(512);
    expect(liveRamMarginMB()).toBe(1024);
    expect(liveRamMarginMB('aggressive')).toBe(512);
  });
});

describe('textMemoryFootprint — one text estimate, x1.1 once', () => {
  const MiB = 1024 * 1024;
  const parts = { weightsBytes: 4000 * MiB, anonBytes: 1000 * MiB };

  it('resident = weights + anonymous bytes, no factor; peak = resident x 1.1', () => {
    const f = textMemoryFootprint(parts, false);
    expect(INCOMING_PEAK_FACTOR).toBe(1.1);
    expect(f.residentMB).toBe(5000);
    expect(f.peakMB).toBe(5500);
    expect(f.liveResidentMB).toBe(5000);
    expect(f.livePeakMB).toBe(5500);
  });

  it('reclaimable mmapped weights are left out of the live charge only', () => {
    const f = textMemoryFootprint(parts, true);
    expect(f.residentMB).toBe(5000);
    expect(f.peakMB).toBe(5500);
    expect(f.liveResidentMB).toBe(1000);
    expect(f.livePeakMB).toBe(1100);
  });

  it('Android: mmapped weights are reclaimable only on the CPU backend (no offloaded layer)', () => {
    expect(textWeightsReclaimable({ useMmap: true, gpuLayers: 0, platform: 'android' })).toBe(true);
    expect(textWeightsReclaimable({ useMmap: true, gpuLayers: 99, platform: 'android' })).toBe(false);
    expect(textWeightsReclaimable({ useMmap: false, gpuLayers: 0, platform: 'android' })).toBe(false);
    expect(textWeightsReclaimable({ useMmap: true, gpuLayers: 99, platform: 'ios' })).toBe(true);
  });
});

describe('modelBudgetFraction', () => {
  it('keeps low-RAM devices conservative (≈2GB on 4GB)', () => {
    expect(modelBudgetFraction(4, 'ios')).toBe(0.50);
    expect(modelBudgetFraction(4, 'android')).toBe(0.50);
  });

  it('keeps 6-8GB devices at the prior 0.60 (unchanged)', () => {
    expect(modelBudgetFraction(6, 'ios')).toBe(0.60);
    expect(modelBudgetFraction(8, 'android')).toBe(0.60);
  });

  it('raises the fraction for high-RAM devices, higher on iOS (entitlement)', () => {
    expect(modelBudgetFraction(12, 'ios')).toBeGreaterThan(0.60);
    expect(modelBudgetFraction(12, 'ios')).toBeGreaterThan(modelBudgetFraction(12, 'android'));
  });
});

describe('modelMemoryBudgetMB', () => {
  it('lets a ~7GB model fit on a 12GB iPhone (the E4B regression)', () => {
    const budget = modelMemoryBudgetMB(12 * GB, 'ios');
    expect(budget).toBeGreaterThan(7 * GB); // 7GB model now fits
  });

  it('caps the 4GB budget at ~2GB (0.50; jetsam-safe, dynamic guard tightens further)', () => {
    // 0.50 * 4096 = 2048; reserve cap (4096-1500=2596) is looser, so fraction binds.
    expect(modelMemoryBudgetMB(4 * GB, 'ios')).toBeCloseTo(0.50 * 4 * GB, 0);
  });

  it('never commits past the reserve floor', () => {
    const total = 12 * GB;
    expect(modelMemoryBudgetMB(total, 'ios')).toBeLessThanOrEqual(total - MEMORY_RESERVE_MB);
  });
});

describe('modelWarningThresholdMB', () => {
  it('is always at or below the hard budget', () => {
    for (const gb of [4, 6, 8, 12, 16]) {
      const total = gb * GB;
      expect(modelWarningThresholdMB(total, 'ios')).toBeLessThanOrEqual(modelMemoryBudgetMB(total, 'ios'));
    }
  });
});

describe('load policy — aggressive vs balanced', () => {
  it("defaults to balanced (behaviour-neutral) when policy omitted", () => {
    for (const gb of [4, 8, 12, 24]) {
      expect(modelBudgetFraction(gb, 'android')).toBe(modelBudgetFraction(gb, 'android', 'balanced'));
      expect(modelMemoryBudgetMB(gb * GB, 'android')).toBe(modelMemoryBudgetMB(gb * GB, 'android', 'balanced'));
    }
  });

  it('aggressive commits a strictly larger fraction at every tier', () => {
    for (const [gb, plat] of [[4, 'android'], [8, 'android'], [12, 'android'], [12, 'ios'], [24, 'android']] as const) {
      expect(modelBudgetFraction(gb, plat, 'aggressive')).toBeGreaterThan(modelBudgetFraction(gb, plat, 'balanced'));
    }
  });

  it('aggressive holds a smaller (but non-zero) OS reserve — the lenient safeguard', () => {
    expect(memoryReserveMB('aggressive')).toBe(AGGRESSIVE_RESERVE_MB);
    expect(memoryReserveMB('balanced')).toBe(MEMORY_RESERVE_MB);
    expect(AGGRESSIVE_RESERVE_MB).toBeGreaterThan(0);
    expect(AGGRESSIVE_RESERVE_MB).toBeLessThan(MEMORY_RESERVE_MB);
  });

  it('fails-before/passes-after: a 21GB model is rejected on a 24GB phone under balanced but fits under aggressive', () => {
    const total = 24 * GB;
    const model = 21 * GB;
    // Balanced: 24GB * 0.70 = 16.8GB budget → 21GB does NOT fit.
    expect(modelMemoryBudgetMB(total, 'android', 'balanced')).toBeLessThan(model);
    // Aggressive: pushes near the physical ceiling → 21GB fits (Nico's Qwen3 MoE case).
    expect(modelMemoryBudgetMB(total, 'android', 'aggressive')).toBeGreaterThanOrEqual(model);
  });

  it('aggressive still never commits past its own reserve floor', () => {
    const total = 24 * GB;
    expect(modelMemoryBudgetMB(total, 'android', 'aggressive')).toBeLessThanOrEqual(total - AGGRESSIVE_RESERVE_MB);
  });
});

describe('awaitMemoryReclaim — unload does not return until native memory is freed (device 2026-07-14)', () => {
  const noSleep = () => Promise.resolve(); // no real timers; drives the loop instantly

  it('returns as soon as the process footprint drops by the threshold', async () => {
    // footprint: 5000 (before) → 4900 → 4700 (dropped 300 ≥ 200) — the reclaim landed on the 2nd poll.
    const readings = [{ footprintMB: 5000 }, { footprintMB: 4900 }, { footprintMB: 4700 }, { footprintMB: 4700 }];
    let i = 0;
    const getProcessMemory = jest.fn(async () => readings[Math.min(i++, readings.length - 1)]);
    await awaitMemoryReclaim(getProcessMemory, { sleep: noSleep });
    // It polled the before + at least two more reads, and did NOT run to the full timeout.
    expect(getProcessMemory.mock.calls.length).toBeLessThanOrEqual(4);
    expect(getProcessMemory.mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it('waits the full bounded window when the footprint never settles (then proceeds)', async () => {
    const getProcessMemory = jest.fn(async () => ({ footprintMB: 5000 })); // never drops
    await awaitMemoryReclaim(getProcessMemory, { sleep: noSleep, timeoutMs: 600, intervalMs: 120 });
    // Polled before + one per interval across the window, then returned (never hangs).
    expect(getProcessMemory.mock.calls.length).toBe(1 + 600 / 120);
  });

  it('falls back to a fixed beat when the RAM sensor is unavailable', async () => {
    const sleep = jest.fn(async () => {});
    const getProcessMemory = jest.fn(async () => null); // no sensor
    await awaitMemoryReclaim(getProcessMemory, { sleep });
    expect(getProcessMemory).toHaveBeenCalledTimes(1); // read once, saw null, bailed to the fixed wait
    expect(sleep).toHaveBeenCalledWith(250);
  });
});

describe('awaitMemoryReclaim with a pre-release baseline', () => {
  const noSleep = jest.fn(async () => {});

  it('returns at once when the memory was already back when the release returned', async () => {
    const getProcessMemory = jest.fn(async () => ({ footprintMB: 1000 }));
    await awaitMemoryReclaim(getProcessMemory, { baseline: { footprintMB: 4000 }, sleep: noSleep });
    expect(getProcessMemory).toHaveBeenCalledTimes(1);
    expect(noSleep).not.toHaveBeenCalled();
  });

  it('keeps polling from the baseline when the drop has not happened yet', async () => {
    const readings = [{ footprintMB: 4000 }, { footprintMB: 3950 }, { footprintMB: 3000 }];
    let i = 0;
    const getProcessMemory = jest.fn(async () => readings[Math.min(i++, readings.length - 1)]);
    await awaitMemoryReclaim(getProcessMemory, { baseline: { footprintMB: 4000 }, sleep: noSleep, minDropMB: 200 });
    expect(getProcessMemory).toHaveBeenCalledTimes(3);
  });
});
