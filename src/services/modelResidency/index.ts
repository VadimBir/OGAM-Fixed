/**
 * ModelResidencyManager
 *
 * Keeps resident on-device models within a RAM budget. Callers ask to make a
 * model resident; the manager evicts others per `planEviction` (unloading via
 * each resident's registered unload fn) before loading the new one. Load/unload
 * are injected by the caller, so this stays decoupled from the text/image/
 * whisper/tts services and is unit-testable.
 *
 * See docs/design/MODEL_ROUTING.md §5.1–5.2.
 */
import { AppState, Platform } from 'react-native';
import { hardwareService } from '../hardware';
import logger from '../../utils/logger';
import {
  planEviction,
  computeBudgetMB,
  Resident,
  EvictionPlan,
  LiveRam,
} from './policy';
import { LoadPolicy, liveRamMarginMB } from '../memoryBudget';
import {
  TrimSeverity,
  classifyTrim,
  trimReleaseTiers,
  subscribeToAndroidTrim,
} from './memoryPressure';
import {
  formatMakeRoomForLine,
  formatOverrideForcingLine,
  formatOverrideForcedLine,
} from './logging';
import {
  UnloadFn,
  MIN_BUDGET_MB,
  DIRTY_AVAILABILITY_HEADROOM_MB,
  AGGRESSIVE_DIRTY_HEADROOM_MB,
  SIDECAR_TYPES,
  RegisteredResident,
  ResidentSpec,
  EnsureResult,
  stripUnload,
} from './residents';

class ModelResidencyManager {
  private readonly residents = new Map<string, RegisteredResident>();
  private readonly residentRegistrations = new Map<string, symbol>();
  private budgetOverrideMB: number | null = null;
  /**
   * Current load policy (single owner). The View (settings screen) dispatches an
   * intent via setLoadPolicy; the manager - not a reactive store snapshot - is the
   * authoritative source the memory math reads, so no imperative decision is made
   * off a store value multiple writers can desync.
   */
  private loadPolicy: LoadPolicy = 'balanced';
  constructor() {
    // Residency owns the memory-pressure response (single owner of model memory).
    // It used to be scattered - e.g. the Kokoro bridge had its own memoryWarning
    // listener freeing itself. Now one place reclaims idle models on a warning.
    try {
      AppState.addEventListener('memoryWarning', () => {
        this.handleMemoryWarning().catch(() => {});
      });
    } catch {
      /* non-RN env (some tests) - no AppState */
    }
    // Android has no AppState 'memoryWarning'; DeviceMemoryModule forwards onTrimMemory /
    // onLowMemory instead. Same single owner of the pressure response.
    try {
      subscribeToAndroidTrim(p => {
        const severity = classifyTrim(p);
        logger.log(
          `[ModelResidency] trim level=${p.level} source=${p.source ?? 'trim'} lowMemory=${!!p.lowMemory} availMB=${p.availMB ?? '?'} → ${severity}`,
        );
        this.handleTrimMemory(severity).catch(() => {});
      });
    } catch {
      /* native module absent (tests / iOS) */
    }
  }

  /** Residents as the pure policy sees them, with a live `canEvict()===false`
   *  treated as pinned so capacity eviction never unloads a model that's in use. */
  private planningResidents(): Resident[] {
    return [...this.residents.values()].map(r => ({
      ...stripUnload(r),
      pinned: r.pinned || (r.canEvict ? !r.canEvict() : false),
    }));
  }

  /**
   * Memory-warning response: reclaim idle SIDECAR models (TTS/STT/embedding) -
   * small and cheap to reload - but never one whose owner vetoes via canEvict()
   * (e.g. TTS is actively playing). Generation models and pinned residents are
   * left alone. This is what the Kokoro bridge's own listener used to do, now
   * centralized so the eviction decision lives in one place.
   */
  async handleMemoryWarning(): Promise<void> {
    // Run under the same FIFO lock as every load/unload: mutating `residents` and
    // driving native unloads concurrently with an in-flight load is exactly the
    // race the lock exists to prevent. The sidecar unloads here don't re-acquire the
    // lock, so this can't deadlock.
    await this.runExclusive('memory-warning', async () => {
      for (const [key, r] of [...this.residents.entries()]) {
        if (r.pinned || !SIDECAR_TYPES.has(r.type)) continue;
        if (r.canEvict && !r.canEvict()) continue; // in use - owner vetoes
        logger.log(
          `[ModelResidency] memory warning → reclaiming idle ${r.type} (${key})`,
        );
        await r
          .unload()
          .catch(err =>
            logger.log(
              `[ModelResidency] memory-warning unload ${key} failed:`,
              err,
            ),
          );
        this.residents.delete(key);
      }
    });
  }

  /**
   * Android trim response (onTrimMemory / onLowMemory, see ./memoryPressure). Releases idle
   * residents tier by tier: sidecars, then the image model, then - critical only - the text
   * model. Skips pinned residents, an owner veto (canEvict), and a resident whose generation
   * is in flight (isBusy). A rejected native unload keeps the resident counted. Returns the
   * released keys. Runs under the FIFO lock like every load/unload.
   */
  async handleTrimMemory(severity: TrimSeverity): Promise<string[]> {
    const tiers = trimReleaseTiers(severity);
    if (tiers.length === 0) return [];
    return this.runExclusive(`trim:${severity}`, async () => {
      const released: string[] = [];
      for (const tier of tiers) {
        for (const [key, r] of [...this.residents.entries()]) {
          if (r.pinned || !tier.includes(r.type)) continue;
          if (r.canEvict && !r.canEvict()) continue;
          const busy = r.isBusy ? await Promise.resolve(r.isBusy()).catch(() => true) : false;
          if (busy) {
            logger.log(`[ModelResidency] trim ${severity}: ${key} busy (generation in flight) - kept`);
            continue;
          }
          try {
            await r.unload();
            this.residents.delete(key);
            released.push(key);
            logger.log(`[ModelResidency] trim ${severity}: released idle ${r.type} (${key})`);
          } catch (err) {
            logger.log(`[ModelResidency] trim ${severity}: unload ${key} failed:`, err);
          }
        }
      }
      return released;
    });
  }

  /**
   * After an image generation finishes (Android only): release the idle image model when
   * live free RAM is below the live safety margin, so the next text load or the OS does not
   * have to fight the multi-GB dirty image working set. Returns true when released.
   */
  async releaseIdleImageIfMemoryLow(): Promise<boolean> {
    if (Platform.OS !== 'android') return false;
    if (!this.residents.has('image')) return false;
    await hardwareService.refreshMemoryInfo().catch(() => {});
    const availMB = Math.round(hardwareService.getAvailableMemoryGB() * 1024);
    const marginMB = liveRamMarginMB(this.loadPolicy);
    if (availMB >= marginMB) return false;
    return this.runExclusive('release:image-low-memory', async () => {
      const r = this.residents.get('image');
      if (!r || r.pinned) return false;
      if (r.canEvict && !r.canEvict()) return false;
      const busy = r.isBusy ? await Promise.resolve(r.isBusy()).catch(() => true) : false;
      if (busy) return false;
      try {
        await r.unload();
      } catch (err) {
        logger.log('[ModelResidency] post-generation image release failed:', err);
        return false;
      }
      this.residents.delete('image');
      logger.log(`[ModelResidency] released idle image after generation: availMB=${availMB} < marginMB=${marginMB}`);
      return true;
    });
  }

  /**
   * M4 gate, read on fresh RAM, evicts nothing: may every OTHER resident (the text model) stay in
   * RAM while resident `key` (the image model) runs a generation that still has to allocate
   * `pendingMB`? Two checks, both the budget's own numbers:
   *  - live: pendingMB fits real free RAM minus the live headroom (the generating model's working
   *    set is dirty memory, so it is live-gated on every platform, like capacityForSpec);
   *  - physical: all residents together fit the physical ceiling of the current policy.
   * The load-time makeRoomFor already applied the budget; this re-asks it at generation time,
   * because free RAM moves after the load and sd.cpp allocates its weights only on first use.
   * Absent `key` resident or nothing else resident: true (nothing competes).
   */
  async canKeepOthersDuringGeneration(key: string, pendingMB: number): Promise<boolean> {
    if (!this.residents.has(key) || this.residents.size < 2) return true;
    await hardwareService.refreshMemoryInfo().catch(() => {});
    const residentMB = [...this.residents.values()].reduce((sum, r) => sum + r.sizeMB, 0);
    const availMB = Math.round(hardwareService.getAvailableMemoryGB() * 1024);
    const headroomMB =
      this.loadPolicy === 'aggressive' ? AGGRESSIVE_DIRTY_HEADROOM_MB : DIRTY_AVAILABILITY_HEADROOM_MB;
    const physicalCapMB =
      this.budgetOverrideMB ??
      computeBudgetMB(hardwareService.getTotalMemoryGB() * 1024, { policy: this.loadPolicy });
    const liveFits = pendingMB <= availMB - headroomMB;
    const physicalFits = residentMB <= physicalCapMB;
    logger.log(
      `[ModelResidency] generation gate ${key}: pendingMB=${Math.round(pendingMB)} availMB=${availMB} headroomMB=${headroomMB} residentMB=${Math.round(residentMB)} capMB=${Math.round(physicalCapMB)} → ${liveFits && physicalFits ? 'keep others' : 'others must yield'}`,
    );
    return liveFits && physicalFits;
  }

  /**
   * Global FIFO lock. Every model load/unload (text, image, whisper, tts,
   * classifier) runs through here, so only ONE heavy model operation touches
   * memory at a time. This is what makes the budget safe to enforce: makeRoomFor
   * + the actual load + register happen atomically, never racing a second load.
   *
   * Re-entrancy rule: an eviction unload (registered via `register`) runs INSIDE
   * a held lock, so it must be the NON-locking internal unload - it must never
   * call runExclusive again, or it deadlocks. Public load/unload methods acquire
   * the lock; the internal `_do…` variants they call do not.
   */
  private opChain: Promise<void> = Promise.resolve();

  async runExclusive<T>(label: string, fn: () => Promise<T>): Promise<T> {
    const prev = this.opChain;
    let release: () => void = () => {};
    this.opChain = new Promise<void>(resolve => {
      release = resolve;
    });
    await prev.catch(() => {});
    try {
      return await fn();
    } finally {
      release();
    }
  }

  /** Force a specific budget (tests / low-memory tuning). null → derive from device RAM. */
  setBudgetOverrideMB(mb: number | null): void {
    this.budgetOverrideMB = mb;
  }

  /**
   * Set the load policy. Called (as an intent) when the user toggles "aggressive
   * model loading" and at boot from the persisted setting. 'aggressive' commits a
   * larger fraction of RAM and a smaller reserve so big models load; the numbers
   * themselves live in the memoryBudget owner, never branched on here.
   */
  setLoadPolicy(policy: LoadPolicy): void {
    this.loadPolicy = policy;
  }

  getLoadPolicy(): LoadPolicy {
    return this.loadPolicy;
  }

  getBudgetMB(): number {
    if (this.budgetOverrideMB != null) return this.budgetOverrideMB;
    // iOS: the budget is the device + platform PHYSICAL-RAM cap (a fraction of total RAM).
    // Android: live RAM (liveBudgetMB) - see the Android branch below.
    //
    // On iOS we do NOT min() this with os_proc_available_memory. That metric is the DIRTY
    // (anonymous) memory headroom before jetsam - but llama.cpp mmaps the GGUF, so a
    // model's weights are CLEAN, file-backed pages that the OS pages in/out freely and
    // that do NOT count against the jetsam limit. Budgeting the full model size against
    // the dirty headroom was a category error: it refused an 8GB mmap'd model (whose
    // real dirty cost is ~1-2GB of KV+compute) on a 12GB phone that runs it fine. A
    // GGUF's loadability is bounded by physical RAM (its weights fit as clean pages),
    // which is exactly computeBudgetMB. Floored so a small model always loads.
    const physicalCapMB = computeBudgetMB(
      hardwareService.getTotalMemoryGB() * 1024,
      { policy: this.loadPolicy },
    );
    // Android: live RAM drives the budget for every model (see capacityForSpec).
    if (Platform.OS === 'android') return this.liveBudgetMB(physicalCapMB);
    return Math.round(Math.max(MIN_BUDGET_MB, physicalCapMB));
  }

  /**
   * Live-RAM budget: real free RAM now + what evicting our own residents would free, minus
   * the live headroom, capped at the physical ceiling. No small-model floor: at critically
   * low free RAM even a small load must wait or be overridden.
   */
  private liveBudgetMB(physicalCapMB: number): number {
    const availableMB = hardwareService.getAvailableMemoryGB() * 1024;
    const residentMB = [...this.residents.values()].reduce(
      (sum, r) => sum + r.sizeMB,
      0,
    );
    // Aggressive mode holds a smaller headroom (the lenient safeguard). Still non-zero.
    const headroomMB =
      this.loadPolicy === 'aggressive'
        ? AGGRESSIVE_DIRTY_HEADROOM_MB
        : DIRTY_AVAILABILITY_HEADROOM_MB;
    const dynamicMB = availableMB + residentMB - headroomMB;
    return Math.round(Math.max(0, Math.min(physicalCapMB, dynamicMB)));
  }

  /**
   * Capacity for loading a SPECIFIC model, branching on its memory characteristic (data, not type):
   *  - physical: the device + platform RAM cap (computeBudgetMB), for every load;
   *  - live: real free RAM minus the live headroom, plus what the planned evictions release
   *    (planEviction). Applied on Android to every load; on iOS to a dirty (CoreML/ONNX image)
   *    load, to any load while a dirty model is resident, and to a load that states its live part
   *    (`liveMB`: a GGUF text model, whose mmapped weights are clean and left out of it).
   * On iOS an mmapped load with no dirty pressure and no live part stays bounded by physical RAM
   * only (clean, file-backed weights page in even when instantaneous available is low).
   */
  private capacityForSpec(spec: ResidentSpec): { budgetMB: number; live?: LiveRam } {
    if (this.budgetOverrideMB != null) return { budgetMB: this.budgetOverrideMB };
    const physicalCapMB = computeBudgetMB(
      hardwareService.getTotalMemoryGB() * 1024,
      { policy: this.loadPolicy },
    );
    // Dirty-memory PRESSURE - the incoming model is dirty, OR a dirty model (CoreML/ONNX
    // image) is already resident. A dirty model's working set/compile spike can't be
    // paged out like clean mmap weights, so while one is present EVERY load (even an
    // mmap sidecar) must also respect real free RAM, or stacking onto the spike jetsams
    // the app.
    const dirtyPressure =
      !!spec.dirtyMemory ||
      [...this.residents.values()].some(r => r.dirtyMemory);
    // Android: EVERY load is live-gated. The physical-RAM-only path admitted an 8GB model at
    // ~3GB free on a 12GB phone (75% used by other apps); the load filled RAM faster than the
    // low-memory killer reclaimed it and the app itself was killed.
    if (!dirtyPressure && spec.liveMB == null && Platform.OS !== 'android') {
      return { budgetMB: Math.round(Math.max(MIN_BUDGET_MB, physicalCapMB)) };
    }
    // Aggressive mode holds a smaller headroom (the lenient safeguard). Still non-zero.
    const headroomMB =
      this.loadPolicy === 'aggressive' ? AGGRESSIVE_DIRTY_HEADROOM_MB : DIRTY_AVAILABILITY_HEADROOM_MB;
    return {
      budgetMB: Math.round(physicalCapMB),
      live: { availableMB: hardwareService.getAvailableMemoryGB() * 1024, headroomMB },
    };
  }

  /** Dirty footprint the plan keeps (the incoming dirty peak + dirty residents that stay) vs the
   *  BALANCED physical ceiling; see makeRoomFor. */
  private dirtyCeiling(spec: ResidentSpec, residents: Resident[], plan: EvictionPlan, totalMB: number) {
    const ceilingMB = computeBudgetMB(totalMB, { policy: 'balanced' });
    // Dirty residents that STAY, excluding the incoming's own slot: on a reload (e.g. an image
    // model re-loaded after a thread change) the incoming REPLACES that resident, so spec.sizeMB
    // already accounts for its footprint; counting the resident too double-charges the model (G4).
    const keptDirtyMB = residents
      .filter(r => r.dirtyMemory && r.key !== spec.key && !plan.evict.some(e => e.key === r.key))
      .reduce((sum, r) => sum + r.sizeMB, 0);
    const footprintMB = (spec.dirtyMemory ? spec.sizeMB : 0) + keptDirtyMB;
    return { ceilingMB, keptDirtyMB, exceeded: !!spec.dirtyMemory && footprintMB > ceilingMB };
  }

  getResidents(): Resident[] {
    return [...this.residents.values()].map(stripUnload);
  }

  isResident(key: string): boolean {
    return this.residents.has(key);
  }

  /**
   * Whether `spec` fits the budget alongside everything already resident,
   * WITHOUT evicting anything. Used by the boot preloader so warming a
   * lower-priority model never kicks out a higher-priority one.
   */
  canLoadWithoutEviction(spec: { key: string; sizeMB: number }): boolean {
    if (this.residents.has(spec.key)) return true;
    const usedMB = [...this.residents.values()].reduce(
      (sum, r) => sum + r.sizeMB,
      0,
    );
    return usedMB + spec.sizeMB <= this.getBudgetMB();
  }

  markUsed(key: string, now: number = Date.now()): void {
    const r = this.residents.get(key);
    if (r) r.lastUsedAt = now;
  }

  /**
   * Register a model that's already loaded elsewhere (e.g. a pinned classifier
   * or a model loaded before the manager existed) so it's accounted for.
   */
  register(
    spec: ResidentSpec,
    unload: UnloadFn,
    now: number = Date.now(),
  ): symbol {
    const registration = Symbol(spec.key);
    this.residents.set(spec.key, { ...spec, lastUsedAt: now, unload });
    this.residentRegistrations.set(spec.key, registration);
    return registration;
  }

  /** Forget a capability-owned resident after that capability has released its native engine. */
  unregister(key: string, registration: symbol): boolean {
    if (this.residentRegistrations.get(key) !== registration) return false;
    return this.residentRegistrations.delete(key) && this.residents.delete(key);
  }

  /**
   * Make `spec` resident, evicting others to fit the budget. `load` runs only
   * if the model isn't already resident; `unload` is stored for future eviction.
   */
  /**
   * Evict residents (per the budget + mutual-exclusion policy) to make room for
   * `spec`, WITHOUT loading it. For callers that own the actual load themselves
   * (e.g. activeModelService) but want the manager to enforce memory. Returns
   * the evicted keys.
   */
  async makeRoomFor(
    spec: ResidentSpec,
    opts?: { override?: boolean },
  ): Promise<{ evicted: string[]; fits: boolean; neededMB?: number; availableMB?: number }> {
    // Re-read real free RAM so the decision reflects current pressure, not a stale
    // boot-time snapshot (other apps may have grabbed memory since).
    await hardwareService.refreshMemoryInfo().catch(() => {});
    // "Load Anyway" (opts.override) covers THIS load only: nothing is remembered, so the next load
    // of any model is planned normally again.
    const override = !!opts?.override;
    const capacity = this.capacityForSpec(spec);
    const budgetMB = capacity.budgetMB;
    const residents = this.planningResidents();
    const availMB = Math.round(hardwareService.getAvailableMemoryGB() * 1024);
    const totalMB = Math.round(hardwareService.getTotalMemoryGB() * 1024);
    // Conservative = ONE model at a time (evict everything else). Balanced and aggressive co-reside
    // whatever fits (aggressive with a larger RAM budget).
    const conservative = this.loadPolicy === 'conservative';
    let plan = planEviction(residents, spec, budgetMB, { singleModel: conservative, live: capacity.live });
    // SECOND GATE: the dirty physical ceiling. Aggressive mode's larger RAM fraction is safe for
    // CLEAN (mmap, pageable) weights, but NOT for a DIRTY model whose un-pageable GPU/anonymous
    // pages can't be zram-backed. So a dirty model's footprint (the incoming + any dirty residents
    // that STAY) is bounded by the BALANCED physical ceiling regardless of policy (overridable via
    // Load Anyway). Independent of the total-budget check above.
    let dirty = this.dirtyCeiling(spec, residents, plan, totalMB);
    // Load Anyway: a load that fits is planned exactly like any other (it evicts nothing extra);
    // only a load that does not fit frees every evictable resident to give it the most RAM.
    const forced = override && (!plan.fits || dirty.exceeded);
    if (forced && !conservative) {
      plan = planEviction(residents, spec, budgetMB, { singleModel: true, live: capacity.live });
      dirty = this.dirtyCeiling(spec, residents, plan, totalMB);
    }
    // [MEM-SM] trace (kept forever): the exact numbers behind every fit decision, plus the raw
    // free/total RAM so a refusal is explainable.
    logger.log(
      formatMakeRoomForLine({ spec, budgetMB, availMB, totalMB, residents, plan, live: capacity.live }),
    );

    if ((!plan.fits || dirty.exceeded) && !override) {
      // Won't fit even after the planned evictions (physical, live, or the dirty ceiling): DON'T
      // evict. The free-RAM decision is made here, BEFORE any eviction, so a refused load never
      // costs a resident model. The caller blocks the load (overridable). Report the binding check
      // (needed vs. the room the incoming model would get), in MB.
      const liveBinds = plan.liveRoomMB !== undefined && plan.incomingLiveMB > plan.liveRoomMB;
      let neededMB = liveBinds ? plan.incomingLiveMB : spec.sizeMB;
      let roomMB = liveBinds && plan.liveRoomMB !== undefined ? plan.liveRoomMB : plan.physicalRoomMB;
      if (!liveBinds && dirty.exceeded) {
        neededMB = spec.sizeMB;
        roomMB = Math.min(roomMB, dirty.ceilingMB - dirty.keptDirtyMB);
      }
      return {
        evicted: [],
        fits: false,
        neededMB: Math.round(neededMB),
        availableMB: Math.max(0, Math.round(roomMB)),
      };
    }
    if (forced) logger.log(formatOverrideForcingLine(spec.key, plan.evict));
    const actuallyEvicted: string[] = [];
    let unloadFailed = false;
    for (const victim of plan.evict) {
      const reg = this.residents.get(victim.key);
      if (!reg) continue;
      try {
        await reg.unload();
        this.residents.delete(victim.key);
        actuallyEvicted.push(victim.key);
      } catch (err) {
        // The native unload REJECTED: the victim still holds its RAM. Do NOT delete it
        // from the budget map (counting phantom-freed memory over-commits the incoming
        // load, OOM). Keep it resident and abort the fit.
        logger.log(`[ModelResidency] unload ${victim.key} failed:`, err);
        unloadFailed = true;
        break;
      }
    }
    if (unloadFailed) {
      return { evicted: actuallyEvicted, fits: false };
    }
    // Load Anyway is unconditional: the user explicitly accepted the risk, so a load that does not
    // fit evicts everything else and loads. NO survival floor, NO refusal.
    if (forced) logger.log(formatOverrideForcedLine(spec.key, plan.evict));
    return { evicted: actuallyEvicted, fits: true };
  }

  async ensureResident(
    spec: ResidentSpec,
    handlers: { load: () => Promise<void>; unload: UnloadFn },
    now: number = Date.now(),
  ): Promise<EnsureResult> {
    const { evicted, fits } = await this.makeRoomFor(spec);

    if (this.residents.has(spec.key)) {
      this.markUsed(spec.key, now);
      return { loaded: false, evicted };
    }

    // Honor the fit verdict: a model that does not fit must NOT be loaded (the caller
    // used to invoke the gate then load regardless — the STT/OOM bug class).
    if (!fits) {
      return { loaded: false, evicted };
    }

    await handlers.load();
    this.residents.set(spec.key, {
      ...spec,
      lastUsedAt: now,
      unload: handlers.unload,
    });
    return { loaded: true, evicted };
  }

  /** Forget a resident the owner has already unloaded (no unload call). */
  release(key: string): void {
    this.residents.delete(key);
  }

  /** Eject a single resident and ACTUALLY unload it from RAM (unlike release(), which only forgets the
   *  accounting). Used by the model selector's per-model Eject. Runs under the FIFO lock like every other
   *  load/unload. Returns true if a resident was found and ejected. */
  async evictByKey(key: string): Promise<boolean> {
    return this.runExclusive(`evict:${key}`, async () => {
      const r = this.residents.get(key);
      if (!r) return false;
      await r.unload().catch(err => logger.log(`[ModelResidency] evict ${key} unload failed:`, err));
      this.residents.delete(key);
      logger.log(`[ModelResidency] evicted ${r.type} (${key}) by user request`);
      return true;
    });
  }

  /**
   * A generation turn is starting: the mic (STT/Whisper) model is idle while the
   * LLM runs, and its RAM is better spent on the LLM's inference working set (which
   * the file-size budget doesn't capture). On a memory-tight device, free it so the
   * generation working set doesn't tip the app past the jetsam limit (the 4GB
   * resend OOM). STT reloads on the next record. Roomy devices keep it warm.
   * Centralizes the "evict idle audio sidecar for generation" decision here.
   */
  async reclaimSttForGeneration(): Promise<void> {
    // Best-effort memory optimization in the generation hot path - must NEVER throw
    // into it (e.g. if the hardware service isn't available). Bail quietly instead.
    let totalGB: number;
    try {
      totalGB = hardwareService.getTotalMemoryGB();
    } catch {
      return;
    }
    if (totalGB > 6) return; // roomy: keep STT warm
    if (!this.residents.has('whisper')) return;
    // Serialize with load/unload: this fires on the generation hot path (every send /
    // regenerate), so without the lock it can race an in-flight whisper load and desync
    // the residents map. whisper's unload doesn't re-acquire the lock, so no deadlock.
    await this.runExclusive('reclaim:stt', async () => {
      const w = this.residents.get('whisper');
      if (!w) return; // reclaimed by another op while we waited for the lock
      if (w.canEvict && !w.canEvict()) return; // in use (e.g. finalizing a transcription) - owner vetoes
      logger.log(
        '[ModelResidency] reclaiming idle STT for generation turn (memory-tight)',
      );
      try {
        await w.unload();
        this.residents.delete('whisper');
      } catch (err) {
        // Native unload REJECTED — whisper still holds its RAM. Keep it counted resident so
        // the next load sizes against real (not phantom-freed) memory rather than OOMing.
        logger.log('[ModelResidency] STT reclaim failed:', err);
      }
    });
  }

  /** Test helper. */
  _reset(): void {
    this.residents.clear();
    this.residentRegistrations.clear();
    this.budgetOverrideMB = null;
    this.opChain = Promise.resolve();
  }
}

export const modelResidencyManager = new ModelResidencyManager();
;
