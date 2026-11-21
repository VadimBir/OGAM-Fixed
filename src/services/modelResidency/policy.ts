/**
 * Model residency policy (pure functions).
 *
 * Decides which on-device models stay in RAM. A phone can't hold every model
 * at once, so before loading a model we evict others to fit a RAM budget:
 *  - text and image generation models co-reside when both fit; when the incoming
 *    one does not, the lower-priority / least-recently-used one is evicted;
 *  - speech (whisper) and TTS are small co-resident sidecars: they stay warm
 *    alongside whichever generation model is loaded and loading one never evicts
 *    anything; they're evicted only as a last resort when a heavy generation
 *    model can't otherwise fit;
 *  - pinned models (e.g. the ~100MB SMOL classifier) are never evicted;
 *  - when an incoming generation model doesn't fit, evict least-recently-used
 *    (non-sidecar, non-pinned) residents until it does.
 *
 * See docs/design/MODEL_ROUTING.md §4–5.2. Pure + deterministic so the policy
 * can be unit-tested without touching native model loading.
 */

import { modelMemoryBudgetMB, LoadPolicy } from '../memoryBudget';

export type ResidentType = 'text' | 'image' | 'whisper' | 'tts' | 'classifier' | 'embedding';

export interface Resident {
  /** Unique model id. */
  key: string;
  type: ResidentType;
  /** The downloaded-model id behind the slot, when the owner registered one (text). */
  modelId?: string;
  /** Resident memory cost in MB (resident bytes: no peak factor, no transient compute). */
  sizeMB: number;
  /** Part of sizeMB that consumes live free RAM; absent = all of it. Smaller for a text model whose
   *  mmapped weights are reclaimable page cache (memoryBudget.textMemoryFootprint). */
  liveMB?: number;
  /** Pinned residents are never evicted (e.g. the classifier). */
  pinned?: boolean;
  /** True for dirty-memory models (CoreML/ONNX image): their working set can't be
   *  paged out like clean mmap weights, so their presence triggers the live os_proc
   *  gate on other loads. */
  dirtyMemory?: boolean;
  /** Epoch ms of last use, for LRU. */
  lastUsedAt: number;
}

export interface IncomingModel {
  key: string;
  type: ResidentType;
  /** The downloaded-model id; a resident in the same slot with another id is being replaced. */
  modelId?: string;
  /** Peak MB of the load (the one place the peak factor applies). */
  sizeMB: number;
  /** Part of sizeMB that consumes live free RAM; absent = all of it. */
  liveMB?: number;
  /** True for real committed RAM that cannot page (CoreML/ONNX image, LiteRT); a clean
   *  (mmap GGUF) incoming pages in and only competes for the physical-RAM ceiling. */
  dirtyMemory?: boolean;
}

export interface EvictionPlan {
  /** Residents to unload, in eviction order. */
  evict: Resident[];
  /** Whether the incoming model fits the budget after eviction (physical AND live). */
  fits: boolean;
  freedMB: number;
  /** Physical room left for the incoming model after the planned evictions. */
  physicalRoomMB: number;
  /** Live room left (free RAM - headroom + what the evictions release); undefined = not live-gated. */
  liveRoomMB?: number;
  /** MB the incoming model is charged physically (0 when the same model is already resident). */
  incomingMB: number;
  /** MB the incoming model is charged against live free RAM. */
  incomingLiveMB: number;
}

/** Live free RAM a load may use: the reading and the headroom that must stay free. */
export interface LiveRam {
  availableMB: number;
  headroomMB: number;
}

const liveOf = (r: { sizeMB: number; liveMB?: number }): number => r.liveMB ?? r.sizeMB;

// Speech (whisper), TTS, and the RAG/MCP embedding model are small sidecars.
const SIDECAR_TYPES = new Set<ResidentType>(['whisper', 'tts', 'embedding']);

// Priority (what to KEEP): text is highest, then image, then the STT/TTS/embedding
// sidecars (equal, lowest), then pinned helpers. Eviction takes the lowest first.
const PRIORITY: Record<ResidentType, number> = {
  text: 3, image: 2, whisper: 1, tts: 1, embedding: 1, classifier: 0,
};

/**
 * The SINGLE rule for "which resident to evict next" — lowest priority, then
 * least-recently-used, honoring pinned + the sidecar restriction. Used by both the
 * pure budget planner (planEviction) and the manager's measure-after-evict loop, so
 * the two can never disagree on eviction order. Returns undefined when nothing is
 * evictable (caller stops and reports fits=false).
 */
function selectEvictionVictim(
  current: Resident[],
  incoming: IncomingModel,
  isEvicted: (r: Resident) => boolean,
): Resident | undefined {
  const incomingIsSidecar = SIDECAR_TYPES.has(incoming.type);
  return current
    .filter(r =>
      !r.pinned && r.key !== incoming.key && !isEvicted(r) &&
      // A sidecar incoming may only reclaim from peer sidecars (never a
      // generation model); a generation incoming may evict anything non-pinned.
      (!incomingIsSidecar || SIDECAR_TYPES.has(r.type)),
    )
    .sort((a, b) => {
      const pa = PRIORITY[a.type] ?? 0;
      const pb = PRIORITY[b.type] ?? 0;
      if (pa !== pb) return pa - pb;        // lowest priority evicted first
      return a.lastUsedAt - b.lastUsedAt;   // then least-recently-used
    })[0];
}

/**
 * Compute a RAM budget for resident models from total device RAM, leaving
 * headroom for the OS and the rest of the app.
 */
export function computeBudgetMB(
  totalRamMB: number,
  opts?: { reserveMB?: number; fraction?: number; policy?: LoadPolicy },
): number {
  // Explicit overrides keep the old escape hatch; otherwise defer to the single
  // device + platform aware budget owner (so residency and the pre-load memory
  // check can never disagree, and high-RAM/iOS-entitled devices get their larger
  // safe fraction instead of a flat 60%). The load policy ('aggressive') tunes the
  // fraction + reserve in that one owner — never branched on here.
  if (opts?.fraction != null || opts?.reserveMB != null) {
    const fraction = opts?.fraction ?? 0.6;
    const reserveMB = opts?.reserveMB ?? 1500;
    return Math.max(0, Math.min(totalRamMB * fraction, totalRamMB - reserveMB));
  }
  return modelMemoryBudgetMB(totalRamMB, undefined, opts?.policy);
}

/**
 * Plan which residents to evict so `incoming` fits. Never evicts pinned residents or the incoming
 * model itself. Two checks, both must hold:
 *  - physical: kept residents (resident MB) + the incoming peak <= `budgetMB`;
 *  - live (when `opts.live`): the incoming live peak <= free RAM - headroom + the live MB that the
 *    evictions (and a replaced model in the same slot) release.
 *
 * `opts.singleModel` (conservative mode, or a "Load Anyway" that does not fit): keep ONE model at a
 * time - evict EVERY evictable resident (still honoring pinned + in-use veto).
 */
export function planEviction(
  current: Resident[],
  incoming: IncomingModel,
  budgetMB: number,
  opts?: { singleModel?: boolean; live?: LiveRam },
): EvictionPlan {
  const evict: Resident[] = [];
  const isEvicted = (r: Resident) => evict.some(e => e.key === r.key);
  // Same slot, same model: nothing new is allocated. Same slot, another model: the loader unloads
  // the resident first, so it is not counted and its live RAM is credited.
  const sameSlot = current.filter(r => r.key === incoming.key);
  const alreadyResident = sameSlot.some(
    r => !incoming.modelId || !r.modelId || r.modelId === incoming.modelId,
  );
  const replacedLiveMB = alreadyResident ? 0 : sameSlot.reduce((sum, r) => sum + liveOf(r), 0);

  const usedMB = () =>
    current
      .filter(r => r.key !== incoming.key && !isEvicted(r))
      .reduce((sum, r) => sum + r.sizeMB, 0);
  const incomingMB = alreadyResident ? 0 : incoming.sizeMB;
  const incomingLiveMB = alreadyResident ? 0 : liveOf(incoming);
  const physicalRoom = () => budgetMB - usedMB();
  const liveRoom = () =>
    opts?.live
      ? opts.live.availableMB - opts.live.headroomMB + replacedLiveMB +
        evict.reduce((sum, r) => sum + liveOf(r), 0)
      : undefined;
  const fits = () => {
    const live = liveRoom();
    return incomingMB <= physicalRoom() && (live === undefined || incomingLiveMB <= live);
  };
  const result = (): EvictionPlan => ({
    evict,
    fits: fits(),
    freedMB: evict.reduce((sum, r) => sum + r.sizeMB, 0),
    physicalRoomMB: physicalRoom(),
    liveRoomMB: liveRoom(),
    incomingMB,
    incomingLiveMB,
  });

  if (opts?.singleModel) {
    // One model at a time: evict everything evictable (no co-residency). selectEvictionVictim
    // still skips pinned + in-use (canEvict veto) residents, so the classifier and a
    // playing TTS survive; every other model is unloaded to free the most RAM.
    for (let victim = selectEvictionVictim(current, incoming, isEvicted);
      victim;
      victim = selectEvictionVictim(current, incoming, isEvicted)) {
      evict.push(victim);
    }
    return result();
  }

  // Smart routing (balanced): KEEP as many models co-resident as the budget allows;
  // only when the incoming model doesn't fit do we evict — ONE AT A TIME, lowest
  // priority (then least-recently-used) first (selectEvictionVictim, shared with
  // the manager's measure-after-evict loop).
  //   - Text + image co-reside when they fit (e.g. image-gen with prompt
  //     enhancement keeps both warm) — no forced mutual exclusion.
  //   - A 4th model that needs room swaps out a single lowest-priority/LRU victim,
  //     never a blanket clear.
  //   - A sidecar load (mic/speaker) never evicts a heavier generation model —
  //     that would break an in-flight answer — so it may only reclaim from peer
  //     sidecars; if that's not enough it reports fits=false and the caller bails.
  while (!fits()) {
    const victim = selectEvictionVictim(current, incoming, isEvicted);
    if (!victim) break; // nothing evictable left → fits stays false, caller bails
    evict.push(victim);
  }
  return result();
}
