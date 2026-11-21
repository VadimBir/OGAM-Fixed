/**
 * Residency budget constants + resident types.
 *
 * Extracted from index.ts (behavior-neutral) so the manager file stays within the
 * max-lines budget. These are the top-level constants, interfaces, and the pure
 * `stripUnload` projection the manager uses; moving them here changes no values and
 * no behavior — the manager imports them back unchanged.
 */
import { Resident, ResidentType } from './policy';
import { LIVE_RAM_MARGIN_MB, AGGRESSIVE_LIVE_RAM_MARGIN_MB } from '../memoryBudget';

export type UnloadFn = () => Promise<void>;

/** Physical-budget floor for clean, pageable models on iOS; never force live-gated loads past live RAM. */
export const MIN_BUDGET_MB = 1024;
/** Live-gated loads (DIRTY models on every platform; every model on Android): keep this
 *  much real RAM free for the OS + other apps so a load never spills into swap or the
 *  low-memory killer. Alias of the memoryBudget live margin (one owner of the number). */
export const DIRTY_AVAILABILITY_HEADROOM_MB = LIVE_RAM_MARGIN_MB;
/** Aggressive-mode live headroom - leaner, still non-zero (lenient safeguard). */
export const AGGRESSIVE_DIRTY_HEADROOM_MB = AGGRESSIVE_LIVE_RAM_MARGIN_MB;
/** Small, cheaply-reloadable models reclaimed first under memory pressure. */
export const SIDECAR_TYPES = new Set<ResidentType>(['whisper', 'tts', 'embedding']);

export interface RegisteredResident extends Resident {
  unload: UnloadFn;
  /** Owner's veto: returns false when the model is in use right now (e.g. TTS is
   *  playing) so residency never evicts it mid-use. Absent → always evictable. */
  canEvict?: () => boolean;
  /** Owner's in-flight probe for the memory-pressure (trim) release only: true while a
   *  generation that needs this model is running. Unlike canEvict it never pins the model
   *  for capacity eviction. Absent → never busy. */
  isBusy?: () => boolean | Promise<boolean>;
}

export interface ResidentSpec {
  key: string;
  type: ResidentType;
  /** The specific downloaded-model id. `key` is only the slot/type (e.g. 'text'), so this tells a
   *  reload of the same model (charged 0) from a different model replacing it (charged in full). */
  modelId?: string;
  /** makeRoomFor: the load peak (the one place the peak factor applies). register: the resident
   *  bytes (no factor, no transient compute). */
  sizeMB: number;
  /** Part of sizeMB that consumes live free RAM; absent = all of it. A text model whose mmapped
   *  weights are reclaimable passes only its anonymous bytes (memoryBudget.textMemoryFootprint). */
  liveMB?: number;
  pinned?: boolean;
  /** Owner's veto: returns false while the model is in use (e.g. TTS playing) so
   *  residency never evicts it mid-use. Absent → always evictable. */
  canEvict?: () => boolean;
  /** In-flight probe used only by the trim release (see RegisteredResident.isBusy). */
  isBusy?: () => boolean | Promise<boolean>;
  /**
   * Whether the model's weights occupy DIRTY (anonymous, jetsam-counted) memory -
   * the gap modeled as DATA, not a Platform/type branch in the budget.
   *  - false (default): mmap-backed GGUF (llama text / whisper). Weights are CLEAN,
   *    file-backed pages the OS pages freely; they do NOT pressure os_proc_available.
   *    iOS: bounded by PHYSICAL RAM only - so an 8GB GGUF loads on a 12GB phone.
   *    Android: also bounded by live free RAM (every Android load is live-gated).
   *  - true: CoreML/ONNX image weights load into dirty/GPU memory that DOES count
   *    against the jetsam limit → also bounded by real free RAM (os_proc_available)
   *    so it never loads into swap.
   */
  dirtyMemory?: boolean;
}

export interface EnsureResult {
  loaded: boolean;
  evicted: string[];
}

export const stripUnload = ({
  unload: _unload,
  ...rest
}: RegisteredResident): Resident => rest;
