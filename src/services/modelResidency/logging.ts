/**
 * [MEM-SM] log-line formatting for the residency manager.
 *
 * Extracted from index.ts (behavior-neutral) so the manager file stays within the
 * max-lines budget. These are PURE string builders — they produce the exact same
 * text the inline `logger.log('[MEM-SM] …')` calls produced; the manager still owns
 * WHEN a line is logged (call site + order unchanged), these only build the WHAT.
 */
import { Resident, ResidentType } from './policy';

interface FitSpec {
  key: string;
  sizeMB: number;
  liveMB?: number;
  dirtyMemory?: boolean;
}

interface EvictionPlanLike {
  fits: boolean;
  evict: Array<{ key: string }>;
  liveRoomMB?: number;
}

const liveSuffix = (liveMB?: number): string => (liveMB == null ? '' : `/live${liveMB}`);

/**
 * The primary fit-decision trace: the exact numbers behind every fit decision.
 * capacityForSpec gives the physical budget and, when the load is live-gated, the live
 * free RAM + headroom; planEviction enforces both. Also logs the raw free/total RAM so a
 * refusal is explainable.
 */
export function formatMakeRoomForLine(args: {
  spec: FitSpec;
  budgetMB: number;
  availMB: number;
  totalMB: number;
  residents: Array<Pick<Resident, 'key' | 'sizeMB' | 'pinned'> & { type?: ResidentType; liveMB?: number }>;
  plan: EvictionPlanLike;
  live?: { availableMB: number; headroomMB: number };
}): string {
  const { spec, budgetMB, availMB, totalMB, residents, plan, live } = args;
  const liveText = live
    ? ` liveMB=${spec.liveMB ?? spec.sizeMB} headroomMB=${live.headroomMB} liveRoomMB=${Math.round(plan.liveRoomMB ?? 0)}`
    : ' live=off';
  return `[MEM-SM] makeRoomFor ${spec.key} sizeMB=${
    spec.sizeMB
  } dirty=${!!spec.dirtyMemory} budgetMB=${budgetMB}${liveText} os_procAvailMB=${availMB} totalMB=${totalMB} residents=[${residents
    .map(r => `${r.key}:${r.sizeMB}${liveSuffix(r.liveMB)}${r.pinned ? '(pinned)' : ''}`)
    .join(',')}] fits=${plan.fits} evict=[${plan.evict
    .map(e => e.key)
    .join(',')}]`;
}

/** OVERRIDE pre-eviction trace ("forcing load after evicting …"). */
export function formatOverrideForcingLine(specKey: string, evict: Array<{ key: string }>): string {
  return `[MEM-SM] makeRoomFor ${specKey} OVERRIDE - forcing load after evicting [${evict
    .map(e => e.key)
    .join(',')}]`;
}

/** OVERRIDE post-eviction trace ("forced load after evicting … (no floor)"). */
export function formatOverrideForcedLine(specKey: string, evict: Array<{ key: string }>): string {
  return `[MEM-SM] makeRoomFor ${specKey} OVERRIDE - forced load after evicting [${evict.map(e => e.key).join(',')}] (no floor)`;
}
