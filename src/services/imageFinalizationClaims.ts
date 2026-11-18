/**
 * Which image downloads are being finalized (move + unzip/classify + register) by a live,
 * in-process completion listener right now.
 *
 * Both a live listener and the Models screen's "processing" resume effect react to the same
 * native completion (useDownloads sets `processing` for every image row). Without a shared claim
 * both ran the finalization for one download concurrently: two moves of one staged file, two
 * unzips into one directory, two registrations and two "downloaded" alerts. The live finalizer
 * claims the model id before its first await; the resume effect skips claimed ids and only
 * finalizes downloads nobody in this process is handling (restored after a relaunch).
 */
const claimed = new Set<string>();

/** Claim `modelId` for finalization. False when someone else already holds the claim. */
export function claimImageFinalization(modelId: string): boolean {
  if (claimed.has(modelId)) return false;
  claimed.add(modelId);
  return true;
}

export function releaseImageFinalization(modelId: string): void {
  claimed.delete(modelId);
}

export function isImageFinalizationClaimed(modelId: string): boolean {
  return claimed.has(modelId);
}
