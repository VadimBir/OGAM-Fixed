import logger from '../../utils/logger';

/**
 * The ONE lock every load-modify-save of the model registries (text + image) runs under.
 *
 * Up to three downloads finish concurrently (MAX_CONCURRENT_DOWNLOADS), and each finalize did
 * `load -> push -> save` on the same AsyncStorage key with an await between load and save. Two
 * finalizes interleaving both loaded the old list and the second save dropped the first model;
 * a delete racing a finalize resurrected or lost a row the same way. Writers now queue on one
 * promise chain, so each sees the previous writer's result.
 *
 * NOT re-entrant: code holding the lock must call the unlocked `loadDownloadedModels` /
 * `saveModelsList` / `saveImageModelsList`, never another locked writer (it would wait on itself).
 */
let registryWriteChain: Promise<void> = Promise.resolve();
export function withModelRegistryLock<T>(work: () => Promise<T>): Promise<T> {
  const run = registryWriteChain.then(work);
  // The chain only orders writers; the caller still receives `run`'s own rejection.
  registryWriteChain = run.then(
    () => undefined,
    (error: unknown) => {
      logger.warn(`[ModelManagerStorage] registry write failed: ${error instanceof Error ? error.message : String(error)}`);
    },
  );
  return run;
}
