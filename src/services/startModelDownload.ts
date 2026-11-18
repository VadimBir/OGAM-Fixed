import { modelManager } from './modelManager';
import { mmProjLocalName, splitGgufRefusal, ALREADY_DOWNLOADED_PREFIX } from './modelManager/download';
import logger from '../utils/logger';
import { useDownloadStore, isActiveStatus } from '../stores/downloadStore';
import { useAppStore } from '../stores';
import { activeModelService } from './activeModelService';
import { makeModelKey } from '../utils/modelKey';
import type { ModelFile, DownloadedModel } from '../types';

/** Placeholder downloadId for a row that exists only to represent the QUEUED state
 *  before a real native download has been started (which may wait for a concurrency
 *  slot). Reconciled to the real id by download.ts's retryEntry once the start begins. */
const queuedPlaceholderId = (modelKey: string) => `queued:${modelKey}`;

/**
 * Fetch only the vision projector of a model whose weights were already installed, then report the
 * model (with vision when the fetch worked, as text-needing-repair when it did not).
 */
function fetchMissingProjector(
  modelId: string,
  file: ModelFile,
  model: DownloadedModel,
  opts: StartModelDownloadOpts,
): void {
  modelManager.repairMmProj(modelId, file, { modelKey: model.id })
    .then(async () => {
      const refreshed = (await modelManager.getDownloadedModels()).find(m => m.id === model.id) ?? model;
      useAppStore.getState().addDownloadedModel(refreshed);
      opts.onRegistered?.(refreshed);
    })
    .catch((error: unknown) => {
      logger.warn(`[startModelDownload] projector fetch for installed ${model.id} failed: ${error instanceof Error ? error.message : String(error)}`);
      // Registered as text with the vision gap recorded (mmProjFileName), so Repair Vision is offered.
      opts.onRegistered?.(model);
    });
}

export interface StartModelDownloadOpts {
  /** Select a downloaded user model when the download began without a selection.
   * Hidden support-model downloads leave this off. */
  autoSelectIfEmpty?: boolean;
  /** Screen-specific UI to run AFTER the model is registered (e.g. a success or
   *  vision-repair alert). The standard register + clear has already happened. */
  onRegistered?: (model: DownloadedModel) => void;
  /** Screen-specific error UI (e.g. an alert). The in-flight entry's 'failed' status
   *  is already set when this fires. */
  onError?: (error: Error) => void;
}

/**
 * THE single entry point to start a model download — shared by the Models screen
 * (useTextModels) and the onboarding ModelDownloadScreen so both use the IDENTICAL
 * mechanism instead of two near-duplicate handlers:
 *   duplicate-start guard → modelManager.downloadModelBackground → watchDownload →
 *   register the model (appStore) + clear the in-flight store entry.
 * Each screen owns only its presentation, via onRegistered / onError. The underlying
 * pipeline (backgroundDownloadService, downloadStore progress, mmproj sidecar) was
 * already shared; this collapses the last duplicated wrapper around it.
 */
export async function startModelDownload(
  modelId: string,
  file: ModelFile,
  opts: StartModelDownloadOpts = {},
): Promise<void> {
  // Capture this before the asynchronous download starts. If the user had a
  // selection and explicitly unloads it while this download runs, completion
  // must not replace that decision.
  const mayAutoSelect =
    opts.autoSelectIfEmpty === true &&
    useAppStore.getState().activeModelId === null;
  const modelKey = makeModelKey(modelId, file.name);
  // A split GGUF would fetch part 1 only - a file llama.cpp cannot load. Refuse it up front with a
  // clear reason instead of a "successful" download that never loads.
  const refusal = splitGgufRefusal(file);
  if (refusal) {
    logger.warn(`[startModelDownload] refused split GGUF ${modelId}/${file.name} (${file.splitPartCount} parts)`);
    opts.onError?.(new Error(refusal));
    return;
  }
  // Duplicate-start guard: a download already active for this logical file (rapid
  // double-tap, race after retry) is a no-op. add() also enforces this; checking
  // up-front avoids the unnecessary native call.
  const existing = useDownloadStore.getState().downloads[modelKey];
  if (existing && isActiveStatus(existing.status)) return;

  // Publish a QUEUED row to the store IMMEDIATELY, before starting the (possibly
  // slot-limited) native download. Without this, a queued download had no store entry
  // at all until a concurrency slot freed up — so the Models/onboarding screens (which
  // read the store) showed nothing, and this very guard missed a second tap while
  // queued (letting the same model enqueue twice). The store is the single source of
  // truth all screens read; download.ts reconciles this placeholder id to the real
  // native downloadId via retryEntry once the start begins.
  useDownloadStore.getState().add({
    modelKey,
    downloadId: queuedPlaceholderId(modelKey),
    modelId,
    fileName: file.name,
    quantization: file.quantization,
    modelType: 'text',
    status: 'pending',
    bytesDownloaded: 0,
    totalBytes: file.size,
    combinedTotalBytes: file.size + (file.mmProjFile?.size ?? 0),
    progress: 0,
    createdAt: Date.now(),
    ...(file.mmProjFile && {
      mmProjFileName: mmProjLocalName(file.name, file.mmProjFile.name),
      mmProjFileSize: file.mmProjFile.size,
    }),
  });

  // Until the real native start begins, the failure target is the queued placeholder row.
  let currentDownloadId: string | undefined = queuedPlaceholderId(modelKey);
  const fail = (err: Error) => {
    if (currentDownloadId) {
      useDownloadStore.getState().setStatus(currentDownloadId, 'failed', { message: err.message });
    }
    opts.onError?.(err);
  };

  try {
    // downloadModelBackground writes the row + adds the store entry synchronously
    // after start (add for new, retryEntry for an existing failed one — including the
    // queued placeholder row added above, whose id it reconciles to the real one).
    const info = await modelManager.downloadModelBackground(modelId, file);
    currentDownloadId = info.downloadId;
    modelManager.watchDownload(info.downloadId, (model: DownloadedModel) => {
      // Standard completion: register + clear the in-flight entry so the UI reads
      // "downloaded" from downloadedModels, not a lingering 100% store entry.
      useAppStore.getState().addDownloadedModel(model);
      const app = useAppStore.getState();
      // Select THE MODEL JUST DOWNLOADED when nothing is selected. Picking lastTextModelId (an
      // older choice) or downloadedModels[0] (whatever sorts first) selected some other model,
      // so the download the user just waited for was not the one chat used.
      if (mayAutoSelect && app.activeModelId === null && app.downloadedModels.some(item => item.id === model.id)) {
        activeModelService.selectTextModel(model.id);
      }
      useDownloadStore.getState().remove(modelKey);
      // The main file was already installed and complete, only the projector is missing: fetch the
      // projector ALONE (the repair path) instead of re-downloading - and on cancel deleting - the
      // installed weights.
      const projectorMissing = !!file.mmProjFile && !(model.engine === 'llama' && model.mmProjPath);
      if (String(info.downloadId).startsWith(ALREADY_DOWNLOADED_PREFIX) && projectorMissing && file.mmProjFile?.downloadUrl) {
        fetchMissingProjector(modelId, file, model, opts);
        return;
      }
      opts.onRegistered?.(model);
    }, fail);
  } catch (e) {
    // A start cancelled while still queued (no slot yet) rejects with `.cancelled` — it
    // is not an error, so drop the queued placeholder row and clean up quietly.
    if ((e as Error & { cancelled?: boolean })?.cancelled) {
      useDownloadStore.getState().remove(modelKey);
      return;
    }
    fail(e as Error);
  }
}
