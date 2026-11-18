/**
 * "Download to phone" for a Civitai file: the shared native background downloader (WorkManager:
 * survives app exit, progress, cancel, exact-size check, SHA-256 when Civitai gives one), then the
 * same register step as a .safetensors file import (sdCppFileRegistration.registerSdFileInPlace): a checkpoint becomes a
 * selectable stable-diffusion.cpp image model, a LoRA joins the image LoRA list.
 *
 * The row lives in useDownloadStore under image:<id> with `imageDownloadType: 'sdcpp-file'`, so the
 * Download Manager shows it and imageDownloadResume finalizes it after an app restart.
 */
import RNFS from 'react-native-fs';
import { modelManager, backgroundDownloadService } from '.';
import { sdCppGeneratorService } from './sdCppGenerator';
import { useDownloadStore, isActiveStatus } from '../stores/downloadStore';
import { makeImageModelKey } from '../utils/modelKey';
import { showAlert, type AlertState } from '../utils/alertState';
import { getUserFacingDownloadMessage } from '../utils/downloadErrors';
import type { ONNXImageModel } from '../types';
import { claimImageFinalization, releaseImageFinalization } from './imageFinalizationClaims';
import { withCivitaiDownloadToken, type CivitaiResult } from './civitaiBrowser';
import { missingFreeSpace, formatGB, registerSdFileInPlace } from './sdCppFileRegistration';
import logger from '../utils/logger';

export interface SdCppDownloadDeps {
  addDownloadedImageModel: (model: ONNXImageModel) => void;
  activeImageModelId: string | null;
  setActiveImageModelId: (id: string) => void;
  setAlertState: (state: AlertState) => void;
}

/** Metadata persisted with the store row (no URL: a gated URL would carry the API key). */
export interface SdCppFileDownloadMetadata {
  imageDownloadType: 'sdcpp-file';
  imageModelName: string;
  imageModelDescription: string;
  imageModelSize: number;
  imageModelFileName: string;
  civitaiType: CivitaiResult['type'];
  civitaiBaseModel: string;
}

/** Image-model id of a Civitai file: stable, so a second tap finds the same store row / model. */
export function civitaiImageModelId(result: Pick<CivitaiResult, 'versionId' | 'file'>): string {
  return `civitai_${result.versionId}_${result.file.id}`;
}

/** File name kept on disk: the Civitai name with only safe characters, always .safetensors. */
export function civitaiLocalFileName(result: Pick<CivitaiResult, 'file'>): string {
  const base = result.file.name.replace(/\.safetensors$/i, '').replaceAll(/[^a-zA-Z0-9._-]/g, '_') || 'model';
  return `${base}.safetensors`;
}

export function civitaiDisplayName(result: Pick<CivitaiResult, 'name' | 'versionName'>): string {
  return result.versionName ? `${result.name} ${result.versionName}` : result.name;
}

/**
 * Move the finished file into image_models/<modelId>/ and register it. Used by the live completion
 * listener and by the restart resume (imageDownloadResume). Throws on failure; the caller marks
 * the row failed. Leaves nothing half-registered behind.
 */
export async function finalizeSdCppFileDownload(opts: {
  downloadId: string;
  modelId: string;
  fileName: string;
  displayName: string;
  deps: SdCppDownloadDeps;
}): Promise<void> {
  const { downloadId, modelId, fileName, displayName, deps } = opts;
  const imageModelsDir = modelManager.getImageModelsDirectory();
  const modelDir = `${imageModelsDir}/${modelId}`;
  const dest = `${modelDir}/${fileName}`;
  useDownloadStore.getState().setProcessing(downloadId);
  try {
    if (!(await RNFS.exists(imageModelsDir))) await RNFS.mkdir(imageModelsDir);
    if (!(await RNFS.exists(modelDir))) await RNFS.mkdir(modelDir);
    if (!(await RNFS.exists(dest))) {
      await backgroundDownloadService.moveCompletedDownload(downloadId, dest);
    }
    const outcome = await registerSdFileInPlace(
      { path: dest, modelDir, fileName, modelId, verb: 'downloaded', displayName, source: 'From Civitai' },
      deps,
    );
    // A rejected file was already explained by an alert and removed; the row is done either way.
    logger.log(`[Civitai] ${modelId} finalized as ${outcome.kind}`);
    useDownloadStore.getState().remove(makeImageModelKey(modelId));
  } catch (error) {
    await RNFS.unlink(modelDir).catch(() => {});
    throw error;
  }
}

function failRow(downloadId: string, deps: SdCppDownloadDeps, message?: string) {
  deps.setAlertState(showAlert('Download Failed', getUserFacingDownloadMessage(message)));
  useDownloadStore.getState().setStatus(downloadId, 'failed', { message: message || 'Download failed' });
}

/** Start a background download of a Civitai checkpoint / LoRA to this phone. */
export async function startCivitaiPhoneDownload(
  result: CivitaiResult,
  deps: SdCppDownloadDeps,
  apiKey?: string | null,
): Promise<void> {
  if (!sdCppGeneratorService.isAvailable() || !backgroundDownloadService.isAvailable()) {
    deps.setAlertState(showAlert('Not Supported', 'Civitai checkpoints run on the stable-diffusion.cpp engine, which this build only has on Android.'));
    return;
  }
  const modelId = civitaiImageModelId(result);
  const modelKey = makeImageModelKey(modelId);
  const existing = useDownloadStore.getState().downloads[modelKey];
  if (existing && (isActiveStatus(existing.status) || existing.status === 'paused')) return;

  const shortBy = await missingFreeSpace(result.file.sizeBytes);
  if (shortBy !== null) {
    deps.setAlertState(showAlert(
      'Not Enough Storage',
      `${result.file.name} needs about ${formatGB(shortBy)} more free space. Free up space and try again.`,
    ));
    return;
  }

  const fileName = civitaiLocalFileName(result);
  const displayName = civitaiDisplayName(result);
  const metadata: SdCppFileDownloadMetadata = {
    imageDownloadType: 'sdcpp-file',
    imageModelName: displayName,
    imageModelDescription: `From Civitai (${result.baseModel})`,
    imageModelSize: result.file.sizeBytes,
    imageModelFileName: fileName,
    civitaiType: result.type,
    civitaiBaseModel: result.baseModel,
  };
  const metadataJson = JSON.stringify(metadata);
  const placeholderId = `queued:${modelKey}`;
  if (existing) {
    useDownloadStore.getState().retryEntry(modelKey, placeholderId);
  } else {
    useDownloadStore.getState().add({
      modelKey,
      downloadId: placeholderId,
      modelId: `image:${modelId}`,
      fileName,
      quantization: '',
      modelType: 'image',
      status: 'pending',
      bytesDownloaded: 0,
      totalBytes: result.file.sizeBytes,
      combinedTotalBytes: result.file.sizeBytes,
      progress: 0,
      createdAt: Date.now(),
      metadataJson,
    });
  }

  try {
    const info = await backgroundDownloadService.startDownload({
      url: withCivitaiDownloadToken(result.file.downloadUrl, apiKey),
      fileName,
      modelId: `image:${modelId}`,
      modelKey,
      modelType: 'image',
      totalBytes: result.file.sizeBytes,
      sha256: result.file.sha256?.toLowerCase(),
      metadataJson,
    });
    useDownloadStore.getState().retryEntry(modelKey, info.downloadId);
    const unsubComplete = backgroundDownloadService.onComplete(info.downloadId, async () => {
      unsubComplete(); unsubError();
      // Claimed before the first await: the Models screen's processing-resume effect skips it.
      if (!claimImageFinalization(modelId)) return;
      try {
        await finalizeSdCppFileDownload({ downloadId: info.downloadId, modelId, fileName, displayName, deps });
      } catch (error: any) {
        logger.error(`[Civitai] finalizing ${modelId} failed:`, error?.message);
        failRow(info.downloadId, deps, error?.message || 'Failed to install the downloaded model');
      } finally {
        releaseImageFinalization(modelId);
      }
    });
    const unsubError = backgroundDownloadService.onError(info.downloadId, (ev) => {
      unsubComplete(); unsubError();
      // useDownloads already set the row 'failed' (retry / remove stay available).
      deps.setAlertState(showAlert('Download Failed', getUserFacingDownloadMessage(ev.reason)));
    });
    backgroundDownloadService.startProgressPolling();
  } catch (error: any) {
    if (error?.cancelled === true) {
      useDownloadStore.getState().remove(modelKey);
      return;
    }
    failRow(placeholderId, deps, error?.message);
  }
}
