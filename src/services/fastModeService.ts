/**
 * Fast mode actions: one tap downloads the few-step LoRA (or TAESD) with the app's background
 * downloader (WorkManager row in useDownloadStore: progress, cancel, Download Manager), registers a
 * LoRA through the same path as a Civitai LoRA (civitaiDownload.finalizeSdCppFileDownload ->
 * sdCppFileRegistration.registerSdFileInPlace, so it joins settings.imageLoras), then applies the
 * preset (fastModePresets).
 *
 * LoRA rows use imageDownloadType 'sdcpp-file', so a download that finishes after an app restart is
 * still registered by imageDownloadResume (the preset is then applied on the next tap). TAESD rows
 * use 'fastmode-taesd' (not an SD checkpoint or LoRA: registration would reject it); a TAESD row
 * that finished while the app was closed is finalized by recoverFastModeDownloads.
 */
import RNFS from 'react-native-fs';
import { backgroundDownloadService } from '.';
import { sdCppGeneratorService } from './sdCppGenerator';
import { finalizeSdCppFileDownload, type SdCppDownloadDeps, type SdCppFileDownloadMetadata } from './civitaiDownload';
import { missingFreeSpace, formatGB, safeSdFileId } from './sdCppFileRegistration';
import { claimImageFinalization, releaseImageFinalization } from './imageFinalizationClaims';
import {
  FAST_LORA_PRESETS,
  TAESD_DIR,
  TAESD_FILES,
  fastModeOffUpdate,
  npuFastModeUpdate,
  sdCppFastModeUpdate,
  taesdPathFor,
  type FastLoraFamily,
  type FastLoraPreset,
  type FastSteps,
  type NpuFastPreset,
  type TaesdFamily,
} from './fastModePresets';
import { useAppStore } from '../stores';
import { useDownloadStore, isActiveStatus, type DownloadEntry } from '../stores/downloadStore';
import { makeImageModelKey } from '../utils/modelKey';
import { showAlert } from '../utils/alertState';
import { getUserFacingDownloadMessage } from '../utils/downloadErrors';
import type { ImageLora } from '../types';
import logger from '../utils/logger';

export type FastModeDeps = SdCppDownloadDeps;

const TAESD_DOWNLOAD_TYPE = 'fastmode-taesd';

interface TaesdDownloadMetadata {
  imageDownloadType: typeof TAESD_DOWNLOAD_TYPE;
  imageModelName: string;
  imageModelFileName: string;
  imageModelSize: number;
  taesdFamily: TaesdFamily;
}

/** Download-row id of a Fast mode LoRA (image:<id> in useDownloadStore). */
export const fastLoraDownloadId = (preset: Pick<FastLoraPreset, 'fileName'>) =>
  `fastmode_${safeSdFileId(preset.fileName.replace(/\.safetensors$/i, ''))}`;

export const taesdDownloadId = (family: TaesdFamily) => `fastmode_taesd_${family}`;

/** The registered LoRA of a preset: same file name in image_loras/ (registerLora keeps the name). */
export function findPresetLora(preset: Pick<FastLoraPreset, 'fileName'>, loras: ImageLora[] | undefined): ImageLora | undefined {
  const suffix = `/${preset.fileName}`;
  return (loras ?? []).find(l => l.path.endsWith(suffix));
}

/** The live download row of a Fast mode file, if any (for progress / cancel in the UI). */
export function fastDownloadEntry(downloadModelId: string): DownloadEntry | undefined {
  return useDownloadStore.getState().downloads[makeImageModelKey(downloadModelId)];
}

function failRow(downloadId: string, deps: FastModeDeps, message?: string) {
  deps.setAlertState(showAlert('Download Failed', getUserFacingDownloadMessage(message)));
  useDownloadStore.getState().setStatus(downloadId, 'failed', { message: message || 'Download failed' });
}

interface FastDownloadSpec {
  /** image:<modelId> row; also the staging folder name under image_models/. */
  modelId: string;
  fileName: string;
  url: string;
  sizeBytes: number;
  sha256: string;
  metadata: SdCppFileDownloadMetadata | TaesdDownloadMetadata;
  /** Runs once the bytes are complete (claimed against the Models screen resume). Throws on failure. */
  finalize: (downloadId: string) => Promise<void>;
}

/**
 * Start (or retry) a background download, as civitaiDownload.startCivitaiPhoneDownload does:
 * free-space check, one store row per file, live completion -> finalize. Resolves 'started',
 * 'busy' (already downloading) or 'blocked' (not available / no space; an alert was shown).
 */
async function startFastDownload(spec: FastDownloadSpec, deps: FastModeDeps): Promise<'started' | 'busy' | 'blocked'> {
  if (!backgroundDownloadService.isAvailable()) {
    deps.setAlertState(showAlert('Not Supported', 'Fast mode downloads need the Android background downloader.'));
    return 'blocked';
  }
  const { modelId, fileName } = spec;
  const modelKey = makeImageModelKey(modelId);
  const existing = useDownloadStore.getState().downloads[modelKey];
  if (existing && (isActiveStatus(existing.status) || existing.status === 'paused')) return 'busy';

  const shortBy = await missingFreeSpace(spec.sizeBytes);
  if (shortBy !== null) {
    deps.setAlertState(showAlert('Not Enough Storage', `${fileName} needs about ${formatGB(shortBy)} more free space. Free up space and try again.`));
    return 'blocked';
  }

  const metadataJson = JSON.stringify(spec.metadata);
  const placeholderId = `queued:${modelKey}`;
  if (existing) {
    if (existing.downloadId && existing.downloadId !== placeholderId) {
      await backgroundDownloadService.cancelDownload(existing.downloadId).catch(() => {});
    }
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
      totalBytes: spec.sizeBytes,
      combinedTotalBytes: spec.sizeBytes,
      progress: 0,
      createdAt: Date.now(),
      metadataJson,
    });
  }

  try {
    const info = await backgroundDownloadService.startDownload({
      url: spec.url,
      fileName,
      modelId: `image:${modelId}`,
      modelKey,
      modelType: 'image',
      totalBytes: spec.sizeBytes,
      sha256: spec.sha256,
      metadataJson,
    });
    useDownloadStore.getState().retryEntry(modelKey, info.downloadId);
    const unsubComplete = backgroundDownloadService.onComplete(info.downloadId, async () => {
      unsubComplete(); unsubError();
      // Claimed before the first await: the Models screen's processing-resume effect skips it.
      if (!claimImageFinalization(modelId)) return;
      try {
        await spec.finalize(info.downloadId);
      } catch (error: any) {
        logger.error(`[FastMode] finalizing ${modelId} failed:`, error?.message);
        failRow(info.downloadId, deps, error?.message || 'Failed to install the download');
      } finally {
        releaseImageFinalization(modelId);
      }
    });
    const unsubError = backgroundDownloadService.onError(info.downloadId, ev => {
      unsubComplete(); unsubError();
      if (ev.reasonCode === 'user_cancelled') return;
      deps.setAlertState(showAlert('Download Failed', getUserFacingDownloadMessage(ev.reason, ev.reasonCode)));
    });
    backgroundDownloadService.startProgressPolling();
    return 'started';
  } catch (error: any) {
    if (error?.cancelled === true) {
      useDownloadStore.getState().remove(modelKey);
      return 'blocked';
    }
    failRow(placeholderId, deps, error?.message);
    return 'blocked';
  }
}

/** Cancel a running Fast mode download (same as the Civitai row cancel). */
export async function cancelFastDownload(downloadModelId: string): Promise<void> {
  const modelKey = makeImageModelKey(downloadModelId);
  const entry = useDownloadStore.getState().downloads[modelKey];
  if (!entry) return;
  useDownloadStore.getState().remove(modelKey);
  if (entry.downloadId) await backgroundDownloadService.cancelDownload(entry.downloadId).catch(() => {});
}

// ---- GPU (stable-diffusion.cpp) -----------------------------------------------------------------

/** The GPU choice the user tapped last ('sd1:2'); a download that lands later applies its preset only
 *  while it is still the wanted one (Off / another choice in between = the LoRA is just registered). */
let wantedSdCppChoice: string | null = null;
const choiceKey = (family: FastLoraFamily, steps: FastSteps) => `${family}:${steps}`;

function applySdCppPreset(preset: FastLoraPreset, loraPath: string): void {
  const { settings, updateSettings } = useAppStore.getState();
  updateSettings(sdCppFastModeUpdate(settings, preset, loraPath));
  logger.log(`[FastMode] sdcpp ${preset.family} ${preset.steps} step(s): ${preset.fileName} sampler=${preset.sampler} scheduler=${preset.scheduler} cfg=${preset.cfg}`);
}

function readyMessage(preset: FastLoraPreset): string {
  return `${preset.steps} step${preset.steps === 1 ? '' : 's'}, guidance ${preset.cfg}, sampler ${preset.sampler}, ` +
    `schedule ${preset.scheduler}, LoRA ${preset.fileName.replace(/\.safetensors$/i, '')} on. Turn Fast mode off to restore your settings.`;
}

/**
 * Turn Fast mode on for a stable-diffusion.cpp checkpoint: apply at once when the preset LoRA is
 * already on the phone, else download it, register it and apply when it lands.
 */
export async function selectSdCppFastMode(
  family: FastLoraFamily,
  steps: FastSteps,
  deps: FastModeDeps,
): Promise<'applied' | 'started' | 'busy' | 'blocked'> {
  const preset = FAST_LORA_PRESETS[family][steps];
  wantedSdCppChoice = choiceKey(family, steps);
  const { settings, updateSettings } = useAppStore.getState();
  const known = findPresetLora(preset, settings.imageLoras);
  if (known) {
    if (await RNFS.exists(known.path)) {
      applySdCppPreset(preset, known.path);
      return 'applied';
    }
    // Listed but the file is gone: drop the stale entry so the fresh download registers cleanly.
    updateSettings({ imageLoras: (settings.imageLoras ?? []).filter(l => l.id !== known.id) });
  }
  if (!sdCppGeneratorService.isAvailable()) {
    deps.setAlertState(showAlert('Not Supported', 'Fast mode LoRAs run on the stable-diffusion.cpp engine (Android).'));
    return 'blocked';
  }
  const modelId = fastLoraDownloadId(preset);
  const displayName = preset.fileName.replace(/\.safetensors$/i, '');
  const metadata: SdCppFileDownloadMetadata = {
    imageDownloadType: 'sdcpp-file',
    imageModelName: displayName,
    imageModelDescription: `Fast mode LoRA (${preset.repo})`,
    imageModelSize: preset.sizeBytes,
    imageModelFileName: preset.fileName,
    civitaiType: 'LORA',
    civitaiBaseModel: family === 'sdxl' ? 'SDXL 1.0' : 'SD 1.5',
  };
  return startFastDownload({
    modelId,
    fileName: preset.fileName,
    url: preset.url,
    sizeBytes: preset.sizeBytes,
    sha256: preset.sha256,
    metadata,
    finalize: async downloadId => {
      await finalizeSdCppFileDownload({ downloadId, modelId, fileName: preset.fileName, displayName, deps });
      const lora = findPresetLora(preset, useAppStore.getState().settings.imageLoras);
      // Not registered = registration rejected the file and already explained why in an alert.
      if (!lora) {
        logger.warn(`[FastMode] ${preset.fileName} was not registered as a LoRA; preset not applied`);
        return;
      }
      if (wantedSdCppChoice !== choiceKey(family, steps)) {
        logger.log(`[FastMode] ${preset.fileName} registered; another choice was made meanwhile, preset not applied`);
        return;
      }
      applySdCppPreset(preset, lora.path);
      deps.setAlertState(showAlert('Fast mode on', readyMessage(preset)));
    },
  }, deps);
}

// ---- NPU (LocalDream) ---------------------------------------------------------------------------

export function selectNpuFastMode(preset: NpuFastPreset, steps: FastSteps): void {
  wantedSdCppChoice = null;
  const { settings, updateSettings } = useAppStore.getState();
  updateSettings(npuFastModeUpdate(settings, preset, steps));
  logger.log(`[FastMode] npu ${preset.kind} ${steps} step(s) sampler=${preset.sampler} cfg=${preset.cfg}`);
}

/** Off: the steps / guidance / sampler / schedule from before Fast mode, preset LoRA disabled. */
export function turnOffFastMode(): void {
  wantedSdCppChoice = null;
  const { settings, updateSettings } = useAppStore.getState();
  updateSettings(fastModeOffUpdate(settings));
}

// ---- TAESD --------------------------------------------------------------------------------------

async function finalizeTaesd(downloadId: string, family: TaesdFamily): Promise<void> {
  const dest = taesdPathFor(family);
  const modelKey = makeImageModelKey(taesdDownloadId(family));
  useDownloadStore.getState().setProcessing(downloadId);
  if (!(await RNFS.exists(TAESD_DIR))) await RNFS.mkdir(TAESD_DIR);
  if (!(await RNFS.exists(dest))) {
    try {
      await backgroundDownloadService.moveCompletedDownload(downloadId, dest);
    } catch (error) {
      await RNFS.unlink(dest).catch(() => {});
      throw error;
    }
  }
  useDownloadStore.getState().remove(modelKey);
  // Applies at the next model load (sdCppLoadConfigFromSettings -> loadConfigDiffers -> reload).
  useAppStore.getState().updateSettings({ imageSdTaesd: true });
  logger.log(`[FastMode] TAESD ${family} ready at ${dest}`);
}

/**
 * Fast decode switch. On: uses the family's TAESD when it is on the phone, else downloads it
 * (9.8 MB) and turns the switch on when it lands. Off: back to the full VAE at the next load.
 */
export async function setTaesdEnabled(
  enabled: boolean,
  family: TaesdFamily,
  deps: FastModeDeps,
): Promise<'applied' | 'started' | 'busy' | 'blocked'> {
  const { updateSettings } = useAppStore.getState();
  if (!enabled) {
    // A TAESD download still running would turn the switch back on when it lands.
    await cancelFastDownload(taesdDownloadId(family));
    updateSettings({ imageSdTaesd: false });
    return 'applied';
  }
  if (await RNFS.exists(taesdPathFor(family))) {
    updateSettings({ imageSdTaesd: true });
    return 'applied';
  }
  const file = TAESD_FILES[family];
  const metadata: TaesdDownloadMetadata = {
    imageDownloadType: TAESD_DOWNLOAD_TYPE,
    imageModelName: family === 'sdxl' ? 'TAESDXL (fast decode)' : 'TAESD (fast decode)',
    imageModelFileName: file.localFileName,
    imageModelSize: file.sizeBytes,
    taesdFamily: family,
  };
  return startFastDownload({
    modelId: taesdDownloadId(family),
    fileName: file.localFileName,
    url: file.url,
    sizeBytes: file.sizeBytes,
    sha256: file.sha256,
    metadata,
    finalize: downloadId => finalizeTaesd(downloadId, family),
  }, deps);
}

/**
 * A TAESD download that completed while the app was closed: imageDownloadResume has no branch for
 * it, so it is finalized here (called when the Fast mode section mounts).
 */
export async function recoverFastModeDownloads(deps: FastModeDeps): Promise<void> {
  const rows = Object.values(useDownloadStore.getState().downloads);
  for (const row of rows) {
    if (row.status !== 'completed' && row.status !== 'processing') continue;
    let meta: Partial<TaesdDownloadMetadata> | null = null;
    try { meta = row.metadataJson ? JSON.parse(row.metadataJson) : null; } catch { meta = null; }
    if (meta?.imageDownloadType !== TAESD_DOWNLOAD_TYPE || !meta.taesdFamily) continue;
    const modelId = taesdDownloadId(meta.taesdFamily);
    if (!claimImageFinalization(modelId)) continue;
    try {
      await finalizeTaesd(row.downloadId, meta.taesdFamily);
    } catch (error: any) {
      logger.error(`[FastMode] recovering ${modelId} failed:`, error?.message);
      failRow(row.downloadId, deps, error?.message || 'Failed to install the download');
    } finally {
      releaseImageFinalization(modelId);
    }
  }
}
