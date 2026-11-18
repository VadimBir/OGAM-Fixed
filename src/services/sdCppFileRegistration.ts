/**
 * Register a stable-diffusion.cpp file (.safetensors / .ckpt) that already sits in its own folder
 * under image_models/: an SD checkpoint becomes an image model, an SD LoRA joins the LoRA list,
 * anything else is removed with an explanation. Core-layer owner shared by the file import
 * (screens/ModelsScreen/importSdCheckpoint.ts) and Civitai downloads (civitaiDownload.ts).
 */
import RNFS from 'react-native-fs';
import { showAlert, type AlertState } from '../utils/alertState';
import { modelManager } from './modelManager';
import { sdCppGeneratorService } from './sdCppGenerator';
import {
  loraCompatibility,
  sdFamilyLabel,
  type LoraBase,
  type SafetensorsClassification,
} from './sdCppModelDetection';
import { useAppStore } from '../stores';
import type { ImageLora, ONNXImageModel } from '../types';
import logger from '../utils/logger';

export interface SdRegisterDeps {
  addDownloadedImageModel: (model: ONNXImageModel) => void;
  activeImageModelId: string | null;
  setActiveImageModelId: (id: string) => void;
  setAlertState: (s: AlertState) => void;
}

/** LoRA files live outside image_models/ so the orphan scan never offers them for deletion. */
export const IMAGE_LORAS_DIR = `${RNFS.DocumentDirectoryPath}/image_loras`;

export const safeSdFileId = (name: string) => name.replaceAll(/[^a-zA-Z0-9_-]/g, '_');

/** Room kept free beyond the file itself (the OS and the app need headroom to keep running). */
const FREE_SPACE_HEADROOM_BYTES = 256 * 1024 * 1024;

export const formatGB = (bytes: number) => `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;

/**
 * Bytes still missing to copy a file of `size` (plus headroom), or null when it fits. An unknown
 * size, or a device that cannot report free space, is not blocked - the copy itself still fails
 * cleanly (copyFileWithProgress removes its partial file).
 */
export async function missingFreeSpace(size: number | null): Promise<number | null> {
  if (!size || size <= 0) return null;
  try {
    const { freeSpace } = await RNFS.getFSInfo();
    const needed = size + FREE_SPACE_HEADROOM_BYTES;
    return freeSpace >= needed ? null : needed - freeSpace;
  } catch (error) {
    logger.warn('[SDCPP-IMPORT] free space unavailable, copying without the check:', error);
    return null;
  }
}

function rejectionTitle(kind: SafetensorsClassification['kind']): string {
  if (kind === 'video') return 'Video models are not supported';
  if (kind === 'sd-lora-unsupported') return 'LoRA type not supported';
  return 'Not a Stable Diffusion checkpoint';
}

function rejectionHint(kind: SafetensorsClassification['kind'], reason: string): string {
  if (kind === 'video') return 'This app generates still images. Import a Stable Diffusion image checkpoint (SD 1.x, SD 2.x, SDXL) or an SD LoRA.';
  if (kind === 'sd-lora-unsupported') {
    return 'stable-diffusion.cpp applies LoRA / LoCon (lora_down/lora_up), diffusers/PEFT (lora_A/lora_B), LoHa and LoKr files. ' +
      'This LoRA type (for example OFT or IA3) would load and change nothing, so it was not added.';
  }
  if (kind === 'llm' && /LoRA/.test(reason)) {
    return 'This is a text-model (LLM) LoRA, not an image LoRA. Text-model LoRAs run on the GGUF engine as GGUF adapters ' +
      '(convert it with llama.cpp convert_lora_to_gguf.py) and need the matching base text model.';
  }
  if (kind === 'llm') return 'This file holds text-model (LLM) weights. Text models run on the GGUF engine — import a .gguf build of it.';
  return 'Import a full single-file Stable Diffusion checkpoint (UNet + text encoder + VAE), or an SD LoRA.';
}

async function classify(path: string, fileName: string): Promise<SafetensorsClassification> {
  // .ckpt is a pickled torch zip with no cheap key index; it is (practically) only used for SD.
  if (fileName.toLowerCase().endsWith('.ckpt')) {
    return { kind: 'sd-checkpoint', family: 'unknown', reason: '.ckpt checkpoint' };
  }
  return sdCppGeneratorService.classifySafetensors(path);
}

async function registerLora(
  stagedPath: string,
  stagingDir: string,
  fileName: string,
  enabled: boolean,
): Promise<ImageLora> {
  if (!(await RNFS.exists(IMAGE_LORAS_DIR))) await RNFS.mkdir(IMAGE_LORAS_DIR);
  const dest = `${IMAGE_LORAS_DIR}/${fileName}`;
  if (await RNFS.exists(dest)) throw new Error(`A LoRA named "${fileName}" already exists`);
  await RNFS.moveFile(stagedPath, dest);
  await RNFS.unlink(stagingDir).catch(() => {});
  // A removed LoRA's path can be reused by a new file: never keep its cached base model.
  sdCppGeneratorService.forgetFile(dest);
  // Size feeds only the RAM estimate; a failed stat must not fail the import.
  const stat = await RNFS.stat(dest).catch(() => null);
  const sizeBytes = Number(stat?.size) || undefined;
  const lora: ImageLora = {
    id: `lora_${safeSdFileId(fileName)}_${Date.now()}`,
    name: fileName.replace(/\.safetensors$/i, ''),
    path: dest,
    weight: 1,
    enabled,
    sizeBytes,
  };
  const { settings, updateSettings } = useAppStore.getState();
  updateSettings({ imageLoras: [...(settings.imageLoras ?? []), lora] });
  return lora;
}

/** The "LoRA added" text: its base model and whether it fits the selected checkpoint. */
function loraAddedMessage(lora: ImageLora, base: LoraBase, activeFamily: string | undefined): string {
  const label = sdFamilyLabel(base);
  const fit = loraCompatibility(base, activeFamily);
  if (fit === 'mismatch') {
    return `${lora.name} is a ${label} LoRA. The selected checkpoint is ${sdFamilyLabel(activeFamily)}, so the LoRA was added disabled. ` +
      `Enable it in Image Generation settings with a ${label} checkpoint.`;
  }
  const head = `${lora.name} (${label}) is enabled for stable-diffusion.cpp checkpoints (weight 1.0). Adjust or disable it in Image Generation settings.`;
  return base === 'unknown'
    ? `${head} Its base model (SD 1.x / SDXL / ...) could not be read from the file: use it only with the checkpoint family it was trained for.`
    : head;
}

export type SdFileOutcome =
  | { kind: 'checkpoint'; model: ONNXImageModel }
  | { kind: 'lora'; lora: ImageLora }
  | { kind: 'rejected'; reason: string };

/**
 * Classify a .safetensors / .ckpt that already sits at `path` inside its own `modelDir` (copied by
 * an import, or moved there by a finished download) and register it with NO conversion: an SD
 * checkpoint becomes an sd.cpp image model (selected when none is), an SD LoRA joins the LoRA list,
 * anything else is removed with an explanation. Shared by the file import and Civitai downloads so
 * both register identically.
 */
export async function registerSdFileInPlace(
  opts: {
    path: string;
    modelDir: string;
    fileName: string;
    modelId: string;
    verb: 'imported' | 'downloaded';
    /** Display name; default = file name without extension, `_`/`-` as spaces. */
    displayName?: string;
    /** Prefix for the model description, e.g. "From Civitai". */
    source?: string;
  },
  deps: SdRegisterDeps,
): Promise<SdFileOutcome> {
  const { path, modelDir, fileName, modelId, verb } = opts;
  const result = await classify(path, fileName);
  logger.log(`[SDCPP-IMPORT] ${fileName}: ${result.kind}/${result.family} (${result.reason})`);

  if (result.kind === 'sd-lora') {
    const base = result.loraBase ?? 'unknown';
    const state = useAppStore.getState();
    const activeFamily = (state.downloadedImageModels ?? []).find(m => m.id === deps.activeImageModelId)?.sdFamily;
    const lora = await registerLora(path, modelDir, fileName, loraCompatibility(base, activeFamily) !== 'mismatch');
    const note = result.loraWarning ? `\n\n${result.loraWarning}` : '';
    deps.setAlertState(showAlert('LoRA added', loraAddedMessage(lora, base, activeFamily) + note));
    return { kind: 'lora', lora };
  }
  if (result.kind !== 'sd-checkpoint') {
    await RNFS.unlink(modelDir).catch(() => {});
    deps.setAlertState(showAlert(
      rejectionTitle(result.kind),
      `${result.reason}.\n\n${rejectionHint(result.kind, result.reason)}`,
    ));
    return { kind: 'rejected', reason: result.reason };
  }
  const size = (await RNFS.stat(path)).size;
  const engineText = `Stable Diffusion checkpoint (${result.family}) — runs directly on stable-diffusion.cpp`;
  const imageModel: ONNXImageModel = {
    id: modelId,
    name: opts.displayName?.trim() || fileName.replace(/\.(safetensors|ckpt)$/i, '').replaceAll(/[_-]/g, ' '),
    description: opts.source ? `${opts.source}. ${engineText}` : engineText,
    modelPath: path,
    downloadedAt: new Date().toISOString(),
    size: Number(size) || 0,
    backend: 'sdcpp',
    sdFamily: result.family,
  };
  await modelManager.addDownloadedImageModel(imageModel);
  deps.addDownloadedImageModel(imageModel);
  if (!deps.activeImageModelId) deps.setActiveImageModelId(imageModel.id);
  // GPU-only build: libsdcpp_jni.so is built with SDCPP_GPU_ONLY (CMakeLists.txt) and loads on the
  // Vulkan "gpu" backend, refusing with ERR_NO_GPU when no GPU exists (sdcpp_jni.cpp load_ctx).
  deps.setAlertState(showAlert('Success', `${imageModel.name} ${verb}. It runs on the stable-diffusion.cpp engine (GPU).`));
  return { kind: 'checkpoint', model: imageModel };
}
