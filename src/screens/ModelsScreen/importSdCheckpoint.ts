import RNFS from 'react-native-fs';
import { showAlert, AlertState } from '../../components/CustomAlert';
import { modelManager } from '../../services';
import { copyFileWithProgress } from '../../services/modelManager/copyFile';
import { sdCppGeneratorService } from '../../services/sdCppGenerator';
import type { SafetensorsClassification } from '../../services/sdCppModelDetection';
import { useAppStore } from '../../stores';
import { ImageLora, ONNXImageModel } from '../../types';
import logger from '../../utils/logger';

export type SdImportDeps = {
  addDownloadedImageModel: (model: ONNXImageModel) => void;
  activeImageModelId: string | null;
  setActiveImageModelId: (id: string | null) => void;
  setImportProgress: (p: { fraction: number; fileName: string } | null) => void;
  setAlertState: (s: AlertState) => void;
};

/** LoRA files live outside image_models/ so the orphan scan never offers them for deletion. */
export const IMAGE_LORAS_DIR = `${RNFS.DocumentDirectoryPath}/image_loras`;

const safeId = (name: string) => name.replaceAll(/[^a-zA-Z0-9_-]/g, '_');

/** Room kept free beyond the file itself (the OS and the app need headroom to keep running). */
const FREE_SPACE_HEADROOM_BYTES = 256 * 1024 * 1024;

const formatGB = (bytes: number) => `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;

/**
 * Bytes still missing to copy a file of `size` (plus headroom), or null when it fits. An unknown
 * size, or a device that cannot report free space, is not blocked - the copy itself still fails
 * cleanly (copyFileWithProgress removes its partial file).
 */
async function missingFreeSpace(size: number | null): Promise<number | null> {
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

async function classify(path: string, fileName: string): Promise<SafetensorsClassification> {
  // .ckpt is a pickled torch zip with no cheap key index; it is (practically) only used for SD.
  if (fileName.toLowerCase().endsWith('.ckpt')) {
    return { kind: 'sd-checkpoint', family: 'unknown', reason: '.ckpt checkpoint' };
  }
  return sdCppGeneratorService.classifySafetensors(path);
}

async function registerLora(stagedPath: string, stagingDir: string, fileName: string): Promise<ImageLora> {
  if (!(await RNFS.exists(IMAGE_LORAS_DIR))) await RNFS.mkdir(IMAGE_LORAS_DIR);
  const dest = `${IMAGE_LORAS_DIR}/${fileName}`;
  if (await RNFS.exists(dest)) throw new Error(`A LoRA named "${fileName}" already exists`);
  await RNFS.moveFile(stagedPath, dest);
  await RNFS.unlink(stagingDir).catch(() => {});
  const lora: ImageLora = {
    id: `lora_${safeId(fileName)}_${Date.now()}`,
    name: fileName.replace(/\.safetensors$/i, ''),
    path: dest,
    weight: 1,
    enabled: true,
  };
  const { settings, updateSettings } = useAppStore.getState();
  updateSettings({ imageLoras: [...(settings.imageLoras ?? []), lora] });
  return lora;
}

/**
 * Import one raw .safetensors / .ckpt file with NO conversion. The header decides what it is:
 * an SD checkpoint becomes an image model on the stable-diffusion.cpp engine, an SD LoRA joins
 * the LoRA list, and LLM weights are refused (text models run as GGUF).
 */
export async function importSdCheckpointOrLora(
  file: { uri: string; name: string; size?: number | null },
  deps: SdImportDeps,
): Promise<void> {
  const { addDownloadedImageModel, activeImageModelId, setActiveImageModelId, setImportProgress, setAlertState } = deps;
  if (!sdCppGeneratorService.isAvailable()) {
    setAlertState(showAlert('Not Supported', 'Raw Stable Diffusion checkpoints (.safetensors/.ckpt) run on Android only.'));
    return;
  }
  const fileName = file.name;
  const modelId = `sdcpp_${safeId(fileName.replace(/\.(safetensors|ckpt)$/i, ''))}_${Date.now()}`;
  const imageModelsDir = modelManager.getImageModelsDirectory();
  const modelDir = `${imageModelsDir}/${modelId}`;
  const dest = `${modelDir}/${fileName}`;
  // A checkpoint is 2-7 GB. Copying it into a full disk fails late (after minutes) or fills the
  // device so other apps / the OS start failing. Check before writing a byte.
  const shortBy = await missingFreeSpace(file.size ?? null);
  if (shortBy !== null) {
    setAlertState(showAlert(
      'Not Enough Storage',
      `${fileName} needs about ${formatGB(shortBy)} more free space. Free up space and try again.`,
    ));
    return;
  }
  if (!(await RNFS.exists(imageModelsDir))) await RNFS.mkdir(imageModelsDir);
  await RNFS.mkdir(modelDir);
  try {
    const source = file.uri.startsWith('content://') ? file.uri : decodeURIComponent(file.uri);
    await copyFileWithProgress(source, dest, {
      knownTotalBytes: file.size ?? null,
      onProgress: fraction => setImportProgress({ fraction: fraction * 0.9, fileName }),
    });
    const result = await classify(dest, fileName);
    logger.log(`[SDCPP-IMPORT] ${fileName}: ${result.kind}/${result.family} (${result.reason})`);

    if (result.kind === 'sd-lora') {
      const lora = await registerLora(dest, modelDir, fileName);
      setImportProgress({ fraction: 1, fileName });
      setAlertState(showAlert('LoRA added', `${lora.name} is enabled for Stable Diffusion checkpoints (weight 1.0). Adjust or disable it in Image Generation settings.`));
      return;
    }
    if (result.kind !== 'sd-checkpoint') {
      await RNFS.unlink(modelDir).catch(() => {});
      const hint = result.kind === 'llm'
        ? 'This file holds text-model (LLM) weights. Text models run on the GGUF engine — import a .gguf build of it.'
        : 'Import a full single-file Stable Diffusion checkpoint (UNet + text encoder + VAE), or an SD LoRA.';
      setAlertState(showAlert('Not a Stable Diffusion checkpoint', `${result.reason}.\n\n${hint}`));
      return;
    }
    const size = (await RNFS.stat(dest)).size;
    const imageModel: ONNXImageModel = {
      id: modelId,
      name: fileName.replace(/\.(safetensors|ckpt)$/i, '').replaceAll(/[_-]/g, ' '),
      description: `Stable Diffusion checkpoint (${result.family}) — runs directly on stable-diffusion.cpp`,
      modelPath: dest,
      downloadedAt: new Date().toISOString(),
      size: Number(size) || 0,
      backend: 'sdcpp',
      sdFamily: result.family,
    };
    await modelManager.addDownloadedImageModel(imageModel);
    addDownloadedImageModel(imageModel);
    if (!activeImageModelId) setActiveImageModelId(imageModel.id);
    setImportProgress({ fraction: 1, fileName });
    setAlertState(showAlert('Success', `${imageModel.name} imported. It runs on the stable-diffusion.cpp engine (CPU).`));
  } catch (e) {
    await RNFS.unlink(modelDir).catch(() => {});
    throw e;
  }
}
