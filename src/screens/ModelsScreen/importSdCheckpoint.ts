import RNFS from 'react-native-fs';
import { showAlert, AlertState } from '../../components/CustomAlert';
import { modelManager } from '../../services';
import { copyFileWithProgress } from '../../services/modelManager/copyFile';
import { sdCppGeneratorService } from '../../services/sdCppGenerator';
import {
  formatGB,
  missingFreeSpace,
  registerSdFileInPlace,
  safeSdFileId,
} from '../../services/sdCppFileRegistration';
import { ONNXImageModel } from '../../types';

// Re-exported for existing importers; the owner is services/sdCppFileRegistration.ts.
export { IMAGE_LORAS_DIR, registerSdFileInPlace } from '../../services/sdCppFileRegistration';
export type { SdFileOutcome } from '../../services/sdCppFileRegistration';

export type SdImportDeps = {
  addDownloadedImageModel: (model: ONNXImageModel) => void;
  activeImageModelId: string | null;
  setActiveImageModelId: (id: string | null) => void;
  setImportProgress: (p: { fraction: number; fileName: string } | null) => void;
  setAlertState: (s: AlertState) => void;
};

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
  const modelId = `sdcpp_${safeSdFileId(fileName.replace(/\.(safetensors|ckpt)$/i, ''))}_${Date.now()}`;
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
    const outcome = await registerSdFileInPlace(
      { path: dest, modelDir, fileName, modelId, verb: 'imported' },
      { addDownloadedImageModel, activeImageModelId, setActiveImageModelId, setAlertState },
    );
    if (outcome.kind !== 'rejected') setImportProgress({ fraction: 1, fileName });
  } catch (e) {
    await RNFS.unlink(modelDir).catch(() => {});
    throw e;
  }
}
