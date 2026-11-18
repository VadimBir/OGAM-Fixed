import RNFS from 'react-native-fs';
import { statFile } from '../../utils/fileStat';
import {
  DownloadedModel,
  LiteRTDownloadedModel,
  LlamaDownloadedModel,
  ModelEngine,
  ModelFile,
} from '../../types';
import { buildDownloadedModel, persistDownloadedModel } from './storage';
import { copyFileWithProgress } from './copyFile';
import { isLiteRTFileName } from '../../utils/modelHelpers';
import { parseQuantization, stripQuantization } from '../../utils/quantization';
import logger from '../../utils/logger';

export interface ImportLocalModelOpts {
  sourceUri: string;
  fileName: string;
  modelsDir: string;
  sourceSize?: number | null;
  engine?: ModelEngine;
  liteRTVision?: boolean;
  onProgress?: (progress: { fraction: number; fileName: string }) => void;
  mmProjSourceUri?: string;
  mmProjFileName?: string;
  mmProjSourceSize?: number | null;
}


/**
 * Bringing a file the USER picked into the library.
 *
 * A different job from scan.ts, which adopts files already sitting in the models dir after a crash or a
 * reinstall. Split out when scan.ts crossed the size cap: one file was doing recovery, projector linking,
 * image-zip reconciliation AND local import.
 */
function resolveUri(uri: string): string {
  // Android content:// URIs are passed directly to RNFS.copyFile — no cache copy needed.
  // iOS file:// URIs need decoding (%20 → space) so RNFS can find the file on disk.
  if (uri.startsWith('content://')) {
    return uri;
  }
  return decodeURIComponent(uri);
}


export async function importLocalModel(opts: ImportLocalModelOpts): Promise<DownloadedModel> { // NOSONAR
  const { sourceUri, fileName, modelsDir, sourceSize, engine: _engine, liteRTVision, onProgress, mmProjSourceUri, mmProjFileName, mmProjSourceSize } = opts;

  const isLitert = isLiteRTFileName(fileName);
  if (!fileName.toLowerCase().endsWith('.gguf') && !isLitert) {
    throw new Error('Only .gguf and .litertlm files can be imported');
  }

  const resolvedSource = resolveUri(sourceUri);
  const resolvedMmProjSource = mmProjSourceUri ? resolveUri(mmProjSourceUri) : undefined;

  const destPath = `${modelsDir}/${fileName}`;
  const destExists = await RNFS.exists(destPath);
  if (destExists) throw new Error(`A model file named "${fileName}" already exists`);
  if (mmProjFileName && await RNFS.exists(`${modelsDir}/${mmProjFileName}`)) {
    throw new Error(`A file named "${mmProjFileName}" already exists`);
  }

  // Copy main model: progress 0→0.5 when mmproj present, 0→1 otherwise
  const mainProgressScale = mmProjFileName ? 0.5 : 1;
  await copyFileWithProgress(resolvedSource, destPath, {
    knownTotalBytes: sourceSize ?? null,
    onProgress: onProgress ? (fraction: number) => onProgress({ fraction: fraction * mainProgressScale, fileName }) : undefined,
  });

  // From here on the main file is OURS on disk: any failure before the row is saved must remove it,
  // or a failed import leaves an untracked multi-GB file that the next import refuses ("already
  // exists") and the next scan adopts as a recovered model.
  try {
    return await registerImportedModel({
      fileName, destPath, modelsDir, isLitert, liteRTVision, onProgress,
      mmProjFileName, resolvedMmProjSource, mmProjSourceSize,
    });
  } catch (error) {
    logger.warn(`[ImportLocalModel] import of ${fileName} failed after copy; removing the copied file: ${error instanceof Error ? error.message : String(error)}`);
    // The projector path was checked free above, so a file there now is this import's copy.
    const copied = [destPath, ...(mmProjFileName ? [`${modelsDir}/${mmProjFileName}`] : [])];
    for (const path of copied) {
      if (!(await RNFS.exists(path).catch(() => false))) continue;
      await RNFS.unlink(path).catch((cleanupError: unknown) => {
        logger.error(`[ImportLocalModel] could not remove ${path}: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}`);
      });
    }
    throw error;
  }
}

interface RegisterImportOpts {
  fileName: string;
  destPath: string;
  modelsDir: string;
  isLitert: boolean;
  liteRTVision?: boolean;
  onProgress?: (progress: { fraction: number; fileName: string }) => void;
  mmProjFileName?: string;
  resolvedMmProjSource?: string;
  mmProjSourceSize?: number | null;
}

async function registerImportedModel(opts: RegisterImportOpts): Promise<DownloadedModel> {
  const { fileName, destPath, modelsDir, isLitert, liteRTVision, onProgress, mmProjFileName, resolvedMmProjSource, mmProjSourceSize } = opts;
  // The shared label rule (utils/quantization): IQ4_XS / BF16 / UD-Q4_K_XL read as themselves.
  const quantization = parseQuantization(fileName);
  const modelName = stripQuantization(fileName);
  const fileSize = (await statFile(destPath))?.size ?? 0;

  const pseudoFile: ModelFile = { name: fileName, size: fileSize, quantization, downloadUrl: '' };
  const baseModel = await buildDownloadedModel({ modelId: 'local_import', file: pseudoFile, resolvedLocalPath: destPath });
  const baseFields = {
    id: `local_import/${fileName}`,
    name: modelName,
    author: 'Local Import',
    credibility: { source: 'community' as const, isOfficial: false, isVerifiedQuantizer: false },
  };

  if (isLitert) {
    const liteRTModel: LiteRTDownloadedModel = {
      ...baseModel, ...baseFields, engine: 'litert', liteRTVision: liteRTVision ?? false,
    };
    await persistDownloadedModel(liteRTModel, modelsDir);
    return liteRTModel;
  }

  const llamaModel: LlamaDownloadedModel = { ...baseModel, ...baseFields, engine: 'llama' };

  // Copy mmproj and link it to the model: progress 0.5→1
  if (mmProjFileName && resolvedMmProjSource) {
    const mmProjDestPath = `${modelsDir}/${mmProjFileName}`;
    await copyFileWithProgress(resolvedMmProjSource, mmProjDestPath, {
      knownTotalBytes: mmProjSourceSize ?? null,
      onProgress: onProgress
        ? (fraction: number) => onProgress({ fraction: 0.5 + fraction * 0.5, fileName: mmProjFileName })
        : undefined,
    });
    llamaModel.mmProjPath = mmProjDestPath;
    llamaModel.mmProjFileName = mmProjFileName;
    llamaModel.mmProjFileSize = (await statFile(mmProjDestPath))?.size ?? 0;
    llamaModel.isVisionModel = true;
  }

  await persistDownloadedModel(llamaModel, modelsDir);
  return llamaModel;
}
