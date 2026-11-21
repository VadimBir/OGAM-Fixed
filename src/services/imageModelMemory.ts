/**
 * Single owner of an image model's runtime-RAM estimate. stable-diffusion.cpp checkpoints are
 * sized from their architecture + the current load/generate options (weight type, flash
 * attention, VAE tiling, W×H); LocalDream SDXL NPU from its UNet size (estimateSdxlNpuPeakBytes);
 * other LocalDream / Core ML models keep the file-size multiplier.
 */
import type { ONNXImageModel } from '../types';
import { hardwareService } from './hardware';
import { useAppStore } from '../stores';
import { estimateSdCppMemory, type SdCppMemoryEstimate } from './sdCppMemory';
import { snapSdCppDimension } from './imageEngineParams';
import { INCOMING_PEAK_FACTOR } from './memoryBudget';
import { getObservedImageModelLayout } from '../utils/imageModelIntegrity';

const GiB = 1024 * 1024 * 1024;
/** unet.bin / zip size, for an SDXL NPU model whose unet.bin was not listed yet (Phase A ratio). */
export const SDXL_UNET_OF_ZIP = 0.72;
/** HTP spill-fill scratch the lowram SDXL UNet context reserves. */
export const SDXL_SPILL_FILL_BYTES = 0.09 * GiB;
/** token/pos embedding tables (both CLIPs) the core keeps in RAM (main.cpp:720 loadEmbeddingTables). */
export const SDXL_EMBEDDINGS_BYTES = 0.2 * GiB;
/** The core process itself: QNN runtime libs (~165 MB, LocalDreamModule prepareRuntimeDir), the
 *  11 MB core, HTTP server, tokenizer. A budget, not a device measurement. */
export const LOCALDREAM_PROCESS_BASELINE_BYTES = 0.25 * GiB;

/** SDXL NPU model (local-dream `--type sdxl`): seen as such on disk, or a catalog id from an SDXL repo. */
export function isSdxlNpuModel(model: Pick<ONNXImageModel, 'id' | 'backend' | 'modelPath'>): boolean {
  if (model.backend !== 'qnn') return false;
  const seen = model.modelPath ? getObservedImageModelLayout(model.modelPath) : undefined;
  if (seen) return seen.kind === 'sdxl';
  return model.id.includes('_sdxl_npu_');
}

/**
 * Peak bytes of an SDXL NPU model under `--lowram` (one stage model resident at a time, the UNet
 * being the largest): unet.bin + HTP spill-fill + embedding tables + the process baseline. Uses the
 * real unet.bin size once validateImageModelDir has listed it, else zip size x SDXL_UNET_OF_ZIP.
 */
export function estimateSdxlNpuPeakBytes(model: Pick<ONNXImageModel, 'size' | 'modelPath'>): number {
  const seenUnet = model.modelPath ? getObservedImageModelLayout(model.modelPath)?.unetBytes : undefined;
  const unetBytes = seenUnet && seenUnet > 0 ? seenUnet : (model.size || 0) * SDXL_UNET_OF_ZIP;
  return unetBytes + SDXL_SPILL_FILL_BYTES + SDXL_EMBEDDINGS_BYTES + LOCALDREAM_PROCESS_BASELINE_BYTES;
}

export function estimateSdCppModelMemory(model: Pick<ONNXImageModel, 'size' | 'sdFamily'>): SdCppMemoryEstimate {
  const s = useAppStore.getState().settings ?? ({} as Record<string, never>);
  const width = snapSdCppDimension(s.imageWidth);
  return estimateSdCppMemory({
    fileSizeBytes: model.size || 0,
    family: model.sdFamily,
    weightType: s.imageSdWeightType ?? 'auto',
    width,
    height: snapSdCppDimension(s.imageHeight, width),
    // The diffusion model always runs flash attention (sdcpp_jni.cpp diffusion_flash_attn = true); the
    // setting only covers the text encoder + VAE, so UNet attention scores are never materialised.
    flashAttn: true,
    vaeTiling: s.imageSdVaeTiling ?? true,
    threads: s.imageThreads,
    loraBytes: (s.imageLoras ?? [])
      .filter(l => l.enabled && l.weight !== 0)
      .reduce((sum, l) => sum + (l.sizeBytes ?? 0), 0),
  });
}

export function estimateImageModelRamBytes(model: ONNXImageModel): number {
  if (model.backend === 'sdcpp') return estimateSdCppModelMemory(model).totalMB * 1024 * 1024;
  if (isSdxlNpuModel(model)) return estimateSdxlNpuPeakBytes(model);
  return hardwareService.estimateImageModelRam(model) || 0;
}

/**
 * MB the residency gate charges for an image model: `peakMB` when it is loaded, `residentMB` once it
 * is resident. sd.cpp: the load peak (x1.1, no compute) and the resident bytes (no factor, no
 * compute) from estimateSdCppMemory. SDXL NPU: its generation peak as resident (each generation
 * reloads the UNet, so that RAM must stay available) and that peak x INCOMING_PEAK_FACTOR, once, as
 * the load charge. Other backends have no weights/working-set split here, so both are the file-size
 * estimate.
 */
export function imageModelResidencyMB(model: ONNXImageModel): { peakMB: number; residentMB: number } {
  if (model.backend === 'sdcpp') {
    const est = estimateSdCppModelMemory(model);
    return { peakMB: est.loadPeakMB, residentMB: est.residentMB };
  }
  if (isSdxlNpuModel(model)) {
    const bytes = estimateSdxlNpuPeakBytes(model);
    return {
      peakMB: Math.ceil((bytes * INCOMING_PEAK_FACTOR) / (1024 * 1024)),
      residentMB: Math.ceil(bytes / (1024 * 1024)),
    };
  }
  const mb = Math.round((hardwareService.estimateImageModelRam(model) || 0) / (1024 * 1024));
  return { peakMB: mb, residentMB: mb };
}
