/**
 * Single owner of an image model's runtime-RAM estimate. stable-diffusion.cpp checkpoints are
 * sized from their architecture + the current load/generate options (weight type, flash
 * attention, VAE tiling, W×H); LocalDream/Core ML keep the file-size multiplier.
 */
import type { ONNXImageModel } from '../types';
import { hardwareService } from './hardware';
import { useAppStore } from '../stores';
import { estimateSdCppMemory, type SdCppMemoryEstimate } from './sdCppMemory';
import { snapSdCppDimension } from './imageEngineParams';

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
  return hardwareService.estimateImageModelRam(model) || 0;
}
