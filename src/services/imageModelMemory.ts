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
    flashAttn: s.imageSdFlashAttn ?? true,
    vaeTiling: s.imageSdVaeTiling ?? true,
  });
}

export function estimateImageModelRamBytes(model: ONNXImageModel): number {
  if (model.backend === 'sdcpp') return estimateSdCppModelMemory(model).totalMB * 1024 * 1024;
  return hardwareService.estimateImageModelRam(model) || 0;
}
