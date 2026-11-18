import type { ImageLora } from '../types';
import type { EngineImageOptions, GenerateImageParams } from './imageGenerationTypes';

/** Samplers the LocalDream core (QNN/MNN) accepts in its `scheduler` field; default 'dpm'. */
export const LOCALDREAM_SAMPLERS = ['dpm', 'euler_a'] as const;

export const SDCPP_SIZE_MULTIPLE = 64;
export const SDCPP_MIN_SIZE = 64;
export const SDCPP_MAX_SIZE = 1024;

/** SD latents are 1/8 of the image and the UNet downsamples 3x → multiples of 64 px. */
export function snapSdCppDimension(px: number | null | undefined, fallback = 512): number {
  const v = typeof px === 'number' && Number.isFinite(px) && px > 0 ? px : fallback;
  const clamped = Math.min(SDCPP_MAX_SIZE, Math.max(SDCPP_MIN_SIZE, v));
  return Math.max(SDCPP_MIN_SIZE, Math.round(clamped / SDCPP_SIZE_MULTIPLE) * SDCPP_SIZE_MULTIPLE);
}

/**
 * What an engine will really render for a request, known BEFORE generating, so UI and callers
 * never discover a silent substitution afterwards (QNN without a resolution patch -> 512x512,
 * MNN -> square 128..512, OpenCL only at exactly 512x512).
 */
export interface EffectiveImageSize {
  engine: 'localdream' | 'sdcpp';
  requestedWidth: number;
  requestedHeight: number;
  width: number;
  height: number;
  /** true when width/height differ from what was requested. */
  adjusted: boolean;
  reason: string | null;
  /** Whether OpenCL will really run; null when the engine cannot say (sd.cpp, no model, iOS). */
  useOpenCL: boolean | null;
}

/** sd.cpp honours W and H independently on the 64 px grid (64..SDCPP_MAX_SIZE). */
export function resolveSdCppEffectiveSize(params: { width?: number | null; height?: number | null }): EffectiveImageSize {
  const requestedWidth = params.width || 512;
  const requestedHeight = params.height || 512;
  const width = snapSdCppDimension(params.width);
  const height = snapSdCppDimension(params.height);
  const adjusted = width !== requestedWidth || height !== requestedHeight;
  return {
    engine: 'sdcpp',
    requestedWidth,
    requestedHeight,
    width,
    height,
    adjusted,
    reason: adjusted ? `stable-diffusion.cpp uses a ${SDCPP_SIZE_MULTIPLE} px grid, ${SDCPP_MIN_SIZE}-${SDCPP_MAX_SIZE} px` : null,
    useOpenCL: null,
  };
}

export interface EngineImageSettings {
  imageWidth?: number | null;
  imageHeight?: number | null;
  imageSampler?: string;
  imageScheduler?: string;
  imageLoras?: ImageLora[];
  imageSdVaeTiling?: boolean;
}

export interface EngineImageRequest {
  width: number;
  height: number;
  engineOptions: EngineImageOptions;
}

/**
 * The single place a request's engine signature is decided. Every entry point (chat button, auto
 * detection, generate_image tool call, retry) reaches here through imageGenerationService, so the
 * params always match the engine the model's format selected:
 *  - sdcpp: free W×H (64-px grid, no 256 floor), any sd.cpp sampler/scheduler, enabled LoRAs.
 *  - LocalDream: the existing square size policy; sampler only when the core supports it.
 */
export function resolveEngineImageRequest(
  model: { backend?: string },
  settings: EngineImageSettings,
  params: Pick<GenerateImageParams, 'sampler' | 'scheduler' | 'width' | 'height'>,
  squareSize: number,
): EngineImageRequest {
  const sampler = (params.sampler ?? settings.imageSampler ?? '').trim();
  if (model.backend === 'sdcpp') {
    const width = snapSdCppDimension(params.width ?? settings.imageWidth);
    const height = snapSdCppDimension(params.height ?? settings.imageHeight, width);
    const loras = (settings.imageLoras ?? [])
      .filter(l => l.enabled && l.path && Number.isFinite(l.weight) && l.weight !== 0)
      .map(l => ({ path: l.path, weight: l.weight }));
    return {
      width,
      height,
      engineOptions: {
        sampler,
        scheduler: (params.scheduler ?? settings.imageScheduler ?? '').trim(),
        loras,
        vaeTiling: settings.imageSdVaeTiling ?? true,
      },
    };
  }
  const localDreamSampler = (LOCALDREAM_SAMPLERS as readonly string[]).includes(sampler)
    ? sampler
    : undefined;
  return {
    width: squareSize,
    height: squareSize,
    engineOptions: localDreamSampler ? { sampler: localDreamSampler } : {},
  };
}
