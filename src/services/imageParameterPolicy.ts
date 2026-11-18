import {
  effectiveImageParameter,
  resolveImageParameters,
  type ImageParameterStore,
} from '@offgrid/models';
import { SWEET_SPOT_SIZE } from '../utils/imageGenAdvice';

export interface MobileImageParameterSettings {
  imageSteps?: number | null;
  imageGuidanceScale?: number | null;
  imageWidth?: number | null;
}

export interface MobileImageParameterRequest {
  steps?: number;
  guidanceScale?: number;
}

const positiveFinite = (value: number | null | undefined): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value > 0
    ? value
    : undefined;

/** Adapt Mobile's current settings shape into the shared model-specific policy. */
export function resolveMobileImageParameters(
  model: { id: string; name?: string },
  settings: MobileImageParameterSettings,
  request: MobileImageParameterRequest = {},
): { steps: number; guidanceScale: number; size: number } {
  const store: ImageParameterStore = {
    [model.id]: {
      steps: positiveFinite(settings.imageSteps),
      cfgScale: positiveFinite(settings.imageGuidanceScale),
      size: positiveFinite(settings.imageWidth),
    },
  };
  const resolved = resolveImageParameters(model, store);
  return {
    steps: effectiveImageParameter(
      positiveFinite(request.steps),
      resolved.steps,
    ),
    guidanceScale: effectiveImageParameter(
      positiveFinite(request.guidanceScale),
      resolved.cfgScale,
    ),
    size: Math.max(SWEET_SPOT_SIZE, resolved.size),
  };
}

export interface ImageDefaultSettings {
  imageNegativePrompt?: string | null;
  imageSeed?: number | null;
}

/**
 * The user's image defaults applied to a request that names none: a request's own negative prompt
 * / seed (generate_image tool, retry) wins. Seed -1 (or unset) leaves the seed out, so each engine
 * picks a random one as before (sdCppGenerator / remoteImageGeneration generateRandomSeed,
 * LocalDreamModule.kt random when no "seed" key).
 */
export function applyImageDefaults<T extends { negativePrompt?: string; seed?: number }>(
  params: T,
  settings: ImageDefaultSettings,
): T {
  const negative = params.negativePrompt?.trim()
    ? params.negativePrompt
    : (settings.imageNegativePrompt ?? '').trim() || params.negativePrompt;
  const fixedSeed =
    typeof settings.imageSeed === 'number' && Number.isInteger(settings.imageSeed) && settings.imageSeed >= 0
      ? settings.imageSeed
      : undefined;
  const seed = params.seed ?? fixedSeed;
  return {
    ...params,
    ...(negative !== undefined ? { negativePrompt: negative } : {}),
    ...(seed !== undefined ? { seed } : {}),
  };
}
