/**
 * Advice for the GPU (mnn) image-generation path — the slow, quality-sensitive one used
 * when the device has no compatible NPU. On a mid-tier GPU a full SD1.5 model is a hard
 * speed/quality trade the user has to steer manually:
 *  - too few steps look muddy (undercooked denoise),
 *  - a large size is very slow (observed: 512x512 @ 22 steps ~= 18 min on a Snapdragon 765G).
 * So when the active model runs on the GPU path, nudge the user toward the settings that
 * matter. The NPU (qnn) and CoreML (ANE) paths are fast + fixed-shape, so no advice.
 *
 * Pure + data-driven: it branches on the backend AS DATA (not a Platform/device check),
 * so it's unit-testable and the same rule drives the card on every surface.
 */
export interface ImageGenAdvice {
  /** Surface the advisory card at all (any tip applies). */
  show: boolean;
  /** Steps are below the quality floor for the GPU path. */
  raiseSteps: boolean;
  /** Resolution is above the sweet spot → slow on a mid-tier GPU. */
  lowerSize: boolean;
  /** Resolution is below what SD1.5 can render coherently → garbage output. */
  raiseSize: boolean;
}

/** Below this, GPU-path output is visibly undercooked (muddy). */
export const QUALITY_STEP_FLOOR = 20;
/**
 * The usable sweet-spot resolution for SD1.5 on the GPU path. SD1.5 is trained at 512;
 * 256 stays coherent while generating far faster, but BELOW 256 the model produces
 * garbage (observed on-device: 128x128 was fast but incoherent). So 256 is both the
 * "smaller = faster" target when the user is at 512, and the floor below which quality
 * collapses.
 */
export const SWEET_SPOT_SIZE = 256;

/** Slider maximum only. It is NOT the default (that was the old bug: 50 steps out of the box). */
export const MAX_IMAGE_STEPS = 50;

/**
 * THE default step count for every image engine and every layer (store, JS generators, and the
 * native LocalDream/sd.cpp fallbacks, which are 20 and are contract-tested against this value).
 * 20 = QUALITY_STEP_FLOOR: below it output is muddy, above it time grows linearly for little gain
 * (512x512 @ 22 steps ~= 18 min on a Snapdragon 765G, so 50 is never a sane out-of-box value).
 */
export const DEFAULT_IMAGE_STEPS = QUALITY_STEP_FLOOR;

/** One owner for the default. Persisted user values still take precedence. */
export function defaultImageSteps(_platform?: string): number {
  return DEFAULT_IMAGE_STEPS;
}

export function getImageGenAdvice(opts: {
  backend?: string | null;
  steps: number;
  width: number;
}): ImageGenAdvice {
  // Only the mnn (GPU/CPU) path is slow + step/size-sensitive. qnn (NPU) / coreml (ANE)
  // are fast and fixed-resolution, so tuning there isn't the same trade-off.
  if (opts.backend !== 'mnn') {
    return { show: false, raiseSteps: false, lowerSize: false, raiseSize: false };
  }
  const raiseSteps = opts.steps < QUALITY_STEP_FLOOR;
  const lowerSize = opts.width > SWEET_SPOT_SIZE;
  const raiseSize = opts.width > 0 && opts.width < SWEET_SPOT_SIZE;
  return { show: raiseSteps || lowerSize || raiseSize, raiseSteps, lowerSize, raiseSize };
}
