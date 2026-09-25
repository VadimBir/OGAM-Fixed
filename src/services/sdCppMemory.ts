/**
 * Runtime RAM estimate for a stable-diffusion.cpp checkpoint, from the architecture instead of a
 * flat file-size multiplier. Pure, so it is unit-testable and shared by the load gate, the
 * residency register and the settings readout.
 *
 * Peak = weights (after the load-time weight-type conversion) + the larger of the UNet and VAE
 * compute buffers (sd.cpp frees each graph's buffer before the next stage) + CLIP/runtime overhead.
 * Source rules (sd.cpp model_loader.cpp tensor_should_be_converted): a block quant (q8_0/q5_0/q4_0,
 * block 32) converts only tensors whose row length is a multiple of 32 and skips biases, norms,
 * embeddings and the time/label embedders; 3x3 conv kernels (row length 3) therefore stay in the
 * file's type. VAE (first_stage_model) weights are all convs/norms, so they never convert.
 */

export type SdWeightType = 'auto' | 'q8_0' | 'q5_0' | 'q4_0';
export const SD_WEIGHT_TYPES: readonly SdWeightType[] = ['auto', 'q8_0', 'q5_0', 'q4_0'];

/** Bytes per parameter of each ggml block type (block 32: fp16 scale [+ fp16 min] + packed bits). */
const BYTES_PER_PARAM: Record<Exclude<SdWeightType, 'auto'>, number> = {
  q8_0: 34 / 32,
  q5_0: 22 / 32,
  q4_0: 18 / 32,
};

interface FamilyShape {
  unetParams: number;
  /** Share of UNet params in linear/attention weights (convertible); the rest are 3x3 convs. */
  unetLinearShare: number;
  textParams: number;
  vaeParams: number;
  /** Latent downscale at which the highest-resolution attention runs (sd1/sd2: 8, sdxl: 16). */
  attnDownscale: number;
  heads: number;
}

const SD1: FamilyShape = { unetParams: 860e6, unetLinearShare: 0.55, textParams: 123e6, vaeParams: 84e6, attnDownscale: 8, heads: 8 };
const FAMILIES: Record<string, FamilyShape> = {
  sd1: SD1,
  sd2: { unetParams: 865e6, unetLinearShare: 0.55, textParams: 354e6, vaeParams: 84e6, attnDownscale: 8, heads: 5 },
  sdxl: { unetParams: 2567e6, unetLinearShare: 0.7, textParams: 817e6, vaeParams: 84e6, attnDownscale: 16, heads: 10 },
};

const MB = 1024 * 1024;
/** UNet activations (excluding attention scores) at 512x512 for sd1; scaled by pixel count. */
const UNET_ACT_MB_AT_512 = 150;
/** Non-tiled VAE decode compute buffer at 512x512 (sd1/sd2/sdxl share the same VAE decoder).
 *  Conservative: the CPU conv2d path im2cols the 256-ch 512x512 input (512*512*2304*2 B = 1.2 GB)
 *  on top of the activations. Unmeasured on device — sd.cpp logs the real value ("vae compute
 *  buffer size", verbose → logcat DEBUG); recalibrate from that. Under-estimating here OOM-kills. */
const VAE_DECODE_MB_AT_512 = 2000;
/** sd.cpp default VAE tile: 32 latent px = 256 image px (vae.hpp get_tile_sizes). */
const VAE_TILE_PX = 256;
const RUNTIME_OVERHEAD_MB = 96;
const SAFETY_FACTOR = 1.1;

export interface SdCppMemoryInput {
  fileSizeBytes: number;
  family?: string;
  weightType?: SdWeightType;
  width: number;
  height: number;
  flashAttn: boolean;
  vaeTiling: boolean;
}

export interface SdCppMemoryEstimate {
  weightsMB: number;
  unetComputeMB: number;
  vaeComputeMB: number;
  totalMB: number;
  /** File bytes per parameter inferred from size (auto keeps the file's own types). */
  sourceBytesPerParam: number;
}

export function estimateSdCppMemory(input: SdCppMemoryInput): SdCppMemoryEstimate {
  const shape = FAMILIES[input.family ?? ''] ?? SD1;
  const totalParams = shape.unetParams + shape.textParams + shape.vaeParams;
  // The file's own precision: ~2 B/param fp16, ~4 fp32, <1.1 already quantized. Clamped because
  // EMA copies or bundled extras inflate the file (conservative: counts them as real weights).
  const sourceBytesPerParam = Math.min(4, Math.max(0.5, input.fileSizeBytes / totalParams));
  const wt = input.weightType ?? 'auto';
  const target = wt === 'auto' ? sourceBytesPerParam : Math.min(sourceBytesPerParam, BYTES_PER_PARAM[wt]);
  const unetLinear = shape.unetParams * shape.unetLinearShare;
  const weightsBytes =
    unetLinear * target +
    (shape.unetParams - unetLinear) * sourceBytesPerParam +
    shape.textParams * target +
    shape.vaeParams * sourceBytesPerParam;

  const pixels = Math.max(64 * 64, input.width * input.height);
  const scale512 = pixels / (512 * 512);
  // Attention scores: tokens^2 x heads x f32 at the highest-resolution attention level. Flash
  // attention streams them, so only the activations remain.
  const attnTokens = pixels / (shape.attnDownscale * shape.attnDownscale);
  const attnScoresMB = input.flashAttn ? 0 : (attnTokens * attnTokens * shape.heads * 4) / MB;
  const unetComputeMB = UNET_ACT_MB_AT_512 * scale512 * (shape === SD1 ? 1 : 1.5) + attnScoresMB;
  const vaePixels = input.vaeTiling ? Math.min(pixels, VAE_TILE_PX * VAE_TILE_PX) : pixels;
  const vaeComputeMB = VAE_DECODE_MB_AT_512 * (vaePixels / (512 * 512));

  const weightsMB = weightsBytes / MB;
  const totalMB = Math.ceil(
    (weightsMB + Math.max(unetComputeMB, vaeComputeMB) + RUNTIME_OVERHEAD_MB) * SAFETY_FACTOR,
  );
  return {
    weightsMB: Math.round(weightsMB),
    unetComputeMB: Math.round(unetComputeMB),
    vaeComputeMB: Math.round(vaeComputeMB),
    totalMB,
    sourceBytesPerParam,
  };
}
