/**
 * Runtime RAM estimate for a stable-diffusion.cpp checkpoint, from the architecture instead of a
 * flat file-size multiplier. Pure, so it is unit-testable and shared by the load gate, the
 * residency register and the settings readout.
 *
 * Peak = weights (after the load-time weight-type conversion) + the larger of the UNet and VAE
 * compute buffers (sd.cpp frees each graph's buffer before the next stage) + CLIP/runtime overhead.
 * Source rules (docs/notes/sdcpp-memory-audit.md section 1, sd.cpp @ c92d73c): the chosen weight
 * type converts ONLY linear weights (row length % 32 == 0, name not excluded). Every convolution
 * (3x3 and 1x1, UNet and VAE) is f16 in RAM whatever the file type or weight type; biases and
 * norms are f32 (negligible, not counted); embeddings (names containing `embedding`, `time_embed.`,
 * `label_emb.`) keep the file type. When tensors are converted, each loader thread also holds a
 * read buffer plus an f32 copy of the largest converted tensor (~75 MiB per thread, audit section 2).
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
  /** Share of UNet params in linear/attention weights (convertible); the rest are convolutions (f16). */
  unetLinearShare: number;
  textParams: number;
  /** Text-encoder token + position embedding params: they keep the file type, never the chosen one.
   *  sd1 38.0e6 (audit); sd2/sdxl derived as 49408 x width + 77 x width per encoder
   *  (OpenCLIP-H 1024: 50.7e6; CLIP-L 768 + OpenCLIP-bigG 1280: 38.0e6 + 63.3e6). */
  textEmbedParams: number;
  vaeParams: number;
  /** Latent downscale at which the highest-resolution attention runs (sd1/sd2: 8, sdxl: 16). */
  attnDownscale: number;
  heads: number;
}

// unetLinearShare: audit calc/sd15.py, calc/sd2.py, calc/sdxl.py (linear params / UNet params).
const SD1: FamilyShape = { unetParams: 860e6, unetLinearShare: 0.312, textParams: 123e6, textEmbedParams: 38.0e6, vaeParams: 84e6, attnDownscale: 8, heads: 8 };
const FAMILIES: Record<string, FamilyShape> = {
  sd1: SD1,
  sd2: { unetParams: 865e6, unetLinearShare: 0.346, textParams: 354e6, textEmbedParams: 50.7e6, vaeParams: 84e6, attnDownscale: 8, heads: 5 },
  sdxl: { unetParams: 2567e6, unetLinearShare: 0.867, textParams: 817e6, textEmbedParams: 101.3e6, vaeParams: 84e6, attnDownscale: 16, heads: 10 },
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
/** Bytes per parameter of a convolution kernel in RAM: always f16, whatever the file or weight type. */
const CONV_BYTES_PER_PARAM = 2;
/** Source bytes/param from which a file is read as fp32 (fp16 ~2, with bundled extras still < 3). */
const FP32_FILE_MIN_BYTES_PER_PARAM = 3;
/** Per loader thread while converting: read buffer + f32 copy of the largest converted tensor
 *  (SD1.5 ff.net.0.proj, 13.1 M params: 25 MiB f16 + 50 MiB f32). Audit section 2. */
const LOAD_BUFFER_MB_PER_THREAD = 75;
/** Loader threads when the caller does not say (appStore imageThreads default). */
const DEFAULT_LOAD_THREADS = 4;
const SAFETY_FACTOR = 1.1;

export interface SdCppMemoryInput {
  fileSizeBytes: number;
  family?: string;
  weightType?: SdWeightType;
  width: number;
  height: number;
  flashAttn: boolean;
  vaeTiling: boolean;
  /** Loader threads (n_threads); sizes the load-time conversion buffers. Default 4. */
  threads?: number;
  /** Sum of the enabled LoRA file sizes. */
  loraBytes?: number;
}

export interface SdCppMemoryEstimate {
  weightsMB: number;
  /** Enabled LoRA tensors (file size; 0 for a LoRA imported before sizes were recorded). */
  loraMB: number;
  /** Transient per-thread conversion buffers while loading; 0 when no tensor is converted. */
  loadBufferMB: number;
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
  const textLinear = shape.textParams - shape.textEmbedParams;
  const weightsBytes =
    unetLinear * target +
    (shape.unetParams - unetLinear) * CONV_BYTES_PER_PARAM +
    shape.textEmbedParams * sourceBytesPerParam +
    textLinear * target +
    shape.vaeParams * CONV_BYTES_PER_PARAM;
  // A tensor is read straight into its destination when file type == RAM type; any conversion
  // (a quant override below the file type, or an fp32 file whose convs become f16) goes through the
  // per-thread buffers. An unconverted fp16 `auto` load has none.
  const converts = target < sourceBytesPerParam || sourceBytesPerParam >= FP32_FILE_MIN_BYTES_PER_PARAM;
  const threads = Math.max(1, Math.round(input.threads ?? DEFAULT_LOAD_THREADS));
  const loadBufferMB = converts ? threads * LOAD_BUFFER_MB_PER_THREAD : 0;

  const pixels = Math.max(64 * 64, input.width * input.height);
  const scale512 = pixels / (512 * 512);
  // Attention scores: tokens^2 x heads x f32 at the highest-resolution attention level. Flash
  // attention streams them, so only the activations remain.
  const attnTokens = pixels / (shape.attnDownscale * shape.attnDownscale);
  const attnScoresMB = input.flashAttn ? 0 : (attnTokens * attnTokens * shape.heads * 4) / MB;
  const unetComputeMB = UNET_ACT_MB_AT_512 * scale512 * (shape === SD1 ? 1 : 1.5) + attnScoresMB;
  const vaePixels = input.vaeTiling ? Math.min(pixels, VAE_TILE_PX * VAE_TILE_PX) : pixels;
  const vaeComputeMB = VAE_DECODE_MB_AT_512 * (vaePixels / (512 * 512));

  // Only sd1/sd2/sdxl have a measured shape. Any other family (flux, sd3, unknown) borrows the SD1
  // shape, which with the 4 B/param clamp counts a 17-24 GB Flux file as ~3 GB and lets it OOM.
  // Floor those at the file size: conservative (ignores a quant override), never under-counts.
  const weightsFloor = FAMILIES[input.family ?? ''] ? 0 : input.fileSizeBytes;
  const weightsMB = Math.max(weightsBytes, weightsFloor) / MB;
  // Enabled LoRA tensors: resident for the whole run in runtime mode (quantized weights), and held
  // during the merge in immediate mode (freed after, model_manager.cpp:719), so they count at peak.
  const loraMB = Math.max(0, input.loraBytes ?? 0) / MB;
  const totalMB = Math.ceil(
    (weightsMB + loraMB + loadBufferMB + Math.max(unetComputeMB, vaeComputeMB) + RUNTIME_OVERHEAD_MB) * SAFETY_FACTOR,
  );
  return {
    weightsMB: Math.round(weightsMB),
    loraMB: Math.round(loraMB),
    loadBufferMB,
    unetComputeMB: Math.round(unetComputeMB),
    vaeComputeMB: Math.round(vaeComputeMB),
    totalMB,
    sourceBytesPerParam,
  };
}
