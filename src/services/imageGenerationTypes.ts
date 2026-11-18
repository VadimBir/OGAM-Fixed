/**
 * ImageGenerationService types + the pure phase predicate.
 *
 * Extracted from imageGenerationService.ts (behavior-neutral) so the service file
 * stays within the max-lines budget. imageGenerationService re-exports the public
 * symbols (ImageGenPhase, ImageGenerationState, isInFlight), so every existing
 * `import { ... } from './imageGenerationService'` keeps working unchanged.
 */
import { GeneratedImage, ONNXImageModel, RemoteServer } from '../types';

/**
 * Explicit lifecycle phase — the single source of truth for "what is image
 * generation doing right now". The UI projects this (it never assembles the
 * in-progress view from scattered flags), so the progress indicator can't flash
 * or desync: it's shown for exactly `enhancing | loading | generating | saving`
 * and hidden otherwise.
 */
export type ImageGenPhase =
  | 'idle'
  | 'enhancing'  // running the text model to enrich the prompt
  | 'loading'    // loading the image model into memory
  | 'generating' // diffusion steps running
  | 'saving'     // writing the result + adding the chat message
  | 'done'
  | 'error'
  | 'cancelled';

/** True while a generation is actively in flight (drives the progress indicator). */
export function isInFlight(phase: ImageGenPhase): boolean {
  return phase === 'enhancing' || phase === 'loading' || phase === 'generating' || phase === 'saving';
}

export interface ImageGenerationState {
  phase: ImageGenPhase;
  /** Derived from phase (isInFlight) — kept for back-compat with existing readers. */
  isGenerating: boolean;
  progress: { step: number; totalSteps: number } | null;
  status: string | null;
  previewPath: string | null;
  prompt: string | null;
  conversationId: string | null;
  /** Stable id for the assistant message this live image turn will create. */
  messageId: string | null;
  error: string | null;
  result: GeneratedImage | null;
}

export type ImageGenerationListener = (state: ImageGenerationState) => void;

export interface GenerateImageParams {
  prompt: string;
  conversationId?: string;
  negativePrompt?: string;
  steps?: number;
  guidanceScale?: number;
  seed?: number;
  previewInterval?: number;
  /** img2img reference image (base64, no data-URL prefix). Presence switches the request to img2img. */
  initImage?: string;
  /** img2img denoise strength 0..1 (low = closer to ref). Falls back to settings.imageDenoiseStrength. */
  denoiseStrength?: number;
  /** Optional inpainting mask (base64). */
  maskImage?: string;
  /** Per-request sampler override (wins over settings.imageSampler). */
  sampler?: string;
  /** Per-request noise-schedule override, stable-diffusion.cpp only (wins over settings.imageScheduler). */
  scheduler?: string;
  /** Per-request size override, stable-diffusion.cpp only (LocalDream keeps its square size policy). */
  width?: number;
  height?: number;
}

/**
 * Why a generate request produced no image. One code per cause, so a caller (the generate_image
 * tool) can tell the model what to do instead of one generic "make sure a model is selected".
 */
export type ImageGenFailureReason =
  | 'no_model'
  | 'load_failed'
  | 'out_of_memory'
  | 'busy'
  | 'cancelled'
  | 'engine_error';

/** The values the engine actually ran with, after every clamp/policy step. A field is absent when
 *  the engine is not sent that value (the remote path sends prompt + size only). */
export interface ImageGenUsedParams {
  engine: 'local' | 'remote';
  width: number;
  height: number;
  steps?: number;
  guidance?: number;
  seed?: number;
  /** '' / undefined = the engine's default sampler. */
  sampler?: string;
}

/** The engine a generate request would run on right now; null = none available. */
export type ActiveImageEngine =
  | { kind: 'remote'; server: RemoteServer }
  | { kind: 'local'; model: ONNXImageModel }
  | null;

export type ImageGenerationOutcome =
  | { ok: true; image: GeneratedImage; used: ImageGenUsedParams }
  | { ok: false; reason: ImageGenFailureReason; message: string };

/** Engine-specific knobs resolved once per request and forwarded to the format-selected engine. */
export interface EngineImageOptions {
  sampler?: string;
  scheduler?: string;
  loras?: Array<{ path: string; weight: number }>;
  /** sd.cpp: decode the VAE in tiles (bounded memory at any resolution). */
  vaeTiling?: boolean;
}

export interface ActiveImageModel {
  id: string;
  name: string;
  modelPath: string;
  backend?: string;
}

export interface RunGenerationOptions {
  params: GenerateImageParams;
  enhancedPrompt: string;
  activeImageModel: ActiveImageModel;
  steps: number;
  guidanceScale: number;
  imageWidth: number;
  imageHeight: number;
  useOpenCL: boolean;
  engineOptions: EngineImageOptions;
}
