/**
 * generate_image tool: model arguments -> image request -> model-facing result.
 *
 * Every setting lives in the app. The model may only REQUEST a value; this file bounds the request,
 * imageGenerationService applies the engine policy (imageParameterPolicy / imageEngineParams), and the
 * result reports the values the engine actually USED, so the model never believes it got a size or a
 * step count it did not get. An omitted argument falls back to the user's configured setting.
 */
import type { ToolCall } from './types';
import type {
  ActiveImageEngine,
  GenerateImageParams,
  ImageGenFailureReason,
  ImageGenUsedParams,
  ImageGenerationOutcome,
} from '../imageGenerationTypes';

/** Size range the tool accepts from the model (owner rule: 64x64 up to 512x512, W and H independent). */
export const IMAGE_TOOL_MIN_SIZE = 64;
export const IMAGE_TOOL_MAX_SIZE = 512;
const STEPS_MIN = 1;
const STEPS_MAX = 50;
const GUIDANCE_MIN = 1;
const GUIDANCE_MAX = 20;
const SEED_MAX = 2147483647;
const DETAIL_MAX_CHARS = 160;

/** What the model is told to do for each failure. "Do not retry" stops a small model from looping. */
const FAILURE_ADVICE: Record<ImageGenFailureReason, string> = {
  no_model: 'No image model is selected. Tell the user to choose one in Models. Do not retry.',
  load_failed: 'The image model failed to load. Tell the user. Do not retry now.',
  out_of_memory:
    'Not enough memory for the image model. Tell the user to close other apps or ask for a smaller size. Do not retry now.',
  busy: 'Another image is still being generated. Do not retry now; tell the user to wait for it.',
  cancelled: 'The user cancelled the image. Do not retry.',
  engine_error: 'The image engine failed. Tell the user. Do not retry with the same arguments.',
};

function imageService(): {
  activeImageEngine: () => ActiveImageEngine;
  generateImageWithOutcome: (p: GenerateImageParams) => Promise<ImageGenerationOutcome>;
} {
  return require('../imageGenerationService').imageGenerationService; // NOSONAR (cycle-safe lazy import)
}

function toNumber(v: unknown): number | undefined {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() && Number.isFinite(Number(v))) return Number(v);
  return undefined;
}

/** One numeric argument: absent -> undefined (user setting); out of range -> clamped + noted. */
function boundArg(
  name: string,
  raw: unknown,
  range: { min: number; max: number; integer: boolean },
  notes: string[],
): number | undefined {
  if (raw === undefined || raw === null || raw === '') return undefined;
  const n = toNumber(raw);
  if (n === undefined) {
    notes.push(`${name} "${String(raw)}" ignored (not a number)`);
    return undefined;
  }
  const asked = range.integer ? Math.round(n) : n;
  const v = Math.min(range.max, Math.max(range.min, asked));
  if (v !== asked) notes.push(`${name} ${asked} clamped to ${v}`);
  return v;
}

/** Samplers the active engine accepts; null = the engine is not sent a sampler (remote). */
function engineSamplers(engine: ActiveImageEngine): readonly string[] | null {
  if (!engine || engine.kind === 'remote') return null;
  if (engine.model.backend === 'sdcpp') return require('../sdCppGenerator').SDCPP_SAMPLERS; // NOSONAR
  return require('../imageEngineParams').LOCALDREAM_SAMPLERS; // NOSONAR
}

export interface ImageToolRequest {
  params: Omit<GenerateImageParams, 'prompt' | 'conversationId'>;
  /** Values as the model asked for them (after the tool's own bounds), to compare with USED. */
  asked: { width?: number; height?: number; steps?: number; guidance?: number; seed?: number; sampler?: string };
  notes: string[];
}

/** Bound the model's arguments. Exported for tests. */
export function resolveImageToolRequest(
  args: Record<string, any>,
  engine: ActiveImageEngine,
): ImageToolRequest {
  const notes: string[] = [];
  const size = { min: IMAGE_TOOL_MIN_SIZE, max: IMAGE_TOOL_MAX_SIZE, integer: true };
  const width = boundArg('width', args.width, size, notes);
  const height = boundArg('height', args.height, size, notes);
  const steps = boundArg('steps', args.steps, { min: STEPS_MIN, max: STEPS_MAX, integer: true }, notes);
  const guidance = boundArg(
    'guidance',
    args.guidance,
    { min: GUIDANCE_MIN, max: GUIDANCE_MAX, integer: false },
    notes,
  );
  const rawSeed = boundArg('seed', args.seed, { min: -1, max: SEED_MAX, integer: true }, notes);
  const seed = rawSeed !== undefined && rawSeed >= 0 ? rawSeed : undefined;

  let sampler: string | undefined;
  const rawSampler = typeof args.sampler === 'string' ? args.sampler.trim().toLowerCase() : '';
  if (rawSampler) {
    const accepted = engineSamplers(engine);
    if (accepted === null || accepted.includes(rawSampler)) sampler = rawSampler;
    else notes.push(`sampler "${rawSampler}" not supported here; user default used (one of: ${accepted.join(', ')})`);
  }
  const negative = typeof args.negative_prompt === 'string' ? args.negative_prompt.trim() : '';

  return {
    params: {
      ...(negative ? { negativePrompt: negative } : {}),
      ...(width !== undefined ? { width } : {}),
      ...(height !== undefined ? { height } : {}),
      ...(steps !== undefined ? { steps } : {}),
      ...(guidance !== undefined ? { guidanceScale: guidance } : {}),
      ...(seed !== undefined ? { seed } : {}),
      ...(sampler ? { sampler } : {}),
    },
    asked: { width, height, steps, guidance, seed, sampler },
    notes,
  };
}

/** Compact JSON the model reads back: the values USED, plus a note for each value it did not get. */
export function imageToolSuccessContent(
  used: ImageGenUsedParams,
  asked: ImageToolRequest['asked'],
  notes: string[],
): string {
  const out = [...notes];
  const differs = (key: 'width' | 'height' | 'steps' | 'guidance' | 'seed', value?: number) => {
    const a = asked[key];
    if (a !== undefined && value !== undefined && a !== value) out.push(`${key} ${a} -> ${value} (engine limit)`);
  };
  differs('width', used.width);
  differs('height', used.height);
  if (used.engine === 'remote') {
    const unsent = (['steps', 'guidance', 'seed', 'sampler'] as const).filter(k => asked[k] !== undefined);
    if (unsent.length) out.push(`${unsent.join(', ')} not sent: the remote engine takes prompt and size only`);
  } else {
    differs('steps', used.steps);
    differs('guidance', used.guidance);
    differs('seed', used.seed);
  }
  return JSON.stringify({
    status: 'ok',
    shown_in_chat: true,
    width: used.width,
    height: used.height,
    ...(used.steps !== undefined ? { steps: used.steps } : {}),
    ...(used.guidance !== undefined ? { guidance: used.guidance } : {}),
    ...(used.seed !== undefined ? { seed: used.seed } : {}),
    ...(used.engine === 'local' ? { sampler: used.sampler || 'default' } : {}),
    ...(out.length ? { notes: out } : {}),
    next: 'The image is already shown to the user. Do not call generate_image again unless asked for another image.',
  });
}

/** Model-facing failure text: reason code, what to do, and a short engine detail. */
export function imageToolFailureMessage(reason: ImageGenFailureReason, detail: string): string {
  const withDetail = reason === 'load_failed' || reason === 'out_of_memory' || reason === 'engine_error';
  const d = detail.replace(/\s+/g, ' ').trim().slice(0, DETAIL_MAX_CHARS);
  return `generate_image ${reason}: ${FAILURE_ADVICE[reason]}${withDetail && d ? ` Detail: ${d}` : ''}`;
}

/**
 * Run the tool. Success returns the compact USED-values JSON; any failure THROWS, so the loop records
 * a typed error result (status 'error'), never a success the model and the turn-end logic believe.
 */
export async function handleGenerateImage(prompt: string, call: ToolCall): Promise<string> {
  const conversationId = call.context?.conversationId;
  // Without a conversation the generated image has nowhere to render.
  if (!conversationId) {
    throw new Error('generate_image needs an open chat to show the image. Do not retry.');
  }
  const service = imageService();
  const engine = service.activeImageEngine();
  if (!engine) throw new Error(imageToolFailureMessage('no_model', ''));
  const request = resolveImageToolRequest(call.arguments ?? {}, engine);
  const outcome = await service.generateImageWithOutcome({
    prompt,
    conversationId,
    ...request.params,
  });
  if (!outcome.ok) throw new Error(imageToolFailureMessage(outcome.reason, outcome.message));
  return imageToolSuccessContent(outcome.used, request.asked, request.notes);
}

export type ImagePromptStyle = 'tags' | 'natural' | null;

/** Prompt style for the selected image model's architecture. Only sd.cpp checkpoints record it
 *  (sdFamily); LocalDream / Core ML / remote models carry no architecture -> null (neutral text). */
export function imagePromptStyle(engine: ActiveImageEngine): ImagePromptStyle {
  if (engine?.kind !== 'local') return null;
  const family = engine.model.sdFamily;
  if (family === 'sd1') return 'tags';
  if (family === 'sd2' || family === 'sdxl' || family === 'sd3' || family === 'flux') return 'natural';
  return null;
}

const PROMPT_DESCRIPTIONS: Record<'tags' | 'natural' | 'neutral', string> = {
  tags: 'English, comma-separated tags, most important first: subject, details, style, lighting. E.g. "red fox, snowy forest, golden hour, detailed fur, photo".',
  natural: 'English, one or two plain sentences: subject, setting, style, lighting.',
  neutral: 'English description: subject, setting, style, lighting.',
};

/** The `prompt` argument description for the image model selected right now. */
export function imagePromptDescription(): string {
  let style: ImagePromptStyle = null;
  try {
    style = imagePromptStyle(imageService().activeImageEngine());
  } catch {
    // Store/service not initialised (e.g. schema preview at startup) -> neutral text.
  }
  return PROMPT_DESCRIPTIONS[style ?? 'neutral'];
}
