import { remoteServerManager } from './remoteServerManager';
import type { RemoteMediaModelIds, RemoteServer } from '../types';
import {
  REMOTE_FETCH_REDIRECT_POLICY,
  remoteApiUrl,
  remoteAuthorizationHeaders,
} from './remoteTransportPolicy';
import { remoteHttpErrorMessage } from './httpClient';
import { OverridableMemoryError } from './modelLoadErrors';

export interface RemoteImageResult {
  base64?: string;
  url?: string;
  /** The values this request actually carried, plus a seed the server echoed back. */
  used?: RemoteImageUsed;
}

/** Parameters the app decided for one image request. The server only executes them. */
export interface RemoteImageRequest {
  prompt: string;
  /** OpenAI-compatible `WxH`; derived from width/height when omitted. */
  size?: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  cfgScale?: number;
  seed?: number;
  sampler?: string;
  scheduler?: string;
  loras?: Array<{ name: string; weight: number }>;
  /** <= 0 means "model default" and is not sent. */
  clipSkip?: number;
}

/** What reached the server. A field is absent when the request did not carry it. */
export interface RemoteImageUsed {
  width?: number;
  height?: number;
  steps?: number;
  cfgScale?: number;
  seed?: number;
  sampler?: string;
  scheduler?: string;
  negativePrompt?: string;
}

export interface RemoteVoiceResult {
  audio: ArrayBuffer;
  contentType: string;
}

export interface RemoteMediaRequestOptions {
  signal?: AbortSignal;
  override?: boolean;
  onImageProgress?: (step: number, total: number) => void;
}

/** One image HTTP request (synchronous generate or async submit). A PC renders big images for minutes. */
const IMAGE_REQUEST_TIMEOUT_MS = 10 * 60 * 1000;
/** One Desktop status poll. */
const IMAGE_POLL_REQUEST_TIMEOUT_MS = 15 * 1000;
/** Whole Desktop async job, submit to completed. */
const IMAGE_POLL_DEADLINE_MS = 20 * 60 * 1000;
const IMAGE_POLL_INTERVAL_MS = 1000;
/** Consecutive transient poll failures tolerated before the job is reported failed. */
const IMAGE_POLL_MAX_RETRIES = 3;

const CANCELLED = 'Remote request cancelled';

type RemoteHttpError = Error & { status?: number; timedOut?: boolean };

function endpoint(server: RemoteServer, path: string): string {
  return remoteApiUrl(server.endpoint, path);
}

async function request<T>(
  input: {
    server: RemoteServer;
    path: string;
    init: RequestInit;
    signal?: AbortSignal;
    /** 0 / undefined = no timeout (transcription and voice keep their behaviour). */
    timeoutMs?: number;
  },
  consume: (response: Response) => Promise<T>,
): Promise<T> {
  const { server, path, init, signal, timeoutMs } = input;
  if (signal?.aborted) throw new Error(CANCELLED);
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  let timedOut = false;
  const timer = timeoutMs && timeoutMs > 0
    ? setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, timeoutMs)
    : undefined;
  try {
    const apiKey = await remoteServerManager.getApiKey(server.id);
    if (controller.signal.aborted) throw new Error(CANCELLED);
    const response = await fetch(endpoint(server, path), {
      ...init,
      headers: {
        Accept: 'application/json',
        ...init.headers,
        ...remoteAuthorizationHeaders(server.endpoint, apiKey),
      },
      signal: controller.signal,
      redirect: REMOTE_FETCH_REDIRECT_POLICY,
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => '');
      const message = remoteHttpErrorMessage(detail, response.status);
      try {
        const body = JSON.parse(detail) as { error?: { code?: unknown }; code?: unknown };
        const marker = 'OFFGRID_IMAGE_MEMORY_LIMIT:';
        if ((body.error?.code ?? body.code) === 'OFFGRID_IMAGE_MEMORY_LIMIT' || message.includes(marker)) {
          throw Object.assign(new OverridableMemoryError(message.replace(marker, '').trim()), { remote: true });
        }
      } catch (error) {
        if (error instanceof OverridableMemoryError) throw error;
      }
      throw Object.assign(new Error(message), { status: response.status });
    }
    // Keep caller cancellation attached until the response body is
    // consumed. A successful header is not a completed image/audio transfer.
    return await consume(response);
  } catch (error) {
    if (timedOut && !signal?.aborted) {
      throw Object.assign(new Error('Remote server did not answer in time'), { timedOut: true });
    }
    if (controller.signal.aborted) throw new Error(CANCELLED);
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}

/** Network failure, timeout or 5xx: worth polling again. A 4xx, a memory refusal or a cancel is final. */
function isTransientPollError(error: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted || !(error instanceof Error)) return false;
  if (error instanceof OverridableMemoryError) return false;
  const { status, timedOut } = error as RemoteHttpError;
  if (timedOut) return true;
  if (typeof status === 'number') return status >= 500;
  return error.message !== CANCELLED;
}

function waitForNextPoll(signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error(CANCELLED));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', abort);
      resolve();
    }, IMAGE_POLL_INTERVAL_MS);
    const abort = () => {
      clearTimeout(timer);
      reject(new Error(CANCELLED));
    };
    signal?.addEventListener('abort', abort, { once: true });
  });
}

function requiredModel(
  server: RemoteServer,
  kind: keyof RemoteMediaModelIds,
): string {
  const model = server.mediaModels?.[kind]?.trim();
  if (!model) throw new Error(`No remote ${kind} model is configured`);
  return model;
}

const positive = (value: number | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

/** The JSON fields this app decides for an image request; an absent value is not sent. */
function imageParameterBody(input: RemoteImageRequest): {
  body: Record<string, unknown>;
  used: RemoteImageUsed;
} {
  const body: Record<string, unknown> = {};
  const used: RemoteImageUsed = {};
  const negativePrompt = input.negativePrompt?.trim();
  if (negativePrompt) {
    body.negative_prompt = negativePrompt;
    used.negativePrompt = negativePrompt;
  }
  if (positive(input.width) && positive(input.height)) {
    used.width = Math.round(input.width);
    used.height = Math.round(input.height);
    body.width = used.width;
    body.height = used.height;
  }
  if (positive(input.steps)) {
    used.steps = Math.round(input.steps);
    body.steps = used.steps;
  }
  if (positive(input.cfgScale)) {
    used.cfgScale = input.cfgScale;
    body.cfg_scale = input.cfgScale;
  }
  if (typeof input.seed === 'number' && Number.isInteger(input.seed) && input.seed >= 0) {
    used.seed = input.seed;
    body.seed = input.seed;
  }
  const sampler = input.sampler?.trim();
  if (sampler) {
    used.sampler = sampler;
    body.sampler = sampler;
  }
  const scheduler = input.scheduler?.trim();
  if (scheduler) {
    used.scheduler = scheduler;
    body.scheduler = scheduler;
  }
  const loras = (input.loras ?? []).filter(
    lora => lora.name.trim() && Number.isFinite(lora.weight) && lora.weight !== 0,
  );
  if (loras.length > 0) {
    body.loras = loras.map(lora => ({ name: lora.name.trim(), weight: lora.weight }));
  }
  if (positive(input.clipSkip)) body.clip_skip = Math.round(input.clipSkip);
  return { body, used };
}

function sizeOf(input: RemoteImageRequest): string {
  if (input.size) return input.size;
  if (positive(input.width) && positive(input.height)) {
    return `${Math.round(input.width)}x${Math.round(input.height)}`;
  }
  return '1024x1024';
}

/** A seed the server reports back wins over the one sent: it is what actually rendered. */
function echoedSeed(result: { seed?: unknown; data?: Array<{ seed?: unknown }> }): number | undefined {
  const seed = result.data?.[0]?.seed ?? result.seed;
  return typeof seed === 'number' && Number.isInteger(seed) && seed >= 0 ? seed : undefined;
}

/**
 * OpenAI-compatible media adapters. The app decides every request parameter (model, size, steps,
 * seed, ...); the server only executes. The server record, stored in the app, names the endpoint.
 */
export const remoteMediaRuntime = {
  async generateImage(
    server: RemoteServer,
    input: RemoteImageRequest,
    options: RemoteMediaRequestOptions = {},
  ): Promise<RemoteImageResult> {
    if (options.signal?.aborted) throw new Error(CANCELLED);
    const openRouter = new URL(server.endpoint).hostname === 'openrouter.ai';
    const desktop = server.modelManagement === 'offgrid-desktop-v1';
    type ImagePayload = {
      data?: Array<{ b64_json?: string; url?: string; seed?: unknown }>;
      choices?: Array<{ message?: { images?: Array<{ image_url?: { url?: string } }> } }>;
      seed?: unknown;
    };
    type PollState = {
      status: string;
      result?: ImagePayload;
      error?: { message?: string };
      progress?: { step: number; total: number };
    };
    const parameters = imageParameterBody(input);
    // OpenRouter's chat route takes the prompt (and a seed) only; it records only what it sent.
    const used: RemoteImageUsed = openRouter
      ? (parameters.used.seed !== undefined ? { seed: parameters.used.seed } : {})
      : parameters.used;
    const payload = await request({
      server,
      path: openRouter ? '/v1/chat/completions' : '/v1/images/generations',
      init: {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(openRouter
          ? {
              model: requiredModel(server, 'image'),
              messages: [{ role: 'user', content: input.prompt }],
              modalities: ['image', 'text'],
              stream: false,
              ...(used.seed !== undefined ? { seed: used.seed } : {}),
            }
          : {
              model: requiredModel(server, 'image'),
              prompt: input.prompt,
              // Kept for plain OpenAI-compatible servers; width/height carry the same values.
              size: sizeOf(input),
              ...parameters.body,
              response_format: 'b64_json',
              ...(desktop ? { async: true } : {}),
              ...(options.override ? { allow_unsafe_memory_override: true } : {}),
            }),
      },
      signal: options.signal,
      timeoutMs: IMAGE_REQUEST_TIMEOUT_MS,
    }, response => response.json() as Promise<ImagePayload & { request_id?: string }>);
    let result: ImagePayload = payload;
    if (desktop && payload.request_id) {
      const deadline = Date.now() + IMAGE_POLL_DEADLINE_MS;
      let failures = 0;
      while (true) {
        if (options.signal?.aborted) throw new Error(CANCELLED);
        // No Desktop cancel endpoint is known here; a cancelled or expired job is left to the server.
        if (Date.now() > deadline) throw new Error('Remote image generation did not finish in time');
        let state: PollState;
        try {
          state = await request({
            server,
            path: `/v1/requests/${encodeURIComponent(payload.request_id)}`,
            init: { method: 'GET' },
            signal: options.signal,
            timeoutMs: IMAGE_POLL_REQUEST_TIMEOUT_MS,
          }, response => response.json() as Promise<PollState>);
          failures = 0;
        } catch (error) {
          failures += 1;
          if (!isTransientPollError(error, options.signal) || failures > IMAGE_POLL_MAX_RETRIES) throw error;
          await waitForNextPoll(options.signal);
          continue;
        }
        if (state.progress) options.onImageProgress?.(state.progress.step, state.progress.total);
        if (state.status === 'completed') {
          result = state.result ?? {};
          break;
        }
        if (state.status === 'failed') throw new Error(state.error?.message ?? 'Remote image generation failed');
        await waitForNextPoll(options.signal);
      }
    }
    const image = result.data?.[0];
    const imageUrl = image?.url ?? result.choices?.[0]?.message?.images?.[0]?.image_url?.url;
    if (!image?.b64_json && !imageUrl) throw new Error('Remote server returned no image');
    const seed = echoedSeed(result) ?? echoedSeed(payload);
    return {
      base64: image?.b64_json,
      url: imageUrl,
      used: seed !== undefined ? { ...used, seed } : used,
    };
  },

  async transcribe(
    server: RemoteServer,
    input: { fileUri: string; language?: string },
    options: RemoteMediaRequestOptions = {},
  ): Promise<string> {
    const body = new FormData();
    body.append('model', requiredModel(server, 'transcription'));
    if (input.language) body.append('language', input.language);
    body.append('file', {
      uri: input.fileUri,
      name: 'recording.wav',
      type: 'audio/wav',
    } as unknown as Blob);
    const payload = await request({
      server,
      path: '/v1/audio/transcriptions',
      init: { method: 'POST', body },
      signal: options.signal,
    }, response => response.json() as Promise<{ text?: unknown }>);
    if (typeof payload.text !== 'string') {
      throw new TypeError('Remote server returned no transcript');
    }
    return payload.text.trim();
  },

  async synthesizeVoice(
    server: RemoteServer,
    input: { text: string; voice?: string },
    options: RemoteMediaRequestOptions = {},
  ): Promise<RemoteVoiceResult> {
    const openRouter = new URL(server.endpoint).hostname === 'openrouter.ai';
    const voice = input.voice || (openRouter
      ? (await remoteMediaRuntime.listVoices(server, options))[0]
      : undefined);
    if (openRouter && !voice) throw new Error('This remote model has no available speakers.');
    return request({
      server,
      path: '/v1/audio/speech',
      init: {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: requiredModel(server, 'voice'),
          input: input.text,
          ...(voice ? { voice } : {}),
          ...(openRouter ? { response_format: 'mp3' } : {}),
        }),
      },
      signal: options.signal,
    }, async response => ({
      audio: await response.arrayBuffer(),
      contentType: response.headers.get('content-type') ?? 'audio/mpeg',
    }));
  },

  async listVoices(server: RemoteServer, options: RemoteMediaRequestOptions = {}): Promise<string[]> {
    try {
      const modelId = requiredModel(server, 'voice');
      const catalog = await request({
        server,
        path: '/v1/models?output_modalities=speech',
        init: { method: 'GET' },
        signal: options.signal,
      }, response => response.json() as Promise<{
        data?: Array<{ id?: string; supported_voices?: unknown; voices?: unknown }>;
      }>);
      const model = catalog.data?.find(entry => entry.id === modelId);
      const listed = model?.supported_voices ?? model?.voices;
      if (Array.isArray(listed)) {
        return listed.filter((voice): voice is string => typeof voice === 'string');
      }
    } catch (error) {
      if (options.signal?.aborted) throw error;
    }
    const payload = await request({
      server,
      path: '/v1/audio/voices',
      init: { method: 'GET' },
      signal: options.signal,
    }, response => response.json() as Promise<{ voices?: unknown }>);
    return Array.isArray(payload.voices)
      ? payload.voices.filter((voice): voice is string => typeof voice === 'string')
      : [];
  },
};
