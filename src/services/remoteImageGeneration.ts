import RNFS from 'react-native-fs';
import type { GeneratedImage, RemoteServer } from '../types';
import { useAppStore } from '../stores';
import { generateId, generateRandomSeed } from '../utils/generateId';
import type { GenerateImageParams, ImageGenerationState } from './imageGenerationTypes';
import { resolveMobileImageParameters } from './imageParameterPolicy';
import { clipSkipOf } from './imageEngineParams';
import { remoteMediaRuntime, type RemoteImageUsed } from './remoteMediaRuntime';
import { remoteServerManager } from './remoteServerManager';
import { remoteAuthorizationHeaders, validateRemoteEndpoint } from './remoteTransportPolicy';
import {
  completedImageGenerationState,
  saveImageGenerationResult,
} from './imageGenerationResult';

interface RemoteImageGenerationDeps {
  updateState: (state: Partial<ImageGenerationState>) => void;
  fail: (message: string, cause?: unknown) => null;
  isCancelled: () => boolean;
  setRequest: (controller: AbortController | null) => void;
  /** Optional: the values the server was actually sent (or echoed), for the caller's report. */
  onUsed?: (used: RemoteImageUsed) => void;
}

/** Remote img2img / inpaint is not wired to any server contract; refuse instead of a silent txt2img. */
export const REMOTE_IMG2IMG_UNSUPPORTED =
  'The remote image server takes text-to-image only. The reference image was not sent, so no image was made.';

const positiveFinite = (value: number | null | undefined): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined;

/** Resolve a server-returned image URL (absolute or relative) and refuse a public cleartext host. */
function resolveImageUrl(server: RemoteServer, url: string): URL {
  // A data: URL reaching here did not match the base64 PNG / JPEG / WebP form read above.
  const dataUrl = /^\s*data:([^;,]*)/i.exec(url);
  if (dataUrl) {
    throw new Error(
      `Remote server returned an image this app cannot read (${dataUrl[1] || 'unknown type'}). Expected base64 PNG, JPEG, or WebP.`,
    );
  }
  const resolved = new URL(url, server.endpoint);
  validateRemoteEndpoint(resolved.toString());
  return resolved;
}

/** Bearer only to the server's own origin, and only over TLS (remoteAuthorizationHeaders). */
async function downloadHeaders(server: RemoteServer, url: URL): Promise<Record<string, string>> {
  if (url.origin !== new URL(server.endpoint).origin) return {};
  const apiKey = await remoteServerManager.getApiKey(server.id);
  return remoteAuthorizationHeaders(url.toString(), apiKey);
}

export async function runRemoteImageGeneration(
  params: GenerateImageParams,
  server: RemoteServer,
  deps: RemoteImageGenerationDeps,
  options: { override?: boolean; enhancedPrompt?: string } = {},
): Promise<GeneratedImage | null> {
  const modelId = server.mediaModels?.image;
  if (!modelId) return deps.fail('No remote image model is configured');
  if (params.initImage || params.maskImage) return deps.fail(REMOTE_IMG2IMG_UNSUPPORTED);
  const settings = useAppStore.getState().settings;
  // steps / guidance come from the model policy. Its size is NOT used: that one is square and floored
  // to the on-device sweet spot, and a PC server renders the exact W x H the app asks for.
  const imageParameters = resolveMobileImageParameters(
    { id: modelId, name: modelId }, settings, params,
  );
  const requestedWidth = positiveFinite(params.width) ?? positiveFinite(settings.imageWidth);
  const requestedHeight = positiveFinite(params.height) ?? positiveFinite(settings.imageHeight);
  const width = Math.round(requestedWidth ?? requestedHeight ?? imageParameters.size);
  const height = Math.round(requestedHeight ?? requestedWidth ?? imageParameters.size);
  const { steps, guidanceScale } = imageParameters;
  const seed = params.seed ?? generateRandomSeed();
  const sampler = (params.sampler ?? settings.imageSampler ?? '').trim() || undefined;
  const scheduler = (params.scheduler ?? settings.imageScheduler ?? '').trim() || undefined;
  const loras = (settings.imageLoras ?? [])
    .filter(lora => lora.enabled && lora.name && Number.isFinite(lora.weight) && lora.weight !== 0)
    .map(lora => ({ name: lora.name, weight: lora.weight }));
  const messageId = params.conversationId ? generateId() : null;
  const startTime = Date.now();
  deps.updateState({
    phase: 'generating', prompt: params.prompt, conversationId: params.conversationId || null,
    messageId, status: `Creating image on ${server.name}...`, previewPath: null,
    progress: null, error: null, result: null,
  });
  const controller = new AbortController();
  deps.setRequest(controller);
  let imagePath: string | null = null;
  let downloadJobId: number | null = null;
  const stopDownload = () => {
    if (downloadJobId !== null) RNFS.stopDownload(downloadJobId);
  };
  controller.signal.addEventListener('abort', stopDownload, { once: true });
  const cancelled = () => controller.signal.aborted || deps.isCancelled();
  const discardPartial = async () => {
    if (imagePath) await RNFS.unlink(imagePath).catch(() => undefined);
  };
  try {
    const remote = await remoteMediaRuntime.generateImage(
      server,
      {
        prompt: options.enhancedPrompt ?? params.prompt,
        size: `${width}x${height}`,
        width,
        height,
        negativePrompt: params.negativePrompt,
        steps,
        cfgScale: guidanceScale,
        seed,
        sampler,
        scheduler,
        loras,
        clipSkip: clipSkipOf(settings),
      },
      {
        signal: controller.signal,
        override: options.override,
        onImageProgress: (step, total) => {
          if (!deps.isCancelled() && total > 0) {
            deps.updateState({
              progress: { step, totalSteps: total },
              status: `Generating image (${step}/${total})...`,
            });
          }
        },
      },
    );
    const used = remote.used ?? {};
    const dataUrl = remote.url?.match(/^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$/i);
    const base64 = dataUrl?.[2] ?? remote.base64;
    const downloadUrl = base64 || !remote.url ? null : resolveImageUrl(server, remote.url);
    if (!base64 && !downloadUrl) throw new Error('Remote server returned no image data');
    if (cancelled()) return null;
    const id = generateId();
    const directory = `${RNFS.DocumentDirectoryPath}/generated_images`;
    const urlExtension = downloadUrl?.pathname.match(/\.(png|jpe?g|webp)$/i)?.[1]?.toLowerCase();
    const extension = dataUrl?.[1]?.toLowerCase() === 'image/jpeg'
      ? 'jpg'
      : dataUrl?.[1]?.toLowerCase().replace('image/', '') ?? urlExtension ?? 'png';
    let fileName = `${id}.${extension}`;
    imagePath = `${directory}/${fileName}`;
    await RNFS.mkdir(directory);
    if (base64) {
      await RNFS.writeFile(imagePath, base64, 'base64');
    } else if (downloadUrl) {
      const headers = await downloadHeaders(server, downloadUrl);
      if (cancelled()) {
        await discardPartial();
        return null;
      }
      const transfer = RNFS.downloadFile({ fromUrl: downloadUrl.toString(), toFile: imagePath, headers });
      downloadJobId = transfer.jobId;
      const outcome = await transfer.promise;
      if (cancelled()) {
        await discardPartial();
        return null;
      }
      if (outcome.statusCode < 200 || outcome.statusCode >= 300) {
        throw new Error(`Image download returned HTTP ${outcome.statusCode}`);
      }
      const contentType = Object.entries(outcome.headers ?? {})
        .find(([key]) => key.toLowerCase() === 'content-type')?.[1]
        ?.split(';')[0]?.toLowerCase();
      const receivedExtension = contentType === 'image/jpeg' ? 'jpg'
        : contentType === 'image/png' ? 'png'
        : contentType === 'image/webp' ? 'webp'
        : undefined;
      if (contentType && !receivedExtension) {
        throw new Error('Remote server returned an unsupported image format');
      }
      if (receivedExtension && receivedExtension !== extension) {
        fileName = `${id}.${receivedExtension}`;
        const correctedPath = `${directory}/${fileName}`;
        await RNFS.moveFile(imagePath, correctedPath);
        imagePath = correctedPath;
      }
    }
    if (cancelled()) {
      await discardPartial();
      return null;
    }
    deps.onUsed?.(used);
    // Record what the server was sent (or echoed). On the user's own server the seed and steps are
    // sent; a public standard server is sent neither, so the app values below are a fallback only
    // (GeneratedImage.seed / steps are required numbers with no "unknown" value).
    const result: GeneratedImage = {
      id, prompt: params.prompt, negativePrompt: used.negativePrompt ?? params.negativePrompt, imagePath, fileName,
      width: used.width ?? width, height: used.height ?? height, steps: used.steps ?? steps,
      seed: used.seed ?? seed, modelId, createdAt: new Date().toISOString(),
    };
    deps.updateState(completedImageGenerationState(result));
    return saveImageGenerationResult(result, {
      params,
      activeImageModel: {
        id: modelId, name: `${server.name} / ${modelId}`, modelPath: server.endpoint, backend: 'remote',
      },
      messageId, steps: used.steps ?? steps, guidanceScale: used.cfgScale ?? guidanceScale,
      useOpenCL: false, startTime, isRemote: true,
    });
  } catch (error) {
    if (cancelled()) {
      await discardPartial();
      return null;
    }
    await discardPartial();
    return deps.fail(error instanceof Error ? error.message : 'Remote image generation failed', error);
  } finally {
    controller.signal.removeEventListener('abort', stopDownload);
    deps.setRequest(null);
  }
}
