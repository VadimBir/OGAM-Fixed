/**
 * Image Studio state: image-only generation (no chat, no text model).
 *
 * The Studio calls the same public API as the chat path (imageGenerationService.generateImage,
 * useChatGenerationActions.ts) without a conversationId, so no chat message is written. Steps, size,
 * guidance and seed are NOT threaded from a render snapshot: the service reads them fresh from
 * useAppStore at generate time (imageGenerationService _generate -> resolveMobileImageParameters /
 * applyImageDefaults), the same contract the chat path documents. The text model is touched only by
 * the service's own prompt enhancement, which runs only while settings.enhanceImagePrompts is on
 * (imagePromptEnhancement.ts enhanceImagePrompt).
 *
 * Reference (img2img): initImage (base64, no data-URL prefix) + denoiseStrength = 1 - keep.
 * stable-diffusion.cpp rejects initImage (sdCppGenerator.ts SdCppUnsupportedFeatureError) and so does
 * the remote image server path (remoteImageGeneration.ts REMOTE_IMG2IMG_UNSUPPORTED), so for those
 * engines the reference is disabled and never sent.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import RNFS from 'react-native-fs';
import { launchImageLibrary } from 'react-native-image-picker';
import { useAppStore, useRemoteServerStore } from '../../stores';
import { DEFAULT_SETTINGS } from '../../stores/appStore';
import {
  imageGenerationService,
  type ImageGenerationState,
} from '../../services/imageGenerationService';
import type {
  ActiveImageEngine,
  GenerateImageParams,
} from '../../services/imageGenerationTypes';
import type { GeneratedImage } from '../../types';
import logger from '../../utils/logger';

export type StudioReference =
  | { source: 'device'; uri: string; base64: string }
  | { source: 'gallery'; uri: string; imagePath: string };

/** Most Studio results the strip remembers (newest first). */
export const STUDIO_STRIP_LIMIT = 60;
export const REFERENCE_BLOCKED_NOTE = 'Reference images need an NPU model for now';

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Keep % shown by the slider for a stored denoise strength (keep = 1 - denoise). */
export function keepPercentFromDenoise(denoise: number | undefined): number {
  const d =
    typeof denoise === 'number' && Number.isFinite(denoise)
      ? denoise
      : DEFAULT_SETTINGS.imageDenoiseStrength;
  return Math.round((1 - clamp(d, 0, 1)) * 100);
}

/** Denoise strength stored for a keep % (rounded to 2 decimals: keep 60 -> 0.4). */
export function denoiseFromKeepPercent(keepPercent: number): number {
  return Math.round((100 - clamp(keepPercent, 0, 100))) / 100;
}

/** img2img is not available on stable-diffusion.cpp models nor on the remote image server. */
export function isReferenceBlocked(engine: ActiveImageEngine): boolean {
  if (!engine) return false;
  if (engine.kind === 'remote') return true;
  return engine.model.backend === 'sdcpp';
}

function stripFileScheme(path: string): string {
  return path.startsWith('file://') ? path.slice('file://'.length) : path;
}

async function referenceBase64(ref: StudioReference): Promise<string | null> {
  if (ref.source === 'device') return ref.base64 || null;
  try {
    const data = await RNFS.readFile(stripFileScheme(ref.imagePath), 'base64');
    return data || null;
  } catch (e) {
    logger.warn('[ImageStudio] Failed to read the reference image:', e);
    return null;
  }
}

function recordStudioImage(id: string): void {
  const { settings, updateSettings } = useAppStore.getState();
  const ids = [id, ...(settings.imageStudioImageIds ?? []).filter(x => x !== id)].slice(
    0,
    STUDIO_STRIP_LIMIT,
  );
  updateSettings({ imageStudioImageIds: ids });
}

export function galleryReference(image: GeneratedImage): StudioReference {
  return {
    source: 'gallery',
    uri: `file://${stripFileScheme(image.imagePath)}`,
    imagePath: image.imagePath,
  };
}

/** The engine a request would use now, re-evaluated whenever its inputs change. */
function useActiveImageEngine(): ActiveImageEngine {
  // Subscribed only so the screen re-renders when the selection changes; the answer itself comes
  // from the service's single owner (activeImageEngine), never a second copy of its order.
  useAppStore(s => s.activeImageModelId);
  useAppStore(s => s.downloadedImageModels);
  useRemoteServerStore(s => s.activeRemoteMediaServerIds);
  useRemoteServerStore(s => s.servers);
  return imageGenerationService.activeImageEngine();
}

export function useImageStudio() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [reference, setReference] = useState<StudioReference | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [referenceError, setReferenceError] = useState<string | null>(null);
  const [genState, setGenState] = useState<ImageGenerationState>(() =>
    imageGenerationService.getState(),
  );

  useEffect(() => imageGenerationService.subscribe(setGenState), []);

  const engine = useActiveImageEngine();
  const referenceBlocked = isReferenceBlocked(engine);

  const denoise = useAppStore(s => s.settings.imageDenoiseStrength);
  const chain = useAppStore(s => s.settings.imageStudioChain ?? false);
  const studioIds = useAppStore(s => s.settings.imageStudioImageIds);
  const generatedImages = useAppStore(s => s.generatedImages);

  const studioImages = useMemo(() => {
    const byId = new Map(generatedImages.map(img => [img.id, img]));
    return (studioIds ?? [])
      .map(id => byId.get(id))
      .filter((img): img is GeneratedImage => !!img);
  }, [generatedImages, studioIds]);

  const keepPercent = keepPercentFromDenoise(denoise);
  const setKeepPercent = useCallback((keep: number) => {
    useAppStore.getState().updateSettings({ imageDenoiseStrength: denoiseFromKeepPercent(keep) });
  }, []);
  const setChain = useCallback((on: boolean) => {
    useAppStore.getState().updateSettings({ imageStudioChain: on });
  }, []);

  // A Studio request carries no conversationId; a chat request does.
  const studioInFlight = genState.isGenerating && !genState.conversationId;
  const busyElsewhere = genState.isGenerating && !!genState.conversationId;

  const pickFromDevice = useCallback(async () => {
    setReferenceError(null);
    try {
      const result = await launchImageLibrary({
        mediaType: 'photo',
        includeBase64: true,
        quality: 0.9,
        maxWidth: 1024,
        maxHeight: 1024,
      });
      const asset = result.assets?.[0];
      if (!asset) return;
      if (asset.base64 && asset.uri) {
        setReference({ source: 'device', uri: asset.uri, base64: asset.base64 });
      } else {
        setReferenceError('Could not read that image.');
      }
    } catch (e) {
      logger.warn('[ImageStudio] Device picker failed:', e);
    }
  }, []);

  const generate = useCallback(async () => {
    const text = prompt.trim();
    if (!text || imageGenerationService.getState().isGenerating) return;
    setReferenceError(null);
    const params: GenerateImageParams = { prompt: text, previewInterval: 2 };
    const negative = negativePrompt.trim();
    if (negative) params.negativePrompt = negative;
    // Re-check the engine at press time: a reference is never sent to an engine that rejects it.
    const blockedNow = isReferenceBlocked(imageGenerationService.activeImageEngine());
    if (reference && !blockedNow) {
      const base64 = await referenceBase64(reference);
      if (!base64) {
        setReferenceError('Could not read the reference image.');
        return;
      }
      params.initImage = base64;
      params.denoiseStrength = useAppStore.getState().settings.imageDenoiseStrength;
    }
    const image = await imageGenerationService.generateImage(params);
    if (!image) return;
    recordStudioImage(image.id);
    setSelectedId(image.id);
    const { settings } = useAppStore.getState();
    if (
      settings.imageStudioChain &&
      !isReferenceBlocked(imageGenerationService.activeImageEngine())
    ) {
      setReference(galleryReference(image));
    }
  }, [prompt, negativePrompt, reference]);

  const cancel = useCallback(() => {
    imageGenerationService.cancelGeneration().catch(e => {
      logger.warn('[ImageStudio] Cancel failed:', e);
    });
  }, []);

  const selectedImage =
    studioImages.find(img => img.id === selectedId) ?? studioImages[0] ?? null;

  return {
    prompt,
    setPrompt,
    negativePrompt,
    setNegativePrompt,
    engine,
    referenceBlocked,
    reference,
    setReference,
    referenceError,
    pickFromDevice,
    keepPercent,
    setKeepPercent,
    chain,
    setChain,
    genState,
    studioInFlight,
    busyElsewhere,
    studioImages,
    selectedImage,
    setSelectedId,
    generate,
    cancel,
  };
}
