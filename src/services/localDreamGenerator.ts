import { NativeModules, NativeEventEmitter, Platform } from 'react-native';
import RNFS from 'react-native-fs';
import {
  ImageGenerationParams,
  ImageGenerationProgress,
  GeneratedImage,
} from '../types';
import { generateRandomSeed } from '../utils/generateId';
import logger from '../utils/logger';
import { resolveOwnedDocumentPath } from '../utils/resolveDocumentPath';
import { DEFAULT_IMAGE_STEPS } from '../utils/imageGenAdvice';
import type { EffectiveImageSize } from './imageEngineParams';

const { LocalDreamModule, CoreMLDiffusionModule } = NativeModules;
const PROGRESS_LOG_SAMPLE_STEPS = 5;

// Pick the right native module per platform
const DiffusionModule = Platform.select({
  ios: CoreMLDiffusionModule,
  android: LocalDreamModule,
  default: null,
});

type ProgressCallback = (progress: ImageGenerationProgress) => void;

/** Native LocalDreamStatus event (LocalDreamModule.kt EVENT_STATUS): server lifecycle changes JS did not ask for. */
export interface LocalDreamStatusEvent {
  status: 'restarting' | 'ready' | 'failed' | 'fallback';
  message: string | null;
}
type StatusListener = (event: LocalDreamStatusEvent) => void;
type PreviewCallback = (preview: { previewPath: string; step: number; totalSteps: number }) => void;

/**
 * LocalDream-based image generator service.
 * Replaces ONNX Runtime with local-dream's subprocess HTTP server.
 *
 * The native module (LocalDreamModule) manages:
 * - Server process lifecycle (spawn/kill)
 * - HTTP POST + SSE parsing for image generation
 * - RGB→PNG conversion and file management
 *
 * Progress events are emitted via NativeEventEmitter from the native side.
 */
class LocalDreamGeneratorService {
  private loadedThreads: number | null = null;
  /** The run JS currently owns (null = none). cancelGeneration releases it at once so a new request
   *  can queue natively behind the cancelled run (native refuses the cancelled one by run id); a
   *  superseded run's finally must not clear the newer run's ownership. */
  private activeRun: number | null = null;
  private runCounter = 0;
  private eventEmitter: NativeEventEmitter | null = null;
  private statusSubscription: { remove(): void } | null = null;
  private statusListeners = new Set<StatusListener>();
  private lastStatus: LocalDreamStatusEvent | null = null;

  /**
   * Persistent (not per-generation): the native server restart after a cancel emits "restarting" /
   * "ready" AFTER generateImage has already rejected, so a per-generation listener would miss it.
   * Android only; the iOS Core ML module has no such event.
   */
  private ensureStatusListener(): void {
    if (this.statusSubscription || Platform.OS !== 'android' || !this.isAvailable()) return;
    this.statusSubscription = this.getEmitter().addListener('LocalDreamStatus', (event: LocalDreamStatusEvent) => {
      this.lastStatus = event;
      logger.log(`[LocalDream] status=${event?.status}${event?.message ? `: ${event.message}` : ''}`);
      this.statusListeners.forEach(l => {
        try { l(event); } catch (e) { logger.warn('[LocalDream] status listener threw', e); }
      });
    });
  }

  /** Subscribe to server lifecycle status (restarting|ready|failed|fallback). Returns unsubscribe. */
  subscribeToStatus(listener: StatusListener): () => void {
    this.ensureStatusListener();
    this.statusListeners.add(listener);
    return () => { this.statusListeners.delete(listener); };
  }

  getLastStatus(): LocalDreamStatusEvent | null {
    return this.lastStatus;
  }

  private getEmitter(): NativeEventEmitter {
    if (!this.eventEmitter) {
      this.eventEmitter = new NativeEventEmitter(DiffusionModule);
    }
    return this.eventEmitter;
  }

  isAvailable(): boolean {
    return DiffusionModule != null;
  }

  async isModelLoaded(): Promise<boolean> {
    if (!this.isAvailable()) return false;
    try {
      return await DiffusionModule.isModelLoaded();
    } catch {
      return false;
    }
  }

  async getLoadedModelPath(): Promise<string | null> {
    if (!this.isAvailable()) return null;
    try {
      return await DiffusionModule.getLoadedModelPath();
    } catch {
      return null;
    }
  }

  async loadModel(modelPath: string, threads?: number, opts: { backend?: 'mnn' | 'qnn' | 'auto'; cpuOnly?: boolean; attentionVariant?: 'split_einsum' | 'original'; preferGpu?: boolean } = {}): Promise<boolean> {
    if (!this.isAvailable()) {
      throw new Error('LocalDream image generation is not available on this platform');
    }

    const backend = opts.backend ?? 'auto';
    const params: { modelPath: string; threads?: number; backend: string; cpuOnly?: boolean; attentionVariant?: string; preferGpu?: boolean } = {
      modelPath,
      backend,
    };
    if (typeof threads === 'number') {
      params.threads = threads;
    }
    if (opts.cpuOnly) {
      params.cpuOnly = true;
    }
    if (opts.attentionVariant) {
      params.attentionVariant = opts.attentionVariant;
    }
    // iOS Core ML only: select the compute path (GPU vs Neural Engine) the
    // residency estimate was sized for, so the native load matches the budget.
    if (typeof opts.preferGpu === 'boolean') {
      params.preferGpu = opts.preferGpu;
    }

    this.ensureStatusListener();
    const result = await DiffusionModule.loadModel(params);
    this.loadedThreads = typeof threads === 'number' ? threads : this.loadedThreads;
    return result;
  }

  getLoadedThreads(): number | null {
    return this.loadedThreads;
  }

  async unloadModel(): Promise<boolean> {
    if (!this.isAvailable()) return true;
    try {
      const result = await DiffusionModule.unloadModel();
      this.loadedThreads = null;
      return result;
    } catch (e) {
      logger.log('[LocalDream] unloadModel failed (bridge may be torn down):', e);
      this.loadedThreads = null;
      return false;
    }
  }

  private subscribeToProgress(onProgress?: ProgressCallback, onPreview?: PreviewCallback): any {
    return this.getEmitter().addListener(
      'LocalDreamProgress',
      (event: { step: number; totalSteps: number; progress: number; previewPath?: string }) => {
        if (
          !Number.isInteger(event.step) ||
          event.step < 1 ||
          event.step === 1 ||
          event.step === event.totalSteps ||
          event.step % PROGRESS_LOG_SAMPLE_STEPS === 0
        ) {
          logger.log(`[WIRE-IMAGE-PROGRESS] ${JSON.stringify(event)}`); // [WIRE] raw LocalDreamProgress event shape
        }
        onProgress?.({
          step: event.step,
          totalSteps: event.totalSteps,
          progress: event.progress,
        });
        if (event.previewPath && onPreview) {
          onPreview({ previewPath: event.previewPath, step: event.step, totalSteps: event.totalSteps });
        }
      },
    );
  }

  private buildNativeParams(params: ImageGenerationParams & { previewInterval?: number }, prompt: string) {
    const np = {
      prompt,
      negativePrompt: params.negativePrompt || '',
      steps: params.steps || DEFAULT_IMAGE_STEPS,
      guidanceScale: params.guidanceScale || 7.5,
      seed: params.seed ?? generateRandomSeed(),
      // KNOWN ISSUE: sizes < 512 render as rainbow noise on the native engine (MNN CPU/GPU still
      // OPEN; NPU guarded natively). See docs/notes/native-image-sub512-rainbow.md.
      width: params.width || 512,
      height: params.height || 512,
      previewInterval: params.previewInterval ?? 2,
      // Default OFF: the on-device GPU VAE only decodes cleanly at 512x512; at smaller sizes it
      // shears into a rainbow. Native sends use_opencl=true only at exactly 512x512 (see
      // LocalDreamModule.effectiveUseOpenCL); getEffectiveSize() reports that decision up front.
      useOpenCL: params.useOpenCL ?? false,
      // Sampler: the core only knows 'dpm' (default) and 'euler_a'; anything else is not sent.
      ...(params.sampler === 'dpm' || params.sampler === 'euler_a' ? { scheduler: params.sampler } : {}),
      // img2img: only attach when a reference image is present, so a plain request stays txt2img.
      ...(params.initImage
        ? {
            initImage: params.initImage,
            denoiseStrength: params.denoiseStrength ?? 0.6,
            ...(params.maskImage ? { maskImage: params.maskImage } : {}),
          }
        : {}),
    };
    logger.log(`[WIRE-IMAGE-PARAMS] ${JSON.stringify({ requested: { steps: params.steps, guidanceScale: params.guidanceScale, width: params.width, height: params.height }, native: { ...np, prompt: undefined } })}`); // [WIRE] settings→native image params
    return np;
  }

  /**
   * What this request will REALLY render, computed natively before generating (no restart):
   * NPU without a resolution patch -> 512x512, MNN -> square 128..512, OpenCL only at 512x512.
   * Without a loaded model, on iOS, or on an older native build the request is echoed unadjusted.
   */
  async getEffectiveSize(
    params: { width?: number; height?: number; useOpenCL?: boolean },
  ): Promise<EffectiveImageSize> {
    const requestedWidth = params.width || 512;
    const requestedHeight = params.height || 512;
    const echo: EffectiveImageSize = {
      engine: 'localdream',
      requestedWidth,
      requestedHeight,
      width: requestedWidth,
      height: requestedHeight,
      adjusted: false,
      reason: null,
      useOpenCL: null,
    };
    if (!this.isAvailable() || typeof DiffusionModule.getEffectiveImageSize !== 'function') return echo;
    try {
      const r = await DiffusionModule.getEffectiveImageSize({
        width: requestedWidth,
        height: requestedHeight,
        useOpenCL: params.useOpenCL ?? false,
      });
      return {
        ...echo,
        requestedWidth: r.requestedWidth ?? requestedWidth,
        requestedHeight: r.requestedHeight ?? requestedHeight,
        width: r.width ?? requestedWidth,
        height: r.height ?? requestedHeight,
        adjusted: !!r.adjusted,
        reason: r.reason ?? null,
        useOpenCL: typeof r.useOpenCL === 'boolean' ? r.useOpenCL : null,
      };
    } catch (e) {
      logger.log('[LocalDream] getEffectiveImageSize failed, echoing request:', e);
      return echo;
    }
  }

  private buildResult(params: ImageGenerationParams, result: any): GeneratedImage {
    logger.log(`[WIRE-IMAGE] ${JSON.stringify(result)}`); // [WIRE] raw native generateImage result shape from-device
    if (result?.width !== (params.width || 512) || result?.height !== (params.height || 512)) {
      logger.log(`[IMG-SIZE] requested ${params.width || 512}x${params.height || 512}, rendered ${result?.width}x${result?.height}, useOpenCL=${result?.useOpenCL}`);
    }
    return {
      id: result.id,
      prompt: params.prompt,
      negativePrompt: params.negativePrompt,
      imagePath: result.imagePath,
      width: result.width,
      height: result.height,
      steps: params.steps || DEFAULT_IMAGE_STEPS,
      seed: result.seed,
      modelId: '',
      // ISO-8601, the one form every consumer of this field can read - the gallery's date, and the
      // sync descriptor that a peer validates with Date.parse. Epoch milliseconds as text passes the
      // `string` type and fails every reader.
      createdAt: new Date().toISOString(),
    };
  }

  async generateImage(
    params: ImageGenerationParams & { previewInterval?: number },
    onProgress?: ProgressCallback,
    onPreview?: PreviewCallback,
  ): Promise<GeneratedImage> {
    if (!this.isAvailable()) {
      throw new Error('LocalDream image generation is not available on this platform');
    }
    if (this.activeRun !== null) {
      throw new Error('Image generation already in progress');
    }
    const trimmedPrompt = (params.prompt || '').trim();
    if (!trimmedPrompt) {
      throw new Error('Cannot generate image with an empty prompt');
    }

    const run = ++this.runCounter;
    this.activeRun = run;
    this.ensureStatusListener();
    const progressSubscription = this.subscribeToProgress(onProgress, onPreview);

    try {
      const result = await DiffusionModule.generateImage(this.buildNativeParams(params, trimmedPrompt));
      // Native side releases the CoreML pipeline after generation to free
      // memory, so clear TS-side state so the next request triggers a reload.
      this.loadedThreads = null;
      return this.buildResult(params, result);
    } catch (error: any) {
      const msg = error?.message || '';
      if (msg.includes('ERR_NO_MODEL') || msg.includes('unloaded') || msg.includes('Pipeline failed')) {
        this.loadedThreads = null;
      }
      throw error;
    } finally {
      if (this.activeRun === run) this.activeRun = null;
      progressSubscription?.remove();
    }
  }

  async cancelGeneration(): Promise<boolean> {
    if (!this.isAvailable()) return true;
    this.activeRun = null;
    return await DiffusionModule.cancelGeneration();
  }

  async isGenerating(): Promise<boolean> {
    return this.activeRun !== null;
  }

  async getGeneratedImages(): Promise<GeneratedImage[]> {
    if (!this.isAvailable()) return [];
    try {
      const images = await DiffusionModule.getGeneratedImages();
      return images.map((img: any) => ({
        id: img.id,
        prompt: img.prompt || '',
        imagePath: img.imagePath,
        width: img.width || 512,
        height: img.height || 512,
        steps: img.steps || 20,
        seed: img.seed || 0,
        modelId: img.modelId || '',
        createdAt: img.createdAt,
      }));
    } catch {
      return [];
    }
  }

  async deleteGeneratedImage(imageId: string, storedImagePath?: string): Promise<boolean> {
    if (!this.isAvailable()) return false;

    if (storedImagePath !== undefined) {
      const generatedImageDirectory = `${RNFS.DocumentDirectoryPath}/generated_images`;
      const resolvedPath = resolveOwnedDocumentPath(
        storedImagePath,
        generatedImageDirectory,
      );
      const expectedPath = `${generatedImageDirectory}/${imageId}.png`;
      if (
        !resolvedPath ||
        resolvedPath !== expectedPath ||
        !imageId ||
        imageId.includes('/') ||
        imageId.includes('\0')
      ) {
        return false;
      }

      try {
        if (!(await RNFS.exists(resolvedPath))) return true;
      } catch {
        // Let the native store decide when the filesystem check is unavailable.
      }
    }

    return await DiffusionModule.deleteGeneratedImage(imageId);
  }

  async clearOpenCLCache(modelPath: string): Promise<number> {
    if (Platform.OS !== 'android' || !this.isAvailable()) return 0;
    return await DiffusionModule.clearOpenCLCache(modelPath);
  }

  async hasKernelCache(modelPath: string): Promise<boolean> {
    if (Platform.OS !== 'android' || !this.isAvailable()) return true;
    return await DiffusionModule.hasOpenCLCache(modelPath);
  }

  getConstants() {
    if (!this.isAvailable()) {
      return {
        DEFAULT_STEPS: DEFAULT_IMAGE_STEPS,
        DEFAULT_GUIDANCE_SCALE: 7.5,
        DEFAULT_WIDTH: 512,
        DEFAULT_HEIGHT: 512,
        SUPPORTED_WIDTHS: [128, 192, 256, 320, 384, 448, 512],
        SUPPORTED_HEIGHTS: [128, 192, 256, 320, 384, 448, 512],
      };
    }
    const __c = DiffusionModule.getConstants();
    logger.log(`[WIRE-IMAGE-CONSTANTS] ${JSON.stringify(__c)}`); // [WIRE] raw native diffusion constants (steps/guidance/supported sizes)
    return __c;
  }
}

export const localDreamGeneratorService = new LocalDreamGeneratorService();
