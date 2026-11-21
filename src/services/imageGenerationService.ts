import { Alert } from 'react-native';
import { imageEngineRouter as onnxImageGeneratorService } from './imageEngineRouter';
import { sdCppLoadConfigFromSettings } from './sdCppGenerator';
import { activeModelService } from './activeModelService';
import { modelResidencyManager } from './modelResidency';
import { useAppStore } from '../stores';
import { GeneratedImage } from '../types';
import logger from '../utils/logger';
import { generateId } from '../utils/generateId';
import { useRemoteServerStore } from '../stores/remoteServerStore';
import { runRemoteImageGeneration } from './remoteImageGeneration';
import { applyImageDefaults, resolveMobileImageParameters } from './imageParameterPolicy';
import { resolveEngineImageRequest } from './imageEngineParams';
import {
  generationProgressStatus,
  imagePhaseTransitionLog,
} from './imageGenerationHelpers';
import { enhanceImagePrompt } from './imagePromptEnhancement';
import {
  completedImageGenerationState,
  saveImageGenerationResult,
} from './imageGenerationResult';
import type { LocalDreamStatusEvent } from './localDreamGenerator';
import { reportModelFailure } from './modelFailureHandler';
import { reasonFromLoadError } from './modelFailureReasons';
import { isOverridableMemoryError } from './modelLoadErrors';
import {
  isInFlight,
  ImageGenerationState,
  ImageGenerationListener,
  GenerateImageParams,
  ActiveImageModel,
  RunGenerationOptions,
  ActiveImageEngine,
  ImageGenFailureReason,
  ImageGenUsedParams,
  ImageGenerationOutcome,
} from './imageGenerationTypes';

export { isInFlight } from './imageGenerationTypes';
export type {
  ImageGenPhase,
  ImageGenerationState,
  ImageGenerationOutcome,
  ImageGenFailureReason,
} from './imageGenerationTypes';

/** SdCppUnsupportedFeatureError.code (sdCppGenerator.ts); matched by code so a mocked module still works. */
const SDCPP_UNSUPPORTED_FEATURE_CODE = 'ERR_SDCPP_UNSUPPORTED_FEATURE';
const UNSUPPORTED_IMG2IMG_MESSAGE =
  'img2img/inpaint not supported by this engine';

/** Per-request record of how a generate call ended, filled by every terminal path. */
interface ImageRequestTrack {
  failure: { reason: ImageGenFailureReason; message: string } | null;
  used: Pick<ImageGenUsedParams, 'engine' | 'guidance' | 'sampler' | 'steps' | 'seed'> | null;
  /** Set by cancelGeneration. Per request, not per service: after a cancel the phase is idle at
   *  once, so a new request can start while the cancelled one is still unwinding (enhance, load,
   *  native generate); a shared flag would be reset by the new request and the old run would go on
   *  to drive the new run's state. */
  cancelled: boolean;
}

/** Memory pressure wins over the call site's own reason: the same wall shows up as a load error
 *  or as an engine error, and the model must be told "free memory", not "retry". */
function imageFailureReason(
  message: string,
  cause: unknown,
  fallback: ImageGenFailureReason,
): ImageGenFailureReason {
  if (
    isOverridableMemoryError(cause) ||
    reasonFromLoadError(message) === 'insufficient-memory' ||
    (cause !== undefined && reasonFromLoadError(cause) === 'insufficient-memory')
  ) {
    return 'out_of_memory';
  }
  return fallback;
}

class ImageGenerationService {
  // The ONLY stored state is `phase` (+ the data fields). `isGenerating` is NOT
  // stored — there's no second source to desync. It's computed from phase in
  // getState() (see below) for back-compat readers.
  private state: Omit<ImageGenerationState, 'isGenerating'> = {
    phase: 'idle',
    progress: null,
    status: null,
    previewPath: null,
    prompt: null,
    conversationId: null,
    messageId: null,
    error: null,
    result: null,
  };

  private readonly listeners: Set<ImageGenerationListener> = new Set();
  /** Unsubscribe for the image engine's own lifecycle status (LocalDream server restarts). */
  private engineStatusUnsubscribe: (() => void) | null = null;
  private remoteRequest: AbortController | null = null;
  /** Last generate request, so a failure card's Retry button can re-run it. */
  private _lastParams: GenerateImageParams | null = null;
  /** The in-flight request's outcome record (one request at a time: the busy guard). */
  private request: ImageRequestTrack | null = null;

  /** Public snapshot: isGenerating is computed from phase, never stored. */
  getState(): ImageGenerationState {
    return { ...this.state, isGenerating: isInFlight(this.state.phase) };
  }

  isGeneratingFor(conversationId: string): boolean {
    return (
      isInFlight(this.state.phase) &&
      this.state.conversationId === conversationId
    );
  }

  subscribe(listener: ImageGenerationListener): () => void {
    this.listeners.add(listener);
    listener(this.getState());
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    const state = this.getState();
    this.listeners.forEach(listener => listener(state));
  }

  private updateState(partial: Partial<ImageGenerationState>): void {
    // Strip any derived field a caller might pass — phase is the only stored truth.
    const { isGenerating: _ignored, ...rest } = partial;
    const prevPhase = this.state.phase;
    this.state = { ...this.state, ...rest };
    // [IMG-SM] state-machine trace (kept forever, like [TTS-SM]): every phase
    // transition logs one line so one repro reads as a linear state machine and a
    // silent stall/flash is never undiagnosable again.
    if ('phase' in partial && this.state.phase !== prevPhase) {
      logger.log(imagePhaseTransitionLog(prevPhase, this.state));
    }
    this.notifyListeners();
  }

  /** Own the terminal error state and its retry actions. */
  private _fail(
    error: string,
    opts?: { cause?: unknown; remote?: boolean; reason?: ImageGenFailureReason },
  ): null {
    if (this.request) {
      this.request.failure = {
        reason:
          opts?.reason ??
          imageFailureReason(
            error,
            opts?.cause,
            /no (remote )?image model/i.test(error) ? 'no_model' : 'engine_error',
          ),
        message: error,
      };
    }
    this.updateState({
      phase: 'error',
      progress: null,
      status: null,
      previewPath: null,
      error,
    });
    // Surface the FULL error (incl. native server diagnostics: backend, exit code, recent
    // server log) in a popup so a crash is diagnosable on-device instead of a generic card line.
    // Guarded so a benign cancellation never pops.
    if (!/cancel/i.test(error)) {
      try {
        Alert.alert('Image generation error', error, [{ text: 'OK' }]);
      } catch {
        /* Alert unavailable (non-UI context) — the error state already carries the message. */
      }
    }
    // On a memory-pressure failure the card offers "Free memory & Retry" — so the retry
    // must ACTUALLY free memory (eject resident models) before re-running, not just
    // re-run into the same wall. Derive memory-pressure from the SAME single source the
    // card's label uses (reasonFromLoadError) so the label and the eject can never
    // disagree — no second regex to drift.
    const memoryPressure = reasonFromLoadError(error) === 'insufficient-memory';
    const onRetry = this._lastParams
      ? async () => {
          if (memoryPressure && !opts?.remote)
            await activeModelService.ejectAll().catch(() => {});
          await this.generateImage(this._lastParams as GenerateImageParams);
        }
      : undefined;
    // Load Anyway: only when the cause is the overridable memory gate. Re-run the last
    // request forcing the image-model load past the budget (override evicts every
    // evictable resident, then loads). reportModelFailure ignores onLoadAnyway unless
    // the cause is actually overridable, so passing it here is safe for other errors.
    const onLoadAnyway =
      isOverridableMemoryError(opts?.cause) && this._lastParams
        ? async () => {
            await this.generateImage(this._lastParams as GenerateImageParams, {
              override: true,
            });
          }
        : undefined;
    // Report with the typed cause (not the wrapped string) so the overridable
    // discriminant survives; `message` keeps the user-facing wrapped text.
    reportModelFailure('image', opts?.cause ?? error, {
      message: error,
      title: opts?.remote ? 'Remote image error' : undefined,
      remote: opts?.remote,
      onRetry,
      onLoadAnyway,
    });
    return null;
  }

  private _setEnhancementState(
    params: GenerateImageParams,
    status: string,
  ): void {
    this.updateState({
      phase: 'enhancing',
      prompt: params.prompt,
      conversationId: params.conversationId || null,
      status,
      previewPath: null,
      progress: null,
      error: null,
      result: null,
    });
  }

  private async _enhancePrompt(
    params: GenerateImageParams,
  ): Promise<string> {
    return enhanceImagePrompt(params, status =>
      this._setEnhancementState(params, status),
    );
  }

  private async _ensureImageModelLoaded(
    activeImageModelId: string | null,
    activeImageModel: ActiveImageModel,
    opts: { desiredThreads: number; override?: boolean },
    request: ImageRequestTrack,
  ): Promise<boolean> {
    const isImageModelLoaded = await onnxImageGeneratorService.isModelLoaded();
    const loadedPath = await onnxImageGeneratorService.getLoadedModelPath();
    const loadedThreads = onnxImageGeneratorService.getLoadedThreads();
    const needsThreadReload =
      loadedThreads == null || loadedThreads !== opts.desiredThreads ||
      onnxImageGeneratorService.sdCppConfigDiffers(
        sdCppLoadConfigFromSettings(useAppStore.getState().settings ?? {}),
      );
    if (
      isImageModelLoaded &&
      loadedPath === activeImageModel.modelPath &&
      !needsThreadReload
    )
      return true;
    if (!activeImageModelId) {
      this._fail('No image model selected');
      return false;
    }
    try {
      this.updateState({
        phase: 'loading',
        status: `Loading ${activeImageModel.name}...`,
      });
      await activeModelService.loadImageModel(
        activeImageModelId,
        undefined,
        opts.override ? { override: true } : undefined,
      );
      return true;
    } catch (error: any) {
      // Cancelled while loading: end quietly, never fail a newer request's state.
      if (request.cancelled) return false;
      // Pass the TYPED error as `cause` — an OverridableMemoryError here is what lets
      // the failure card offer "Load Anyway". Stringifying it (as before) hid it.
      const message = `Failed to load image model: ${error?.message || 'Unknown error'}`;
      this._fail(message, {
        cause: error,
        reason: imageFailureReason(message, error, 'load_failed'),
      });
      return false;
    }
  }

  /** A cancelled request ends quietly. cancelGeneration already reset the state; reset it again
   *  only while no newer request owns it (a superseded run must not reset the new run's card). */
  private _endCancelled(request: ImageRequestTrack, message = 'Image generation was cancelled.'): null {
    if (!request.failure) request.failure = { reason: 'cancelled', message };
    if (this.request === request) this.resetState();
    return null;
  }

  /** Engine lifecycle status (LocalDream: server restart after a cancel, NPU resolution-patch
   *  restart, CPU fallback) shown in the in-flight card. Subscribed lazily through the engine
   *  router (no new import edge, nothing at module load). */
  private _ensureEngineStatusSubscription(): void {
    if (this.engineStatusUnsubscribe) return;
    try {
      const unsubscribe = onnxImageGeneratorService.subscribeToStatus(event =>
        this._onEngineStatus(event),
      );
      if (typeof unsubscribe === 'function') this.engineStatusUnsubscribe = unsubscribe;
    } catch (e) {
      logger.warn('[ImageGen] Engine status subscription failed:', e);
    }
  }

  private _onEngineStatus(event: LocalDreamStatusEvent | undefined): void {
    if (!event || !isInFlight(this.state.phase) || this.request?.cancelled) return;
    const status =
      event.status === 'ready'
        ? 'Image engine ready, generating...'
        : event.message ||
          (event.status === 'restarting'
            ? 'Restarting the image engine...'
            : `Image engine: ${event.status}`);
    this.updateState({ status });
  }

  /** Ask the loaded engine what it will really render (LocalDream: NPU 512 or a resolution-patch
   *  size, MNN square 128-512 on a 64 px grid; sd.cpp: 64 px grid), so the size sent is the size
   *  rendered. Any failure keeps the request. */
  private async _planEngineSize(
    width: number,
    height: number,
    useOpenCL: boolean,
  ): Promise<{ width: number; height: number }> {
    try {
      const plan = await onnxImageGeneratorService.getEffectiveSize({ width, height, useOpenCL });
      if (
        plan &&
        Number.isFinite(plan.width) && plan.width > 0 &&
        Number.isFinite(plan.height) && plan.height > 0
      ) {
        if (plan.adjusted) {
          logger.log(`[IMG-SIZE] ${width}x${height} -> ${plan.width}x${plan.height}: ${plan.reason}`);
        }
        return { width: plan.width, height: plan.height };
      }
    } catch (e) {
      logger.warn('[ImageGen] Effective size plan failed, sending the request size:', e);
    }
    return { width, height };
  }

  private async _runGenerationAndSave(
    opts: RunGenerationOptions,
    request: ImageRequestTrack,
  ): Promise<GeneratedImage | null> {
    const {
      params,
      enhancedPrompt,
      activeImageModel,
      steps,
      guidanceScale,
      imageWidth,
      imageHeight,
      useOpenCL,
      engineOptions,
    } = opts;

    // The first generation for a model compiles/warms the backend and takes ~120s.
    // This is platform-agnostic: on iOS the CoreML model compiles on first use, on
    // Android the OpenCL kernels compile. The persisted `warmedImageModels` flag is
    // the single cross-platform signal (so the notice shows once on every device);
    // the OpenCL kernel-cache check is an extra Android signal in case the cache was
    // cleared after the flag was set.
    let isFirstRun = !useAppStore
      .getState()
      .warmedImageModels.includes(activeImageModel.id);
    if (useOpenCL) {
      try {
        const hasCache = await onnxImageGeneratorService.hasKernelCache(
          activeImageModel.modelPath,
        );
        isFirstRun = isFirstRun || !hasCache;
      } catch (e) {
        // If check fails, don't add a false first-run signal (keep the warmed-flag result).
        logger.warn('[ImageGen] Failed to check for OpenCL kernel cache:', e);
      }
    }

    this.updateState({
      phase: 'generating',
      status: isFirstRun
        ? 'Optimizing GPU for your device (~120s, one-time)...'
        : 'Starting image generation...',
    });
    const startTime = Date.now();
    try {
      const result = await onnxImageGeneratorService.generateImage(
        {
          prompt: enhancedPrompt,
          negativePrompt: params.negativePrompt || '',
          steps,
          guidanceScale,
          seed: params.seed,
          width: imageWidth,
          height: imageHeight,
          previewInterval: params.previewInterval ?? 2,
          useOpenCL,
          // Format-selected engine knobs (sampler/scheduler/LoRA) resolved by resolveEngineImageRequest.
          ...engineOptions,
          // img2img: forward a reference image + the configurable denoise strength from settings
          // (per-request override wins). Dormant for plain txt2img (no initImage) — the native
          // body only becomes img2img when `image` is present (LocalDreamModule.kt).
          ...(params.initImage
            ? {
                initImage: params.initImage,
                denoiseStrength:
                  params.denoiseStrength ??
                  useAppStore.getState().settings.imageDenoiseStrength,
                ...(params.maskImage ? { maskImage: params.maskImage } : {}),
              }
            : {}),
        },
        progress => {
          if (request.cancelled) return;
          const displayStep = Math.min(progress.step, steps);
          // Once steps are advancing it IS generating — don't mislabel it "GPU
          // optimization" (which read as if generation hadn't started). On the first run
          // the GPU is still warming, so note that as a one-time aside, not the headline.
          const status = generationProgressStatus(
            displayStep,
            steps,
            isFirstRun,
          );
          this.updateState({
            progress: { step: displayStep, totalSteps: steps },
            status,
          });
        },
        preview => {
          if (request.cancelled) return;
          const displayStep = Math.min(preview.step, steps);
          this.updateState({
            previewPath: `file://${preview.previewPath}?t=${Date.now()}`,
            status: `Refining image (${displayStep}/${steps})...`,
          });
        },
      );
      if (request.cancelled) return this._endCancelled(request);
      if (!result?.imagePath) {
        this.resetState();
        return null;
      }
      this.updateState(completedImageGenerationState(result));
      activeModelService.noteImageGenerationCompleted?.(); // weights now resident (M4 gate sizing)
      // Android: hand the multi-GB idle image working set back when live free RAM is low.
      modelResidencyManager.releaseIdleImageIfMemoryLow().catch(err => {
        logger.warn('[ImageGenerationService] Post-generation image release failed:', err);
      });
      return saveImageGenerationResult(result, {
        params,
        activeImageModel,
        messageId: this.state.messageId,
        steps,
        guidanceScale,
        useOpenCL,
        startTime,
      });
    } catch (error: any) {
      const errorMsg = error?.message || 'Image generation failed';
      // A cancelled request's error is the cancel (e.g. LocalDream's dropped connection), whatever
      // its text: it must not pop an error or fail a newer request.
      if (request.cancelled || errorMsg.includes('cancelled')) {
        this._endCancelled(request, errorMsg);
      } else if (error?.code === SDCPP_UNSUPPORTED_FEATURE_CODE) {
        // sd.cpp has no img2img/inpaint parameters (SdCppUnsupportedFeatureError). Not a crash:
        // say so plainly instead of the generic engine-failure path.
        logger.warn('[ImageGenerationService] Unsupported sd.cpp feature:', error);
        this._fail(UNSUPPORTED_IMG2IMG_MESSAGE, {
          cause: error,
          reason: 'engine_error',
        });
      } else {
        logger.error('[ImageGenerationService] Generation error:', error);

        // If the pipeline crashed or the model was unloaded, surface a
        // user-friendly message and allow retry (model will auto-reload).
        const isPipelineCrash =
          errorMsg.includes('Pipeline failed') ||
          errorMsg.includes('unloaded') ||
          errorMsg.includes('ERR_NO_MODEL') ||
          errorMsg.includes('TextEncoder');
        const userMessage = isPipelineCrash
          ? 'Image generation failed — the model encountered an error and was unloaded. Please try again.'
          : errorMsg;

        // Classify from the RAW engine message: the friendly pipeline-crash text drops the memory cue.
        this._fail(userMessage, {
          reason: imageFailureReason(errorMsg, error, 'engine_error'),
        });
      }
      return null;
    }
  }

  /**
   * The engine a generate request would run on right now, in generateImage's own order: a remote
   * image server with a model, else the selected downloaded local model. The single availability
   * answer (the generate_image tool is offered only when this is non-null).
   */
  activeImageEngine(): ActiveImageEngine {
    const remoteServer = useRemoteServerStore
      .getState()
      .getActiveRemoteMediaServer('image');
    if (remoteServer?.mediaModels?.image) {
      return { kind: 'remote', server: remoteServer };
    }
    const { activeImageModelId, downloadedImageModels } =
      useAppStore.getState();
    const model = downloadedImageModels.find(m => m.id === activeImageModelId);
    return model ? { kind: 'local', model } : null;
  }

  /**
   * Generate an image. Runs independently of UI lifecycle.
   * If conversationId is provided, the result will be added as a chat message.
   */
  async generateImage(
    params: GenerateImageParams,
    opts?: { override?: boolean },
  ): Promise<GeneratedImage | null> {
    const outcome = await this.generateImageWithOutcome(params, opts);
    return outcome.ok ? outcome.image : null;
  }

  /**
   * generateImage with a typed result: the image plus the values actually used, or the failure
   * reason. Callers that must report back (the generate_image tool) use this; UI callers keep
   * generateImage. The UI side effects (state, failure card) are identical for both.
   */
  async generateImageWithOutcome(
    params: GenerateImageParams,
    opts?: { override?: boolean },
  ): Promise<ImageGenerationOutcome> {
    if (isInFlight(this.state.phase)) {
      logger.log(
        '[ImageGenerationService] Already generating, ignoring request',
      );
      return {
        ok: false,
        reason: 'busy',
        message: 'Another image is still being generated.',
      };
    }
    const request: ImageRequestTrack = { failure: null, used: null, cancelled: false };
    this.request = request;
    try {
      const image = await this._generate(params, opts, request);
      if (image) {
        const local = request.used?.engine !== 'remote';
        return {
          ok: true,
          image,
          used: {
            engine: local ? 'local' : 'remote',
            width: image.width,
            height: image.height,
            // Local: what the engine ran. Remote: only what the server was really sent / echoed
            // (onUsed); the image's steps / seed are app fallbacks for a public server, not used.
            steps: local ? image.steps : request.used?.steps,
            seed: local ? image.seed : request.used?.seed,
            guidance: request.used?.guidance,
            sampler: request.used?.sampler,
          },
        };
      }
      if (request.failure) return { ok: false, ...request.failure };
      return request.cancelled
        ? { ok: false, reason: 'cancelled', message: 'Image generation was cancelled.' }
        : { ok: false, reason: 'engine_error', message: 'The image engine returned no image.' };
    } finally {
      if (this.request === request) this.request = null;
    }
  }

  private async _generate(
    requested: GenerateImageParams,
    opts: { override?: boolean } | undefined,
    request: ImageRequestTrack,
  ): Promise<GeneratedImage | null> {
    // The user's default negative prompt / fixed seed fill a request that names none; one place for
    // every engine (remote, sd.cpp, LocalDream, Core ML) and every entry point.
    const params = applyImageDefaults(requested, useAppStore.getState().settings);
    this._lastParams = params; // so a failure card's Retry can re-run this exact request
    const engine = this.activeImageEngine();
    if (engine?.kind === 'remote') {
      request.used = { engine: 'remote' };
      const remoteServer = engine.server;
      const enhancedPrompt = await this._enhancePrompt(params);
      if (request.cancelled) return this._endCancelled(request);
      return runRemoteImageGeneration(params, remoteServer, {
        updateState: state => this.updateState(state),
        fail: (message, cause) => this._fail(message, { cause, remote: true }),
        isCancelled: () => request.cancelled,
        setRequest: controller => {
          this.remoteRequest = controller;
        },
        // What the server was really sent (or echoed), per server kind (remoteMediaRuntime).
        onUsed: used => {
          request.used = {
            engine: 'remote', steps: used.steps, seed: used.seed, guidance: used.cfgScale, sampler: used.sampler,
          };
        },
      }, { ...opts, enhancedPrompt });
    }
    if (!engine) return this._fail('No image model selected');
    const activeImageModel = engine.model;
    const { settings, activeImageModelId } = useAppStore.getState();

    const messageId = params.conversationId ? generateId() : null;

    const imageParameters = resolveMobileImageParameters(
      activeImageModel,
      settings,
      params,
    );
    const { steps, guidanceScale } = imageParameters;
    const {
      width: imageWidth,
      height: imageHeight,
      engineOptions,
    } = resolveEngineImageRequest(
      activeImageModel,
      settings,
      params,
      imageParameters.size,
    );
    request.used = {
      engine: 'local',
      guidance: guidanceScale,
      sampler: engineOptions.sampler,
    };

    this.updateState({
      phase: settings.enhanceImagePrompts ? 'enhancing' : 'loading',
      prompt: params.prompt,
      conversationId: params.conversationId || null,
      messageId,
      status: settings.enhanceImagePrompts
        ? 'Preparing prompt enhancement...'
        : 'Preparing image generation...',
      previewPath: null,
      progress: null,
      error: null,
      result: null,
    });

    const enhancedPrompt = await this._enhancePrompt(params);
    logger.log(
      '[ImageGen] enhanceImagePrompts setting:',
      settings.enhanceImagePrompts,
    );
    // Stop can arrive while prompt enhancement owns the text engine. Do not clear that request and
    // continue into the image model after the user already pressed X.
    if (request.cancelled) return this._endCancelled(request);

    // Establish the generating state unconditionally — not only when enhancement
    // is off. When enhancement is ON but _enhancePrompt bailed early (e.g. no text
    // model loaded, so enhancement was skipped), it never set isGenerating, so the
    // in-progress card never appeared. Setting it here fixes that; on the
    // enhancement-ran path this just swaps the 'Enhancing…' status for 'Preparing…'
    // before the image model loads.
    this.updateState({
      phase: 'loading',
      prompt: params.prompt,
      conversationId: params.conversationId || null,
      status: 'Preparing image generation...',
      previewPath: null,
      progress: null,
      error: null,
      result: null,
    });

    this._ensureEngineStatusSubscription();
    // M4: the text model loaded BEFORE the image work (prompt enhancement above still used it). The
    // image load may evict it and suspendTextForImageGeneration may unload it; either way it is
    // reloaded once this request ends (restoreTextAfterImageGeneration honours the loading mode).
    const textBefore = activeModelService.getLoadedModelIds?.()?.textModelId ?? null;
    try {
      const loaded = await this._ensureImageModelLoaded(
        activeImageModelId,
        activeImageModel,
        { desiredThreads: settings.imageThreads ?? 4, override: opts?.override },
        request,
      );
      if (request.cancelled) return this._endCancelled(request);
      if (!loaded) return null;
      // M4: the LLM must not compete for RAM with the generation working set.
      await Promise.resolve(activeModelService.suspendTextForImageGeneration?.()).catch(err => {
        logger.warn('[ImageGenerationService] M4 text suspend failed:', err);
      });
      if (request.cancelled) return this._endCancelled(request);

      const useOpenCL = settings.imageUseOpenCL ?? false;
      // The loaded backend decides what it can render (NPU patch sizes, MNN square): plan after load.
      const planned = await this._planEngineSize(imageWidth, imageHeight, useOpenCL);
      if (request.cancelled) return this._endCancelled(request);

      return await this._runGenerationAndSave({
        params,
        enhancedPrompt,
        activeImageModel,
        steps,
        guidanceScale,
        imageWidth: planned.width,
        imageHeight: planned.height,
        useOpenCL,
        engineOptions,
      }, request);
    } finally {
      await activeModelService.restoreTextAfterImageGeneration?.(textBefore);
    }
  }

  async cancelGeneration(): Promise<void> {
    if (!isInFlight(this.state.phase)) return;
    if (this.request) this.request.cancelled = true;
    this.remoteRequest?.abort();
    // Publish the terminal while conversation identity is still present. Sync subscribers run
    // synchronously, so every peer can remove its live image card before native cancellation waits.
    this.updateState({
      phase: 'cancelled',
      progress: null,
      status: null,
      previewPath: null,
      error: null,
    });
    try {
      await onnxImageGeneratorService.cancelGeneration();
    } catch {
      /* Ignore */
    } finally {
      this.resetState();
    }
  }

  private resetState(): void {
    this.updateState({
      phase: 'idle',
      progress: null,
      status: null,
      previewPath: null,
      prompt: null,
      conversationId: null,
      messageId: null,
      error: null,
      // Keep result so the last generated image is still accessible
    });
  }
}

export const imageGenerationService = new ImageGenerationService();
