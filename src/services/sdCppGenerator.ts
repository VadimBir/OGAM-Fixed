import { NativeModules, NativeEventEmitter, Platform } from 'react-native';
import {
  ImageGenerationParams,
  ImageGenerationProgress,
  GeneratedImage,
} from '../types';
import { generateRandomSeed } from '../utils/generateId';
import logger from '../utils/logger';
import { DEFAULT_IMAGE_STEPS } from '../utils/imageGenAdvice';
import { snapSdCppDimension } from './imageEngineParams';
import type { SdWeightType } from './sdCppMemory';
import {
  classifySafetensorsKeys,
  type SafetensorsClassification,
} from './sdCppModelDetection';

const { SdCppImageModule } = NativeModules;

/** Native reject code for a user cancel (SdCppImageModule.kt). */
export const SDCPP_CANCELLED_CODE = 'CANCELLED';

type ProgressCallback = (progress: ImageGenerationProgress) => void;

/**
 * Thrown (before any native call) when a request needs img2img/inpainting, which the sd.cpp JNI
 * (SdCppNative.nativeGenerate) has no parameters for. Typed so callers can show a real message
 * instead of the reference image being silently dropped and a plain txt2img being returned.
 */
export class SdCppUnsupportedFeatureError extends Error {
  readonly code = 'ERR_SDCPP_UNSUPPORTED_FEATURE';
  readonly feature: 'initImage' | 'maskImage';
  constructor(feature: 'initImage' | 'maskImage') {
    super(
      feature === 'initImage'
        ? 'This image model (stable-diffusion.cpp) cannot do image-to-image yet. Use a NPU/GPU image model, or remove the reference image.'
        : 'This image model (stable-diffusion.cpp) cannot do inpainting masks yet.',
    );
    this.name = 'SdCppUnsupportedFeatureError';
    this.feature = feature;
  }
}

/** Load-time sd.cpp options; a change requires a reload (weights are converted at load). */
export interface SdCppLoadConfig {
  weightType: SdWeightType;
  /** Text encoder + VAE only: the diffusion model always runs flash attention (sdcpp_jni.cpp). */
  flashAttn: boolean;
  /** Where module weights live between runs: unset / 'ram' = resident (upstream default),
   *  'disk' or 'te=disk,vae=disk,diffusion=disk' = reloaded per run (less RAM, slower). */
  paramsBackend?: string;
  /** LoRA merge timing: 'auto' (default) | 'immediately' | 'at_runtime'. */
  loraApplyMode?: 'auto' | 'immediately' | 'at_runtime';
  /** Direct conv2d in the VAE (no im2col buffer; may be slower). Default false. */
  vaeConvDirect?: boolean;
}

const DEFAULT_LORA_APPLY_MODE = 'auto';

/** Config fields with their defaults applied, so "unset" and "the default" compare equal. */
function normalizedLoadConfig(c: SdCppLoadConfig) {
  return {
    weightType: c.weightType,
    flashAttn: c.flashAttn,
    paramsBackend: !c.paramsBackend || c.paramsBackend === 'ram' ? '' : c.paramsBackend,
    loraApplyMode: c.loraApplyMode ?? DEFAULT_LORA_APPLY_MODE,
    vaeConvDirect: c.vaeConvDirect ?? false,
  };
}

export function sdCppLoadConfigFromSettings(settings: {
  imageSdWeightType?: SdWeightType;
  imageSdFlashAttn?: boolean;
}): SdCppLoadConfig {
  return { weightType: settings.imageSdWeightType ?? 'auto', flashAttn: settings.imageSdFlashAttn ?? true };
}

/** Sampler names stable-diffusion.cpp accepts ('' = the checkpoint's default). */
export const SDCPP_SAMPLERS = [
  'euler_a', 'euler', 'dpm++2m', 'dpm++2mv2', 'dpm++2s_a', 'dpm++2m_sde', 'heun', 'dpm2',
  'lms', 'ipndm', 'ipndm_v', 'ddim_trailing', 'tcd', 'lcm', 'res_multistep', 'er_sde',
] as const;

/** Noise schedules stable-diffusion.cpp accepts ('' = the checkpoint's default). */
export const SDCPP_SCHEDULERS = [
  'discrete', 'karras', 'exponential', 'ays', 'gits', 'sgm_uniform', 'simple', 'smoothstep',
  'kl_optimal', 'lcm', 'beta',
] as const;

/**
 * Second, isolated image engine: stable-diffusion.cpp running raw SD checkpoints directly
 * (android only). Same method surface as LocalDreamGeneratorService so imageEngineRouter can
 * swap them by model format without the callers knowing.
 */
class SdCppGeneratorService {
  private loadedThreads: number | null = null;
  private loadedConfig: SdCppLoadConfig | null = null;
  /** The run JS currently owns (null = none). cancelGeneration releases it at once so a new request
   *  can queue natively behind the cancelled run (native refuses the cancelled one by run id); a
   *  superseded run's finally must not clear the newer run's ownership. */
  private activeRun: number | null = null;
  private runCounter = 0;
  private eventEmitter: NativeEventEmitter | null = null;

  private getEmitter(): NativeEventEmitter {
    if (!this.eventEmitter) this.eventEmitter = new NativeEventEmitter(SdCppImageModule);
    return this.eventEmitter;
  }

  isAvailable(): boolean {
    return Platform.OS === 'android' && SdCppImageModule != null;
  }

  async isModelLoaded(): Promise<boolean> {
    if (!this.isAvailable()) return false;
    try {
      return await SdCppImageModule.isModelLoaded();
    } catch {
      return false;
    }
  }

  async getLoadedModelPath(): Promise<string | null> {
    if (!this.isAvailable()) return null;
    try {
      return await SdCppImageModule.getLoadedModelPath();
    } catch {
      return null;
    }
  }

  getLoadedThreads(): number | null {
    return this.loadedThreads;
  }

  /** Read a .safetensors header and classify it (SD checkpoint / SD LoRA / LLM / ...). */
  async classifySafetensors(path: string): Promise<SafetensorsClassification> {
    if (!this.isAvailable()) {
      throw new Error('stable-diffusion.cpp engine is not available on this platform');
    }
    const header: { keys: string[] } = await SdCppImageModule.readSafetensorsHeader(path);
    const result = classifySafetensorsKeys(header.keys ?? []);
    logger.log(`[SDCPP-DETECT] ${path.split('/').pop()} keys=${header.keys?.length ?? 0} -> ${result.kind}/${result.family} (${result.reason})`);
    return result;
  }

  /** True when the loaded context was created with a different weight type / flash attention. */
  loadConfigDiffers(config: SdCppLoadConfig): boolean {
    const c = this.loadedConfig;
    if (c == null) return false;
    const a = normalizedLoadConfig(c);
    const b = normalizedLoadConfig(config);
    return (
      a.weightType !== b.weightType ||
      a.flashAttn !== b.flashAttn ||
      a.paramsBackend !== b.paramsBackend ||
      a.loraApplyMode !== b.loraApplyMode ||
      a.vaeConvDirect !== b.vaeConvDirect
    );
  }

  async loadModel(modelPath: string, threads?: number, config: SdCppLoadConfig = { weightType: 'auto', flashAttn: true }): Promise<boolean> {
    if (!this.isAvailable()) {
      throw new Error('stable-diffusion.cpp image generation is not available on this platform');
    }
    const params: {
      modelPath: string; threads?: number; mmap: boolean; weightType: string; flashAttn: boolean;
      paramsBackend?: string; loraApplyMode: string; vaeConvDirect: boolean;
    } = {
      modelPath, mmap: true, weightType: config.weightType, flashAttn: config.flashAttn,
      loraApplyMode: config.loraApplyMode ?? DEFAULT_LORA_APPLY_MODE,
      vaeConvDirect: config.vaeConvDirect ?? false,
    };
    // Unset / 'ram' = resident weights: the field is left out so native keeps upstream's default.
    if (config.paramsBackend && config.paramsBackend !== 'ram') params.paramsBackend = config.paramsBackend;
    if (typeof threads === 'number') params.threads = threads;
    logger.log(`[SDCPP-LOAD] ${JSON.stringify(params)}`);
    const result = await SdCppImageModule.loadModel(params);
    this.loadedThreads = typeof threads === 'number' ? threads : this.loadedThreads;
    this.loadedConfig = { ...config };
    return result;
  }

  async unloadModel(): Promise<boolean> {
    if (!this.isAvailable()) return true;
    try {
      return await SdCppImageModule.unloadModel();
    } catch (e) {
      logger.log('[SdCpp] unloadModel failed (bridge may be torn down):', e);
      return false;
    } finally {
      this.loadedThreads = null;
      this.loadedConfig = null;
    }
  }

  private buildNativeParams(params: ImageGenerationParams, prompt: string) {
    const np = {
      prompt,
      negativePrompt: params.negativePrompt || '',
      steps: params.steps || DEFAULT_IMAGE_STEPS,
      guidanceScale: params.guidanceScale || 7,
      seed: params.seed ?? generateRandomSeed(),
      // Same 64 px grid snap the native side applies (SdCppImageModule.snapDimension), done here so
      // the size reported by getEffectiveSize / the router is exactly what native receives.
      width: snapSdCppDimension(params.width),
      height: snapSdCppDimension(params.height),
      sampler: params.sampler || '',
      scheduler: params.scheduler || '',
      clipSkip: params.clipSkip ?? -1,
      loras: (params.loras ?? []).map(l => ({ path: l.path, weight: l.weight })),
      vaeTiling: params.vaeTiling ?? true,
    };
    logger.log(`[WIRE-IMAGE-PARAMS] ${JSON.stringify({ engine: 'sdcpp', native: { ...np, prompt: undefined } })}`); // [WIRE] settings→native image params
    return np;
  }

  async generateImage(
    params: ImageGenerationParams & { previewInterval?: number },
    onProgress?: ProgressCallback,
  ): Promise<GeneratedImage> {
    if (!this.isAvailable()) {
      throw new Error('stable-diffusion.cpp image generation is not available on this platform');
    }
    if (this.activeRun !== null) throw new Error('Image generation already in progress');
    const trimmedPrompt = (params.prompt || '').trim();
    if (!trimmedPrompt) throw new Error('Cannot generate image with an empty prompt');
    if (params.initImage) throw new SdCppUnsupportedFeatureError('initImage');
    if (params.maskImage) throw new SdCppUnsupportedFeatureError('maskImage');

    const run = ++this.runCounter;
    this.activeRun = run;
    const subscription = this.getEmitter().addListener(
      'SdCppProgress',
      (event: { step: number; totalSteps: number; progress: number }) => {
        onProgress?.({ step: event.step, totalSteps: event.totalSteps, progress: event.progress });
      },
    );
    try {
      const native = this.buildNativeParams(params, trimmedPrompt);
      const result = await SdCppImageModule.generateImage(native);
      logger.log(`[WIRE-IMAGE] ${JSON.stringify(result)}`); // [WIRE] raw native generateImage result shape
      return {
        id: result.id,
        prompt: params.prompt,
        negativePrompt: params.negativePrompt,
        imagePath: result.imagePath,
        width: result.width,
        height: result.height,
        steps: native.steps,
        seed: result.seed,
        modelId: '',
        createdAt: new Date().toISOString(),
      };
    } catch (error: any) {
      // Native rejects with code CANCELLED (SdCppImageModule.generateImage) for a user cancel.
      // imageGenerationService treats an error message containing "cancelled" as a cancel, not a failure.
      if (error?.code === SDCPP_CANCELLED_CODE) {
        const cancelled = new Error('Image generation was cancelled') as Error & { code: string };
        cancelled.code = SDCPP_CANCELLED_CODE;
        throw cancelled;
      }
      throw error;
    } finally {
      if (this.activeRun === run) this.activeRun = null;
      subscription.remove();
    }
  }

  async cancelGeneration(): Promise<boolean> {
    if (!this.isAvailable()) return true;
    this.activeRun = null;
    return await SdCppImageModule.cancelGeneration();
  }

  async isGenerating(): Promise<boolean> {
    return this.activeRun !== null;
  }
}

export const sdCppGeneratorService = new SdCppGeneratorService();
