import { NativeModules, NativeEventEmitter, Platform } from 'react-native';
import {
  ImageGenerationParams,
  ImageGenerationProgress,
  GeneratedImage,
} from '../types';
import { generateRandomSeed } from '../utils/generateId';
import logger from '../utils/logger';
import {
  classifySafetensorsKeys,
  type SafetensorsClassification,
} from './sdCppModelDetection';

const { SdCppImageModule } = NativeModules;

type ProgressCallback = (progress: ImageGenerationProgress) => void;

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
  private generating = false;
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

  async loadModel(modelPath: string, threads?: number): Promise<boolean> {
    if (!this.isAvailable()) {
      throw new Error('stable-diffusion.cpp image generation is not available on this platform');
    }
    const params: { modelPath: string; threads?: number; mmap: boolean } = { modelPath, mmap: true };
    if (typeof threads === 'number') params.threads = threads;
    logger.log(`[SDCPP-LOAD] ${JSON.stringify(params)}`);
    const result = await SdCppImageModule.loadModel(params);
    this.loadedThreads = typeof threads === 'number' ? threads : this.loadedThreads;
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
    }
  }

  private buildNativeParams(params: ImageGenerationParams, prompt: string) {
    const np = {
      prompt,
      negativePrompt: params.negativePrompt || '',
      steps: params.steps || 20,
      guidanceScale: params.guidanceScale || 7,
      seed: params.seed ?? generateRandomSeed(),
      width: params.width || 512,
      height: params.height || 512,
      sampler: params.sampler || '',
      scheduler: params.scheduler || '',
      clipSkip: params.clipSkip ?? -1,
      loras: (params.loras ?? []).map(l => ({ path: l.path, weight: l.weight })),
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
    if (this.generating) throw new Error('Image generation already in progress');
    const trimmedPrompt = (params.prompt || '').trim();
    if (!trimmedPrompt) throw new Error('Cannot generate image with an empty prompt');

    this.generating = true;
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
    } finally {
      this.generating = false;
      subscription.remove();
    }
  }

  async cancelGeneration(): Promise<boolean> {
    if (!this.isAvailable()) return true;
    this.generating = false;
    return await SdCppImageModule.cancelGeneration();
  }

  async isGenerating(): Promise<boolean> {
    return this.generating;
  }
}

export const sdCppGeneratorService = new SdCppGeneratorService();
