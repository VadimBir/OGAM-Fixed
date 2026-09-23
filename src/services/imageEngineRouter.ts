import { localDreamGeneratorService } from './localDreamGenerator';
import { sdCppGeneratorService } from './sdCppGenerator';
import { ImageGenerationParams, ImageGenerationProgress, GeneratedImage } from '../types';
import logger from '../utils/logger';

type ProgressCallback = (progress: ImageGenerationProgress) => void;
type PreviewCallback = (preview: { previewPath: string; step: number; totalSteps: number }) => void;
type LoadOpts = {
  backend?: 'mnn' | 'qnn' | 'auto' | 'sdcpp';
  cpuOnly?: boolean;
  attentionVariant?: 'split_einsum' | 'original';
  preferGpu?: boolean;
};
export type ImageEngineId = 'localdream' | 'sdcpp';

/**
 * The one image-generator seam every caller uses (manual/auto chat, tool call, retry — all funnel
 * through imageGenerationService → here). The loaded model's FORMAT picks the engine at load time:
 * backend 'sdcpp' (raw .safetensors/.ckpt checkpoint) → stable-diffusion.cpp; everything else
 * (QNN/MNN dirs, Core ML) → LocalDream, byte-for-byte the previous path. No manual toggle.
 */
class ImageEngineRouter {
  private active: ImageEngineId = 'localdream';

  getActiveEngine(): ImageEngineId {
    return this.active;
  }

  private get engine() {
    return this.active === 'sdcpp' ? sdCppGeneratorService : localDreamGeneratorService;
  }

  isAvailable(): boolean {
    return localDreamGeneratorService.isAvailable() || sdCppGeneratorService.isAvailable();
  }

  isModelLoaded(): Promise<boolean> {
    return this.engine.isModelLoaded();
  }

  getLoadedModelPath(): Promise<string | null> {
    return this.engine.getLoadedModelPath();
  }

  getLoadedThreads(): number | null {
    return this.engine.getLoadedThreads();
  }

  async loadModel(modelPath: string, threads?: number, opts: LoadOpts = {}): Promise<boolean> {
    if (opts.backend === 'sdcpp') {
      if (this.active !== 'sdcpp') await localDreamGeneratorService.unloadModel();
      this.active = 'sdcpp';
      logger.log(`[IMG-ENGINE] sdcpp <- ${modelPath}`);
      return sdCppGeneratorService.loadModel(modelPath, threads);
    }
    if (this.active === 'sdcpp') await sdCppGeneratorService.unloadModel();
    this.active = 'localdream';
    const { backend, ...rest } = opts;
    return localDreamGeneratorService.loadModel(modelPath, threads, { ...rest, backend: backend ?? 'auto' });
  }

  unloadModel(): Promise<boolean> {
    return this.engine.unloadModel();
  }

  generateImage(
    params: ImageGenerationParams & { previewInterval?: number },
    onProgress?: ProgressCallback,
    onPreview?: PreviewCallback,
  ): Promise<GeneratedImage> {
    if (this.active === 'sdcpp') return sdCppGeneratorService.generateImage(params, onProgress);
    return localDreamGeneratorService.generateImage(params, onProgress, onPreview);
  }

  cancelGeneration(): Promise<boolean> {
    return this.engine.cancelGeneration();
  }

  isGenerating(): Promise<boolean> {
    return this.engine.isGenerating();
  }

  /** sd.cpp has no OpenCL kernel cache to warm; LocalDream keeps its own check. */
  hasKernelCache(modelPath: string): Promise<boolean> {
    if (this.active === 'sdcpp') return Promise.resolve(true);
    return localDreamGeneratorService.hasKernelCache(modelPath);
  }

  // Both engines write filesDir/generated_images/<id>.png, so the store is LocalDream's.
  getGeneratedImages() {
    return localDreamGeneratorService.getGeneratedImages();
  }

  deleteGeneratedImage(imageId: string, storedImagePath?: string) {
    return localDreamGeneratorService.deleteGeneratedImage(imageId, storedImagePath);
  }

  clearOpenCLCache(modelPath: string) {
    return localDreamGeneratorService.clearOpenCLCache(modelPath);
  }

  getConstants() {
    return localDreamGeneratorService.getConstants();
  }
}

export const imageEngineRouter = new ImageEngineRouter();
