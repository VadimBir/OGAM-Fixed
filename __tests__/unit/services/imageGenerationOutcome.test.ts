/**
 * imageGenerationService.generateImageWithOutcome: every terminal path reports a typed reason
 * (no_model / load_failed / out_of_memory / busy / cancelled / engine_error) or the values used.
 *
 * Drives the REAL service; only the native generator, the model loaders, the LLM and the failure
 * card are mocked (same boundaries as hardening/batch4-phase-state-machine).
 * Falsify: let any null path fall through untyped -> its reason assertion goes red.
 */
import { useAppStore } from '../../../src/stores/appStore';
import { imageGenerationService } from '../../../src/services/imageGenerationService';
import { localDreamGeneratorService } from '../../../src/services/localDreamGenerator';
import { activeModelService } from '../../../src/services/activeModelService';
import { llmService } from '../../../src/services/llm';
import { resetStores, flushPromises } from '../../utils/testHelpers';
import { createONNXImageModel } from '../../utils/factories';

jest.mock('../../../src/services/localDreamGenerator');
jest.mock('../../../src/services/activeModelService');
jest.mock('../../../src/services/llm');
jest.mock('../../../src/services/modelFailureHandler', () => ({ reportModelFailure: jest.fn() }));

const mockDream = localDreamGeneratorService as jest.Mocked<typeof localDreamGeneratorService>;
const mockActive = activeModelService as jest.Mocked<typeof activeModelService>;
const mockLlm = llmService as jest.Mocked<typeof llmService>;

const OK_RESULT = {
  id: 'r1', prompt: 'p', imagePath: '/mock/out.png', width: 256, height: 256,
  steps: 8, seed: 77, modelId: 'img-1', createdAt: new Date().toISOString(),
};

function setupModel() {
  const model = createONNXImageModel({ id: 'img-1', modelPath: '/mock/img-model' });
  useAppStore.setState({
    downloadedImageModels: [model],
    activeImageModelId: 'img-1',
    generatedImages: [],
    warmedImageModels: ['img-1'],
    settings: {
      imageSteps: 8, imageGuidanceScale: 2, imageWidth: 256, imageHeight: 256, imageThreads: 4,
      enhanceImagePrompts: false,
    } as any,
  });
  return model;
}

beforeEach(async () => {
  resetStores();
  jest.clearAllMocks();
  mockDream.isModelLoaded.mockResolvedValue(true);
  mockDream.getLoadedModelPath.mockResolvedValue('/mock/img-model');
  mockDream.getLoadedThreads.mockReturnValue(4);
  mockDream.isAvailable.mockReturnValue(true);
  mockDream.generateImage.mockResolvedValue(OK_RESULT as any);
  mockDream.cancelGeneration.mockResolvedValue(true);
  mockActive.loadImageModel.mockResolvedValue();
  mockLlm.isModelLoaded.mockReturnValue(false);
  mockLlm.isCurrentlyGenerating.mockReturnValue(false);
  await imageGenerationService.cancelGeneration().catch(() => {});
});

describe('activeImageEngine', () => {
  it('is null without a selected downloaded model, local with one', () => {
    useAppStore.setState({ downloadedImageModels: [], activeImageModelId: null });
    expect(imageGenerationService.activeImageEngine()).toBeNull();
    setupModel();
    expect(imageGenerationService.activeImageEngine()).toMatchObject({ kind: 'local', model: { id: 'img-1' } });
  });
});

describe('generateImageWithOutcome', () => {
  it('success reports the values the engine used', async () => {
    setupModel();
    const outcome = await imageGenerationService.generateImageWithOutcome({ prompt: 'apple' });
    expect(outcome.ok).toBe(true);
    if (!outcome.ok) return;
    expect(outcome.used).toMatchObject({ engine: 'local', width: 256, height: 256, steps: 8, seed: 77 });
    expect(typeof outcome.used.guidance).toBe('number');
  });

  it('no selected model -> no_model', async () => {
    useAppStore.setState({ downloadedImageModels: [], activeImageModelId: null });
    const outcome = await imageGenerationService.generateImageWithOutcome({ prompt: 'x' });
    expect(outcome).toMatchObject({ ok: false, reason: 'no_model' });
  });

  it('model load error -> load_failed; memory load error -> out_of_memory', async () => {
    setupModel();
    mockDream.isModelLoaded.mockResolvedValue(false);
    mockActive.loadImageModel.mockRejectedValueOnce(new Error('bad file header'));
    expect(await imageGenerationService.generateImageWithOutcome({ prompt: 'x' })).toMatchObject({
      ok: false, reason: 'load_failed',
    });
    mockActive.loadImageModel.mockRejectedValueOnce(new Error('Insufficient memory to load the image model'));
    expect(await imageGenerationService.generateImageWithOutcome({ prompt: 'x' })).toMatchObject({
      ok: false, reason: 'out_of_memory',
    });
  });

  it('engine error -> engine_error, even behind the friendly pipeline-crash message', async () => {
    setupModel();
    mockDream.generateImage.mockRejectedValueOnce(new Error('Pipeline failed: unet'));
    expect(await imageGenerationService.generateImageWithOutcome({ prompt: 'x' })).toMatchObject({
      ok: false, reason: 'engine_error',
    });
    mockDream.generateImage.mockRejectedValueOnce(new Error('Pipeline failed: out of memory'));
    expect(await imageGenerationService.generateImageWithOutcome({ prompt: 'x' })).toMatchObject({
      ok: false, reason: 'out_of_memory',
    });
  });

  it('sd.cpp unsupported img2img/inpaint -> engine_error with a clear message', async () => {
    setupModel();
    const err: any = new Error('This image model (stable-diffusion.cpp) cannot do image-to-image yet.');
    err.code = 'ERR_SDCPP_UNSUPPORTED_FEATURE';
    mockDream.generateImage.mockRejectedValueOnce(err);
    expect(await imageGenerationService.generateImageWithOutcome({ prompt: 'x' })).toMatchObject({
      ok: false,
      reason: 'engine_error',
      message: 'img2img/inpaint not supported by this engine',
    });
  });

  it('a second request while one is in flight -> busy; cancel -> cancelled', async () => {
    setupModel();
    let finish!: (v: any) => void;
    mockDream.generateImage.mockImplementation(async () => new Promise(r => { finish = r; }));
    const first = imageGenerationService.generateImageWithOutcome({ prompt: 'first' });
    await flushPromises();
    expect(await imageGenerationService.generateImageWithOutcome({ prompt: 'second' })).toMatchObject({
      ok: false, reason: 'busy',
    });
    await imageGenerationService.cancelGeneration();
    finish(OK_RESULT);
    expect(await first).toMatchObject({ ok: false, reason: 'cancelled' });
  });

  it('generateImage keeps its GeneratedImage | null contract', async () => {
    setupModel();
    expect(await imageGenerationService.generateImage({ prompt: 'x' })).toMatchObject({ imagePath: '/mock/out.png' });
    useAppStore.setState({ downloadedImageModels: [], activeImageModelId: null });
    expect(await imageGenerationService.generateImage({ prompt: 'x' })).toBeNull();
  });
});
