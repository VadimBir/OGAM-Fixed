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
import { reportModelFailure } from '../../../src/services/modelFailureHandler';

jest.mock('../../../src/services/localDreamGenerator');
jest.mock('../../../src/services/activeModelService');
jest.mock('../../../src/services/llm');
jest.mock('../../../src/services/modelFailureHandler', () => ({ reportModelFailure: jest.fn() }));
const mockRunRemote = jest.fn();
jest.mock('../../../src/services/remoteImageGeneration', () => ({
  runRemoteImageGeneration: (...args: any[]) => mockRunRemote(...args),
}));

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

describe('cancel isolation: a cancelled run never drives the next request', () => {
  // Falsify: one service-wide cancel flag reset by the new request -> the old run's late error
  // fails / resets the new request.
  it('the cancelled run failing late does not fail or reset the request started after the cancel', async () => {
    setupModel();
    let rejectFirst!: (e: unknown) => void;
    let resolveSecond!: (v: any) => void;
    mockDream.generateImage
      .mockImplementationOnce(async () => new Promise((_, rej) => { rejectFirst = rej; }))
      .mockImplementationOnce(async () => new Promise(res => { resolveSecond = res; }));
    const first = imageGenerationService.generateImageWithOutcome({ prompt: 'first' });
    await flushPromises();
    await imageGenerationService.cancelGeneration();
    const second = imageGenerationService.generateImageWithOutcome({ prompt: 'second' });
    await flushPromises();
    expect(imageGenerationService.getState().phase).toBe('generating');

    // LocalDream rejects a disconnected run with a non-"cancelled" text.
    rejectFirst(new Error('Failed to generate image: [SocketException] Socket closed'));
    expect(await first).toMatchObject({ ok: false, reason: 'cancelled' });
    expect(imageGenerationService.getState()).toMatchObject({ phase: 'generating', prompt: 'second' });
    expect(reportModelFailure).not.toHaveBeenCalled();

    resolveSecond(OK_RESULT);
    expect(await second).toMatchObject({ ok: true });
  });
});

describe('size planning and engine status', () => {
  // Falsify: skip the planner -> native gets 320x320 although the NPU renders 512x512.
  it('sends the size the loaded engine plans for the tool request', async () => {
    setupModel();
    (mockDream.getEffectiveSize as jest.Mock).mockResolvedValueOnce({
      engine: 'localdream', requestedWidth: 320, requestedHeight: 320, width: 512, height: 512,
      adjusted: true, reason: 'NPU model has no 320x320 resolution patch', useOpenCL: false,
    });
    await imageGenerationService.generateImageWithOutcome({ prompt: 'x', width: 320, height: 320 });
    expect(mockDream.getEffectiveSize).toHaveBeenCalledWith(expect.objectContaining({ width: 320, height: 320 }));
    expect(mockDream.generateImage).toHaveBeenCalledWith(
      expect.objectContaining({ width: 512, height: 512 }), expect.any(Function), expect.any(Function),
    );
  });

  it('a non-square tool request reaches LocalDream when the planner keeps it', async () => {
    setupModel();
    (mockDream.getEffectiveSize as jest.Mock).mockResolvedValueOnce({
      engine: 'localdream', width: 256, height: 512, adjusted: false, reason: null, useOpenCL: false,
    });
    await imageGenerationService.generateImageWithOutcome({ prompt: 'x', width: 256, height: 512 });
    expect(mockDream.generateImage).toHaveBeenCalledWith(
      expect.objectContaining({ width: 256, height: 512 }), expect.any(Function), expect.any(Function),
    );
  });

  it('a planner failure keeps the requested size', async () => {
    setupModel();
    (mockDream.getEffectiveSize as jest.Mock).mockRejectedValueOnce(new Error('boom'));
    await imageGenerationService.generateImageWithOutcome({ prompt: 'x', width: 192, height: 64 });
    expect(mockDream.generateImage).toHaveBeenCalledWith(
      expect.objectContaining({ width: 192, height: 64 }), expect.any(Function), expect.any(Function),
    );
  });

  // Falsify: drop the subscription -> the NPU restart status never reaches the card.
  it('LocalDream lifecycle status shows in the in-flight state, never when idle', async () => {
    setupModel();
    let statusListener: ((e: any) => void) | undefined;
    (mockDream.subscribeToStatus as jest.Mock).mockImplementation((l: (e: any) => void) => {
      statusListener = l;
      return jest.fn();
    });
    let finish!: (v: any) => void;
    mockDream.generateImage.mockImplementationOnce(async () => new Promise(r => { finish = r; }));
    const run = imageGenerationService.generateImageWithOutcome({ prompt: 'x' });
    await flushPromises();
    expect(statusListener).toBeDefined();
    statusListener!({ status: 'restarting', message: 'Reloading the NPU model for 320x320' });
    expect(imageGenerationService.getState().status).toBe('Reloading the NPU model for 320x320');
    statusListener!({ status: 'ready', message: null });
    expect(imageGenerationService.getState().status).toBe('Image engine ready, generating...');
    finish(OK_RESULT);
    await run;
    statusListener!({ status: 'restarting', message: 'Stopping the cancelled generation' });
    expect(imageGenerationService.getState().status).not.toBe('Stopping the cancelled generation');
  });
});

describe('remote outcome reports only what the server was sent', () => {
  it('steps / seed / guidance / sampler come from onUsed, not from the image fallbacks', async () => {
    const spy = jest.spyOn(imageGenerationService, 'activeImageEngine')
      .mockReturnValue({ kind: 'remote', server: { id: 's', name: 'PC' } } as any);
    mockRunRemote.mockImplementationOnce(async (_p: any, _s: any, deps: any) => {
      deps.onUsed({ width: 384, height: 256, steps: 12, seed: 5, cfgScale: 6, sampler: 'euler' });
      return { ...OK_RESULT, width: 384, height: 256, steps: 12, seed: 5 };
    });
    expect(await imageGenerationService.generateImageWithOutcome({ prompt: 'x' })).toMatchObject({
      ok: true, used: { engine: 'remote', width: 384, height: 256, steps: 12, seed: 5, guidance: 6, sampler: 'euler' },
    });
    // A public server is sent no steps / seed: the image carries app fallbacks, which are not reported.
    mockRunRemote.mockImplementationOnce(async (_p: any, _s: any, deps: any) => {
      deps.onUsed({ width: 512, height: 512 });
      return { ...OK_RESULT, width: 512, height: 512, steps: 20, seed: 123 };
    });
    const outcome = await imageGenerationService.generateImageWithOutcome({ prompt: 'x' });
    expect(outcome.ok && outcome.used).toMatchObject({ engine: 'remote', width: 512, height: 512 });
    if (outcome.ok) {
      expect(outcome.used.steps).toBeUndefined();
      expect(outcome.used.seed).toBeUndefined();
    }
    spy.mockRestore();
  });
});
