export {};

/**
 * Image engine wiring: use_opencl reaches native, one default step count, effective size known up
 * front, sd.cpp honours 64..512 independent W/H and refuses img2img loudly.
 *
 * Priority: P0 - a dropped setting or a silent size/feature substitution looks like a broken engine.
 */

const mockLocalDreamModule = {
  generateImage: jest.fn(),
  cancelGeneration: jest.fn(),
  getConstants: jest.fn(),
  getEffectiveImageSize: jest.fn(),
};
const mockSdCppModule = {
  generateImage: jest.fn(),
};
const mockAddListener = jest.fn().mockReturnValue({ remove: jest.fn() });

jest.mock('react-native', () => ({
  NativeModules: {
    LocalDreamModule: mockLocalDreamModule,
    CoreMLDiffusionModule: null,
    SdCppImageModule: mockSdCppModule,
  },
  NativeEventEmitter: jest.fn().mockImplementation(() => ({
    addListener: mockAddListener,
    removeAllListeners: jest.fn(),
  })),
  Platform: { OS: 'android', select: (opts: any) => opts.android ?? opts.default },
}));

// Native LocalDreamModule.DEFAULT_STEPS / SdCppImageModule fallback (Kotlin, both literally 20).
const NATIVE_DEFAULT_STEPS = 20;

const nativeResult = { id: 'i', imagePath: '/p.png', width: 512, height: 512, seed: 1 };

describe('LocalDream: use_opencl reaches the native request', () => {
  let service: any;
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalDreamModule.generateImage.mockResolvedValue(nativeResult);
    jest.isolateModules(() => {
      service = require('../../../src/services/localDreamGenerator').localDreamGeneratorService;
    });
  });

  it('forwards useOpenCL=true unchanged (native decides the 512x512 gate)', async () => {
    await service.generateImage({ prompt: 'a cat', width: 512, height: 512, useOpenCL: true });
    expect(mockLocalDreamModule.generateImage).toHaveBeenCalledWith(
      expect.objectContaining({ useOpenCL: true, width: 512, height: 512 }),
    );
  });

  it('keeps useOpenCL=true even below 512 (the gate is native, not silently dropped in JS)', async () => {
    await service.generateImage({ prompt: 'a cat', width: 256, height: 256, useOpenCL: true });
    expect(mockLocalDreamModule.generateImage).toHaveBeenCalledWith(
      expect.objectContaining({ useOpenCL: true, width: 256, height: 256 }),
    );
  });

  it('defaults useOpenCL to false when the caller sends nothing', async () => {
    await service.generateImage({ prompt: 'a cat' });
    expect(mockLocalDreamModule.generateImage).toHaveBeenCalledWith(
      expect.objectContaining({ useOpenCL: false }),
    );
  });
});

// Falsify: one shared `generating` boolean -> the cancelled run's finally frees the new run.
describe('LocalDream run ownership across a cancel', () => {
  it('the cancelled run ending later does not free the run started after the cancel', async () => {
    let service: any;
    jest.isolateModules(() => {
      service = require('../../../src/services/localDreamGenerator').localDreamGeneratorService;
    });
    let rejectFirst!: (e: unknown) => void;
    let resolveSecond!: (v: unknown) => void;
    mockLocalDreamModule.cancelGeneration.mockResolvedValue(true);
    mockLocalDreamModule.generateImage
      .mockImplementationOnce(() => new Promise((_, rej) => { rejectFirst = rej; }))
      .mockImplementationOnce(() => new Promise(res => { resolveSecond = res; }));
    const first = service.generateImage({ prompt: 'one' }).catch((e: any) => e);
    await service.cancelGeneration();
    expect(await service.isGenerating()).toBe(false);
    const second = service.generateImage({ prompt: 'two' });
    rejectFirst(new Error('Generation cancelled'));
    await first;
    expect(await service.isGenerating()).toBe(true);
    resolveSecond(nativeResult);
    await expect(second).resolves.toMatchObject({ id: 'i' });
    expect(await service.isGenerating()).toBe(false);
  });
});

describe('one source of truth for default steps', () => {
  it('DEFAULT_IMAGE_STEPS equals the native default and the quality floor', () => {
    const adv = require('../../../src/utils/imageGenAdvice');
    expect(adv.DEFAULT_IMAGE_STEPS).toBe(NATIVE_DEFAULT_STEPS);
    expect(adv.DEFAULT_IMAGE_STEPS).toBe(adv.QUALITY_STEP_FLOOR);
    expect(adv.DEFAULT_IMAGE_STEPS).toBeLessThan(adv.MAX_IMAGE_STEPS);
    expect(adv.defaultImageSteps('android')).toBe(adv.DEFAULT_IMAGE_STEPS);
    expect(adv.defaultImageSteps('ios')).toBe(adv.DEFAULT_IMAGE_STEPS);
  });

  it('LocalDream and sd.cpp generators both fall back to it, not to 8 / a private literal', async () => {
    mockLocalDreamModule.generateImage.mockResolvedValue(nativeResult);
    mockSdCppModule.generateImage.mockResolvedValue(nativeResult);
    let ld: any;
    let sd: any;
    jest.isolateModules(() => {
      ld = require('../../../src/services/localDreamGenerator').localDreamGeneratorService;
      sd = require('../../../src/services/sdCppGenerator').sdCppGeneratorService;
    });
    await ld.generateImage({ prompt: 'x' });
    await sd.generateImage({ prompt: 'x' });
    expect(mockLocalDreamModule.generateImage).toHaveBeenLastCalledWith(expect.objectContaining({ steps: NATIVE_DEFAULT_STEPS }));
    expect(mockSdCppModule.generateImage).toHaveBeenLastCalledWith(expect.objectContaining({ steps: NATIVE_DEFAULT_STEPS }));
  });
});

describe('LocalDream getEffectiveSize (computed natively, up front)', () => {
  let service: any;
  beforeEach(() => {
    jest.clearAllMocks();
    jest.isolateModules(() => {
      service = require('../../../src/services/localDreamGenerator').localDreamGeneratorService;
    });
  });

  it('returns the native plan: NPU without a patch renders 512x512 and OpenCL is off', async () => {
    mockLocalDreamModule.getEffectiveImageSize.mockResolvedValue({
      requestedWidth: 320, requestedHeight: 320, width: 512, height: 512, adjusted: true,
      reason: 'NPU model has no 320x320 resolution patch (available: 512x512)', backend: 'qnn',
      openClRequested: true, useOpenCL: false,
    });
    const r = await service.getEffectiveSize({ width: 320, height: 320, useOpenCL: true });
    expect(mockLocalDreamModule.getEffectiveImageSize).toHaveBeenCalledWith({ width: 320, height: 320, useOpenCL: true });
    expect(r).toMatchObject({
      engine: 'localdream', requestedWidth: 320, width: 512, height: 512, adjusted: true, useOpenCL: false,
    });
    expect(r.reason).toContain('resolution patch');
  });

  it('MNN non-square request is reported as adjusted square', async () => {
    mockLocalDreamModule.getEffectiveImageSize.mockResolvedValue({
      requestedWidth: 512, requestedHeight: 256, width: 256, height: 256, adjusted: true,
      reason: 'MNN renders square 128-512 px on a 64 px grid', backend: 'mnn', useOpenCL: false,
    });
    const r = await service.getEffectiveSize({ width: 512, height: 256 });
    expect(r).toMatchObject({ width: 256, height: 256, adjusted: true });
  });

  it('echoes the request (adjusted=false) when the native method is missing or throws', async () => {
    const saved = mockLocalDreamModule.getEffectiveImageSize;
    (mockLocalDreamModule as any).getEffectiveImageSize = undefined;
    const missing = await service.getEffectiveSize({ width: 384, height: 128 });
    expect(missing).toMatchObject({ width: 384, height: 128, adjusted: false, useOpenCL: null });
    (mockLocalDreamModule as any).getEffectiveImageSize = saved;

    saved.mockRejectedValue(new Error('boom'));
    const failed = await service.getEffectiveSize({ width: 384, height: 128 });
    expect(failed).toMatchObject({ width: 384, height: 128, adjusted: false });
  });
});

describe('sd.cpp: 64x64 .. 512x512 with independent W/H reaches native unchanged', () => {
  let service: any;
  beforeEach(() => {
    jest.clearAllMocks();
    mockSdCppModule.generateImage.mockResolvedValue(nativeResult);
    jest.isolateModules(() => {
      service = require('../../../src/services/sdCppGenerator').sdCppGeneratorService;
    });
  });

  it.each([
    [64, 64],
    [512, 512],
    [64, 512],
    [512, 64],
    [192, 320],
  ])('%ix%i is sent as-is', async (w, h) => {
    await service.generateImage({ prompt: 'x', width: w, height: h });
    expect(mockSdCppModule.generateImage).toHaveBeenLastCalledWith(expect.objectContaining({ width: w, height: h }));
  });

  it('off-grid sizes snap to the 64 px grid exactly as getEffectiveSize reports', async () => {
    const { resolveSdCppEffectiveSize } = require('../../../src/services/imageEngineParams');
    const eff = resolveSdCppEffectiveSize({ width: 300, height: 70 });
    expect(eff).toMatchObject({ engine: 'sdcpp', width: 320, height: 64, adjusted: true });
    await service.generateImage({ prompt: 'x', width: 300, height: 70 });
    expect(mockSdCppModule.generateImage).toHaveBeenLastCalledWith(
      expect.objectContaining({ width: eff.width, height: eff.height }),
    );
  });

  it('64x64 and 512x512 are not adjusted', () => {
    const { resolveSdCppEffectiveSize } = require('../../../src/services/imageEngineParams');
    expect(resolveSdCppEffectiveSize({ width: 64, height: 64 })).toMatchObject({ width: 64, height: 64, adjusted: false, reason: null });
    expect(resolveSdCppEffectiveSize({ width: 512, height: 512 })).toMatchObject({ width: 512, height: 512, adjusted: false });
  });
});

describe('sd.cpp refuses img2img loudly instead of dropping initImage', () => {
  let service: any;
  let ErrorClass: any;
  beforeEach(() => {
    jest.clearAllMocks();
    mockSdCppModule.generateImage.mockResolvedValue(nativeResult);
    jest.isolateModules(() => {
      const mod = require('../../../src/services/sdCppGenerator');
      service = mod.sdCppGeneratorService;
      ErrorClass = mod.SdCppUnsupportedFeatureError;
    });
  });

  it('throws a typed error and never calls native', async () => {
    const p = service.generateImage({ prompt: 'x', initImage: 'BASE64' });
    await expect(p).rejects.toBeInstanceOf(ErrorClass);
    await expect(p).rejects.toMatchObject({ code: 'ERR_SDCPP_UNSUPPORTED_FEATURE', feature: 'initImage' });
    expect(mockSdCppModule.generateImage).not.toHaveBeenCalled();
  });

  it('a maskImage alone is refused too', async () => {
    await expect(service.generateImage({ prompt: 'x', maskImage: 'M' })).rejects.toMatchObject({ feature: 'maskImage' });
    expect(mockSdCppModule.generateImage).not.toHaveBeenCalled();
  });

  it('the refusal does not wedge the service: a plain txt2img still runs afterwards', async () => {
    await expect(service.generateImage({ prompt: 'x', initImage: 'B' })).rejects.toBeInstanceOf(ErrorClass);
    await expect(service.generateImage({ prompt: 'x' })).resolves.toMatchObject({ id: 'i' });
    expect(mockSdCppModule.generateImage).toHaveBeenCalledTimes(1);
  });
});
