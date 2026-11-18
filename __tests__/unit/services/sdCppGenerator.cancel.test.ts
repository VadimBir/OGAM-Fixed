/** Native reject code CANCELLED maps to a "cancelled" error imageGenerationService treats as a cancel. */
const mockGenerate = jest.fn();

// Minimal react-native: spreading requireActual('react-native') evaluates every lazy export and
// throws on the DevMenu TurboModule under jest.
jest.mock('react-native', () => {
  return {
    Platform: { OS: 'android', select: (o: any) => o.android ?? o.default },
    NativeEventEmitter: jest.fn().mockImplementation(() => ({ addListener: () => ({ remove: jest.fn() }) })),
    NativeModules: {
      SdCppImageModule: {
        generateImage: (p: unknown) => mockGenerate(p),
        cancelGeneration: () => Promise.resolve(true),
        addListener: jest.fn(),
        removeListeners: jest.fn(),
      },
    },
  };
});

import { sdCppGeneratorService } from '../../../src/services/sdCppGenerator';

describe('sdCppGeneratorService cancel mapping', () => {
  it('CANCELLED code -> error whose message includes "cancelled", code kept', async () => {
    mockGenerate.mockRejectedValueOnce(Object.assign(new Error('Generation cancelled'), { code: 'CANCELLED' }));
    const err = await sdCppGeneratorService.generateImage({ prompt: 'x' }).catch(e => e);
    expect(err.message).toMatch(/cancelled/);
    expect(err.code).toBe('CANCELLED');
  });

  it('other native errors pass through unchanged', async () => {
    const boom = Object.assign(new Error('boom'), { code: 'GENERATION_ERROR' });
    mockGenerate.mockRejectedValueOnce(boom);
    await expect(sdCppGeneratorService.generateImage({ prompt: 'x' })).rejects.toBe(boom);
  });
});

// Falsify: go back to one `generating` boolean -> the old run's finally clears the new run and
// isGenerating() reads false while run 2 is still pending.
describe('sdCppGeneratorService run ownership across a cancel', () => {
  it('cancel frees the service at once; the cancelled run ending later does not free the new run', async () => {
    let rejectFirst!: (e: unknown) => void;
    let resolveSecond!: (v: unknown) => void;
    mockGenerate
      .mockImplementationOnce(() => new Promise((_, rej) => { rejectFirst = rej; }))
      .mockImplementationOnce(() => new Promise(res => { resolveSecond = res; }));
    const first = sdCppGeneratorService.generateImage({ prompt: 'one' }).catch(e => e);
    expect(await sdCppGeneratorService.isGenerating()).toBe(true);
    await sdCppGeneratorService.cancelGeneration();
    expect(await sdCppGeneratorService.isGenerating()).toBe(false);

    const second = sdCppGeneratorService.generateImage({ prompt: 'two' });
    rejectFirst(Object.assign(new Error('Generation cancelled'), { code: 'CANCELLED' }));
    expect((await first).code).toBe('CANCELLED');
    expect(await sdCppGeneratorService.isGenerating()).toBe(true);
    await expect(sdCppGeneratorService.generateImage({ prompt: 'three' })).rejects.toThrow('already in progress');

    resolveSecond({ id: '2', imagePath: '/2.png', width: 64, height: 64, seed: 1 });
    await expect(second).resolves.toMatchObject({ id: '2' });
    expect(await sdCppGeneratorService.isGenerating()).toBe(false);
  });
});
