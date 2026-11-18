/** Native reject code CANCELLED maps to a "cancelled" error imageGenerationService treats as a cancel. */
const mockGenerate = jest.fn();

jest.mock('react-native', () => {
  const actual = jest.requireActual('react-native');
  return {
    ...actual,
    Platform: { ...actual.Platform, OS: 'android' },
    NativeEventEmitter: jest.fn().mockImplementation(() => ({ addListener: () => ({ remove: jest.fn() }) })),
    NativeModules: {
      ...actual.NativeModules,
      SdCppImageModule: { generateImage: (p: unknown) => mockGenerate(p), addListener: jest.fn(), removeListeners: jest.fn() },
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
