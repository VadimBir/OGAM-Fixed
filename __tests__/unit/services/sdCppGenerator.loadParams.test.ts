/**
 * sdCppGeneratorService.loadModel forwards the new load options to the native module with defaults
 * (paramsBackend unset, loraApplyMode 'auto', vaeConvDirect false), and loadConfigDiffers treats an
 * unset option and its default as equal.
 */
const mockLoadModel = jest.fn(async (_params: Record<string, unknown>) => true);

// Minimal react-native: spreading requireActual('react-native') evaluates every lazy export and
// throws on the DevMenu TurboModule under jest (same pattern as sdCppGenerator.cancel.test.ts).
jest.mock('react-native', () => ({
  Platform: { OS: 'android', select: (o: any) => o.android ?? o.default },
  NativeEventEmitter: jest.fn().mockImplementation(() => ({ addListener: () => ({ remove: jest.fn() }) })),
  NativeModules: {
    SdCppImageModule: { loadModel: (p: Record<string, unknown>) => mockLoadModel(p), addListener: jest.fn(), removeListeners: jest.fn() },
  },
}));

import { sdCppGeneratorService } from '../../../src/services/sdCppGenerator';

beforeEach(() => mockLoadModel.mockClear());

describe('sdCppGeneratorService.loadModel options', () => {
  it('defaults: no paramsBackend field, loraApplyMode auto, vaeConvDirect false', async () => {
    await sdCppGeneratorService.loadModel('/m.safetensors', 4, { weightType: 'auto', flashAttn: true });
    const sent = mockLoadModel.mock.calls[0][0];
    expect(sent).toMatchObject({ modelPath: '/m.safetensors', threads: 4, loraApplyMode: 'auto', vaeConvDirect: false });
    expect(sent).not.toHaveProperty('paramsBackend');
  });

  it("'ram' is the same as unset; 'disk' and explicit values are forwarded", async () => {
    await sdCppGeneratorService.loadModel('/m.safetensors', 4, { weightType: 'auto', flashAttn: true, paramsBackend: 'ram' });
    expect(mockLoadModel.mock.calls[0][0]).not.toHaveProperty('paramsBackend');
    await sdCppGeneratorService.loadModel('/m.safetensors', 4, {
      weightType: 'q4_0', flashAttn: false, paramsBackend: 'disk', loraApplyMode: 'at_runtime', vaeConvDirect: true,
    });
    expect(mockLoadModel.mock.calls[1][0]).toMatchObject({
      paramsBackend: 'disk', loraApplyMode: 'at_runtime', vaeConvDirect: true, weightType: 'q4_0', flashAttn: false,
    });
  });

  it('loadConfigDiffers: unset equals default, a changed option differs', async () => {
    await sdCppGeneratorService.loadModel('/m.safetensors', 4, { weightType: 'auto', flashAttn: true });
    expect(sdCppGeneratorService.loadConfigDiffers({
      weightType: 'auto', flashAttn: true, paramsBackend: 'ram', loraApplyMode: 'auto', vaeConvDirect: false,
    })).toBe(false);
    expect(sdCppGeneratorService.loadConfigDiffers({ weightType: 'auto', flashAttn: true, vaeConvDirect: true })).toBe(true);
  });
});
