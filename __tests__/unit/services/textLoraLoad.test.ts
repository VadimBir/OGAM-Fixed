/**
 * Text LoRA on the local engine: enabled adapters go to llama.rn as lora_list at init
 * (buildModelParams); a change while a model is loaded is applied live with applyLoraAdapters /
 * removeLoraAdapters (no reload), and a failed apply reports llama.cpp's error.
 */
import { initLlama } from 'llama.rn';
import RNFS from 'react-native-fs';
import { buildModelParams, loraLoadHint } from '../../../src/services/llmHelpers';
import { llmService } from '../../../src/services/llm';
import { hardwareService } from '../../../src/services/hardware';
import { useAppStore } from '../../../src/stores/appStore';
import { getTextLoraStatus, setTextLoraStatus } from '../../../src/services/textLora';
import { patchTextLora } from '../../../src/services/textLoraActions';
import { enabledLoraList, loraListKey } from '../../../src/services/textLoraParams';
import type { TextLora } from '../../../src/services/textLoraTypes';
import { createMockLlamaContext } from '../../utils/testHelpers';

jest.mock('../../../src/utils/logger', () => ({ __esModule: true, default: { log: jest.fn(), warn: jest.fn(), error: jest.fn() } }));

const mockedInitLlama = initLlama as jest.MockedFunction<typeof initLlama>;
const mockedRNFS = RNFS as jest.Mocked<typeof RNFS>;

const lora = (over: Partial<TextLora> = {}): TextLora => ({
  id: 'a', name: 'a', path: '/docs/text_loras/a.gguf', scale: 0.8, enabled: true, ...over,
});

const setLoras = (textLoras: TextLora[]) =>
  useAppStore.setState({ settings: { ...useAppStore.getState().settings, textLoras } });

beforeEach(() => {
  jest.clearAllMocks();
  jest.restoreAllMocks();
  Object.assign(llmService as any, { context: null, currentModelPath: null, isGenerating: false, activeCompletionPromise: null, appliedLoraKey: '', nativeConversationId: null });
  setTextLoraStatus({ lastError: null, cpuFallback: false });
  setLoras([]);
});

describe('buildModelParams lora_list', () => {
  it('sends the enabled adapters (path + scaled) and omits disabled / zero-scale ones', () => {
    const params = buildModelParams('/m.gguf', {
      textLoras: [lora(), lora({ id: 'b', path: '/docs/text_loras/b.gguf', enabled: false }), lora({ id: 'c', path: '/c.gguf', scale: 0 })],
    });
    expect((params.baseParams as any).lora_list).toEqual([{ path: '/docs/text_loras/a.gguf', scaled: 0.8 }]);
    expect(params.loraList).toEqual([{ path: '/docs/text_loras/a.gguf', scaled: 0.8 }]);
  });

  it('omits lora_list when no adapter is enabled', () => {
    const params = buildModelParams('/m.gguf', { textLoras: [lora({ enabled: false })] });
    expect('lora_list' in (params.baseParams as object)).toBe(false);
    expect(params.loraList).toEqual([]);
  });

  it('clamps the scale to 0-2', () => {
    expect(enabledLoraList([lora({ scale: 5 })])).toEqual([{ path: lora().path, scaled: 2 }]);
  });

  it('the load-failure hint names the enabled adapters', () => {
    expect(loraLoadHint({ lora_list: [{ path: '/x/y/my.gguf', scaled: 1 }] })).toContain('my.gguf');
    expect(loraLoadHint({})).toBe('');
  });
});

describe('loadModel passes lora_list to initLlama', () => {
  it('the init params carry the enabled adapters', async () => {
    mockedRNFS.exists.mockResolvedValue(true);
    mockedInitLlama.mockResolvedValue(createMockLlamaContext() as any);
    setLoras([lora()]);
    await llmService.loadModel('/models/test.gguf');
    expect(mockedInitLlama).toHaveBeenCalledWith(
      expect.objectContaining({ lora_list: [{ path: lora().path, scaled: 0.8 }] }),
    );
    expect(llmService.getAppliedTextLoraKey()).toBe(loraListKey([{ path: lora().path, scaled: 0.8 }]));
    await llmService.unloadModel();
  });
});

describe('syncTextLoras (live apply)', () => {
  const loadedContext = () => {
    const ctx = {
      applyLoraAdapters: jest.fn(() => Promise.resolve()),
      removeLoraAdapters: jest.fn(() => Promise.resolve()),
      clearCache: jest.fn(() => Promise.resolve()),
    };
    Object.assign(llmService as any, { context: ctx, currentModelPath: '/m.gguf', nativeConversationId: 'conv-1' });
    jest.spyOn(hardwareService, 'getAppMemoryUsage').mockResolvedValue({ available: 8 * 1024 ** 3, total: 12 * 1024 ** 3, used: 4 * 1024 ** 3 } as any);
    return ctx;
  };

  it('no model loaded: nothing to apply (the next load sends lora_list)', async () => {
    setLoras([lora()]);
    await expect(llmService.syncTextLoras()).resolves.toEqual({ changed: false });
  });

  it('applies an enabled adapter live, clears the KV cache, and forgets the conversation state', async () => {
    const ctx = loadedContext();
    setLoras([lora()]);
    await expect(llmService.syncTextLoras()).resolves.toEqual({ changed: true });
    expect(ctx.applyLoraAdapters).toHaveBeenCalledWith([{ path: lora().path, scaled: 0.8 }]);
    expect(ctx.clearCache).toHaveBeenCalledWith(true);
    expect((llmService as any).nativeConversationId).toBeNull();
    // Same set again: no second native call.
    await expect(llmService.syncTextLoras()).resolves.toEqual({ changed: false });
    expect(ctx.applyLoraAdapters).toHaveBeenCalledTimes(1);
  });

  it('disabling every adapter removes them live', async () => {
    const ctx = loadedContext();
    (llmService as any).appliedLoraKey = loraListKey([{ path: lora().path, scaled: 0.8 }]);
    setLoras([lora({ enabled: false })]);
    await expect(llmService.syncTextLoras()).resolves.toEqual({ changed: true });
    expect(ctx.removeLoraAdapters).toHaveBeenCalled();
    expect(ctx.applyLoraAdapters).not.toHaveBeenCalled();
  });

  it('a scale change through the settings action re-applies with the new scale', async () => {
    const ctx = loadedContext();
    setLoras([lora()]);
    await llmService.syncTextLoras();
    await patchTextLora('a', { scale: 1.5 });
    expect(ctx.applyLoraAdapters).toHaveBeenLastCalledWith([{ path: lora().path, scaled: 1.5 }]);
    expect(useAppStore.getState().settings.textLoras?.[0].scale).toBe(1.5);
    expect(getTextLoraStatus().lastError).toBeNull();
  });

  it('a failed apply keeps the previous set and shows the error', async () => {
    const ctx = loadedContext();
    ctx.applyLoraAdapters.mockRejectedValueOnce(new Error("Failed to apply LoRA adapter '/docs/text_loras/a.gguf'"));
    setLoras([lora()]);
    const error = await patchTextLora('a', { enabled: true });
    expect(error).toContain('Failed to apply LoRA adapter');
    expect(getTextLoraStatus().lastError).toContain('Failed to apply LoRA adapter');
    expect(llmService.getAppliedTextLoraKey()).toBe('');
    expect(ctx.clearCache).not.toHaveBeenCalled();
  });

  it('refuses when free RAM cannot hold the adapter set', async () => {
    const ctx = loadedContext();
    jest.spyOn(hardwareService, 'getAppMemoryUsage').mockResolvedValue({ available: 0, total: 12 * 1024 ** 3, used: 12 * 1024 ** 3 } as any);
    setLoras([lora()]);
    const result = await llmService.syncTextLoras();
    expect(result.changed).toBe(false);
    expect(result.error).toContain('Not enough free RAM');
    expect(ctx.applyLoraAdapters).not.toHaveBeenCalled();
  });
});
