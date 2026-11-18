/**
 * Advanced sampling / load parameters: what reaches local llama.rn (buildCompletionParams,
 * buildModelParams, initContextWithFallback) and remote servers (remoteSamplingOptions ->
 * openAISamplingParams / ollamaSamplingOptions), own server vs third-party.
 */
import { Platform } from 'react-native';
import {
  buildCompletionParams,
  buildModelParams,
  initContextWithFallback,
  resolveUbatch,
  resolveUseMmap,
} from '../../../src/services/llmHelpers';
import {
  SAMPLING_DEFAULTS,
  customStopStrings,
  formatStopStringsText,
  parseStopStringsText,
  remoteSamplingOptions,
  resolveSampling,
} from '../../../src/services/samplingParams';
import { openAISamplingParams } from '../../../src/services/providers/openAICompatibleProvider';
import { ollamaSamplingOptions } from '../../../src/services/providers/openAICompatibleStream';
import { estimateLlamaMemory } from '../../../src/services/llamaMemoryEstimate';

jest.mock('../../../src/utils/logger', () => ({
  __esModule: true,
  default: { log: jest.fn(), warn: jest.fn(), error: jest.fn() },
}));

const origOS = Platform.OS;
afterEach(() => { (Platform as any).OS = origOS; });

describe('resolveSampling / stop strings', () => {
  it('fills every value with the llama.cpp default when unset', () => {
    expect(resolveSampling(undefined)).toEqual(SAMPLING_DEFAULTS);
    expect(SAMPLING_DEFAULTS).toMatchObject({
      topK: 40, minP: 0.05, typicalP: 1, penaltyLastN: 64, presencePenalty: 0, frequencyPenalty: 0,
      dryMultiplier: 0, dryBase: 1.75, dryAllowedLength: 2, dryPenaltyLastN: 64,
      xtcProbability: 0, xtcThreshold: 0.1, mirostat: 0, mirostatTau: 5, mirostatEta: 0.1, seed: -1,
    });
  });

  it('sanitises: integers rounded, bad mirostat -> 0, negative seed -> -1, negative windows -> 0, NaN -> default', () => {
    const s = resolveSampling({ topK: 12.6, mirostat: 7, seed: -42, penaltyLastN: -1, dryPenaltyLastN: -1, minP: Number.NaN });
    expect(s.topK).toBe(13);
    expect(s.mirostat).toBe(0);
    expect(s.seed).toBe(-1);
    expect(s.penaltyLastN).toBe(0);
    expect(s.dryPenaltyLastN).toBe(0);
    expect(s.minP).toBe(0.05);
  });

  it('custom stops: drops empty and duplicate entries, keeps order', () => {
    expect(customStopStrings({ customStopStrings: ['A', '', 'B', 'A'] })).toEqual(['A', 'B']);
    expect(customStopStrings({})).toEqual([]);
  });

  it('stop text: one per line, \\n escape round-trips', () => {
    const list = parseStopStringsText('### User:\n\\n\\nQ:\n\n');
    expect(list).toEqual(['### User:', '\n\nQ:']);
    expect(formatStopStringsText(list)).toBe('### User:\n\\n\\nQ:');
  });
});

describe('buildCompletionParams (local llama.rn)', () => {
  it('sends every sampling field under its llama.rn name, defaults when unset', () => {
    const p = buildCompletionParams({ temperature: 0.7, topP: 0.9, repeatPenalty: 1.1, maxTokens: 256 });
    expect(p).toMatchObject({
      n_predict: 256, temperature: 0.7, top_p: 0.9, penalty_repeat: 1.1,
      top_k: 40, min_p: 0.05, typical_p: 1, penalty_last_n: 64, penalty_present: 0, penalty_freq: 0,
      dry_multiplier: 0, dry_base: 1.75, dry_allowed_length: 2, dry_penalty_last_n: 64,
      xtc_probability: 0, xtc_threshold: 0.1, mirostat: 0, mirostat_tau: 5, mirostat_eta: 0.1, seed: -1,
      stop: ['</s>', '<|end|>', '<|eot_id|>'],
    });
  });

  it('passes the user values through', () => {
    const p = buildCompletionParams({
      topK: 20, minP: 0.1, typicalP: 0.9, penaltyLastN: 128, presencePenalty: 0.5, frequencyPenalty: 0.3,
      dryMultiplier: 0.8, dryBase: 2, dryAllowedLength: 3, dryPenaltyLastN: 256,
      xtcProbability: 0.5, xtcThreshold: 0.2, mirostat: 2, mirostatTau: 4, mirostatEta: 0.2, seed: 1234,
    });
    expect(p).toMatchObject({
      top_k: 20, min_p: 0.1, typical_p: 0.9, penalty_last_n: 128, penalty_present: 0.5, penalty_freq: 0.3,
      dry_multiplier: 0.8, dry_base: 2, dry_allowed_length: 3, dry_penalty_last_n: 256,
      xtc_probability: 0.5, xtc_threshold: 0.2, mirostat: 2, mirostat_tau: 4, mirostat_eta: 0.2, seed: 1234,
    });
  });

  it('adds custom stops after the built-in ones, without duplicates', () => {
    const p = buildCompletionParams({ customStopStrings: ['### User:', '</s>'] });
    expect(p.stop).toEqual(['</s>', '<|end|>', '<|eot_id|>', '### User:']);
  });

  it('returns a fresh stop array each call (llama.rn appends additional_stops to it)', () => {
    const a = buildCompletionParams({});
    a.stop.push('<|im_end|>');
    const b = buildCompletionParams({});
    expect(b.stop).toEqual(['</s>', '<|end|>', '<|eot_id|>']);
  });
});

describe('buildModelParams load fields', () => {
  it('n_ubatch follows n_batch when unset or 0 (old behaviour)', () => {
    expect((buildModelParams('/m.gguf', { nBatch: 256 }).baseParams as any).n_ubatch).toBe(256);
    expect((buildModelParams('/m.gguf', { nBatch: 256, nUbatch: 0 }).baseParams as any).n_ubatch).toBe(256);
  });

  it('n_ubatch set: sent, capped at n_batch', () => {
    expect((buildModelParams('/m.gguf', { nBatch: 512, nUbatch: 128 }).baseParams as any).n_ubatch).toBe(128);
    expect(resolveUbatch(256, 1024)).toBe(256);
    expect(buildModelParams('/m.gguf', { nBatch: 512, nUbatch: 128 }).nUbatch).toBe(128);
  });

  it("cacheTypeV 'same' / unset sends V = K (old behaviour)", () => {
    for (const v of [undefined, 'same']) {
      const r = buildModelParams('/m.gguf', { cacheType: 'q4_0', cacheTypeV: v });
      expect(r.baseParams).toMatchObject({ cache_type_k: 'q4_0', cache_type_v: 'q4_0' });
      expect(r.cacheTypeV).toBe('q4_0');
    }
  });

  it('split K/V: sends both; usesF16Cache when either half is f16', () => {
    const r = buildModelParams('/m.gguf', { cacheType: 'q8_0', cacheTypeV: 'f16' });
    expect(r.baseParams).toMatchObject({ cache_type_k: 'q8_0', cache_type_v: 'f16' });
    expect(r.cacheType).toBe('q8_0');
    expect(r.cacheTypeV).toBe('f16');
    expect(r.usesF16Cache).toBe(true);
    expect(buildModelParams('/m.gguf', { cacheType: 'q8_0', cacheTypeV: 'q4_0' }).usesF16Cache).toBe(false);
  });

  it('flash attention follows the V cache off-Android: FA off allowed only with an f16 V', () => {
    (Platform as any).OS = 'ios';
    // Quantized K, f16 V: llama.cpp accepts FA off (only a quantized V needs it).
    expect((buildModelParams('/m.gguf', { flashAttn: false, cacheType: 'q8_0', cacheTypeV: 'f16' }).baseParams as any).flash_attn_type).toBe('off');
    // f16 K, quantized V: FA cannot be off -> 'auto' (llama.cpp enables it).
    expect((buildModelParams('/m.gguf', { flashAttn: false, cacheType: 'f16', cacheTypeV: 'q8_0' }).baseParams as any).flash_attn_type).toBe('auto');
  });

  it("Android split cache: f16 V -> 'auto' (FA kept on where the accelerator runs it), quantized V -> 'on'", () => {
    (Platform as any).OS = 'android';
    expect((buildModelParams('/m.gguf', { cacheType: 'q8_0', cacheTypeV: 'f16' }).baseParams as any).flash_attn_type).toBe('auto');
    expect((buildModelParams('/m.gguf', { cacheType: 'f16', cacheTypeV: 'q8_0' }).baseParams as any).flash_attn_type).toBe('on');
  });

  describe('mmap modes', () => {
    const repackable = '/models/model-q4_0.gguf';
    const fits = { modelFileSizeBytes: 500 * 1024 * 1024, liveAvailableMB: 8000 };
    it("'auto' / unset = shouldDisableMmap (Android small q4_0 -> mmap off)", () => {
      (Platform as any).OS = 'android';
      expect(resolveUseMmap(repackable, 'auto', fits)).toBe(false);
      expect((buildModelParams(repackable, { ...fits }).baseParams as any).use_mmap).toBe(false);
      expect((buildModelParams('/models/m-q8_0.gguf', { ...fits }).baseParams as any).use_mmap).toBe(true);
    });
    it("'on' forces mmap even where auto would disable it", () => {
      (Platform as any).OS = 'android';
      expect((buildModelParams(repackable, { ...fits, mmapMode: 'on' }).baseParams as any).use_mmap).toBe(true);
    });
    it("'off' disables mmap for any model", () => {
      (Platform as any).OS = 'ios';
      expect((buildModelParams('/models/m-q8_0.gguf', { mmapMode: 'off' }).baseParams as any).use_mmap).toBe(false);
    });
  });
});

describe('initContextWithFallback with split K/V', () => {
  const { initLlama } = require('llama.rn');
  const mockedInitLlama = initLlama as jest.MockedFunction<typeof initLlama>;

  it('flash-attention retry still sends f16 for BOTH halves', async () => {
    mockedInitLlama.mockRejectedValueOnce(new Error('FA failed')).mockResolvedValueOnce({ gpu: true } as any);
    await initContextWithFallback({ model: '/m.gguf', flash_attn_type: 'on', cache_type_k: 'q8_0', cache_type_v: 'q4_0' }, 2048, 24);
    expect(mockedInitLlama.mock.calls[1][0]).toMatchObject({ flash_attn_type: 'auto', cache_type_k: 'f16', cache_type_v: 'f16' });
  });

  it('when every attempt fails with K != V, re-runs the chain once with V = K', async () => {
    const ctx = { gpu: false, release: jest.fn() };
    mockedInitLlama
      .mockRejectedValueOnce(new Error('different K and V')) // GPU
      .mockRejectedValueOnce(new Error('different K and V')) // FA retry
      .mockRejectedValueOnce(new Error('different K and V')) // CPU
      .mockResolvedValueOnce(ctx as any); // V = K, GPU
    const result = await initContextWithFallback({ model: '/m.gguf', flash_attn_type: 'on', cache_type_k: 'q8_0', cache_type_v: 'f16' }, 2048, 24);
    expect(result.context).toBe(ctx);
    expect(mockedInitLlama.mock.calls[3][0]).toMatchObject({ cache_type_k: 'q8_0', cache_type_v: 'q8_0', flash_attn_type: 'on' });
  });

  it("V = K retry turns a flash_attn 'off' into 'auto' when K is quantized", async () => {
    // nGpuLayers 0: the chain is attempt 1 (no GPU) + attempt 2 (CPU); both fail, then V = K.
    mockedInitLlama
      .mockRejectedValueOnce(new Error('attempt 1'))
      .mockRejectedValueOnce(new Error('attempt 2'))
      .mockResolvedValueOnce({ gpu: false } as any);
    await initContextWithFallback({ model: '/m.gguf', flash_attn_type: 'off', cache_type_k: 'q8_0', cache_type_v: 'f16' }, 2048, 0);
    expect(mockedInitLlama).toHaveBeenCalledTimes(3);
    expect(mockedInitLlama.mock.calls[2][0]).toMatchObject({ cache_type_v: 'q8_0', flash_attn_type: 'auto' });
  });

  it('K = V: no extra retry, the error propagates', async () => {
    mockedInitLlama.mockRejectedValueOnce(new Error('attempt 1')).mockRejectedValueOnce(new Error('cpu fail'));
    await expect(initContextWithFallback({ model: '/m.gguf', cache_type_k: 'f16', cache_type_v: 'f16' }, 2048, 0)).rejects.toThrow(/cpu fail/);
    expect(mockedInitLlama).toHaveBeenCalledTimes(2);
  });
});

describe('KV estimate with split K/V', () => {
  const meta = {
    'general.architecture': 'llama',
    'llama.block_count': 32, 'llama.attention.head_count': 32, 'llama.attention.head_count_kv': 8,
    'llama.embedding_length': 4096,
  };
  it('V absent = K type for both halves (unchanged formula)', () => {
    const a = estimateLlamaMemory(meta, { ctxLen: 4096, cacheType: 'q8_0', nBatch: 512, weightsBytes: 0 })!;
    const b = estimateLlamaMemory(meta, { ctxLen: 4096, cacheType: 'q8_0', cacheTypeV: 'q8_0', nBatch: 512, weightsBytes: 0 })!;
    expect(a.kvBytes).toBe(b.kvBytes);
  });
  it('f16 V costs more than q8_0 V, by the V half only', () => {
    const q8 = estimateLlamaMemory(meta, { ctxLen: 4096, cacheType: 'q8_0', nBatch: 512, weightsBytes: 0 })!;
    const split = estimateLlamaMemory(meta, { ctxLen: 4096, cacheType: 'q8_0', cacheTypeV: 'f16', nBatch: 512, weightsBytes: 0 })!;
    const half = 32 * 4096 * 8 * 128;
    expect(split.kvBytes - q8.kvBytes).toBeCloseTo(half * (2 - 34 / 32));
  });
});

describe('remote sampling', () => {
  const settings = {
    temperature: 0.7, topP: 0.9, repeatPenalty: 1.1, maxTokens: 2048,
    topK: 20, minP: 0.1, typicalP: 0.95, penaltyLastN: 128, presencePenalty: 0, frequencyPenalty: 0.4,
    dryMultiplier: 0.8, dryBase: 1.75, dryAllowedLength: 2, dryPenaltyLastN: 64,
    xtcProbability: 0.5, xtcThreshold: 0.1, mirostat: 0, mirostatTau: 5, mirostatEta: 0.1, seed: -1,
    customStopStrings: ['a', 'b', 'c', 'd', 'e'],
  };
  const options = remoteSamplingOptions(settings);

  it('remoteSamplingOptions carries every value', () => {
    expect(options).toMatchObject({
      temperature: 0.7, topP: 0.9, repeatPenalty: 1.1, topK: 20, minP: 0.1, typicalP: 0.95,
      penaltyLastN: 128, presencePenalty: 0, frequencyPenalty: 0.4, dryMultiplier: 0.8,
      xtcProbability: 0.5, mirostat: 0, seed: -1, stopSequences: ['a', 'b', 'c', 'd', 'e'],
    });
  });

  it("own server (llama-server): every value incl. llama.cpp extras, seed -1, all stops", () => {
    const body = openAISamplingParams(options, true, true);
    expect(body).toMatchObject({
      temperature: 0.7, top_p: 0.9, top_k: 20, repeat_penalty: 1.1,
      presence_penalty: 0, frequency_penalty: 0.4, seed: -1,
      min_p: 0.1, typical_p: 0.95, repeat_last_n: 128,
      dry_multiplier: 0.8, dry_base: 1.75, dry_allowed_length: 2, dry_penalty_last_n: 64,
      xtc_probability: 0.5, xtc_threshold: 0.1, mirostat: 0, mirostat_tau: 5, mirostat_eta: 0.1,
      stop: ['a', 'b', 'c', 'd', 'e'],
    });
  });

  it('third-party host: OpenAI fields only, zero penalties and random seed omitted, max 4 stops', () => {
    const body = openAISamplingParams(options, false, false);
    expect(body).toEqual({ temperature: 0.7, top_p: 0.9, frequency_penalty: 0.4, stop: ['a', 'b', 'c', 'd'] });
  });

  it('third-party host with a fixed seed sends it', () => {
    const body = openAISamplingParams(remoteSamplingOptions({ ...settings, seed: 7 }), false, false);
    expect(body.seed).toBe(7);
  });

  it('OpenRouter (extensions, not own server): top_k / repeat_penalty but no llama.cpp extras', () => {
    const body = openAISamplingParams(options, true, false);
    expect(body).toMatchObject({ top_k: 20, repeat_penalty: 1.1 });
    expect(body).not.toHaveProperty('min_p');
    expect(body).not.toHaveProperty('dry_multiplier');
  });

  it('Ollama: its own option names; no DRY / XTC / mirostat / typical_p', () => {
    const body = ollamaSamplingOptions(options);
    expect(body).toEqual({
      temperature: 0.7, top_p: 0.9, top_k: 20, min_p: 0.1, repeat_penalty: 1.1, repeat_last_n: 128,
      presence_penalty: 0, frequency_penalty: 0.4, seed: -1, stop: ['a', 'b', 'c', 'd', 'e'],
    });
  });
});

describe('computePendingSettings: new load fields', () => {
  const { computePendingSettings } = require('../../../src/screens/ChatScreen/pendingSettings');
  it('flags a change once the load snapshot captured the field', () => {
    expect(computePendingSettings('llama', { nUbatch: 128 }, { nUbatch: 0 })).toBe(true);
    expect(computePendingSettings('llama', { mmapMode: 'off' }, { mmapMode: 'auto' })).toBe(true);
    expect(computePendingSettings('llama', { cacheTypeV: 'f16' }, { cacheTypeV: 'same' })).toBe(true);
  });
  it('ignores a field the snapshot did not capture', () => {
    expect(computePendingSettings('llama', { nBatch: 512, nUbatch: 128, mmapMode: 'off', cacheTypeV: 'f16' }, { nBatch: 512 })).toBe(false);
  });
});
