/**
 * Architecture-based RAM estimates: GGUF KV/compute (text) and stable-diffusion.cpp weights +
 * compute (image), plus the quantized-KV ⇒ flash-attention rule in buildModelParams.
 */
jest.mock('@offgrid/models', () => ({ REASONING_BUDGET_AUTO: 0, thinkingBudgetPayload: () => ({}) }), { virtual: true });
jest.mock('../../../src/services/llmNativeLog', () => ({
  ensureNativeLogCapture: jest.fn(), resetNativeLogCapture: jest.fn(), recentNativeLog: jest.fn(),
}));
jest.mock('../../../src/utils/messageContent', () => ({ templateEmitsReasoning: jest.fn() }));
jest.mock('../../../src/utils/logger', () => ({ __esModule: true, default: { log: jest.fn(), warn: jest.fn(), error: jest.fn() } }));

import { estimateLlamaMemory, kvBytesPerElement } from '../../../src/services/llamaMemoryEstimate';
import { estimateSdCppMemory } from '../../../src/services/sdCppMemory';
import { checkMemoryForModel } from '../../../src/services/llmSafetyChecks';
import { buildModelParams } from '../../../src/services/llmHelpers';

const MiB = 1024 * 1024;

// Qwen2.5-1.5B: 28 layers, 12 heads, 2 KV heads, head dim 128, vocab 151936, embedding 1536.
const qwen25 = {
  'general.architecture': 'qwen2',
  'qwen2.block_count': 28,
  'qwen2.attention.head_count': 12,
  'qwen2.attention.head_count_kv': 2,
  'qwen2.embedding_length': 1536,
  'qwen2.vocab_size': 151936,
};
// Qwen3-1.7B: 28 layers, 16 heads, 8 KV heads, head dim 128 (explicit key/value length).
const qwen3 = {
  'general.architecture': 'qwen3',
  'qwen3.block_count': 28,
  'qwen3.attention.head_count': 16,
  'qwen3.attention.head_count_kv': 8,
  'qwen3.attention.key_length': 128,
  'qwen3.attention.value_length': 128,
  'qwen3.embedding_length': 2048,
  'qwen3.vocab_size': 151936,
};

describe('estimateLlamaMemory (GGUF metadata)', () => {
  it('KV at 8K scales f16 : q8_0 : q4_0 = 2 : 34/32 : 18/32 bytes/element', () => {
    const at = (cacheType: string) => estimateLlamaMemory(qwen25, { ctxLen: 8192, cacheType, nBatch: 512, weightsBytes: 1e9 })!.kvBytes;
    expect(at('f16') / MiB).toBeCloseTo(224, 0);
    expect(at('q8_0') / MiB).toBeCloseTo(119, 0);
    expect(at('q4_0') / MiB).toBeCloseTo(63, 0);
    expect(kvBytesPerElement(undefined)).toBe(2);
  });

  it('wide-KV model: Qwen3-1.7B at 8K q8_0 is ~476 MiB (the old 3%/1K heuristic said ~300)', () => {
    const est = estimateLlamaMemory(qwen3, { ctxLen: 8192, cacheType: 'q8_0', nBatch: 512, weightsBytes: 1.1e9 })!;
    expect(est.kvBytes / MiB).toBeCloseTo(476, 0);
    expect(est.computeBytes / MiB).toBeCloseTo((151936 + 2048) * 512 * 4 / MiB, 3);
    expect(est.totalBytes).toBe(Math.ceil((1.1e9 + est.kvBytes + est.computeBytes) * 1.1));
  });

  it('sliding window caps the context; missing fields → null (caller falls back)', () => {
    const swa = { ...qwen3, 'qwen3.attention.sliding_window': 1024 };
    expect(estimateLlamaMemory(swa, { ctxLen: 8192, cacheType: 'f16', nBatch: 512, weightsBytes: 0 })!.context).toBe(1024);
    expect(estimateLlamaMemory({ 'general.architecture': 'x' }, { ctxLen: 8192, cacheType: 'f16', nBatch: 512, weightsBytes: 0 })).toBeNull();
  });
});

describe('llmSafetyChecks.checkMemoryForModel with GGUF bytes', () => {
  const mem = (availMB: number) => async () => ({ available: availMB * MiB, total: 4096 * MiB });
  it('uses the architecture KV + compute instead of the fraction heuristic', async () => {
    const r = await checkMemoryForModel({
      modelFileSize: 1000 * MiB, contextLength: 8192, quantizedCache: true, getAvailableMemory: mem(4000),
      archBytes: { kvBytes: 476 * MiB, computeBytes: 300 * MiB },
    });
    expect(r.estimatedMB).toBeCloseTo(1200 + 476 + 300, 0);
  });
  it('without archBytes keeps the old heuristic (unchanged behaviour)', async () => {
    const r = await checkMemoryForModel({ modelFileSize: 1000 * MiB, contextLength: 8192, quantizedCache: true, getAvailableMemory: mem(4000) });
    expect(r.estimatedMB).toBeCloseTo(1200 + 8 * 1200 * 0.03, 0);
  });
});

describe('buildModelParams — a quantized KV cache always gets flash attention', () => {
  const p = (s: Parameters<typeof buildModelParams>[1]) => buildModelParams('/m.gguf', { nThreads: 4, ...s }).baseParams as Record<string, unknown>;
  it('Flash Attn OFF + q4_0 keeps q4_0 and sends flash_attn_type auto (never off)', () => {
    const b = p({ flashAttn: false, cacheType: 'q4_0' });
    expect(b.cache_type_k).toBe('q4_0');
    expect(b.cache_type_v).toBe('q4_0');
    expect(b.flash_attn_type).toBe('auto');
  });
  it('Flash Attn OFF + f16 sends off; unset cache follows the flag', () => {
    expect(p({ flashAttn: false, cacheType: 'f16' }).flash_attn_type).toBe('off');
    expect(p({ flashAttn: false }).cache_type_k).toBe('f16');
    expect(p({ flashAttn: true }).cache_type_k).toBe('q8_0');
  });
  it('exposes the effective cache type for the estimators', () => {
    expect(buildModelParams('/m.gguf', { nThreads: 4 }).cacheType).toBe('q8_0');
  });
});

describe('estimateSdCppMemory (stable-diffusion.cpp)', () => {
  const base = { fileSizeBytes: 2.13e9, family: 'sd1', width: 512, height: 512, flashAttn: true, vaeTiling: true };
  it('q4_0 shrinks only the convertible share; convs + VAE keep the file precision', () => {
    const f16 = estimateSdCppMemory({ ...base, weightType: 'auto' });
    const q4 = estimateSdCppMemory({ ...base, weightType: 'q4_0' });
    const q8 = estimateSdCppMemory({ ...base, weightType: 'q8_0' });
    expect(f16.sourceBytesPerParam).toBeCloseTo(2, 1);
    expect(q8.weightsMB).toBeLessThan(f16.weightsMB);
    expect(q4.weightsMB).toBeLessThan(q8.weightsMB);
    // conv (45% of UNet) + VAE stay fp16 → q4 weights stay well above a naive 0.56/2 scaling.
    expect(q4.weightsMB).toBeGreaterThan(f16.weightsMB * 0.5);
  });
  it('never "quantizes up": a q8 file stays at its own size under q8_0', () => {
    const q8file = estimateSdCppMemory({ ...base, fileSizeBytes: 1.1e9, weightType: 'q8_0' });
    const auto = estimateSdCppMemory({ ...base, fileSizeBytes: 1.1e9, weightType: 'auto' });
    expect(q8file.weightsMB).toBe(auto.weightsMB);
  });
  it('VAE tiling bounds the decode buffer; no-tiling grows with pixels', () => {
    const tiled = estimateSdCppMemory({ ...base, width: 1024, height: 1024 });
    const full = estimateSdCppMemory({ ...base, width: 1024, height: 1024, vaeTiling: false });
    expect(tiled.vaeComputeMB).toBe(500);
    expect(full.vaeComputeMB).toBe(8000);
    const small = estimateSdCppMemory({ ...base, width: 384, height: 384, vaeTiling: false });
    expect(small.vaeComputeMB).toBe(1125);
  });
  it('flash attention removes the tokens² score matrix', () => {
    const fa = estimateSdCppMemory({ ...base, flashAttn: true });
    const noFa = estimateSdCppMemory({ ...base, flashAttn: false });
    expect(noFa.unetComputeMB - fa.unetComputeMB).toBe(512); // 4096² x 8 heads x 4 B
  });
  it('a 1 GB sd1 file at 512² (q8-sized, FA + tiling) is ~1.7 GB, not the old 2.5x = 2.5 GB', () => {
    const est = estimateSdCppMemory({ ...base, fileSizeBytes: 1e9 });
    expect(est.totalMB).toBeGreaterThan(1600);
    expect(est.totalMB).toBeLessThan(1800);
  });
});
