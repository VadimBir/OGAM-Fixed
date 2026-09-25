/**
 * Runtime-RAM estimate of a GGUF model from its own metadata (layers, KV heads, head dims,
 * sliding window, vocab) — the single formula both the pre-load fit check (llmSafetyChecks) and
 * the residency/budget estimate (activeModelService/memory) use, so the two gates cannot disagree.
 *
 * KV = layers x ctx x kvHeads x (keyDim + valueDim) x bytes/element (f16 2, q8_0 34/32, q4_0 18/32).
 * Compute = (vocab + embedding) x nBatch x 4 (logits + activations of one ubatch, f32).
 */

export interface LlamaMemoryEstimate {
  kvBytes: number;
  computeBytes: number;
  /** (weights + mmproj + KV + compute) x 1.1 */
  totalBytes: number;
  context: number;
}

export function kvBytesPerElement(cacheType: string | undefined): number {
  if (cacheType === 'q4_0') return 18 / 32;
  if (cacheType === 'q8_0') return 34 / 32;
  return 2;
}

/** Returns null when the metadata lacks the fields the formula needs (caller falls back). */
export function estimateLlamaMemory(
  metadata: Record<string, unknown>,
  opts: { ctxLen: number; cacheType: string | undefined; nBatch: number; weightsBytes: number },
): LlamaMemoryEstimate | null {
  const architecture = metadata['general.architecture'];
  if (typeof architecture !== 'string') return null;
  const number = (field: string): number | undefined => {
    const value = metadata[`${architecture}.${field}`];
    const parsed = typeof value === 'number' || typeof value === 'string' ? Number(value) : NaN;
    return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
  };
  const blocks = number('block_count');
  const heads = number('attention.head_count');
  const kvHeads = number('attention.head_count_kv') ?? heads;
  const embedding = number('embedding_length');
  const keyLength = number('attention.key_length') ?? (embedding && heads ? embedding / heads : undefined);
  const valueLength = number('attention.value_length') ?? keyLength;
  const vocabulary = number('vocab_size');
  if (!blocks || !kvHeads || !embedding || !keyLength || !valueLength) return null;
  const context = Math.min(opts.ctxLen, number('attention.sliding_window') ?? opts.ctxLen);
  const kvBytes = blocks * context * kvHeads * (keyLength + valueLength) * kvBytesPerElement(opts.cacheType);
  const computeBytes = ((vocabulary ?? 0) + embedding) * opts.nBatch * 4;
  const totalBytes = Math.ceil((opts.weightsBytes + kvBytes + computeBytes) * 1.1);
  return { kvBytes, computeBytes, totalBytes, context };
}
