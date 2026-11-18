/**
 * utils/quantization: the one file-name -> quant-label rule, and the bits table it reads
 * (constants/models.ts QUANTIZATION_INFO). Block sizes cited in models.ts come from
 * node_modules/llama.rn/vendor/llama.cpp/ggml/src/ggml-common.h.
 */
import {
  findQuantization,
  parseQuantization,
  stripQuantization,
  lookupQuantizationInfo,
  quantizationBitsPerWeight,
} from '../../../src/utils/quantization';

describe('parseQuantization', () => {
  it.each([
    ['Qwen3-4B-Q4_K_M.gguf', 'Q4_K_M'],
    ['model-q4_k_m.gguf', 'Q4_K_M'],
    ['model-Q4KM.gguf', 'Q4_K_M'],
    ['model-Q8_0.gguf', 'Q8_0'],
    ['model-Q80.gguf', 'Q8_0'],
    ['model-IQ4_XS.gguf', 'IQ4_XS'],
    ['model-IQ2_XXS.gguf', 'IQ2_XXS'],
    ['model-BF16.gguf', 'BF16'],
    ['model-FP16.gguf', 'F16'],
    ['model-f32.gguf', 'F32'],
    ['gemma-UD-Q4_K_XL.gguf', 'UD-Q4_K_XL'],
    ['gpt-oss-MXFP4_MOE.gguf', 'MXFP4_MOE'],
    ['model-Q4_0_4_4.gguf', 'Q4_0_4_4'],
    ['model-Q4_0_8_8.gguf', 'Q4_0_8_8'],
    ['model-TQ1_0.gguf', 'TQ1_0'],
    ['model-Q2_K_S.gguf', 'Q2_K_S'],
    ['model-Q5_K.gguf', 'Q5_K'],
  ])('%s -> %s', (name, label) => {
    expect(parseQuantization(name)).toBe(label);
  });

  it('does not read IQ4_XS as Q4_ or BF16 as F16 (the old substring bugs)', () => {
    expect(parseQuantization('m-IQ4_XS.gguf')).not.toBe('Q4_');
    expect(parseQuantization('m-BF16.gguf')).toBe('BF16');
  });

  it('a bare q4/q5/q8 token reads as Q4/Q5/Q8 (the pre-refactor extractor label), not Unknown', () => {
    expect(parseQuantization('model-q4.gguf')).toBe('Q4');
    expect(parseQuantization('model.Q8.gguf')).toBe('Q8');
    expect(parseQuantization('Llama-2-7b-Q5.gguf')).toBe('Q5');
  });

  it('a longer token wins over a bare one in the same name', () => {
    expect(parseQuantization('model-Q4-Q4_K_M.gguf')).toBe('Q4_K_M');
  });

  it('requires a whole token: a quant glued to the name is not one', () => {
    expect(parseQuantization('modelQ80.gguf')).toBe('Unknown');
    expect(parseQuantization('Qwen2-7B.gguf')).toBe('Unknown');
    expect(parseQuantization('model-Q9_xyz.gguf')).toBe('Unknown');
  });

  it('findQuantization addresses the raw token', () => {
    const name = 'Qwen3-4B-Q4_K_M.gguf';
    const found = findQuantization(name)!;
    expect(name.slice(found.index, found.index + found.length)).toBe('Q4_K_M');
  });
});

describe('stripQuantization', () => {
  it('drops the quant token and the extension', () => {
    expect(stripQuantization('Qwen3-4B-Q4_K_M.gguf')).toBe('Qwen3-4B');
    expect(stripQuantization('model-q4.gguf')).toBe('model');
    expect(stripQuantization('plain.gguf')).toBe('plain');
  });
});

describe('lookupQuantizationInfo / quantizationBitsPerWeight', () => {
  it.each([
    // [label, bpw] - block bytes*8 / weights per block (ggml-common.h)
    ['Q2_K_S', 2.625], // block_q2_K 84 B / 256
    ['Q3_K', 3.4375], // block_q3_K 110 B / 256
    ['Q4_K', 4.5], // block_q4_K 144 B / 256
    ['Q5_K', 5.5], // block_q5_K 176 B / 256
    ['Q8_K', 9.125], // block_q8_K 292 B / 256
    ['Q4_0_4_4', 4.5], // block_q4_0 18 B / 32
    ['Q4_0_4_8', 4.5],
    ['Q4_0_8_8', 4.5],
  ])('%s has an entry with %d bits', (label, bpw) => {
    expect(lookupQuantizationInfo(label)?.bitsPerWeight).toBe(bpw);
  });

  it('every parsed llama.cpp quant name has an entry', () => {
    const files = [
      'Q2_K', 'Q2_K_S', 'Q2_K_L', 'Q2_K_XL', 'Q3_K', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L', 'Q3_K_XL',
      'Q4_0', 'Q4_1', 'Q4_K', 'Q4_K_S', 'Q4_K_M', 'Q4_K_L', 'Q4_K_XL', 'Q4_0_4_4', 'Q4_0_4_8', 'Q4_0_8_8',
      'Q5_0', 'Q5_1', 'Q5_K', 'Q5_K_S', 'Q5_K_M', 'Q5_K_L', 'Q5_K_XL', 'Q6_K', 'Q6_K_L', 'Q6_K_XL',
      'Q8_0', 'Q8_K', 'Q8_K_XL', 'IQ1_S', 'IQ1_M', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ2_M', 'IQ3_XXS',
      'IQ3_XS', 'IQ3_S', 'IQ3_M', 'IQ4_XS', 'IQ4_NL', 'TQ1_0', 'TQ2_0', 'MXFP4', 'MXFP4_MOE', 'BF16',
      'F16', 'F32',
    ];
    const missing = files.filter(q => !lookupQuantizationInfo(parseQuantization(`m-${q}.gguf`)));
    expect(missing).toEqual([]);
  });

  it('UD- packaging carries no bits of its own', () => {
    expect(lookupQuantizationInfo('UD-Q4_K_XL')).toBe(lookupQuantizationInfo('Q4_K_XL'));
  });

  it('a raw file name resolves through the parser', () => {
    expect(lookupQuantizationInfo('Qwen3-4B-Q4_K_M.gguf')?.bitsPerWeight).toBe(4.5);
  });

  it('labels that are no llama.cpp type have no entry; the caller fallback applies', () => {
    for (const label of ['Q4', 'IQ4_S', 'IQ4_M', 'Q7_K', 'Unknown']) {
      expect(lookupQuantizationInfo(label)).toBeUndefined();
    }
    expect(quantizationBitsPerWeight('Q4')).toBe(4.5);
    expect(quantizationBitsPerWeight('IQ4_M', 3)).toBe(3);
    expect(quantizationBitsPerWeight(undefined, 7)).toBe(7);
  });
});
