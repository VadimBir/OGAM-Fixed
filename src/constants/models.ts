// Model size recommendations based on device RAM
export const MODEL_RECOMMENDATIONS = {
  // RAM in GB -> max model parameters in billions
  memoryToParams: [
    { minRam: 3, maxRam: 4, maxParams: 1.5, quantization: 'Q4_K_M' },
    { minRam: 4, maxRam: 6, maxParams: 3, quantization: 'Q4_K_M' },
    { minRam: 6, maxRam: 8, maxParams: 4, quantization: 'Q4_K_M' },
    { minRam: 8, maxRam: 12, maxParams: 8, quantization: 'Q4_K_M' },
    { minRam: 12, maxRam: 16, maxParams: 13, quantization: 'Q4_K_M' },
    { minRam: 16, maxRam: Infinity, maxParams: 30, quantization: 'Q4_K_M' },
  ],
};

// Curated list of recommended models for mobile (updated Sep 2026)
// Ordered editorially: MiniCPM5 featured first, then Gemma 4, Qwen 3.5, and others.
export const RECOMMENDED_MODELS = [
  // --- MiniCPM5 (featured) ---
  {
    id: 'openbmb/MiniCPM5-2B-GGUF',
    name: 'MiniCPM5 2B',
    params: 2,
    description: 'Strong coding, reasoning, and tool use with 128K context',
    minRam: 4,
    type: 'text' as const,
    org: 'openbmb',
    isNew: true,
  },
  // --- Gemma 4 (featured) ---
  {
    id: 'unsloth/gemma-4-E2B-it-GGUF',
    name: 'Gemma 4 E2B',
    params: 2,
    description: 'Google\'s latest, thinking mode + vision, MoE architecture',
    minRam: 4,
    type: 'vision' as const,
    org: 'google',
    isNew: true,
  },
  {
    id: 'unsloth/gemma-4-E4B-it-GGUF',
    name: 'Gemma 4 E4B',
    params: 4,
    description: 'Google\'s latest, stronger reasoning + vision',
    minRam: 6,
    type: 'vision' as const,
    org: 'google',
    isNew: true,
  },
  // --- Qwen 3.5 ---
  {
    id: 'unsloth/Qwen3.5-0.8B-GGUF',
    name: 'Qwen 3.5 0.8B',
    params: 0.8,
    description: 'Thinking mode, ultra-light, 262K context',
    minRam: 3,
    type: 'vision' as const,
    org: 'Qwen',
  },
  {
    id: 'unsloth/Qwen3.5-2B-GGUF',
    name: 'Qwen 3.5 2B',
    params: 2,
    description: 'Hybrid thinking + chat, 262K context',
    minRam: 4,
    type: 'vision' as const,
    org: 'Qwen',
  },
  {
    id: 'unsloth/Qwen3.5-9B-GGUF',
    name: 'Qwen 3.5 9B',
    params: 9,
    description: 'Best Qwen 3.5 quality, thinking mode, 262K context',
    minRam: 8,
    type: 'vision' as const,
    org: 'Qwen',
  },
  // --- Others ---
  {
    id: 'ggml-org/SmolLM3-3B-GGUF',
    name: 'SmolLM3 3B',
    params: 3,
    description: 'Purpose-built for constrained devices, 128K context',
    minRam: 6,
    type: 'text' as const,
    org: 'HuggingFaceTB',
  },
  {
    id: 'bartowski/Mistral-7B-Instruct-v0.3-GGUF',
    name: 'Mistral 7B',
    params: 7,
    description: 'Fast, reliable general purpose model',
    minRam: 6,
    type: 'text' as const,
    org: 'mistralai',
  },
  // --- Low-RAM only (≤ 6 GB) ---
  {
    id: 'ggml-org/SmolVLM2-500M-Video-Instruct-GGUF',
    name: 'SmolVLM2 500M',
    params: 0.5,
    description: 'Tiny vision + video model for constrained devices',
    minRam: 3,
    maxRam: 6,
    type: 'vision' as const,
    org: 'HuggingFaceTB',
  },
  {
    id: 'QuantFactory/SmolLM2-360M-Instruct-GGUF',
    name: 'SmolLM2 360M',
    params: 0.36,
    description: 'Ultra-light language model for low-RAM devices',
    minRam: 3,
    maxRam: 6,
    type: 'text' as const,
    org: 'HuggingFaceTB',
  },
  // --- Vision ---
  {
    id: 'ggml-org/SmolVLM-Instruct-GGUF',
    name: 'SmolVLM 2B',
    params: 2,
    description: 'Mobile-optimized vision-language model',
    minRam: 4,
    type: 'vision' as const,
    org: 'HuggingFaceTB',
  },
  {
    id: 'ggml-org/SmolVLM2-2.2B-Instruct-GGUF',
    name: 'SmolVLM2 2.2B',
    params: 2.2,
    description: 'Vision + video understanding',
    minRam: 4,
    type: 'vision' as const,
    org: 'HuggingFaceTB',
  },
];

// Trending model families — one best-fit per family is shown as "trending"
export const TRENDING_FAMILIES: Record<string, string[]> = {
  minicpm5: ['openbmb/MiniCPM5-2B-GGUF'],
  gemma4: ['unsloth/gemma-4-E2B-it-GGUF', 'unsloth/gemma-4-E4B-it-GGUF'],
  qwen35: ['unsloth/Qwen3.5-0.8B-GGUF', 'unsloth/Qwen3.5-2B-GGUF', 'unsloth/Qwen3.5-9B-GGUF'],
};
// Flat list used for badge rendering
export const TRENDING_MODEL_IDS = Object.values(TRENDING_FAMILIES).flat();

// Model organization filter options
export const MODEL_ORGS = [
  { key: 'openbmb', label: 'OpenBMB' },
  { key: 'Qwen', label: 'Qwen' },
  { key: 'meta-llama', label: 'Llama' },
  { key: 'google', label: 'Google' },
  { key: 'microsoft', label: 'Microsoft' },
  { key: 'mistralai', label: 'Mistral' },
  { key: 'deepseek-ai', label: 'DeepSeek' },
  { key: 'HuggingFaceTB', label: 'HuggingFace' },
  { key: 'nvidia', label: 'NVIDIA' },
];

// Quantization levels and their properties
export const QUANTIZATION_INFO: Record<string, {
  bitsPerWeight: number;
  quality: string;
  description: string;
  recommended: boolean;
}> = {
  // bpw = sizeof(block)*8 / weights per block (ggml-common.h): block_q4_0 18 B/32 = 4.5 (:199),
  // block_q8_0 34 B/32 = 8.5 (:256), block_q6_K 210 B/256 = 6.5625 (:368). The old 4 / 8 / 6.5
  // ignored the per-block fp16 scale and under-estimated RAM (Q4_0 by ~11%) -> loads that OOM.
  'Q2_K': { bitsPerWeight: 2.625, quality: 'Low', description: 'Extreme compression, noticeable quality loss', recommended: false },
  'Q3_K_S': { bitsPerWeight: 3.4375, quality: 'Low-Medium', description: 'High compression, some quality loss', recommended: false },
  'Q3_K_M': { bitsPerWeight: 3.4375, quality: 'Medium', description: 'Good compression with acceptable quality', recommended: false },
  'Q4_0': { bitsPerWeight: 4.5, quality: 'Medium', description: 'Basic 4-bit quantization', recommended: false },
  'Q4_K_S': { bitsPerWeight: 4.5, quality: 'Medium-Good', description: 'Good balance of size and quality', recommended: true },
  'Q4_K_M': { bitsPerWeight: 4.5, quality: 'Good', description: 'Optimal for mobile - best balance', recommended: true },
  'Q5_K_S': { bitsPerWeight: 5.5, quality: 'Good-High', description: 'Higher quality, larger size', recommended: false },
  'Q5_K_M': { bitsPerWeight: 5.5, quality: 'High', description: 'Near original quality', recommended: false },
  'Q6_K': { bitsPerWeight: 6.5625, quality: 'Very High', description: 'Minimal quality loss', recommended: false },
  'Q8_0': { bitsPerWeight: 8.5, quality: 'Excellent', description: 'Best quality, largest size', recommended: false },
  // Every llama.cpp quant name src/utils/quantization.ts parses has an entry (bits are llama.cpp's
  // bits per weight; *_K_L / *_K_XL keep embeddings/output at Q8, so they sit above their base).
  // Parsed spellings that are NOT a llama.cpp type (bare Q4/Q5/Q8, IQ4_S, IQ4_M, Q7_K, Q4_0_8_4,
  // ...; llama.h llama_ftype + ggml-common.h list none of them) have no entry on purpose:
  // lookupQuantizationInfo returns undefined and each caller applies its own fallback.
  // Block bpw below = sizeof(block)*8 / weights per block, from
  // node_modules/llama.rn/vendor/llama.cpp/ggml/src/ggml-common.h (QK_K=256 :89, K_SCALE_SIZE=12 :90).
  'Q2_K_S': { bitsPerWeight: 2.625, quality: 'Low', description: 'Smaller Q2_K mix', recommended: false }, // block_q2_K 84 B/256 (:309); ftype 21 (llama.h:138)
  'Q3_K': { bitsPerWeight: 3.4375, quality: 'Low-Medium', description: '3-bit K-quant', recommended: false }, // block_q3_K 110 B/256 (:321)
  'Q4_K': { bitsPerWeight: 4.5, quality: 'Medium-Good', description: '4-bit K-quant', recommended: false }, // block_q4_K 144 B/256 (:338)
  'Q5_K': { bitsPerWeight: 5.5, quality: 'Good-High', description: '5-bit K-quant', recommended: false }, // block_q5_K 176 B/256 (:356)
  'Q8_K': { bitsPerWeight: 9.125, quality: 'Excellent', description: '8-bit K-quant (llama.cpp intermediate type)', recommended: false }, // block_q8_K 292 B/256 (:376)
  // Q4_0 interleaved for ARM repack; removed from GGUF (llama.h:151-153 "use Q4_0 and runtime
  // repack"), old files remain. Same bytes as block_q4_0: 18 B/32 = 4.5 (:199).
  'Q4_0_4_4': { bitsPerWeight: 4.5, quality: 'Medium', description: 'Q4_0 interleaved for ARM (legacy)', recommended: false },
  'Q4_0_4_8': { bitsPerWeight: 4.5, quality: 'Medium', description: 'Q4_0 interleaved for ARM (legacy)', recommended: false },
  'Q4_0_8_8': { bitsPerWeight: 4.5, quality: 'Medium', description: 'Q4_0 interleaved for ARM (legacy)', recommended: false },
  'IQ1_S': { bitsPerWeight: 1.56, quality: 'Very Low', description: 'Importance-matrix 1-bit, extreme compression', recommended: false },
  'IQ1_M': { bitsPerWeight: 1.75, quality: 'Very Low', description: 'Importance-matrix 1-bit, extreme compression', recommended: false },
  'IQ2_XXS': { bitsPerWeight: 2.06, quality: 'Low', description: 'Importance-matrix 2-bit', recommended: false },
  'IQ2_XS': { bitsPerWeight: 2.31, quality: 'Low', description: 'Importance-matrix 2-bit', recommended: false },
  'IQ2_S': { bitsPerWeight: 2.5, quality: 'Low', description: 'Importance-matrix 2-bit', recommended: false },
  'IQ2_M': { bitsPerWeight: 2.7, quality: 'Low', description: 'Importance-matrix 2-bit', recommended: false },
  'IQ3_XXS': { bitsPerWeight: 3.06, quality: 'Low-Medium', description: 'Importance-matrix 3-bit', recommended: false },
  'IQ3_XS': { bitsPerWeight: 3.3, quality: 'Low-Medium', description: 'Importance-matrix 3-bit', recommended: false },
  'IQ3_S': { bitsPerWeight: 3.44, quality: 'Medium', description: 'Importance-matrix 3-bit', recommended: false },
  'IQ3_M': { bitsPerWeight: 3.66, quality: 'Medium', description: 'Importance-matrix 3-bit', recommended: false },
  'IQ4_XS': { bitsPerWeight: 4.25, quality: 'Medium-Good', description: 'Importance-matrix 4-bit, smaller than Q4_K_S', recommended: false },
  'IQ4_NL': { bitsPerWeight: 4.5, quality: 'Medium-Good', description: 'Importance-matrix 4-bit non-linear', recommended: false },
  'TQ1_0': { bitsPerWeight: 1.69, quality: 'Very Low', description: 'Ternary quantization (ternary-trained models only)', recommended: false },
  'TQ2_0': { bitsPerWeight: 2.06, quality: 'Very Low', description: 'Ternary quantization (ternary-trained models only)', recommended: false },
  'Q2_K_L': { bitsPerWeight: 3.0, quality: 'Low', description: 'Q2_K with 8-bit embeddings and output', recommended: false },
  'Q2_K_XL': { bitsPerWeight: 3.0, quality: 'Low', description: 'Dynamic Q2_K with higher-precision key layers', recommended: false },
  'Q3_K_L': { bitsPerWeight: 3.6, quality: 'Medium', description: 'Q3_K with larger attention/output tensors', recommended: false },
  'Q3_K_XL': { bitsPerWeight: 3.75, quality: 'Medium', description: 'Dynamic Q3_K with higher-precision key layers', recommended: false },
  'Q4_1': { bitsPerWeight: 5, quality: 'Medium', description: 'Legacy 4-bit with offset', recommended: false },
  'Q4_K_L': { bitsPerWeight: 4.75, quality: 'Good', description: 'Q4_K_M with 8-bit embeddings and output', recommended: false },
  'Q4_K_XL': { bitsPerWeight: 4.75, quality: 'Good', description: 'Dynamic Q4_K with higher-precision key layers', recommended: false },
  'Q5_0': { bitsPerWeight: 5.5, quality: 'Good', description: 'Legacy 5-bit', recommended: false },
  'Q5_1': { bitsPerWeight: 6, quality: 'Good', description: 'Legacy 5-bit with offset', recommended: false },
  'Q5_K_L': { bitsPerWeight: 5.75, quality: 'High', description: 'Q5_K_M with 8-bit embeddings and output', recommended: false },
  'Q5_K_XL': { bitsPerWeight: 5.75, quality: 'High', description: 'Dynamic Q5_K with higher-precision key layers', recommended: false },
  'Q6_K_L': { bitsPerWeight: 6.75, quality: 'Very High', description: 'Q6_K with 8-bit embeddings and output', recommended: false },
  'Q6_K_XL': { bitsPerWeight: 6.75, quality: 'Very High', description: 'Dynamic Q6_K with higher-precision key layers', recommended: false },
  'Q8_K_XL': { bitsPerWeight: 8.75, quality: 'Excellent', description: 'Dynamic Q8 with 16-bit key layers', recommended: false },
  'MXFP4': { bitsPerWeight: 4.25, quality: 'Good', description: 'Microscaling FP4 (gpt-oss style)', recommended: false },
  'BF16': { bitsPerWeight: 16, quality: 'Original', description: 'Unquantized bfloat16', recommended: false },
  'F16': { bitsPerWeight: 16, quality: 'Original', description: 'Unquantized float16', recommended: false },
  'F32': { bitsPerWeight: 32, quality: 'Original', description: 'Unquantized float32', recommended: false },
};
