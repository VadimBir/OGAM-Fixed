import {
  isAccelerableQuant,
  isNpuAccelerableQuant,
  ggufAcceleratorCoverage,
  ggufAcceleratorLabel,
  ggufCpuFallbackWarning,
  flashAttnPlacement,
  modelSupportsNpuGpu,
  isDownloadedModelAccelerable,
  findAccelerableModel,
  recommendedAccelerator,
  planAcceleration,
  acceleratedBackendFor,
  acceleratedSearchQuery,
} from '../../../src/utils/acceleration';
import { INFERENCE_BACKENDS } from '../../../src/types';

const file = (name: string, quantization: string) => ({ name, size: 1, quantization, downloadUrl: '' });

describe('isAccelerableQuant (GPU = OpenCL weight types, ggml-opencl.cpp:8795-8852)', () => {
  it('accepts every quant whose weight types all run on the OpenCL GPU (case-insensitive)', () => {
    for (const q of ['Q4_0', 'q4_0', 'Q4_1', 'Q4_K_M', 'Q4_K_S', 'Q4_K_L', 'Q5_K_M', 'Q5_0', 'Q5_1', 'Q6_K', 'Q6_K_L', 'Q8_0', 'IQ4_NL', 'MXFP4', 'F16', 'BF16']) {
      expect(isAccelerableQuant(q)).toBe(true);
    }
  });

  it('rejects quants holding a CPU-only weight type (Q4 is not Q4: IQ4_XS is not GPU)', () => {
    for (const q of ['IQ4_XS', 'Q3_K_M', 'Q3_K_S', 'Q3_K_L', 'Q2_K', 'IQ3_M', 'IQ2_XXS', 'IQ1_S', 'TQ1_0', 'Q4_0_4_4']) {
      expect(isAccelerableQuant(q)).toBe(false);
    }
  });

  it('rejects unknown mixes instead of guessing (UD-*, *_XL, bare Q4, Unknown, empty)', () => {
    for (const q of ['UD-Q4_K_XL', 'Q4_K_XL', 'Q4', 'Unknown', '', undefined, null]) {
      expect(isAccelerableQuant(q)).toBe(false);
    }
  });
});

describe('isNpuAccelerableQuant (HTP weight types, ggml-hexagon.cpp:4945-5011)', () => {
  it('accepts quants whose weight types all run on the Hexagon NPU', () => {
    for (const q of ['Q4_0', 'Q4_1', 'Q4_K_M', 'Q6_K', 'Q8_0', 'MXFP4', 'F16']) expect(isNpuAccelerableQuant(q)).toBe(true);
  });

  it('rejects quants holding Q5_K / Q5_0 / BF16 (GPU yes, NPU no)', () => {
    for (const q of ['Q4_K_S', 'Q5_K_M', 'Q5_0', 'IQ4_NL', 'BF16', 'IQ4_XS']) expect(isNpuAccelerableQuant(q)).toBe(false);
  });
});

describe('ggufAcceleratorCoverage / ggufAcceleratorLabel / ggufCpuFallbackWarning', () => {
  it('labels GPU+NPU, GPU-only, CPU fallback, and nothing for an unknown mix', () => {
    expect(ggufAcceleratorLabel('Q4_K_M')).toBe('GPU+NPU');
    expect(ggufAcceleratorLabel('Q4_0')).toBe('GPU+NPU');
    expect(ggufAcceleratorLabel('Q5_K_M')).toBe('GPU');
    expect(ggufAcceleratorLabel('IQ4_XS')).toBe('CPU fallback');
    expect(ggufAcceleratorLabel('Q3_K_M')).toBe('CPU fallback');
    expect(ggufAcceleratorLabel('UD-Q4_K_XL')).toBeUndefined();
    expect(ggufAcceleratorLabel(undefined)).toBeUndefined();
  });

  it('reports coverage per accelerator', () => {
    expect(ggufAcceleratorCoverage('Q4_K_S')).toEqual({ gpu: 'full', npu: 'cpu-fallback' });
    expect(ggufAcceleratorCoverage('IQ4_XS')).toEqual({ gpu: 'cpu-fallback', npu: 'cpu-fallback' });
    expect(ggufAcceleratorCoverage('Q4_K_XL')).toEqual({ gpu: 'unknown', npu: 'unknown' });
  });

  it('warns only for a CPU-fallback quant and names the CPU-only weight types', () => {
    const w = ggufCpuFallbackWarning('iq4_xs');
    expect(w?.title).toBe('Runs partly on CPU');
    expect(w?.message).toContain('IQ4_XS');
    expect(w?.message).not.toContain('Q6_K,');
    expect(ggufCpuFallbackWarning('Q3_K_M')?.message).toContain('Q3_K');
    expect(ggufCpuFallbackWarning('Q4_K_M')).toBeNull();
    expect(ggufCpuFallbackWarning('UD-Q4_K_XL')).toBeNull();
    expect(ggufCpuFallbackWarning(undefined)).toBeNull();
  });
});

describe('flashAttnPlacement (OpenCL ggml-opencl.cpp:8930-9032, HTP ggml-hexagon.cpp:4140-4186)', () => {
  it('OpenCL: same-type f16/q8_0/q4_0 KV runs on the GPU on a non-A7X Adreno', () => {
    for (const t of ['f16', 'q8_0', 'q4_0']) {
      expect(flashAttnPlacement({ backend: 'opencl', cacheTypeK: t, cacheTypeV: t, adrenoModel: 830 }).onAccelerator).toBe(true);
    }
  });

  it('OpenCL: Adreno 730/740/750 run same-type KV flash attention on the CPU', () => {
    for (const m of [730, 740, 750]) {
      const r = flashAttnPlacement({ backend: 'opencl', cacheTypeK: 'q8_0', cacheTypeV: 'q8_0', adrenoModel: m });
      expect(r.onAccelerator).toBe(false);
      expect(r.reason).toContain(`Adreno ${m}`);
    }
    // f32 KV is cast to f16 before FA, so it does not escape the A7X decline.
    expect(flashAttnPlacement({ backend: 'opencl', cacheTypeK: 'f32', cacheTypeV: 'f32', adrenoModel: 740 }).onAccelerator).toBe(false);
  });

  it('OpenCL: unknown Adreno model -> unknown for same-type KV; mixed pair stays on GPU but is flagged slow', () => {
    expect(flashAttnPlacement({ backend: 'opencl', cacheTypeK: 'f16', cacheTypeV: 'f16' }).onAccelerator).toBe('unknown');
    const mixed = flashAttnPlacement({ backend: 'opencl', cacheTypeK: 'q8_0', cacheTypeV: 'f16', adrenoModel: 740 });
    expect(mixed.onAccelerator).toBe(true);
    expect(mixed.reason).toContain('slow');
  });

  it('OpenCL: a KV type without an FA kernel runs on the CPU', () => {
    expect(flashAttnPlacement({ backend: 'opencl', cacheTypeK: 'q5_1', cacheTypeV: 'q5_1', adrenoModel: 830 }).onAccelerator).toBe(false);
  });

  it('HTP: f16/q8_0 KV on the NPU, q4_0 on the CPU', () => {
    expect(flashAttnPlacement({ backend: 'htp', cacheTypeK: 'q8_0', cacheTypeV: 'f16' }).onAccelerator).toBe(true);
    const q4 = flashAttnPlacement({ backend: 'htp', cacheTypeK: 'q4_0', cacheTypeV: 'q4_0' });
    expect(q4.onAccelerator).toBe(false);
    expect(q4.reason).toContain('CPU');
  });
});

describe('modelSupportsNpuGpu', () => {
  it('true when the model offers a GGUF file whose quant runs fully on the GPU', () => {
    expect(modelSupportsNpuGpu({ files: [file('m-IQ4_XS.gguf', 'IQ4_XS'), file('m-Q4_0.gguf', 'Q4_0')] })).toBe(true);
    expect(modelSupportsNpuGpu({ files: [file('m-Q8_0.gguf', 'Q8_0')] })).toBe(true);
    expect(modelSupportsNpuGpu({ files: [file('m-Q4_K_M.gguf', 'Q4_K_M'), file('m-Q6_K.gguf', 'Q6_K')] })).toBe(true);
  });

  it('true for a LiteRT model (runs on GPU regardless of quant)', () => {
    expect(modelSupportsNpuGpu({ files: [file('gemma-4-E2B-it.litertlm', 'LiteRT')] })).toBe(true);
    expect(modelSupportsNpuGpu({ files: [file('x.litertlm', 'Unknown')] })).toBe(true);
  });

  it('false when every quant offered holds a CPU-only weight type', () => {
    expect(modelSupportsNpuGpu({ files: [file('m-IQ4_XS.gguf', 'IQ4_XS'), file('m-Q3_K_M.gguf', 'Q3_K_M')] })).toBe(false);
  });

  it('false when there are no files', () => {
    expect(modelSupportsNpuGpu({ files: [] })).toBe(false);
    expect(modelSupportsNpuGpu({ files: undefined as never })).toBe(false);
  });
});

describe('isDownloadedModelAccelerable / findAccelerableModel', () => {
  const m = (id: string, name: string, engine: string, quantization: string) => ({ id, name, engine, quantization });

  it('true only for llama models in an accelerable quant (GPU by default, NPU on request)', () => {
    expect(isDownloadedModelAccelerable({ engine: 'llama', quantization: 'Q4_0' })).toBe(true);
    expect(isDownloadedModelAccelerable({ engine: 'llama', quantization: 'Q4_K_M' })).toBe(true);
    expect(isDownloadedModelAccelerable({ engine: 'llama', quantization: 'IQ4_XS' })).toBe(false);
    expect(isDownloadedModelAccelerable({ engine: 'litert', quantization: 'Q4_0' })).toBe(false);
    expect(isDownloadedModelAccelerable({ engine: 'llama', quantization: 'Q5_K_M' })).toBe(true);
    expect(isDownloadedModelAccelerable({ engine: 'llama', quantization: 'Q5_K_M' }, 'npu')).toBe(false);
  });

  it('finds an accelerable build of the SAME base model (same display name), not a different one', () => {
    const models = [
      m('e4b/IQ4_XS', 'gemma-4-E4B-it-GGUF', 'llama', 'IQ4_XS'),
      m('e2b/Q4_0', 'gemma-4-E2B-it-GGUF', 'llama', 'Q4_0'),   // different model — must NOT match
      m('e4b/Q4_0', 'gemma-4-E4B-it-GGUF', 'llama', 'Q4_0'),   // same model, accelerable — the match
    ];
    const active = { id: 'e4b/IQ4_XS', name: 'gemma-4-E4B-it-GGUF' };
    expect(findAccelerableModel(models, active)?.id).toBe('e4b/Q4_0');
  });

  it('NPU target skips a GPU-only build (Q5_K_M) of the same model', () => {
    const models = [m('l/Q5_K_M', 'Llama-3-8B', 'llama', 'Q5_K_M'), m('l/Q4_K_M', 'Llama-3-8B', 'llama', 'Q4_K_M')];
    expect(findAccelerableModel(models, { id: 'l/IQ4_XS', name: 'Llama-3-8B' }, 'npu')?.id).toBe('l/Q4_K_M');
  });

  it('returns null when only a DIFFERENT model has an accelerable build (no cross-model downgrade)', () => {
    const models = [m('e2b/Q4_0', 'gemma-4-E2B-it-GGUF', 'llama', 'Q4_0')];
    expect(findAccelerableModel(models, { id: 'e4b/Q4_K_M', name: 'gemma-4-E4B-it-GGUF' })).toBeNull();
    expect(findAccelerableModel(models, undefined)).toBeNull();
  });
});

describe('recommendedAccelerator (GPU-first; NPU only for Llama-no-GPU)', () => {
  it('recommends GPU whenever the device has one, regardless of model', () => {
    expect(recommendedAccelerator({ hasNpu: true, hasGpu: true }, 'gemma-4-E4B-it-GGUF')).toBe('gpu');
    expect(recommendedAccelerator({ hasNpu: false, hasGpu: true }, 'Llama-3-8B')).toBe('gpu');
  });

  it('recommends NPU only when there is no GPU AND the model is Llama-family', () => {
    expect(recommendedAccelerator({ hasNpu: true, hasGpu: false }, 'Llama-3-8B-Instruct')).toBe('npu');
    expect(recommendedAccelerator({ hasNpu: true, hasGpu: false }, 'gemma-4-E4B-it-GGUF')).toBeNull();
    expect(recommendedAccelerator({ hasNpu: false, hasGpu: false }, 'Llama-3-8B')).toBeNull();
  });
});

describe('planAcceleration', () => {
  // Default test device: GPU (Adreno) — so the recommended accelerator is GPU.
  const base = {
    engine: 'llama', isRemote: false, inferenceBackend: INFERENCE_BACKENDS.CPU,
    capability: { hasNpu: false, hasGpu: true }, activeQuant: 'IQ4_XS',
    modelName: 'gemma-4-E4B-it-GGUF', downloadedAccelerable: null,
  };

  it('enable (GPU): the active model is already an accelerable quant', () => {
    const plan = planAcceleration({ ...base, activeQuant: 'Q4_0' });
    expect(plan.action).toBe('enable');
    expect(plan.backend).toBe('gpu');
    expect(planAcceleration({ ...base, activeQuant: 'Q4_K_M' }).action).toBe('enable');
  });

  it('NPU target judges the quant against the NPU: Q5_K_M is not NPU-accelerable', () => {
    const npuOnly = { ...base, capability: { hasNpu: true, hasGpu: false }, modelName: 'Llama-3-8B-Instruct' };
    expect(planAcceleration({ ...npuOnly, activeQuant: 'Q5_K_M' }).action).toBe('download');
    expect(planAcceleration({ ...npuOnly, activeQuant: 'Q4_K_M' }).action).toBe('enable');
  });

  it('switch: CPU-fallback quant active but an accelerable build is downloaded → recommends GPU', () => {
    const plan = planAcceleration({ ...base, downloadedAccelerable: { id: 'x/E4B-Q4_0', name: 'gemma-4-E4B-it-GGUF' } });
    expect(plan.action).toBe('switch');
    expect(plan.backend).toBe('gpu');
    expect(plan.targetModelId).toBe('x/E4B-Q4_0');
  });

  it('download: K-quant active and nothing accelerable downloaded', () => {
    expect(planAcceleration(base).action).toBe('download');
  });

  it('fallback: GPU-capable device but an accelerator was selected and the K-quant runs on CPU', () => {
    const plan = planAcceleration({ ...base, inferenceBackend: INFERENCE_BACKENDS.OPENCL });
    expect(plan.action).toBe('download');
    expect(plan.fellBack).toBe(true);
  });

  it('NEVER recommends NPU for Gemma even on an NPU device with no GPU (hidden)', () => {
    expect(planAcceleration({ ...base, capability: { hasNpu: true, hasGpu: false } }).action).toBe('hidden');
  });

  it('recommends NPU for a Llama model on an NPU-only device', () => {
    const plan = planAcceleration({
      ...base, capability: { hasNpu: true, hasGpu: false }, modelName: 'Llama-3-8B-Instruct', activeQuant: 'Q4_0',
    });
    expect(plan.action).toBe('enable');
    expect(plan.backend).toBe('npu');
  });

  it('hidden when genuinely accelerated (accelerated backend + accelerable model)', () => {
    expect(planAcceleration({ ...base, inferenceBackend: INFERENCE_BACKENDS.OPENCL, activeQuant: 'Q4_0' }).action).toBe('hidden');
  });

  it('hidden when the device has neither NPU nor GPU', () => {
    expect(planAcceleration({ ...base, capability: { hasNpu: false, hasGpu: false } }).action).toBe('hidden');
  });

  it('hidden for LiteRT and remote models', () => {
    expect(planAcceleration({ ...base, engine: 'litert' }).action).toBe('hidden');
    expect(planAcceleration({ ...base, isRemote: true }).action).toBe('hidden');
    expect(planAcceleration({ ...base, engine: undefined }).action).toBe('hidden');
  });
});

describe('acceleratedBackendFor', () => {
  it('maps GPU→OpenCL and NPU→HTP per the recommended accelerator', () => {
    expect(acceleratedBackendFor({ hasNpu: true, hasGpu: true }, 'gemma')).toBe(INFERENCE_BACKENDS.OPENCL); // GPU-first
    expect(acceleratedBackendFor({ hasNpu: true, hasGpu: false }, 'Llama-3-8B')).toBe(INFERENCE_BACKENDS.HTP);
    expect(acceleratedBackendFor({ hasNpu: false, hasGpu: true }, 'anything')).toBe(INFERENCE_BACKENDS.OPENCL);
  });
});

describe('acceleratedSearchQuery', () => {
  it('strips the author prefix and quant suffix, then appends Q4_0', () => {
    expect(acceleratedSearchQuery('unsloth/Qwen3-4B-Instruct-Q4_K_M')).toBe('Qwen3-4B-Instruct Q4_0');
    expect(acceleratedSearchQuery('org/gemma-3-4b-it')).toBe('gemma-3-4b-it Q4_0');
  });

  it('falls back to a bare Q4_0 search when the id is missing', () => {
    expect(acceleratedSearchQuery(undefined)).toBe('Q4_0');
    expect(acceleratedSearchQuery(null)).toBe('Q4_0');
  });
});
