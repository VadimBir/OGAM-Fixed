/**
 * Hardware-acceleration eligibility — the SINGLE source of truth for "can this
 * model actually use the GPU/NPU?".
 *
 * Grounded in the native code llama.rn 0.13.0-rc.4 ships (vendored llama.cpp b11010 @ 4bc272f,
 * node_modules/llama.rn/vendor/VERSIONS; the Android OpenCL .so carries the same "4bc272f" string):
 *  - GPU (OpenCL, Adreno only): weight types the backend runs MUL_MAT for are
 *    vendor/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp:8795-8852 — F32, F16, BF16, Q1_0, Q4_0,
 *    Q4_1, Q5_0, Q5_1, MXFP4, IQ4_NL, Q4_K, Q5_K, Q6_K, Q8_0. Anything else (Q2_K, Q3_K, IQ1/2/3_*,
 *    IQ4_XS, TQ*) returns false there and that matmul runs on the CPU.
 *  - NPU (HTP / Hexagon): ggml-hexagon/ggml-hexagon.cpp:4945-5011 — F32, F16, Q4_0, Q4_1, Q8_0,
 *    IQ4_NL, MXFP4, Q4_K, Q6_K. No Q5_K / Q5_0 / Q5_1 / BF16.
 *  - LiteRT (.litertlm) models run on the GPU natively.
 * A GGUF file label (Q4_K_M, IQ4_XS, ...) is a MIX of tensor types, so "Q4" is not one thing: the
 * mix per label comes from llama-quant.cpp (output.weight -> Q6_K unless the file is Q8_0, :456-477;
 * Q4_K_M ffn_down/attn_v -> Q6_K, :556-557 / :612-618; Q4_K_S -> Q5_K, :558 / :624-626; IQ4_NL /
 * IQ4_XS attn_v -> Q5_K, :553-555). The label is a prediction; the GGUF header holds the truth.
 */
import { ModelInfo, ModelFile, InferenceBackend, INFERENCE_BACKENDS } from '../types';
import { isLiteRTFileName } from './modelHelpers';

/** ggml weight types the OpenCL backend runs MUL_MAT for (ggml-opencl.cpp:8795-8852). */
const OPENCL_WEIGHT_TYPES: ReadonlySet<string> = new Set([
  'F32', 'F16', 'BF16', 'Q1_0', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'MXFP4', 'IQ4_NL', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_0',
]);
/** ggml weight types the Hexagon NPU runs MUL_MAT for (ggml-hexagon.cpp:4945-5011). */
const HTP_WEIGHT_TYPES: ReadonlySet<string> = new Set([
  'F32', 'F16', 'Q4_0', 'Q4_1', 'Q8_0', 'IQ4_NL', 'MXFP4', 'Q4_K', 'Q6_K',
]);

/**
 * Weight types a llama.cpp-quantized file of this label holds (dense, non-Falcon models; llama-quant.cpp
 * :456-680 and the ftype default types :849-886). `_L` labels are the base mix with Q8_0 embeddings
 * and output (constants/models.ts QUANTIZATION_INFO). Labels absent here (UD-*, *_XL dynamic mixes,
 * bare Q4, Unknown) have no fixed mix: their support is unknown until the GGUF header is read.
 */
const FILE_LABEL_TENSOR_TYPES: Readonly<Record<string, readonly string[]>> = {
  F32: ['F32'], F16: ['F16'], BF16: ['BF16'],
  Q8_0: ['Q8_0'],
  Q6_K: ['Q6_K'], Q6_K_L: ['Q6_K', 'Q8_0'],
  Q5_K_M: ['Q5_K', 'Q6_K'], Q5_K_S: ['Q5_K', 'Q6_K'], Q5_K: ['Q5_K', 'Q6_K'], Q5_K_L: ['Q5_K', 'Q6_K', 'Q8_0'],
  Q5_0: ['Q5_0', 'Q5_1', 'Q6_K'], Q5_1: ['Q5_1', 'Q6_K'],
  // Q4_K_M: Q4_K + Q6_K. A fused attn_qkv tensor (phi3-style) becomes Q5_K (:659) - not on the NPU.
  Q4_K_M: ['Q4_K', 'Q6_K'], Q4_K: ['Q4_K', 'Q6_K'], Q4_K_L: ['Q4_K', 'Q6_K', 'Q8_0'],
  Q4_K_S: ['Q4_K', 'Q5_K', 'Q6_K'],
  // Q4_0 with an imatrix stores the first ffn_down layers as Q4_1 (:627-632).
  Q4_0: ['Q4_0', 'Q4_1', 'Q6_K'], Q4_1: ['Q4_1', 'Q6_K'],
  IQ4_NL: ['IQ4_NL', 'Q5_K', 'Q6_K'], IQ4_XS: ['IQ4_XS', 'Q5_K', 'Q6_K'],
  MXFP4: ['MXFP4', 'Q8_0'], MXFP4_MOE: ['MXFP4', 'Q8_0'],
  Q3_K_S: ['Q3_K', 'Q6_K'], Q3_K_M: ['Q3_K', 'Q4_K', 'Q5_K', 'Q6_K'], Q3_K_L: ['Q3_K', 'Q5_K', 'Q6_K'], Q3_K: ['Q3_K', 'Q6_K'],
  Q2_K: ['Q2_K', 'Q3_K', 'Q4_K', 'Q6_K'], Q2_K_S: ['Q2_K', 'Q4_K', 'Q6_K'], Q2_K_L: ['Q2_K', 'Q3_K', 'Q4_K', 'Q8_0'],
  IQ3_M: ['IQ3_S', 'Q4_K', 'Q6_K'], IQ3_S: ['IQ3_S', 'Q6_K'], IQ3_XS: ['IQ3_S', 'IQ3_XXS', 'Q6_K'], IQ3_XXS: ['IQ3_XXS', 'IQ3_S', 'Q6_K'],
  IQ2_M: ['IQ2_S', 'IQ3_S', 'Q5_K'], IQ2_S: ['IQ2_XS', 'IQ3_S', 'Q5_K'], IQ2_XS: ['IQ2_XS', 'Q2_K', 'Q5_K'], IQ2_XXS: ['IQ2_XXS', 'Q2_K', 'Q5_K'],
  IQ1_M: ['IQ1_M', 'Q2_K', 'Q5_K'], IQ1_S: ['IQ1_S', 'Q2_K', 'Q5_K'],
  TQ1_0: ['TQ1_0', 'Q4_K', 'Q6_K'], TQ2_0: ['TQ2_0', 'Q4_K', 'Q6_K'],
  // Interleaved Q4_0 types were removed from ggml (constants/models.ts: llama.h:151-153).
  Q4_0_4_4: ['Q4_0_4_4'], Q4_0_4_8: ['Q4_0_4_8'], Q4_0_8_8: ['Q4_0_8_8'],
};

/** 'full' = every weight type of the label runs on the accelerator; 'cpu-fallback' = at least one
 *  type runs on the CPU; 'unknown' = the label has no fixed mix (read the GGUF header). */
export type AcceleratorCoverage = 'full' | 'cpu-fallback' | 'unknown';

function coverage(quant: string | undefined | null, supported: ReadonlySet<string>): AcceleratorCoverage {
  const types = quant ? FILE_LABEL_TENSOR_TYPES[quant.toUpperCase()] : undefined;
  if (!types) return 'unknown';
  return types.every(t => supported.has(t)) ? 'full' : 'cpu-fallback';
}

/** GPU (OpenCL) and NPU (HTP) coverage of a GGUF file quant label (parseQuantization output). */
export function ggufAcceleratorCoverage(quant: string | undefined | null): { gpu: AcceleratorCoverage; npu: AcceleratorCoverage } {
  return { gpu: coverage(quant, OPENCL_WEIGHT_TYPES), npu: coverage(quant, HTP_WEIGHT_TYPES) };
}

/**
 * Short picker label: 'GPU+NPU' (runs fully on both), 'GPU' (fully on the GPU, part on CPU under the
 * NPU), 'CPU fallback' (part of the weights run on the CPU even with the GPU selected), or undefined
 * when the label's mix is unknown (never guessed).
 */
export function ggufAcceleratorLabel(quant: string | undefined | null): 'GPU+NPU' | 'GPU' | 'CPU fallback' | undefined {
  const c = ggufAcceleratorCoverage(quant);
  if (c.gpu === 'unknown') return undefined;
  if (c.gpu === 'cpu-fallback') return 'CPU fallback';
  return c.npu === 'full' ? 'GPU+NPU' : 'GPU';
}

/** Confirm-download copy for a GGUF whose quant makes part of the model run on the CPU; null otherwise. */
export function ggufCpuFallbackWarning(quant: string | undefined | null): { title: string; message: string } | null {
  if (ggufAcceleratorCoverage(quant).gpu !== 'cpu-fallback') return null;
  const types = FILE_LABEL_TENSOR_TYPES[String(quant).toUpperCase()] ?? [];
  const cpuTypes = types.filter(t => !OPENCL_WEIGHT_TYPES.has(t));
  return {
    title: 'Runs partly on CPU',
    message: `${String(quant).toUpperCase()} stores weights as ${cpuTypes.join(', ')}, which the GPU and NPU backends cannot run, so those layers run on the CPU. Pick Q4_0, Q4_K_M, Q6_K or Q8_0 to run fully on the GPU.`,
  };
}

/** True when a GGUF quant label runs fully on the GPU (OpenCL). Unknown mixes are false. */
export function isAccelerableQuant(quant: string | undefined | null): boolean {
  return ggufAcceleratorCoverage(quant).gpu === 'full';
}

/** True when a GGUF quant label runs fully on the NPU (HTP). Unknown mixes are false. */
export function isNpuAccelerableQuant(quant: string | undefined | null): boolean {
  return ggufAcceleratorCoverage(quant).npu === 'full';
}

/**
 * Where flash attention runs for a KV cache type pair, from the backends' supports_op:
 *  - OpenCL (ggml-opencl.cpp:8930-9032): K/V f16, q8_0 or q4_0 (same type), or a mixed pair of those
 *    (mixed = host-side F16->F32 conversion every call, :16461-16511, slow). Adreno 730/740/750 ("A7X",
 *    :292-296) decline every same-type pair (:8984-8987, :9007-9010) and an E17 compiler declines all
 *    FA (:8931-8934). f32 KV is cast to f16 before FA (src/llama-graph.cpp:2624-2630), so it is no way out.
 *  - HTP (ggml-hexagon.cpp:4140-4186): K and V each f16 or q8_0; q4_0 runs on the CPU.
 * With flash_attn_type 'on' a declined FA op runs on the CPU with no warning: llama.cpp only probes
 * FA placement for 'auto' (src/llama-context.cpp:555-558). 'unknown' when the Adreno model is unknown
 * and it decides the answer.
 */
export function flashAttnPlacement(opts: {
  backend: 'opencl' | 'htp';
  cacheTypeK: string;
  cacheTypeV: string;
  /** Adreno model number (e.g. 740) when known. */
  adrenoModel?: number;
}): { onAccelerator: boolean | 'unknown'; reason: string } {
  // f32 K/V reach FA as f16 (src/llama-graph.cpp:2624-2630).
  const asFaType = (t: string) => (t.toLowerCase() === 'f32' ? 'f16' : t.toLowerCase());
  const k = asFaType(opts.cacheTypeK);
  const v = asFaType(opts.cacheTypeV);
  if (opts.backend === 'htp') {
    const ok = (t: string) => t === 'f16' || t === 'q8_0';
    return ok(k) && ok(v)
      ? { onAccelerator: true, reason: 'NPU flash attention supports f16/q8_0 KV' }
      : { onAccelerator: false, reason: `NPU flash attention supports only f16/q8_0 KV (got K ${k}, V ${v}); attention runs on the CPU` };
  }
  const kvOk = (t: string) => t === 'f16' || t === 'q8_0' || t === 'q4_0';
  if (!kvOk(k) || !kvOk(v)) {
    return { onAccelerator: false, reason: `GPU flash attention has no kernel for K ${k} / V ${v}; attention runs on the CPU` };
  }
  if (k !== v) {
    return { onAccelerator: true, reason: `GPU flash attention with mixed K ${k} / V ${v} converts the f16 side to f32 on the host every call (slow)` };
  }
  if (opts.adrenoModel == null) {
    return { onAccelerator: 'unknown', reason: 'Adreno 730/740/750 run same-type KV flash attention on the CPU; the GPU model is unknown' };
  }
  if (opts.adrenoModel === 730 || opts.adrenoModel === 740 || opts.adrenoModel === 750) {
    return { onAccelerator: false, reason: `Adreno ${opts.adrenoModel} runs flash attention with ${k} KV on the CPU (driver compiler crash guard)` };
  }
  return { onAccelerator: true, reason: `GPU flash attention supports ${k} KV` };
}

/** A LiteRT model file runs on the GPU regardless of GGUF quant. */
function isLiteRTFile(file: ModelFile): boolean {
  return isLiteRTFileName(file.name) || file.quantization?.toLowerCase() === 'litert';
}

/**
 * Whether a model can run on the GPU/NPU: it ships a LiteRT file, or a GGUF file whose quant runs
 * fully on the GPU (isAccelerableQuant: Q4_0, Q4_K_M, Q6_K, Q8_0, ...). A model that only offers
 * quants with CPU-only weight types (IQ4_XS, Q3_K_*, IQ2_*, ...) returns false — selecting the GPU
 * for it would run part of it on the CPU, so we neither badge nor prioritize it.
 */
export function modelSupportsNpuGpu(model: Pick<ModelInfo, 'files'>): boolean {
  return (model.files ?? []).some(f => isLiteRTFile(f) || isAccelerableQuant(f.quantization));
}

/** The device's llama.rn acceleration options (from hardwareService.getAccelerationCapability). */
export interface AccelerationCapability {
  hasNpu: boolean;
  hasGpu: boolean;
}

/** A downloaded llama model whose quant runs fully on the GPU, or on the NPU when `backend` is 'npu'. */
export function isDownloadedModelAccelerable(m: { engine?: string; quantization?: string }, backend: 'gpu' | 'npu' = 'gpu'): boolean {
  if (m.engine !== 'llama') return false;
  return backend === 'npu' ? isNpuAccelerableQuant(m.quantization) : isAccelerableQuant(m.quantization);
}

/**
 * An accelerable (fully GPU-capable quant) build of the SAME base model as `active`, if one is
 * downloaded. Must be the same model — suggesting a different, smaller model (E2B when
 * the user loaded E4B) is a downgrade, not "run your model on the GPU/NPU". The display
 * name is derived from the repo id, so every quant of a repo shares it (e.g. both E4B
 * quants are "gemma-4-E4B-it-GGUF"), which cleanly separates E4B from E2B.
 */
export function findAccelerableModel<T extends { id: string; name: string; engine?: string; quantization?: string }>(
  models: T[],
  active: { id: string; name: string } | undefined,
  backend: 'gpu' | 'npu' = 'gpu',
): T | null {
  if (!active) return null;
  return models.find(
    m => m.id !== active.id && m.name === active.name && isDownloadedModelAccelerable(m, backend),
  ) ?? null;
}

/** Llama-family model (name-based) — the only family we'll suggest the experimental NPU for. */
function isLlamaFamily(modelName: string | undefined): boolean {
  return /llama/i.test(modelName ?? '');
}

/**
 * The accelerator we're willing to RECOMMEND for this model, or null (don't nudge):
 *  - GPU (OpenCL) whenever the device has it — Adreno only: ggml-opencl rejects any other GPU
 *    family (ggml-opencl.cpp:6393-6396) and llama.rn loads the OpenCL library only with an
 *    Adreno hint (android/.../RNLlama.java:193-201).
 *  - NPU (HTP) ONLY on a device with no GPU AND only for Llama-family models — the NPU
 *    is experimental and broken for e.g. Gemma, so we never steer those to it.
 * Everything else → null (no recommendation), so we don't push a backend that won't help.
 */
export function recommendedAccelerator(
  capability: AccelerationCapability,
  modelName: string | undefined,
): 'gpu' | 'npu' | null {
  if (capability.hasGpu) return 'gpu';
  if (capability.hasNpu && isLlamaFamily(modelName)) return 'npu';
  return null;
}

/**
 * What the chat acceleration tip should offer, given the device, the active model, and
 * what's already downloaded:
 *  - `enable`   — on CPU with an already-accelerable model (isAccelerableQuant, or
 *                 isNpuAccelerableQuant for the NPU); flip to the recommended backend.
 *  - `switch`   — the active quant has weight types the accelerator can't run (IQ4_XS,
 *                 Q3_K_M, ...) but an accelerable build of the SAME model is downloaded.
 *  - `download` — such a quant is active and no accelerable build exists locally.
 *  - `hidden`   — remote/LiteRT model, no RECOMMENDABLE accelerator (see
 *                 recommendedAccelerator — GPU-first, NPU only for Llama), or already
 *                 genuinely accelerated.
 *
 * `fellBack` distinguishes the two ways `switch`/`download` arise: on CPU it's a "go
 * faster" nudge; when the user HAS selected an accelerator but the active quant can't
 * fully use it, it's a "we're on CPU" warning. `backend` is the recommended accelerator so the
 * copy names the right one (never NPU on a GPU device).
 */
type AccelerationAction = 'enable' | 'switch' | 'download' | 'hidden';

export interface AccelerationPlan {
  action: AccelerationAction;
  /** The accelerator being recommended — drives the copy (GPU vs NPU). */
  backend: 'gpu' | 'npu';
  /** True when an accelerated backend is selected but the active model can't use it. */
  fellBack: boolean;
  /** For `switch`: the downloaded accelerable model to activate. */
  targetModelId?: string;
  targetModelName?: string;
}

function isAcceleratedBackend(backend: InferenceBackend | undefined): boolean {
  return backend === INFERENCE_BACKENDS.HTP || backend === INFERENCE_BACKENDS.OPENCL;
}

export function planAcceleration(params: {
  engine: string | undefined;
  isRemote: boolean;
  inferenceBackend: InferenceBackend | undefined;
  capability: AccelerationCapability;
  activeQuant: string | undefined;
  modelName: string | undefined;
  downloadedAccelerable: { id: string; name: string } | null;
}): AccelerationPlan {
  const { engine, isRemote, inferenceBackend, capability, activeQuant, modelName, downloadedAccelerable } = params;
  const rec = recommendedAccelerator(capability, modelName);
  const hidden: AccelerationPlan = { action: 'hidden', backend: rec ?? 'gpu', fellBack: false };
  if (isRemote || engine !== 'llama') return hidden;
  if (!rec) return hidden; // no accelerator worth recommending for this model

  const accelerated = isAcceleratedBackend(inferenceBackend);
  // Judged against the recommended accelerator: Q5_K_M runs fully on the GPU but not on the NPU.
  const activeAccelerable = rec === 'npu' ? isNpuAccelerableQuant(activeQuant) : isAccelerableQuant(activeQuant);
  // Genuinely accelerated (accelerated backend + a model that can use it) → nothing to do.
  if (accelerated && activeAccelerable) return hidden;
  // On CPU with an accelerable model → offer to turn the recommended accelerator on.
  if (!accelerated && activeAccelerable) return { action: 'enable', backend: rec, fellBack: false };

  // Remaining: the active quant has weight types the accelerator cannot run (or an unknown mix).
  // Either on CPU (nudge) or an accelerator is selected but those layers run on the CPU (fallback
  // warning). Route to an accelerable build of the SAME model — switch if downloaded, else download.
  const fellBack = accelerated;
  if (downloadedAccelerable) {
    return { action: 'switch', backend: rec, fellBack, targetModelId: downloadedAccelerable.id, targetModelName: downloadedAccelerable.name };
  }
  return { action: 'download', backend: rec, fellBack };
}

/** The concrete backend to switch to for the recommended accelerator (GPU → OpenCL, NPU → HTP). */
export function acceleratedBackendFor(
  capability: AccelerationCapability,
  modelName: string | undefined,
): InferenceBackend {
  return recommendedAccelerator(capability, modelName) === 'npu'
    ? INFERENCE_BACKENDS.HTP
    : INFERENCE_BACKENDS.OPENCL;
}

/**
 * The HuggingFace search term to prefill on the Models tab so the user can grab an
 * accelerable (Q4_0) build of the model they're on. Strips a trailing quant suffix
 * (…-Q4_K_M) from the model id and its author prefix, then appends the target quant.
 */
export function acceleratedSearchQuery(modelId: string | undefined | null): string {
  if (!modelId) return 'Q4_0';
  const base = modelId.split('/').pop() ?? modelId;
  const withoutQuant = base.replace(/[-_.]?Q\d[_.].*$/i, '').replace(/[-_.]?(gguf|litertlm)$/i, '');
  return `${withoutQuant.trim()} Q4_0`.trim();
}
