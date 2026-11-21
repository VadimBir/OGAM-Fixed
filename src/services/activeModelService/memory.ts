/**
 * Text-model memory estimate for ActiveModelService (residency gate + register).
 * Pure/standalone: state comes from the store and the model record.
 */

import { DownloadedModel, INFERENCE_BACKENDS } from '../../types';
import { loadLlamaModelInfo } from 'llama.rn';
import { buildModelParams, modelMaxContextFromMetadata, resolveUseMmap } from '../llmHelpers';
import { estimateLlamaMemory } from '../llamaMemoryEstimate';
import { enabledTextLoraBytes } from '../textLora';
import { hardwareService } from '../hardware';
import { textOverheadMultiplier } from './types';
import { useAppStore } from '../../stores';
import {
  TextMemoryParts,
  TextMemoryFootprint,
  textMemoryFootprint,
  textWeightsReclaimable,
} from '../memoryBudget';

/**
 * A text model's memory inputs, computed once per load:
 *  - parts: the GGUF split (weights vs anonymous mmproj + LoRA + KV + compute); null when the model
 *    is not llama or its header can't be read;
 *  - fallbackMB: the file-size heuristic used when parts is null (already padded, so it is both
 *    the resident and the peak charge);
 *  - predictedReclaimable: whether the load is expected to keep the weights as reclaimable mmapped
 *    pages (same mmap decision as buildModelParams; Android also needs the CPU backend).
 */
export interface TextModelMemoryEstimate {
  parts: TextMemoryParts | null;
  fallbackMB: number;
  predictedReclaimable: boolean;
}

export async function estimateTextModelMemory(model: DownloadedModel): Promise<TextModelMemoryEstimate> {
  const settings = useAppStore.getState().settings;
  const fallbackMB = Math.ceil(
    hardwareService.estimateModelRam(model, textOverheadMultiplier(settings.inferenceBackend)) / (1024 * 1024),
  );
  const none: TextModelMemoryEstimate = { parts: null, fallbackMB, predictedReclaimable: false };
  if (model.engine !== 'llama') return none;
  try {
    const metadata = await loadLlamaModelInfo(model.filePath) as Record<string, unknown>;
    const params = buildModelParams(model.filePath, { ...settings, modelMaxContext: modelMaxContextFromMetadata(metadata) });
    // Enabled text LoRA adapters load beside the weights (llama-adapter.cpp:335, same buffer type).
    const loraBytes = await enabledTextLoraBytes(undefined, params.loraList.map(l => l.path));
    const mmProjBytes = model.mmProjFileSize ?? 0;
    // params.cacheType is the type actually sent (an unset setting resolves to q8_0, not f16).
    const estimate = estimateLlamaMemory(metadata, {
      ctxLen: params.ctxLen, cacheType: params.cacheType, cacheTypeV: params.cacheTypeV, nBatch: params.nBatch,
      weightsBytes: model.fileSize + mmProjBytes + loraBytes,
    });
    if (!estimate) return none;
    const useMmap = resolveUseMmap(model.filePath, settings.mmapMode, {
      modelFileSizeBytes: model.fileSize,
      liveAvailableMB: hardwareService.getAvailableMemoryGB() * 1024,
    });
    // Android: only the CPU backend keeps the tensors in the mapping (llm.ts initConfiguredContext
    // offloads 0 layers there); OpenCL / HTP copy them into anonymous buffers.
    const backend = settings.inferenceBackend ?? INFERENCE_BACKENDS.CPU;
    const gpuLayers = backend === INFERENCE_BACKENDS.CPU ? 0 : 1;
    return {
      parts: {
        weightsBytes: model.fileSize,
        anonBytes: mmProjBytes + loraBytes + estimate.kvBytes + estimate.computeBytes,
      },
      fallbackMB,
      predictedReclaimable: textWeightsReclaimable({ useMmap, gpuLayers }),
    };
  } catch {
    return none;
  }
}

/** The MB charges for `est` given whether the weights are reclaimable (predicted or actual). */
export function textFootprintOf(est: TextModelMemoryEstimate, weightsReclaimable: boolean): TextMemoryFootprint {
  if (!est.parts) {
    const mb = est.fallbackMB;
    return { residentMB: mb, peakMB: mb, liveResidentMB: mb, livePeakMB: mb };
  }
  return textMemoryFootprint(est.parts, weightsReclaimable);
}

/** Peak MB a load of `model` is charged (the residency gate's text number; also shown in the selector). */
export async function estimateTextModelMemoryMB(model: DownloadedModel): Promise<number> {
  const est = await estimateTextModelMemory(model);
  return textFootprintOf(est, est.predictedReclaimable).peakMB;
}
