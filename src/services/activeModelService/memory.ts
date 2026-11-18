/**
 * Memory check helpers for ActiveModelService.
 * All functions are pure/standalone — they receive state via parameter objects.
 */

import { DownloadedModel, ONNXImageModel } from '../../types';
import { loadLlamaModelInfo } from 'llama.rn';
import { buildModelParams, modelMaxContextFromMetadata } from '../llmHelpers';
import { estimateLlamaMemory } from '../llamaMemoryEstimate';
import { hardwareService } from '../hardware';
import { llmService } from '../llm';
import { liteRTService } from '../litert';
import {
  ModelType,
  MemoryCheckResult,
  MemoryCheckSeverity,
  textOverheadMultiplier,
  IMAGE_MODEL_OVERHEAD_MULTIPLIER,
} from './types';
import { useAppStore } from '../../stores';
import { estimateImageModelRamBytes } from '../imageModelMemory';
import { Platform } from 'react-native';
import logger from '../../utils/logger';
import {
  modelMemoryBudgetMB,
  modelWarningThresholdMB,
  effectiveAvailableMB,
  preloadSafetyMarginMB,
  LoadPolicy,
} from '../memoryBudget';

// ---------------------------------------------------------------------------
// Budget helpers
// ---------------------------------------------------------------------------

// The pre-load check reads the SAME budget owner as the residency manager, so it
// must honour the SAME load policy — otherwise aggressive mode would relax the
// residency gate while the pre-check kept blocking/warning at balanced limits.
const getMemoryBudgetGB = async (policy: LoadPolicy = 'balanced'): Promise<number> => {
  const deviceInfo = await hardwareService.getDeviceInfo();
  const totalMB = deviceInfo.totalMemory / (1024 * 1024);
  return modelMemoryBudgetMB(totalMB, undefined, policy) / 1024;
};

const getMemoryWarningThresholdGB = async (): Promise<number> => {
  const deviceInfo = await hardwareService.getDeviceInfo();
  const totalMB = deviceInfo.totalMemory / (1024 * 1024);
  return modelWarningThresholdMB(totalMB) / 1024;
};

/**
 * Budget + warning threshold for the single-model pre-load check, from the SAME live-RAM reading
 * and functions the authoritative gate uses (llm.ts getMem -> effectiveAvailableMB, minus
 * preloadSafetyMarginMB), so a "Safe to load" here is not followed by a refusal there.
 *
 * Live available RAM already excludes the models that are loaded now, so their estimate (`otherGB`)
 * is added back: the caller compares required + other against this budget, which reduces to
 * required <= live - margin. The warning threshold keeps its ratio to the budget (from the
 * total-RAM tiers). Falls back to the total-RAM budgets, logged, when no live reading is available.
 */
const getLiveBudgetGB = async (
  policy: LoadPolicy,
  otherGB: number,
): Promise<{ budgetGB: number; warningGB: number }> => {
  const deviceInfo = await hardwareService.getDeviceInfo();
  const totalMB = deviceInfo.totalMemory / (1024 * 1024);
  try {
    const raw = await hardwareService.getAppMemoryUsage();
    if (!raw || !Number.isFinite(raw.available) || !Number.isFinite(raw.total)) {
      throw new Error('no live memory reading');
    }
    const liveMB = effectiveAvailableMB(raw.available / (1024 * 1024), raw.total / (1024 * 1024), {
      platform: Platform.OS,
      policy,
    });
    const marginMB = preloadSafetyMarginMB({ platform: Platform.OS, policy });
    const budgetGB = Math.max(0, liveMB - marginMB) / 1024 + otherGB;
    const totalBudgetMB = modelMemoryBudgetMB(totalMB, undefined, policy);
    const ratio = totalBudgetMB > 0 ? modelWarningThresholdMB(totalMB) / totalBudgetMB : 1;
    return { budgetGB, warningGB: budgetGB * Math.min(1, ratio) };
  } catch (e) {
    logger.warn('[MemoryCheck] live RAM reading unavailable, using total-RAM budget:', e);
    return {
      budgetGB: modelMemoryBudgetMB(totalMB, undefined, policy) / 1024,
      warningGB: modelWarningThresholdMB(totalMB) / 1024,
    };
  }
};

// ---------------------------------------------------------------------------
// Size estimators
// ---------------------------------------------------------------------------

export async function estimateTextModelMemoryMB(model: DownloadedModel): Promise<number> {
  const settings = useAppStore.getState().settings;
  const fallback = hardwareService.estimateModelRam(
    model, textOverheadMultiplier(settings.inferenceBackend),
  );
  if (model.engine !== 'llama') return Math.ceil(fallback / (1024 * 1024));
  try {
    const metadata = await loadLlamaModelInfo(model.filePath) as Record<string, unknown>;
    const params = buildModelParams(model.filePath, { ...settings, modelMaxContext: modelMaxContextFromMetadata(metadata) });
    // params.cacheType is the type actually sent (an unset setting resolves to q8_0, not f16).
    const estimate = estimateLlamaMemory(metadata, {
      ctxLen: params.ctxLen, cacheType: params.cacheType, cacheTypeV: params.cacheTypeV, nBatch: params.nBatch,
      weightsBytes: model.fileSize + (model.mmProjFileSize ?? 0),
    });
    if (!estimate) return Math.ceil(fallback / (1024 * 1024));
    return Math.ceil(estimate.totalBytes / (1024 * 1024));
  } catch {
    return Math.ceil(fallback / (1024 * 1024));
  }
}

async function estimateModelMemoryGB(
  model: DownloadedModel | ONNXImageModel,
  type: ModelType,
): Promise<number> {
  if (type === 'text') return (await estimateTextModelMemoryMB(model as DownloadedModel)) / 1024;
  const imageModel = model as ONNXImageModel;
  // sd.cpp: architecture estimate (weight type, flash attention, VAE tiling, W×H); others keep
  // the image load gate's file-size multiplier.
  const estimate = imageModel.backend === 'sdcpp'
    ? estimateImageModelRamBytes(imageModel)
    : hardwareService.estimateImageModelRam?.(imageModel);
  if (estimate != null) return estimate / (1024 * 1024 * 1024);
  const sizeGB = (imageModel.size || 0) / (1024 * 1024 * 1024);
  return sizeGB * IMAGE_MODEL_OVERHEAD_MULTIPLIER;
}

export interface LoadedModelIds {
  loadedTextModelId: string | null;
  loadedImageModelId: string | null;
}

export interface ModelLists {
  downloadedModels: DownloadedModel[];
  downloadedImageModels: ONNXImageModel[];
}

export async function getCurrentlyLoadedMemoryGB(
  ids: LoadedModelIds,
  lists: ModelLists,
): Promise<number> {
  let totalGB = 0;

  if (ids.loadedTextModelId && (llmService.isModelLoaded() || liteRTService.isModelLoaded())) {
    const textModel = lists.downloadedModels.find(m => m.id === ids.loadedTextModelId);
    if (textModel) {
      totalGB += await estimateModelMemoryGB(textModel, 'text');
    }
  }

  if (ids.loadedImageModelId) {
    const imageModel = lists.downloadedImageModels.find(
      m => m.id === ids.loadedImageModelId,
    );
    if (imageModel) {
      totalGB += await estimateModelMemoryGB(imageModel, 'image');
    }
  }

  return totalGB;
}

/** Memory used by OTHER models already loaded (not the one being replaced). */
async function getOtherLoadedMemoryGB(
  modelType: ModelType,
  ids: LoadedModelIds,
  lists: ModelLists,
): Promise<number> {
  let totalGB = 0;
  if (modelType === 'text' && ids.loadedImageModelId) {
    const imageModel = lists.downloadedImageModels.find(
      m => m.id === ids.loadedImageModelId,
    );
    if (imageModel) {
      totalGB += await estimateModelMemoryGB(imageModel, 'image');
    }
  }
  if (modelType === 'image' && ids.loadedTextModelId && (llmService.isModelLoaded() || liteRTService.isModelLoaded())) {
    const textModel = lists.downloadedModels.find(m => m.id === ids.loadedTextModelId);
    if (textModel) {
      totalGB += await estimateModelMemoryGB(textModel, 'text');
    }
  }
  return totalGB;
}

// ---------------------------------------------------------------------------
// checkMemoryForModel
// ---------------------------------------------------------------------------

export interface CheckMemoryParams {
  modelId: string;
  modelType: ModelType;
  ids: LoadedModelIds;
  lists: ModelLists;
  /** Active load policy. Defaults to balanced so existing callers are unchanged. */
  policy?: LoadPolicy;
  /** User already approved a memory override for this model this session → don't re-prompt. */
  sessionOverride?: boolean;
}

export async function checkMemoryForModel(
  params: CheckMemoryParams,
): Promise<MemoryCheckResult> {
  const { modelId, modelType, ids, lists, policy, sessionOverride } = params;

  const model =
    modelType === 'text'
      ? lists.downloadedModels.find(m => m.id === modelId)
      : lists.downloadedImageModels.find(m => m.id === modelId);

  if (!model) {
    return {
      canLoad: false,
      severity: 'blocked',
      availableMemoryGB: 0,
      requiredMemoryGB: 0,
      currentlyLoadedMemoryGB: 0,
      totalRequiredMemoryGB: 0,
      remainingAfterLoadGB: 0,
      message: 'Model not found',
    };
  }

  const requiredMemoryGB = await estimateModelMemoryGB(model, modelType);
  const currentlyLoadedMemoryGB = await getOtherLoadedMemoryGB(modelType, ids, lists);
  const { budgetGB: memoryBudgetGB, warningGB: warningThresholdGB } = await getLiveBudgetGB(
    policy ?? 'balanced',
    currentlyLoadedMemoryGB,
  );
  const totalRequiredMemoryGB = requiredMemoryGB + currentlyLoadedMemoryGB;
  const remainingBudgetGB = memoryBudgetGB - totalRequiredMemoryGB;

  const modelName = 'name' in model ? model.name : modelId;

  // Already approved for this model this session → skip the gate so the user isn't
  // re-prompted every time it's evicted and reloaded (text↔image↔TTS swaps).
  if (sessionOverride) {
    return {
      canLoad: true, severity: 'safe',
      availableMemoryGB: memoryBudgetGB - currentlyLoadedMemoryGB,
      requiredMemoryGB, currentlyLoadedMemoryGB, totalRequiredMemoryGB,
      remainingAfterLoadGB: remainingBudgetGB,
      message: `${modelName} — load override approved for this session.`,
    };
  }
  const requiredStr = requiredMemoryGB.toFixed(1);
  const totalStr = totalRequiredMemoryGB.toFixed(1);
  const budgetStr = memoryBudgetGB.toFixed(1);

  let severity: MemoryCheckSeverity;
  let canLoad: boolean;
  let message: string;

  if (totalRequiredMemoryGB > memoryBudgetGB) {
    severity = 'critical';
    canLoad = false;
    message =
      currentlyLoadedMemoryGB > 0
        ? `Cannot load ${modelName} (~${requiredStr} GB) while other models are loaded. ` +
          `Total would be ~${totalStr} GB, exceeding your device's ~${budgetStr} GB safe limit. ` +
          `Unload the other model first, or choose a smaller model.`
        : `${modelName} requires ~${requiredStr} GB which exceeds your device's ~${budgetStr} GB safe limit. ` +
          `This model is too large for your device. Choose a smaller model.`;
  } else if (totalRequiredMemoryGB > warningThresholdGB) {
    severity = 'warning';
    canLoad = true;
    message =
      `Loading ${modelName} will use ~${requiredStr} GB. ` +
      `Total model memory will be ~${totalStr} GB, near your device's safe limit. ` +
      `The app may become slow. Continue anyway?`;
  } else {
    severity = 'safe';
    canLoad = true;
    message = `${modelName} requires ~${requiredStr} GB. Safe to load.`;
  }

  return {
    canLoad,
    severity,
    availableMemoryGB: memoryBudgetGB - currentlyLoadedMemoryGB,
    requiredMemoryGB,
    currentlyLoadedMemoryGB,
    totalRequiredMemoryGB,
    remainingAfterLoadGB: remainingBudgetGB,
    message,
  };
}

// ---------------------------------------------------------------------------
// checkMemoryForDualModel
// ---------------------------------------------------------------------------

export interface CheckDualMemoryParams {
  textModelId: string | null;
  imageModelId: string | null;
  lists: ModelLists;
}

export async function checkMemoryForDualModel(
  params: CheckDualMemoryParams,
): Promise<MemoryCheckResult> {
  const { textModelId, imageModelId, lists } = params;
  const memoryBudgetGB = await getMemoryBudgetGB();
  const warningThresholdGB = await getMemoryWarningThresholdGB();

  let totalRequiredGB = 0;
  const modelNames: string[] = [];

  if (textModelId) {
    const textModel = lists.downloadedModels.find(m => m.id === textModelId);
    if (textModel) {
      totalRequiredGB += await estimateModelMemoryGB(textModel, 'text');
      modelNames.push(textModel.name);
    }
  }

  if (imageModelId) {
    const imageModel = lists.downloadedImageModels.find(m => m.id === imageModelId);
    if (imageModel) {
      totalRequiredGB += await estimateModelMemoryGB(imageModel, 'image');
      modelNames.push(imageModel.name);
    }
  }

  const remainingBudgetGB = memoryBudgetGB - totalRequiredGB;
  const namesStr = modelNames.join(' + ');
  const requiredStr = totalRequiredGB.toFixed(1);
  const budgetStr = memoryBudgetGB.toFixed(1);

  let severity: MemoryCheckSeverity;
  let canLoad: boolean;
  let message: string;

  if (totalRequiredGB > memoryBudgetGB) {
    severity = 'critical';
    canLoad = false;
    message =
      `Cannot load both models. ` +
      `${namesStr} would require ~${requiredStr} GB, exceeding your device's ~${budgetStr} GB safe limit.`;
  } else if (totalRequiredGB > warningThresholdGB) {
    severity = 'warning';
    canLoad = true;
    message =
      `Loading ${namesStr} will use ~${requiredStr} GB, near your device's safe limit. ` +
      `Performance may be affected.`;
  } else {
    severity = 'safe';
    canLoad = true;
    message = `${namesStr} will use ~${requiredStr} GB. Safe to load.`;
  }

  return {
    canLoad,
    severity,
    availableMemoryGB: memoryBudgetGB,
    requiredMemoryGB: totalRequiredGB,
    currentlyLoadedMemoryGB: 0,
    totalRequiredMemoryGB: totalRequiredGB,
    remainingAfterLoadGB: remainingBudgetGB,
    message,
  };
}
