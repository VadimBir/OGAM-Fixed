/**
 * Standalone utility helpers for ActiveModelService.
 */

import { useAppStore, useChatStore } from '../../stores';
import { hardwareService } from '../hardware';
import { llmService } from '../llm';
import { liteRTService } from '../litert';
import { imageEngineRouter as onnxImageGeneratorService } from '../imageEngineRouter';
import { ResourceUsage } from './types';

export async function getResourceUsage(): Promise<ResourceUsage> {
  const info = await hardwareService.refreshMemoryInfo();
  const store = useAppStore.getState();
  let estimatedModelMemory = 0;

  if (store.activeModelId) {
    const tm = store.downloadedModels.find(m => m.id === store.activeModelId);
    if (tm?.fileSize) {
      estimatedModelMemory += tm.fileSize * 1.2;
    }
  }
  if (store.activeImageModelId) {
    const im = store.downloadedImageModels.find(m => m.id === store.activeImageModelId);
    if (im?.size) {
      estimatedModelMemory += im.size * 1.3;
    }
  }

  return {
    memoryUsed: info.usedMemory,
    memoryTotal: info.totalMemory,
    memoryAvailable: info.availableMemory,
    memoryUsagePercent: (info.usedMemory / info.totalMemory) * 100,
    estimatedModelMemory,
  };
}

export interface SyncStateTarget {
  setLoadedTextModelId: (id: string | null) => void;
  setLoadedImageModelId: (id: string | null) => void;
  setLoadedImageModelThreads: (n: number | null) => void;
  loadedTextModelId: string | null;
  loadedImageModelId: string | null;
}

export async function syncWithNativeState(target: SyncStateTarget): Promise<void> {
  const store = useAppStore.getState();

  const textModelLoaded = llmService.isModelLoaded() || liteRTService.isModelLoaded();
  if (!textModelLoaded) {
    target.setLoadedTextModelId(null);
  } else if (!target.loadedTextModelId && store.activeModelId) {
    target.setLoadedTextModelId(store.activeModelId);
  }

  const imageModelLoaded = await onnxImageGeneratorService.isModelLoaded();
  if (!imageModelLoaded) {
    target.setLoadedImageModelId(null);
    target.setLoadedImageModelThreads(null);
  } else if (!target.loadedImageModelId && store.activeImageModelId) {
    target.setLoadedImageModelId(store.activeImageModelId);
  }
}

/**
 * ": it needs ~X MB but only ~Y MB is free for it right now" - the needed-versus-available
 * clause of a residency refusal (same "it needs ~" wording as the text pre-load gate).
 * Empty when the gate did not report numbers, so the message reads as before.
 */
export function memoryRefusalDetail(room: { neededMB?: number; availableMB?: number }): string {
  if (room.neededMB == null || room.availableMB == null) return '';
  return `: it needs ~${room.neededMB} MB but only ~${room.availableMB} MB is free for it right now`;
}

/**
 * Trim-release probe for the text model: true while a generation turn is in flight (the
 * llama context is generating, or a chat turn is streaming - which also covers LiteRT and the
 * gaps between tool calls). The residency trim handler never releases the text model then.
 */
export function isTextGenerationInFlight(): boolean {
  return llmService.isCurrentlyGenerating() || useChatStore.getState().streamingForConversationId != null;
}

/** Trim-release probe for the image model: true while the native engine is generating. */
export function isImageGenerationInFlight(): Promise<boolean> {
  return onnxImageGeneratorService.isGenerating();
}
