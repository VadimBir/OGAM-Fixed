/**
 * Text LoRA setting changes from the UI: write settings.textLoras, then apply the new set to the
 * loaded local text model live (LLMService.syncTextLoras). The apply result goes to the status the
 * settings UI shows; a failed apply leaves the previous adapter set active (rn-llama.cpp:762-788).
 */
import RNFS from 'react-native-fs';
import { useAppStore } from '../stores/appStore';
import logger from '../utils/logger';
import { llmService } from './llm';
import { TEXT_LORA_MISMATCH_NOTE, importTextLora, probeGgufAdapter, setTextLoraStatus } from './textLora';
import { clampTextLoraScale } from './textLoraParams';
import { remoteLoraServerKey } from './textLoraRemote';
import type { TextLora } from './textLoraTypes';

/** Apply the current setting to the loaded model and record the outcome. Returns the error, if any. */
export async function syncLoadedTextLoras(): Promise<string | null> {
  try {
    const result = await llmService.syncTextLoras();
    if (result.error) setTextLoraStatus({ lastError: result.error });
    else if (result.changed) setTextLoraStatus({ lastError: null });
    return result.error ?? null;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setTextLoraStatus({ lastError: message });
    return message;
  }
}

function writeTextLoras(next: TextLora[]): Promise<string | null> {
  useAppStore.getState().updateSettings({ textLoras: next });
  return syncLoadedTextLoras();
}

export function patchTextLora(id: string, change: Partial<Pick<TextLora, 'enabled' | 'scale'>>): Promise<string | null> {
  const loras = useAppStore.getState().settings.textLoras ?? [];
  const next = loras.map(l => (l.id === id
    ? { ...l, ...change, ...(change.scale !== undefined ? { scale: clampTextLoraScale(change.scale) } : {}) }
    : l));
  return writeTextLoras(next);
}

/** Remove the row, unload it from the model, then delete the file (text_loras/ is app-owned). */
export async function removeTextLora(id: string): Promise<void> {
  const loras = useAppStore.getState().settings.textLoras ?? [];
  const target = loras.find(l => l.id === id);
  await writeTextLoras(loras.filter(l => l.id !== id));
  if (target) {
    try {
      if (await RNFS.exists(target.path)) await RNFS.unlink(target.path);
    } catch (error) {
      logger.warn(`[TextLoRA] could not delete ${target.path}`, error);
    }
  }
}

export interface TextLoraImportOutcome {
  /** false = the file is not a GGUF adapter; the caller imports it as a model. */
  handled: boolean;
  title?: string;
  message?: string;
}

/**
 * A picked .gguf that is an adapter (general.type "adapter") is a text LoRA import, never a model:
 * a LoRA is copied to text_loras/, added enabled at scale 1 and applied to the loaded model; another
 * adapter type is refused. A GGUF whose header cannot be read is left to the model import.
 */
export async function importPickedTextLora(file: { uri: string; name: string }): Promise<TextLoraImportOutcome> {
  const kind = await probeGgufAdapter(file.uri);
  if (!kind || kind.kind === 'model') return { handled: false };
  if (kind.kind === 'other-adapter') {
    return {
      handled: true,
      title: 'Unsupported adapter',
      message: `This GGUF is an adapter of type "${kind.adapterType}". The text engine loads LoRA adapters only (adapter.type "lora").`,
    };
  }
  const result = await importTextLora(file);
  const applyError = await syncLoadedTextLoras();
  const lines = [
    `${result.lora.name} added as a text LoRA (scale 1.0, enabled). Adjust it in Text Generation settings.`,
    `Trained for architecture: ${result.architecture}.`,
    TEXT_LORA_MISMATCH_NOTE,
    ...result.warnings,
    ...(applyError ? [`Not applied to the loaded model: ${applyError}`] : []),
  ];
  return { handled: true, title: 'Text LoRA added', message: lines.join('\n\n') };
}

/** Per-request scale for one adapter on the user's llama-server. */
export function setRemoteTextLoraScale(
  endpoint: string,
  id: number,
  scale: number,
  serverAdapters: Array<{ id: number; scale: number }>,
): void {
  const { settings, updateSettings } = useAppStore.getState();
  const key = remoteLoraServerKey(endpoint);
  const all = { ...(settings.remoteTextLoras ?? {}) };
  // Every adapter the server lists is sent, each with an explicit scale (untouched ones keep the
  // server's own), so the request never depends on how the server treats an adapter left out
  // (that rule is not in the vendored llama.cpp: UNKNOWN here).
  const current = all[key] ?? [];
  all[key] = serverAdapters.map(a => (a.id === id
    ? { id, scale: clampTextLoraScale(scale) }
    : current.find(r => r.id === a.id) ?? { id: a.id, scale: clampTextLoraScale(a.scale) }));
  updateSettings({ remoteTextLoras: all });
}

/** Drop the overrides for a server: its own global scales apply again (no `lora` field is sent). */
export function clearRemoteTextLoras(endpoint: string): void {
  const { settings, updateSettings } = useAppStore.getState();
  const all = { ...(settings.remoteTextLoras ?? {}) };
  delete all[remoteLoraServerKey(endpoint)];
  updateSettings({ remoteTextLoras: all });
}
