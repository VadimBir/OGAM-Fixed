/**
 * Text LoRA files: import (GGUF adapter detection, copy into the app folder, register), the
 * .safetensors refusal, adapter byte sizes for the memory gate, and the live status the settings UI
 * shows (last llama.cpp error, CPU-fallback warning).
 *
 * llama.rn loads GGUF adapters only (llama_adapter_lora_init opens the file with gguf_init_from_file,
 * vendor src/llama-adapter.cpp:158-161), so a .safetensors LoRA must be converted on a PC first.
 */
import RNFS from 'react-native-fs';
import { addNativeLogListener } from 'llama.rn';
import { useAppStore } from '../stores/appStore';
import { statFile } from '../utils/fileStat';
import logger from '../utils/logger';
import { classifySafetensorsKeys } from './sdCppModelDetection';
import {
  classifyGgufAdapter,
  nonAcceleratedLoraTypes,
  readGgufHeader,
  readSafetensorsKeys,
  type GgufAdapterKind,
} from './textLoraGguf';
import type { TextLora } from './textLoraTypes';

/** Outside models/ so the model scan never adopts an adapter as a model. */
export const TEXT_LORAS_DIR = `${RNFS.DocumentDirectoryPath}/text_loras`;

export const TEXT_LORA_MISMATCH_NOTE =
  'A LoRA only works on the base model it was trained for: same architecture and same layer sizes. ' +
  'llama.cpp refuses a mismatch (for example "model arch and LoRA arch mismatch" or "tensor ... has incorrect shape").';

export const SAFETENSORS_TEXT_LORA_REFUSAL =
  'This .safetensors file is a text LoRA adapter. The on-device text engine (llama.cpp) loads LoRAs in GGUF only.\n\n' +
  "On a PC, convert it with llama.cpp's convert_lora_to_gguf.py (it needs the matching base model), then import the .gguf adapter.";

export const SAFETENSORS_LLM_REFUSAL =
  'This .safetensors file holds text-model (LLM) weights. The on-device text engine (llama.cpp) loads GGUF only.\n\n' +
  "On a PC, convert it with llama.cpp's convert_hf_to_gguf.py (and quantize if needed), then import the .gguf.";

// Resolve picker URIs the way the model import does (modelManager/importLocalModel.ts:37-44).
const resolveUri = (uri: string): string => (uri.startsWith('content://') ? uri : decodeURIComponent(uri));

/**
 * Refusal text for a .safetensors that holds text weights (a text LoRA or an LLM), or null when it is
 * not text (the image import then classifies it). Uses the shared key classifier
 * (sdCppModelDetection.classifySafetensorsKeys: kind 'llm' = model.layers.* / embed_tokens / lm_head).
 */
export async function textSafetensorsRefusal(uri: string): Promise<string | null> {
  let keys: string[];
  try {
    keys = await readSafetensorsKeys(resolveUri(uri));
  } catch (error) {
    logger.warn('[TextLoRA] could not read the safetensors header; leaving it to the image import', error);
    return null;
  }
  const result = classifySafetensorsKeys(keys);
  if (result.kind !== 'llm') return null;
  const isLora = keys.some(k => /\.(lora_A|lora_B|lora_down|lora_up)(\.|$)/.test(k));
  logger.log(`[TextLoRA] refused .safetensors: ${result.reason}`);
  return isLora ? SAFETENSORS_TEXT_LORA_REFUSAL : SAFETENSORS_LLM_REFUSAL;
}

/** GGUF adapter kind of a picked file, or null when the header could not be read. */
export async function probeGgufAdapter(uri: string): Promise<GgufAdapterKind | null> {
  try {
    const header = await readGgufHeader(resolveUri(uri));
    return header ? classifyGgufAdapter(header) : null;
  } catch (error) {
    logger.warn('[TextLoRA] could not read the GGUF header', error);
    return null;
  }
}

export interface TextLoraImportResult {
  lora: TextLora;
  architecture: string;
  /** Non-blocking notes for the user (tensor types not proven on OpenCL/HTP). */
  warnings: string[];
}

const safeId = (name: string) => name.replaceAll(/[^a-zA-Z0-9_-]/g, '_');

/**
 * Copy a GGUF LoRA adapter into text_loras/ and add it to settings.textLoras (enabled, scale 1).
 * The copy is re-read and must still be a LoRA adapter; any failure removes the copy.
 */
export async function importTextLora(file: { uri: string; name: string }): Promise<TextLoraImportResult> {
  if (!file.name.toLowerCase().endsWith('.gguf')) {
    throw new Error('A text LoRA must be a .gguf adapter file.');
  }
  if (!(await RNFS.exists(TEXT_LORAS_DIR))) await RNFS.mkdir(TEXT_LORAS_DIR);
  const dest = `${TEXT_LORAS_DIR}/${file.name}`;
  if (await RNFS.exists(dest)) throw new Error(`A text LoRA named "${file.name}" already exists.`);
  await RNFS.copyFile(resolveUri(file.uri), dest);
  try {
    const header = await readGgufHeader(dest);
    const kind = header ? classifyGgufAdapter(header) : null;
    if (!header || !kind) throw new Error('Not a GGUF file.');
    if (kind.kind !== 'text-lora') {
      throw new Error(kind.kind === 'other-adapter'
        ? `This GGUF adapter is "${kind.adapterType}", not a LoRA (adapter.type must be "lora").`
        : 'This GGUF is a model, not a LoRA adapter (general.type is not "adapter").');
    }
    const odd = nonAcceleratedLoraTypes(header.tensorTypes);
    const warnings = odd.length > 0
      ? [`Adapter tensors are ${odd.join(', ')}. Only F16/F32 LoRA math is confirmed on the GPU (OpenCL) and NPU (HTP); other types may run on the CPU.`]
      : [];
    const lora: TextLora = {
      id: `tlora_${safeId(file.name)}_${Date.now()}`,
      name: file.name.replace(/\.gguf$/i, ''),
      path: dest,
      scale: 1,
      enabled: true,
    };
    const { settings, updateSettings } = useAppStore.getState();
    updateSettings({ textLoras: [...(settings.textLoras ?? []), lora] });
    logger.log(`[TextLoRA] imported ${file.name} (arch ${kind.architecture}, ${header.tensorTypes.length} tensors read)`);
    return { lora, architecture: kind.architecture, warnings };
  } catch (error) {
    await RNFS.unlink(dest).catch(() => {});
    throw error;
  }
}

/** Bytes of the enabled adapters' files: llama.cpp copies every LoRA tensor into its own buffer
 *  (llama-adapter.cpp:385, ggml_backend_alloc_ctx_tensors_from_buft), so this is RAM it uses. */
export async function enabledTextLoraBytes(loras: TextLora[] | undefined, paths?: string[]): Promise<number> {
  const wanted = paths ?? (loras ?? []).filter(l => l.enabled && l.scale > 0).map(l => l.path);
  let total = 0;
  for (const p of wanted) total += (await statFile(p))?.size ?? 0;
  return total;
}

// ---------------------------------------------------------------------------------------------
// Live status for the settings UI (not persisted).
// ---------------------------------------------------------------------------------------------

export interface TextLoraStatus {
  /** The last apply / load error, with llama.cpp's own reason when it logged one. */
  lastError: string | null;
  /** llama.cpp put LoRA tensors on the CPU (llama-adapter.cpp:335-347 "fallback to CPU"). */
  cpuFallback: boolean;
}

let status: TextLoraStatus = { lastError: null, cpuFallback: false };
const listeners = new Set<() => void>();

export function getTextLoraStatus(): TextLoraStatus { return status; }
export function setTextLoraStatus(patch: Partial<TextLoraStatus>): void {
  status = { ...status, ...patch };
  for (const l of listeners) l();
}
export function subscribeTextLoraStatus(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** The warning llama.cpp logs when a LoRA tensor cannot use its base tensor's buffer type. */
export const LORA_CPU_FALLBACK_LINE = /lora for '.*' cannot use buft '.*', fallback to CPU/;

let watching = false;
/** Watch llama.cpp's log for the LoRA CPU fallback. Needs native logging on (ensureNativeLogCapture). */
export function watchLoraCpuFallback(): void {
  if (watching) return;
  watching = true;
  try {
    addNativeLogListener((_level: string, text: string) => {
      if (!status.cpuFallback && LORA_CPU_FALLBACK_LINE.test(text || '')) {
        setTextLoraStatus({ cpuFallback: true });
      }
    });
  } catch (error) {
    watching = false;
    logger.warn('[TextLoRA] could not watch the native log', error);
  }
}

/** The native-log lines that explain a LoRA failure (llama_adapter_lora_init logs the reason). */
export function loraReasonFromLog(log: string): string {
  return log.split('\n').filter(line => /lora|adapter/i.test(line) && /error|fail|mismatch|shape|expect/i.test(line)).join('\n');
}
