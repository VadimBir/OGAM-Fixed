import { LlamaContext } from 'llama.rn';
import RNFS from 'react-native-fs';
import { statFile } from '../utils/fileStat';
import logger from '../utils/logger';
import { OverridableMemoryError } from '../utils/modelLoadErrors';

/**
 * GGUF magic number — first 4 bytes of every valid GGUF file.
 * Used to detect corrupted or truncated model files before loading.
 */
const GGUF_MAGIC = 'GGUF';

/** Minimum plausible GGUF file size (header + at least some tensors) */
const MIN_GGUF_FILE_SIZE = 1024; // 1 KB

function decodeLittleEndianUint32(bytes: string): number | null {
  if (bytes.length < 4) return null;
  const byteValues = Array.from(bytes)
    .slice(0, 4)
    .map(char => char.charCodeAt(0));
  return byteValues.reduce(
    (sum, value, index) => sum + value * 256 ** index,
    0,
  );
}

/**
 * Validate that a model file is a plausible GGUF file.
 * Checks magic bytes and minimum file size to catch corrupted/truncated downloads.
 */
export async function validateModelFile(
  modelPath: string,
): Promise<{ valid: boolean; reason?: string }> {
  try {
    const facts = await statFile(modelPath);
    if (!facts) {
      return { valid: false, reason: `Model file not found at: ${modelPath}` };
    }
    if (!facts.isFile) {
      return { valid: false, reason: `Model path is not a file: ${modelPath}` };
    }
    const fileSize = facts.size;
    const fileSizeMB = (fileSize / (1024 * 1024)).toFixed(1);
    logger.log(`[LLM] Validating model: ${modelPath}`);
    logger.log(`[LLM] Model file size: ${fileSizeMB}MB (${fileSize} bytes)`);
    if (fileSize < MIN_GGUF_FILE_SIZE) {
      return {
        valid: false,
        reason: `Model file too small (${fileSize} bytes) — likely corrupted or incomplete download`,
      };
    }
    // Read first 4 bytes to check GGUF magic number.
    // RNFS.read() has an iOS bridging bug with NSInteger arguments on
    // react-native-fs 2.x, so we catch and skip the magic check if it fails.
    // llama.rn will still validate the file format natively on load.
    let header: string | undefined;
    try {
      header = await RNFS.read(modelPath, 4, 0, 'ascii');
    } catch (readErr) {
      logger.warn(
        '[LLM] RNFS.read() failed for magic check, skipping header validation:',
        readErr,
      );
    }
    if (header !== undefined && !header.startsWith(GGUF_MAGIC)) {
      return {
        valid: false,
        reason: `Invalid model file — not a GGUF file (header: ${header})`,
      };
    }
    if (header !== undefined) {
      logger.log(`[LLM] GGUF magic OK`);
    }
    // Try to read GGUF version (bytes 4-7, little-endian uint32)
    try {
      const versionBytes = await RNFS.read(modelPath, 4, 4, 'ascii');
      if (versionBytes) {
        const version = decodeLittleEndianUint32(versionBytes);
        if (version !== null) logger.log(`[LLM] GGUF version: ${version}`);
      }
    } catch (_e) {
      // Non-critical, just skip
    }
    // Log the model filename for easier identification
    const filename = modelPath.split('/').pop() || modelPath;
    logger.log(`[LLM] Model filename: ${filename}`);
    return { valid: true };
  } catch (e: any) {
    return {
      valid: false,
      reason: `Failed to validate model file: ${e?.message || e}`,
    };
  }
}

/**
 * Live free-RAM check for a text-model load. `requiredMB` is the live part of the load's peak from
 * the ONE text estimate the residency gate also uses (memoryBudget.textMemoryFootprint over
 * estimateLlamaMemory: weights unless mmapped-reclaimable + mmproj + LoRA + KV + compute, x1.1).
 * Safe when free RAM keeps `safetyMarginMB` (the gate's live headroom) after the load.
 */
export interface MemoryCheckArgs {
  requiredMB: number;
  getAvailableMemory: () => Promise<{ available: number; total: number }>;
  /** Live-RAM margin (MB) kept free after the load (memoryBudget.liveRamMarginMB). */
  safetyMarginMB?: number;
}

/** " + NMB safety margin" when a live margin applies; empty otherwise. */
function marginClause(safetyMarginMB: number): string {
  return safetyMarginMB > 0 ? ` + ${Math.round(safetyMarginMB)}MB safety margin` : '';
}

export async function checkMemoryForModel(
  args: MemoryCheckArgs,
): Promise<{
  safe: boolean;
  reason?: string;
  estimatedMB: number;
  availableMB: number;
}> {
  const { requiredMB, getAvailableMemory } = args;
  const safetyMarginMB = Math.max(0, args.safetyMarginMB ?? 0);
  const estimatedMB = Math.max(0, requiredMB);
  try {
    const { available, total } = await getAvailableMemory();
    const availableMB = available / (1024 * 1024);
    const totalMB = total / (1024 * 1024);
    const safe = estimatedMB + safetyMarginMB <= availableMB;
    // [MEM-SM] the pre-load fit decision: the exact "it needs ~X but only Y" call, on device
    // and in tests (DEBUG_LOGS=1).
    logger.log(
      `[MEM-SM] checkMemoryForModel estMB=${Math.round(estimatedMB)} availMB=${Math.round(availableMB)} marginMB=${safetyMarginMB} safe=${safe}`,
    );
    if (!safe) {
      return {
        safe: false,
        reason: `Not enough memory: model needs ~${Math.round(
          estimatedMB,
        )}MB${marginClause(safetyMarginMB)} but only ${Math.round(
          availableMB,
        )}MB available (device total: ${Math.round(
          totalMB,
        )}MB). Try closing other apps or using a smaller model.`,
        estimatedMB,
        availableMB,
      };
    }
    return { safe: true, estimatedMB, availableMB };
  } catch (e: any) {
    // If we can't check memory, proceed anyway but log a warning
    logger.warn('[LLM] Could not check available memory:', e?.message || e);
    return { safe: true, estimatedMB: 0, availableMB: 0 };
  }
}

/** Check the selected context without changing it. A memory refusal can be overridden by the user. */
export async function resolveSafeContext(args: {
  requiredMB: number;
  requestedCtx: number;
  override?: boolean;
  getAvailableMemory: () => Promise<{ available: number; total: number }>;
  safetyMarginMB?: number;
}): Promise<{
  ctxLen: number;
  memCheck: Awaited<ReturnType<typeof checkMemoryForModel>>;
}> {
  const { requiredMB, requestedCtx, override = false, getAvailableMemory, safetyMarginMB } = args;
  const memCheck = await checkMemoryForModel({ requiredMB, getAvailableMemory, safetyMarginMB });
  if (!memCheck.safe && !override) {
    throw new OverridableMemoryError(
      `Not enough memory to load this model at ${requestedCtx.toLocaleString()} context tokens: it needs ~${Math.round(memCheck.estimatedMB)}MB${marginClause(safetyMarginMB ?? 0)} but only ${Math.round(memCheck.availableMB)}MB is available. Close other apps or choose a smaller model.`,
    );
  }
  if (!memCheck.safe && override) {
    logger.warn(
      `[LLM] OVERRIDE — trying selected context ${requestedCtx} despite tight memory`,
    );
  }
  return { ctxLen: requestedCtx, memCheck };
}

/**
 * Wraps a llama.rn completion call with error handling for native crashes.
 * Catches ggml_abort and OOM-style errors and returns a structured error
 * instead of letting the app crash unrecoverably.
 */
export async function safeCompletion<T>(
  context: LlamaContext,
  completionFn: () => Promise<T>,
  label: string = 'completion',
): Promise<T> {
  try {
    return await completionFn();
  } catch (error: any) {
    const msg = error?.message || String(error) || '';
    const isNativeCrash =
      msg.includes('ggml') ||
      msg.includes('abort') ||
      msg.includes('SIGABRT') ||
      msg.includes('tensor') ||
      msg.includes('alloc') ||
      msg.includes('out of memory') ||
      msg.includes('failed to allocate') ||
      msg.includes('OOM');
    if (isNativeCrash) {
      logger.error(`[LLM] Native crash during ${label}: ${msg}`);
      // Try to recover the context by clearing KV cache
      try {
        await (context as any).clearCache(true);
        logger.log(`[LLM] KV cache cleared after native error in ${label}`);
      } catch (clearError) {
        logger.warn(
          `[LLM] Failed to clear KV cache after crash: ${clearError}`,
        );
      }
      throw new Error(
        `Model inference failed (native error). The model's KV cache has been cleared. Please try again, or use a smaller model/context size. (${msg})`,
      );
    }
    throw error;
  }
}
