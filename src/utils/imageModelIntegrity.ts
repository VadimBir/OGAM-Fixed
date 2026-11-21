/**
 * Image-model extraction integrity — the single source of truth for "is a downloaded
 * image model actually complete on disk".
 *
 * Why this exists: a partial unzip (interrupted, killed process, a flaky zip bridge)
 * left a model missing files (observed on-device: `pos_emb.bin` + `clip_v2.mnn.weight`
 * absent from an mnn model), yet the download was marked `_ready` because nothing
 * validated the extraction. At generation time the native server then died and the app
 * showed a MISLEADING "your device may not support this backend" error — masking a
 * plain missing-file bug as a hardware/SoC problem.
 *
 * The required set is the file layout the local-dream v2.8.1 core checks before it starts
 * (main.cpp:326-350, one fixed layout per `--type`):
 *  - every kind: tokenizer.json, pos_emb.bin, token_emb.bin, the UNet and the VAE decoder
 *  - SD1.5 (sd15cpu / sd15npu): clip_v2.mnn. The core no longer accepts clip.mnn or clip.bin, so
 *    an older SD1.5 folder without clip_v2.mnn is reported as needing a re-download.
 *  - SDXL (sdxl, NPU only): clip.mnn, clip_2.mnn + clip_2.mnn.weight, pos_emb_2.bin, token_emb_2.bin
 *  - MNN split-weight pairing (mnn only): every `*.mnn` graph must have its non-zero
 *    `*.mnn.weight` (this is what catches a dropped 156MB `clip_v2.mnn.weight`).
 * The VAE encoder is optional (LocalDreamModule passes --no_img2img without it). The SDXL kind is
 * recognised by upstream's 0-byte `SDXL` marker or by clip_2.mnn; the marker is never required.
 * coreml (iOS) uses a different layout validated elsewhere, so it's not checked here.
 */
import RNFS from 'react-native-fs';
import { unzip } from 'react-native-zip-archive';
import { ImageModelIncompleteError } from './modelLoadErrors';
import logger from './logger';

type ReadDirItem = Awaited<ReturnType<typeof RNFS.readDir>>[number];

export type ImageBackend = 'mnn' | 'qnn' | 'coreml';

export interface ImageDirEntry {
  name: string;
  /** Size in bytes. A present-but-zero file counts as missing (truncated write). */
  size: number;
  isFile: boolean;
}

/** SD1.5 (512 px, sd15cpu / sd15npu) or SDXL (1024 px, NPU only). */
export type ImageModelKind = 'sd15' | 'sdxl';

export interface IntegrityResult {
  complete: boolean;
  /** File names that are absent or zero-byte. Empty when complete. */
  missing: string[];
  /** Model kind read from the listing (mnn/qnn only). */
  kind?: ImageModelKind;
  /** Bytes of unet.bin when present (qnn only), for the SDXL RAM estimate. */
  unetBytes?: number;
}

const BASE_REQUIRED = ['pos_emb.bin', 'token_emb.bin', 'tokenizer.json'] as const;
const SDXL_REQUIRED = [
  'clip.mnn', 'clip_2.mnn', 'clip_2.mnn.weight', 'pos_emb_2.bin', 'token_emb_2.bin', 'vae_decoder.bin',
] as const;
/** The missing-file entry for an older SD1.5 package (clip.mnn / clip.bin, no clip_v2.mnn). */
export const LEGACY_SD15_CLIP_MISSING =
  'clip_v2.mnn (older model package: the updated image engine needs clip_v2.mnn)';

/** SDXL when upstream's `SDXL` marker (0 bytes, so presence only) or the second CLIP graph is listed. */
export function detectImageModelKind(files: ImageDirEntry[]): ImageModelKind {
  return files.some(f => f.isFile && (f.name === 'SDXL' || f.name === 'clip_2.mnn')) ? 'sdxl' : 'sd15';
}

/**
 * Pure completeness check over a flat listing of the RESOLVED model directory (the
 * dir that directly contains unet.mnn / unet.bin). No IO — unit-testable in isolation.
 * `kindHint` = 'sdxl' when the kind was seen outside this listing (a root `SDXL` marker).
 */
export function checkImageModelFiles(
  files: ImageDirEntry[],
  backend: ImageBackend,
  kindHint?: ImageModelKind,
): IntegrityResult {
  // iOS CoreML models have a different structure (validated in coreMLModelUtils); a
  // non-empty dir is the most we assert here without falsely failing them.
  if (backend === 'coreml') {
    return { complete: files.some(f => f.isFile), missing: [] };
  }

  const sizeByName = new Map<string, number>();
  for (const f of files) if (f.isFile) sizeByName.set(f.name, f.size);
  // SDXL ships NPU files only (no sd15cpu-style MNN UNet), so the mnn backend is always SD1.5.
  const kind: ImageModelKind =
    backend === 'qnn' && (kindHint === 'sdxl' || detectImageModelKind(files) === 'sdxl') ? 'sdxl' : 'sd15';
  const unetSize = sizeByName.get('unet.bin');
  const extra = backend === 'qnn' && unetSize != null && unetSize > 0 ? { unetBytes: unetSize } : {};

  const missing: string[] = [];
  const requirePresent = (name: string): void => {
    const size = sizeByName.get(name);
    if (size == null || size <= 0) missing.push(name);
  };

  BASE_REQUIRED.forEach(requirePresent);
  requirePresent(backend === 'mnn' ? 'unet.mnn' : 'unet.bin');

  if (kind === 'sdxl') {
    // main.cpp:327-347. The 0-byte `SDXL` marker is not in this list: the core never reads it.
    SDXL_REQUIRED.forEach(requirePresent);
    return { complete: missing.length === 0, missing: [...new Set(missing)], kind, ...extra };
  }

  // SD1.5: the core loads clip_v2.mnn for both sd15cpu and sd15npu (main.cpp:327). Require the
  // graph itself because the mnn split-weight loop below only fires for graphs that ARE present.
  // A folder that still holds the pre-v2.8 clip.mnn / clip.bin was complete for the old core:
  // say it needs a re-download instead of letting the new core exit on "File not found".
  const requireClip = (): void => {
    if ((sizeByName.get('clip_v2.mnn') ?? 0) > 0) return;
    const legacy = (sizeByName.get('clip.mnn') ?? 0) > 0 || (sizeByName.get('clip.bin') ?? 0) > 0;
    missing.push(legacy ? LEGACY_SD15_CLIP_MISSING : 'clip_v2.mnn');
  };

  if (backend === 'mnn') {
    // sd15cpu loads clip_v2.mnn + vae_decoder.mnn; vae_encoder.mnn is optional (native passes
    // --no_img2img without it), so it is NOT required.
    requirePresent('vae_decoder.mnn');
    requireClip();
    // MNN split-weight pairing: an MNN graph is useless without its `*.mnn.weight`. This is
    // an MNN-FORMAT concern ONLY. QNN packages ship clip_v2.mnn as a MONOLITHIC graph with
    // weights baked in — there is no `.weight` sibling (verified against the xororz/sd-qnn
    // zip and a working on-device qnn model). Running this loop for qnn demanded a
    // clip_v2.mnn.weight that never exists, so every fresh QNN download failed the integrity
    // gate as "incomplete/corrupted" and surfaced a bogus "connection dropped" alert — even
    // though the download and extraction were perfect (Android-only: iOS is coreml).
    for (const [name, size] of sizeByName) {
      if (name.endsWith('.mnn') && size > 0) {
        const weight = `${name}.weight`;
        const weightSize = sizeByName.get(weight);
        if (weightSize == null || weightSize <= 0) missing.push(weight);
      }
    }
  } else {
    // sd15npu: primary is unet.bin (native binary format), vae_decoder.bin is always loaded, and
    // clip_v2.mnn is a monolithic graph (no split weight). vae_encoder is optional (native passes
    // --no_img2img without it).
    requirePresent('vae_decoder.bin');
    requireClip();
  }

  return { complete: missing.length === 0, missing: [...new Set(missing)], kind, ...extra };
}

/** Kind + unet.bin bytes last seen on disk per model path (validateImageModelDir runs before every
 *  image load), so the synchronous RAM estimate can use the real UNet size once it is known. */
const observedLayouts = new Map<string, { kind: ImageModelKind; unetBytes?: number }>();

export function getObservedImageModelLayout(
  modelPath: string,
): { kind: ImageModelKind; unetBytes?: number } | undefined {
  return observedLayouts.get(modelPath);
}

/**
 * Locate the resolved model dir (the one containing the UNet). Checks `modelPath` itself, then two
 * levels of subdirs (output_512/qnn_models_min for SD1.5 qnn, output/qnn_models_sdxl_8gen3 for
 * SDXL). The native LocalDreamModule.resolveModelDir searches deeper (4 levels); every catalog zip
 * resolves within these two. SD1.5 and SDXL NPU share the unet.bin marker: the kind comes from the
 * listing (detectImageModelKind). Returns null if no UNet marker is found anywhere.
 */
export async function resolveImageModelDir(modelPath: string, backend: ImageBackend): Promise<string | null> {
  const marker = backend === 'mnn' ? 'unet.mnn' : 'unet.bin';
  const hasMarker = async (dir: string): Promise<boolean> => {
    // qnn models also ship a clip_v2.mnn; the marker that disambiguates is the unet.
    try { return await RNFS.exists(`${dir}/${marker}`); } catch { return false; }
  };
  if (await hasMarker(modelPath)) return modelPath;
  let items: ReadDirItem[];
  try { items = await RNFS.readDir(modelPath); } catch { return null; }
  for (const item of items) {
    if (item.isDirectory()) {
      if (await hasMarker(item.path)) return item.path;
      // One more level (e.g. output_512/qnn_models_min for qnn).
      let sub: ReadDirItem[];
      try { sub = await RNFS.readDir(item.path); } catch { continue; }
      for (const s of sub) {
        if (s.isDirectory() && await hasMarker(s.path)) return s.path;
      }
    }
  }
  return null;
}

/**
 * IO wrapper: resolve the model dir under `modelPath` and validate its contents.
 * When the UNet marker can't be found at all, that's itself an incomplete extraction.
 */
export async function validateImageModelDir(modelPath: string, backend: ImageBackend): Promise<IntegrityResult> {
  if (backend === 'coreml') return { complete: true, missing: [] };
  const dir = await resolveImageModelDir(modelPath, backend);
  if (!dir) return { complete: false, missing: [backend === 'mnn' ? 'unet.mnn' : 'unet.bin'] };
  let items: ReadDirItem[];
  try { items = await RNFS.readDir(dir); } catch { return { complete: false, missing: ['<unreadable model dir>'] }; }
  const files: ImageDirEntry[] = items.map(i => ({ name: i.name, size: Number(i.size) || 0, isFile: i.isFile() }));
  // Upstream's `SDXL` marker may sit at the download root rather than next to unet.bin.
  let rootMarker = false;
  if (backend === 'qnn' && dir !== modelPath) {
    try { rootMarker = await RNFS.exists(`${modelPath}/SDXL`); } catch { rootMarker = false; }
  }
  const result = checkImageModelFiles(files, backend, rootMarker ? 'sdxl' : undefined);
  if (result.kind) observedLayouts.set(modelPath, { kind: result.kind, unetBytes: result.unetBytes });
  return result;
}

/**
 * Post-unzip completeness gate shared by the primary + resume download paths. On a
 * partial extraction, re-unzip ONCE (handles a transient interrupted write) and re-check;
 * if still incomplete, throw ImageModelIncompleteError so the caller cleans up and the
 * user retries — a partial extraction is NEVER marked `_ready`. No-op for coreml.
 */
export async function ensureImageExtractionComplete(opts: {
  backend: ImageBackend | undefined;
  modelDir: string;
  zipPath: string;
  modelId: string;
}): Promise<void> {
  const { backend, modelDir, zipPath, modelId } = opts;
  if (backend !== 'mnn' && backend !== 'qnn') return;
  let result = await validateImageModelDir(modelDir, backend);
  if (!result.complete) {
    logger.warn(`[ImageDownload] incomplete extraction ${modelId} missing=[${result.missing.join(',')}] — re-unzipping once`);
    await unzip(zipPath, modelDir);
    result = await validateImageModelDir(modelDir, backend);
  }
  if (!result.complete) {
    logger.warn(`[ImageDownload] extraction STILL incomplete ${modelId} missing=[${result.missing.join(',')}] — failing`);
    throw new ImageModelIncompleteError(result.missing);
  }
}
