import type { DownloadedModel } from '../../types';
import type { DownloadEntry } from '../../utils/downloadStatus';
import { basenameOf } from './reconcileStoredPaths';

/**
 * Where a downloaded text model lives on disk, and whose file that is.
 *
 * The models dir is flat, and Hugging Face file names are only unique WITHIN a repo:
 * `unsloth/Qwen3-4B-GGUF` and `Qwen/Qwen3-4B-GGUF` can both publish `Qwen3-4B-Q4_K_M.gguf`. Keyed
 * on the bare name, the second download landed on the first one's path (overwrote it, or was
 * "already downloaded" from the other repo's bytes), and both catalog rows showed "Downloaded".
 *
 * THE RULE (one place, used by download, restore, scan, collapse and the Models screen):
 *  1. The on-disk name of a model is the base name of its file (`basenameOf(filePath)`), and that,
 *     not the catalog `fileName`, is the identity of the artifact. Every row written so far has
 *     `filePath = <modelsDir>/<fileName>`, so existing installs keep their identity and their path.
 *  2. A download first tries the plain name `<basename of file.name>`. It may use it when nothing
 *     is there, or what is there is ITS OWN file: a registry row with the same id, or the same repo
 *     and the same catalog file (sameCatalogFile), or an unattributed row / unregistered file whose
 *     size equals the catalog size (= complete).
 *     An UNREGISTERED file of another size is not a finished copy: the name is used and the file
 *     is fetched again (replaced only when the new download finishes).
 *  3. Otherwise (another repo's registered row, an unattributed registered row of other bytes, or
 *     another repo's or another file's in-flight download targeting that name) it uses a repo-qualified name
 *     `<stem>--<owner>-<hash6>.gguf` (hash of the full repo id), then `…-2`, `…-3` on a further clash.
 *     The qualified name rides in the native row's metadataJson (`localFileName`) so restore after
 *     an app kill finalizes to the same path.
 *  4. The registry row keeps the catalog `fileName` and id `<repo>/<fileName>`; only `filePath`
 *     differs. Projector naming and matching keep using the catalog name, so vision linking is
 *     unchanged.
 */

/** On-disk base name of a registered model (rule 1). */
export function diskFileNameOf(model: Pick<DownloadedModel, 'filePath' | 'fileName'>): string {
  return (model.filePath ? basenameOf(model.filePath) : '') || model.fileName;
}

/**
 * The Hugging Face repo a registry row came from, or undefined when the row does not say
 * (recovered_ scans, local imports, bare ids).
 */
export function repoIdOfModel(model: Pick<DownloadedModel, 'id' | 'fileName' | 'origin'>): string | undefined {
  if (model.origin?.repoId) return model.origin.repoId;
  if (model.fileName && model.id.endsWith(`/${model.fileName}`)) {
    const repo = model.id.slice(0, model.id.length - model.fileName.length - 1);
    // `local_import/<file>` has no owner segment - it names no repo.
    return repo.includes('/') ? repo : undefined;
  }
  return undefined;
}

/**
 * True when a registry row describes the catalog file (repo `modelId`, catalog name `fileName`).
 * An exact id is always ours; an unattributed row (recovered / imported / bare) with the same file
 * name is accepted, as before; a row attributed to ANOTHER repo is never ours.
 */
export function modelRowMatchesCatalogFile(
  model: Pick<DownloadedModel, 'id' | 'fileName' | 'origin'>,
  modelId: string,
  fileName: string,
): boolean {
  if (model.id === `${modelId}/${fileName}`) return true;
  if (model.fileName !== fileName) return false;
  const repo = repoIdOfModel(model);
  return repo === undefined || repo === modelId;
}

/**
 * Whether two catalog names of ONE repo name the same file. A repo can publish the same base name in
 * two folders (`v1/model.gguf`, `v2/model.gguf`); both land on the same flat disk name, so the repo
 * alone does not make a row "ours". A bare name (native download rows keep only the base name,
 * DownloadManagerModule.kt:66; rows restored before restore.catalogFileNameOf) matches by base name.
 */
function sameCatalogFile(a: string, b: string): boolean {
  if (a === b) return true;
  if (a.includes('/') && b.includes('/')) return false;
  return basenameOf(a) === basenameOf(b);
}

function hash6(text: string): string {
  let h = 5381;
  for (let i = 0; i < text.length; i++) h = ((h * 33) ^ text.charCodeAt(i)) >>> 0;
  return h.toString(36).padStart(6, '0').slice(-6);
}

/** Candidate on-disk name number `attempt` (0 = plain) for a catalog file of repo `modelId`. */
export function candidateLocalFileName(modelId: string, catalogFileName: string, attempt: number): string {
  const base = basenameOf(catalogFileName);
  if (attempt === 0) return base;
  const dot = base.lastIndexOf('.');
  const stem = dot > 0 ? base.slice(0, dot) : base;
  const ext = dot > 0 ? base.slice(dot) : '';
  const owner = (modelId.split('/')[0] || 'repo').replaceAll(/[^A-Za-z0-9]/g, '');
  const suffix = attempt === 1 ? '' : `-${attempt}`;
  return `${stem}--${owner || 'repo'}-${hash6(modelId)}${suffix}${ext}`;
}

/** The `localFileName` carried in a native row's metadataJson, if any. */
export function localFileNameFromMetadataJson(metadataJson: string | undefined | null): string | undefined {
  if (!metadataJson) return undefined;
  try {
    const parsed = JSON.parse(metadataJson) as Record<string, unknown>;
    return typeof parsed.localFileName === 'string' && parsed.localFileName ? parsed.localFileName : undefined;
  } catch {
    // Not JSON: the row predates localFileName, so its file lives under the plain name.
    return undefined;
  }
}

export interface LocalFileProbe {
  /** Exists + size of `<modelsDir>/<name>`; size undefined when it cannot be read. */
  stat(name: string): Promise<{ exists: boolean; size?: number }>;
}

export interface ResolveLocalFileOpts {
  modelId: string;
  catalogFileName: string;
  catalogSize: number;
  registry: DownloadedModel[];
  /** Download-store rows (other downloads in flight may already own a name). */
  downloads: DownloadEntry[];
  probe: LocalFileProbe;
}

const MAX_ATTEMPTS = 6;

/**
 * Pick the on-disk name for a download (rules 2-3). `complete` is true when that name already holds
 * this model's finished file - the caller must then NOT fetch or delete it. `exists` is whether any
 * file is at that name now (a cancel must leave such a file alone).
 */
export async function resolveLocalModelFile(
  opts: ResolveLocalFileOpts,
): Promise<{ name: string; complete: boolean; exists: boolean }> {
  const { modelId, catalogFileName, catalogSize, registry, downloads, probe } = opts;
  const targetId = `${modelId}/${catalogFileName}`;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const name = candidateLocalFileName(modelId, catalogFileName, attempt);

    const takenByOtherDownload = downloads.some(d =>
      d.modelType === 'text' &&
      !(d.modelId === modelId && sameCatalogFile(d.fileName, catalogFileName)) &&
      d.status !== 'completed' && d.status !== 'cancelled' &&
      (localFileNameFromMetadataJson(d.metadataJson) ?? basenameOf(d.fileName)) === name,
    );
    if (takenByOtherDownload) continue;

    const rows = registry.filter(m => diskFileNameOf(m) === name);
    const ownRow = rows.find(m =>
      m.id === targetId || (repoIdOfModel(m) === modelId && sameCatalogFile(m.fileName, catalogFileName)));
    const { exists, size } = await probe.stat(name);

    // This repo's registered install: complete when the file is there (its registered size wins
    // over a catalog that may have been re-uploaded since).
    if (ownRow) return { name, complete: exists, exists };
    // Another repo's (or this repo's other file's) registered install: never reuse, overwrite or delete it.
    if (rows.some(m => repoIdOfModel(m) !== undefined)) continue;

    if (!exists) return { name, complete: false, exists };
    const sameBytes = catalogSize > 0 ? size === catalogSize : true;
    if (sameBytes) return { name, complete: true, exists };
    // Unattributed REGISTERED row (local import / recovered) of different bytes: the user's model.
    if (rows.length > 0) continue;
    // Unregistered file of the wrong size: not a finished copy of this file. Fetch again; the
    // finished download replaces it, a cancelled one leaves it untouched (mainPreExisted).
    return { name, complete: false, exists };
  }
  throw new Error(`No free file name for ${catalogFileName} in the models folder`);
}
