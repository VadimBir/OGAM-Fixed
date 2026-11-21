import logger from '../utils/logger';

export interface HFImageModel {
  id: string;
  name: string;
  displayName: string;
  backend: 'mnn' | 'qnn';
  /** sd15 = 512 px SD1.5 (GPU or NPU); sdxl = 1024 px SDXL, NPU only (local-dream v2.8.1 core).
   *  Set on every catalog entry; absent (the iOS Core ML list) reads as sd15. */
  kind?: 'sd15' | 'sdxl';
  variant?: string;
  downloadUrl: string;
  fileName: string;
  size: number;
  repo: string;
}

interface HFTreeEntry {
  type: string;
  path: string;
  size: number;
  lfs?: { oid: string; size: number; pointerSize: number };
}

const REPOS = {
  mnn: 'xororz/sd-mnn',
  qnn: 'xororz/sd-qnn',
} as const;

/**
 * SDXL NPU repos (QNN 2.28 builds for 8 Gen 3-class HTP, `*_qnn2.28_8gen3.zip`). They run on the
 * NPU through the core's `--type sdxl --lowram`, never on the GPU.
 */
export const SDXL_REPOS = [
  'xororz/sdxl-qnn',
  'YuuiKurata/lustify_qnn2.28',
  'YuuiKurata/bigLove_qnn2.28',
  'YuuiKurata/bigLust_qnn2.28',
  'YuuiKurata/ponyRealism_qnn2.28',
  'YuuiKurata/cyberrealisticPony_qnn2.28',
] as const;

/** Only SDXL zips under 4 GB are listed (binary GB, as the catalog shows sizes). */
export const SDXL_MAX_ZIP_BYTES = 4 * 1024 * 1024 * 1024;

const VARIANT_LABELS: Record<string, string> = {
  min: 'For non-flagship Snapdragon chips',
  '8gen1': 'For Snapdragon 8 Gen 1',
  '8gen2': 'For Snapdragon 8 Gen 2/3/4/5',
  '8gen3': 'SDXL 1024 px on the NPU, Snapdragon 8 Gen 3 or newer',
};

// Keyed on skipQnn + includeSdxl: a list fetched WITHOUT a repo must never answer a request that
// wants it (and vice versa), or those models stayed hidden for the whole TTL after one call.
const cache = new Map<string, { models: HFImageModel[]; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function insertSpaces(name: string): string {
  // Insert space before uppercase letters that follow lowercase or digits
  // e.g. "AnythingV5" -> "Anything V5", "AbsoluteReality" -> "Absolute Reality"
  return name.replaceAll(/([a-z\d])([A-Z])/g, '$1 $2');
}

type ParsedFile = Omit<HFImageModel, 'downloadUrl' | 'size' | 'repo'> & { qnnVersion?: string };

function parseFileName(fileName: string, backend: 'mnn' | 'qnn' | 'sdxl'): ParsedFile | null {
  if (!fileName.endsWith('.zip')) return null;

  const baseName = fileName.replace('.zip', '');

  if (backend === 'qnn' || backend === 'sdxl') {
    // NPU: e.g. "AnythingV5_qnn2.28_8gen2.zip"; SDXL NPU: e.g. "illustrious_v16_qnn2.28_8gen3.zip"
    const match = baseName.match(/^(.+?)_qnn([\d.]+)_(.+)$/);
    if (!match) return null;
    const [, name, qnnVersion, variant] = match;
    if (backend === 'sdxl') {
      // `_sdxl_` in the id keeps SDXL ids apart from the SD1.5 repo's and marks the kind on the
      // registered model (imageModelMemory reads it before the files were ever listed).
      return {
        id: `${name.toLowerCase()}_sdxl_npu_${variant}`,
        name,
        displayName: `${insertSpaces(name.replaceAll('_', ' '))} (SDXL NPU)`,
        backend: 'qnn',
        kind: 'sdxl',
        variant,
        fileName,
        qnnVersion,
      };
    }
    const displayVariant = variant === 'min' ? 'non-flagship' : variant;
    return {
      id: `${name.toLowerCase()}_npu_${variant}`,
      name,
      displayName: `${insertSpaces(name)} (NPU ${displayVariant})`,
      backend: 'qnn',
      kind: 'sd15',
      variant,
      fileName,
      qnnVersion,
    };
  }

  // GPU: e.g. "AnythingV5.zip"
  return {
    id: `${baseName.toLowerCase()}_cpu`,
    name: baseName,
    displayName: `${insertSpaces(baseName)} (GPU)`,
    backend: 'mnn',
    kind: 'sd15',
    fileName,
  };
}

/** Upper bound on followed tree pages, so a server that always answers rel="next" cannot loop. */
const MAX_TREE_PAGES = 20;

/** The rel="next" URL of an HF `Link` header, only when it stays on huggingface.co. */
function nextTreePage(response: Response): string | null {
  const link = response.headers?.get?.('link');
  if (!link) return null;
  const match = /<([^>]+)>\s*;\s*rel="?next"?/i.exec(link);
  if (!match) return null;
  return match[1].startsWith('https://huggingface.co/') ? match[1] : null;
}

/**
 * Every top-level entry of the repo. The HF tree API pages long listings through a `Link:
 * rel="next"` header; reading only the first page silently dropped every model after it.
 */
async function fetchRepoFiles(repo: string): Promise<HFTreeEntry[]> {
  const entries: HFTreeEntry[] = [];
  let url: string | null = `https://huggingface.co/api/models/${repo}/tree/main`;
  for (let page = 0; url && page < MAX_TREE_PAGES; page++) {
    const response: Response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${repo}: HTTP ${response.status}`);
    }
    const body: unknown = await response.json();
    if (!Array.isArray(body)) throw new Error(`Failed to fetch ${repo}: unexpected response`);
    entries.push(...(body as HFTreeEntry[]));
    url = nextTreePage(response);
  }
  return entries;
}

/** Numeric compare of dotted versions ("2.31" > "2.28", "2.9" < "2.10"). */
function compareVersions(a: string, b: string): number {
  const pa = a.split('.').map(n => Number.parseInt(n, 10) || 0);
  const pb = b.split('.').map(n => Number.parseInt(n, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (d !== 0) return d;
  }
  return 0;
}

/**
 * SDXL NPU models of every SDXL repo: one card per id (newest QNN SDK build wins within a repo; the
 * first repo keeps an id another repo also parses to), zips of 4 GB or more left out.
 */
function collectSdxlModels(results: PromiseSettledResult<HFTreeEntry[]>[]): HFImageModel[] {
  const byId = new Map<string, { model: HFImageModel; version: string }>();
  results.forEach((result, i) => {
    if (result.status !== 'fulfilled') return;
    const repo = SDXL_REPOS[i];
    for (const entry of result.value) {
      if (entry.type !== 'file') continue;
      const parsed = parseFileName(entry.path, 'sdxl');
      if (!parsed) continue;
      const size = entry.lfs?.size ?? entry.size;
      if (!(size > 0 && size < SDXL_MAX_ZIP_BYTES)) continue;
      const { qnnVersion = '0', ...rest } = parsed;
      const prior = byId.get(rest.id);
      if (prior && (prior.model.repo !== repo || compareVersions(prior.version, qnnVersion) >= 0)) continue;
      byId.set(rest.id, {
        version: qnnVersion,
        model: { ...rest, downloadUrl: `https://huggingface.co/${repo}/resolve/main/${entry.path}`, size, repo },
      });
    }
  });
  return [...byId.values()].map(v => v.model);
}

/**
 * The image catalog. `skipQnn`: no NPU on this device, so no NPU repo is fetched. `includeSdxl`:
 * the SoC runs the SDXL NPU graphs (hardwareService.supportsSdxlNpu), so the SDXL repos are listed.
 */
export async function fetchAvailableModels(
  forceRefresh = false,
  opts?: { skipQnn?: boolean; includeSdxl?: boolean },
): Promise<HFImageModel[]> {
  const skipQnn = !!opts?.skipQnn;
  const includeSdxl = !skipQnn && !!opts?.includeSdxl;
  const cacheKey = `${skipQnn}|${includeSdxl}`;
  const cached = cache.get(cacheKey);
  if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.models;
  }

  // allSettled: one unreachable repo must not hide the other repo's models. Only when EVERY
  // requested repo fails is it an error the screen reports.
  const [mnnResult, qnnResult, ...sdxlResults] = await Promise.allSettled([
    fetchRepoFiles(REPOS.mnn),
    skipQnn ? Promise.resolve([] as HFTreeEntry[]) : fetchRepoFiles(REPOS.qnn),
    ...(includeSdxl ? SDXL_REPOS.map(repo => fetchRepoFiles(repo)) : []),
  ]);
  const failures = [mnnResult, qnnResult, ...sdxlResults].filter(
    (r): r is PromiseRejectedResult => r.status === 'rejected',
  );
  const requested = (skipQnn ? 1 : 2) + sdxlResults.length;
  if (failures.length >= requested) throw failures[0].reason;
  for (const failure of failures) {
    logger.warn('[HFModelBrowser] one image repo failed; showing the others:', failure.reason);
  }
  const mnnFiles = mnnResult.status === 'fulfilled' ? mnnResult.value : [];
  const qnnFiles = qnnResult.status === 'fulfilled' ? qnnResult.value : [];

  const models: HFImageModel[] = [];

  for (const entry of mnnFiles) {
    if (entry.type !== 'file') continue;
    const parsed = parseFileName(entry.path, 'mnn');
    if (!parsed) continue;
    models.push({
      ...parsed,
      downloadUrl: `https://huggingface.co/${REPOS.mnn}/resolve/main/${entry.path}`,
      size: entry.lfs?.size ?? entry.size,
      repo: REPOS.mnn,
    });
  }

  // One card per model + chip variant. Two QNN SDK builds of the same model (e.g. _qnn2.28_ and
  // _qnn2.31_) parse to the same id: both cards rendered with one React key and shared one
  // download-store row. Keep the newest SDK build.
  const qnnById = new Map<string, { model: HFImageModel; version: string }>();
  for (const entry of qnnFiles) {
    if (entry.type !== 'file') continue;
    const parsed = parseFileName(entry.path, 'qnn');
    if (!parsed) continue;
    const { qnnVersion = '0', ...rest } = parsed;
    const prior = qnnById.get(rest.id);
    if (prior && compareVersions(prior.version, qnnVersion) >= 0) continue;
    qnnById.set(rest.id, {
      version: qnnVersion,
      model: {
        ...rest,
        downloadUrl: `https://huggingface.co/${REPOS.qnn}/resolve/main/${entry.path}`,
        size: entry.lfs?.size ?? entry.size,
        repo: REPOS.qnn,
      },
    });
  }
  for (const { model } of qnnById.values()) models.push(model);
  models.push(...collectSdxlModels(sdxlResults));

  // Sort: GPU first, then NPU SD1.5, then NPU SDXL; alphabetically within each group
  const group = (m: HFImageModel) => (m.backend === 'mnn' ? 0 : m.kind === 'sdxl' ? 2 : 1);
  models.sort((a, b) => group(a) - group(b) || a.name.localeCompare(b.name));

  // A partial list is shown but not cached, so the next open retries the repo that failed.
  if (failures.length === 0) cache.set(cacheKey, { models, timestamp: Date.now() });
  return models;
}

export function getVariantLabel(variant?: string): string | undefined {
  return variant ? VARIANT_LABELS[variant] : undefined;
}

export function guessStyle(name: string): string {
  const lower = name.toLowerCase();
  if (
    lower.includes('reality') ||
    lower.includes('realistic') ||
    lower.includes('chillout') ||
    lower.includes('photo')
  ) {
    return 'photorealistic';
  }
  return 'anime';
}
