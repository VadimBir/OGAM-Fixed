import logger from '../utils/logger';

export interface HFImageModel {
  id: string;
  name: string;
  displayName: string;
  backend: 'mnn' | 'qnn';
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

const VARIANT_LABELS: Record<string, string> = {
  min: 'For non-flagship Snapdragon chips',
  '8gen1': 'For Snapdragon 8 Gen 1',
  '8gen2': 'For Snapdragon 8 Gen 2/3/4/5',
};

// Keyed on skipQnn: a list fetched WITHOUT the NPU repo must never answer a request that wants it
// (and vice versa), or the NPU models stayed hidden for the whole TTL after one skipQnn call.
const cache = new Map<boolean, { models: HFImageModel[]; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function insertSpaces(name: string): string {
  // Insert space before uppercase letters that follow lowercase or digits
  // e.g. "AnythingV5" -> "Anything V5", "AbsoluteReality" -> "Absolute Reality"
  return name.replaceAll(/([a-z\d])([A-Z])/g, '$1 $2');
}

type ParsedFile = Omit<HFImageModel, 'downloadUrl' | 'size' | 'repo'> & { qnnVersion?: string };

function parseFileName(fileName: string, backend: 'mnn' | 'qnn'): ParsedFile | null {
  if (!fileName.endsWith('.zip')) return null;

  const baseName = fileName.replace('.zip', '');

  if (backend === 'qnn') {
    // NPU: e.g. "AnythingV5_qnn2.28_8gen2.zip"
    const match = baseName.match(/^(.+?)_qnn([\d.]+)_(.+)$/);
    if (!match) return null;
    const [, name, qnnVersion, variant] = match;
    const displayVariant = variant === 'min' ? 'non-flagship' : variant;
    return {
      id: `${name.toLowerCase()}_npu_${variant}`,
      name,
      displayName: `${insertSpaces(name)} (NPU ${displayVariant})`,
      backend: 'qnn',
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

export async function fetchAvailableModels(forceRefresh = false, opts?: { skipQnn?: boolean }): Promise<HFImageModel[]> {
  const skipQnn = !!opts?.skipQnn;
  const cached = cache.get(skipQnn);
  if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.models;
  }

  // allSettled: one unreachable repo must not hide the other repo's models. Only when EVERY
  // requested repo fails is it an error the screen reports.
  const [mnnResult, qnnResult] = await Promise.allSettled([
    fetchRepoFiles(REPOS.mnn),
    skipQnn ? Promise.resolve([] as HFTreeEntry[]) : fetchRepoFiles(REPOS.qnn),
  ]);
  const failures = [mnnResult, qnnResult].filter(
    (r): r is PromiseRejectedResult => r.status === 'rejected',
  );
  const requested = skipQnn ? 1 : 2;
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

  // Sort: GPU first, then NPU; alphabetically within each group
  models.sort((a, b) => {
    if (a.backend !== b.backend) return a.backend === 'mnn' ? -1 : 1;
    return a.name.localeCompare(b.name);
  });

  // A partial list is shown but not cached, so the next open retries the repo that failed.
  if (failures.length === 0) cache.set(skipQnn, { models, timestamp: Date.now() });
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
