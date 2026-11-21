/**
 * Civitai model search (public REST API, https://civitai.com/api/v1/models).
 *
 * Only what this app can run is offered: single-file SafeTensor checkpoints / LoRAs of the
 * SD 1.x, SD 2.x and SDXL families (the families stable-diffusion.cpp is sized for in
 * sdCppMemory.ts). PickleTensor (.ckpt/.pt) is never offered: unpickling runs arbitrary code.
 * GGUF / Flux / SD3 / video models are skipped (no import path or no memory model for them).
 *
 * The response is untrusted JSON: every field is read defensively and a malformed item is
 * skipped, never thrown on. The optional API key is sent ONLY to https civitai.com hosts.
 */
import logger from '../utils/logger';

export type CivitaiModelType = 'Checkpoint' | 'LORA';
/** Architecture families this app can run on stable-diffusion.cpp. */
export type CivitaiFamily = 'sd1' | 'sd2' | 'sdxl';
export type CivitaiSort = 'Most Downloaded' | 'Highest Rated' | 'Newest';

export interface CivitaiFile {
  id: number;
  name: string;
  sizeBytes: number;
  /** Civitai `metadata.fp` (fp16 / fp32 / bf16), when given. */
  fp?: string;
  downloadUrl: string;
  /** Upper/lower-case hex SHA-256 from `hashes.SHA256`, when given (native download verifies it). */
  sha256?: string;
  primary: boolean;
}

export interface CivitaiResult {
  /** Stable per file: `${modelId}:${versionId}:${fileId}`. */
  key: string;
  modelId: number;
  versionId: number;
  name: string;
  versionName: string;
  type: CivitaiModelType;
  /** Civitai's own label, e.g. "SD 1.5", "SDXL 1.0", "Pony". */
  baseModel: string;
  family: CivitaiFamily;
  creator?: string;
  nsfw: boolean;
  previewUrl?: string;
  downloadCount?: number;
  file: CivitaiFile;
}

export interface CivitaiSearchParams {
  query?: string;
  type: CivitaiModelType;
  family: CivitaiFamily;
  sort?: CivitaiSort;
  cursor?: string | null;
  limit?: number;
  apiKey?: string | null;
}

export interface CivitaiSearchPage {
  results: CivitaiResult[];
  nextCursor: string | null;
  /** Items on this page that were dropped (no runnable file, other base model, NSFW). */
  skipped: number;
}

export const CIVITAI_API_BASE = 'https://civitai.com/api/v1/models';
const REQUEST_TIMEOUT_MS = 15_000;
const DEFAULT_LIMIT = 20;

/** Civitai `baseModel` labels sent as the `baseModels` filter for each family. */
const FAMILY_BASE_MODELS: Record<CivitaiFamily, string[]> = {
  sd1: ['SD 1.5', 'SD 1.4'],
  sd2: ['SD 2.1', 'SD 2.0'],
  sdxl: ['SDXL 1.0', 'Pony', 'Illustrious'],
};

/** The family of a Civitai `baseModel` label, or null when this app cannot run it. */
export function civitaiBaseModelFamily(baseModel: unknown): CivitaiFamily | null {
  if (typeof baseModel !== 'string') return null;
  const b = baseModel.trim().toLowerCase();
  if (/^sd ?1\.[45]\b/.test(b)) return 'sd1';
  if (/^sd ?2\.[01]\b/.test(b)) return 'sd2';
  // Pony / Illustrious / NoobAI are SDXL-architecture fine-tunes.
  if (b.startsWith('sdxl') || b === 'pony' || b === 'illustrious' || b === 'noobai') return 'sdxl';
  return null;
}

const HOST_RE = /^https:\/\/([^/?#:@]+)(?::\d+)?(?:[/?#]|$)/i;

/** True only for an https URL whose host is civitai.com or a subdomain of it. */
export function isCivitaiUrl(url: string): boolean {
  const host = HOST_RE.exec(url)?.[1]?.toLowerCase();
  if (!host) return false;
  return host === 'civitai.com' || host.endsWith('.civitai.com');
}

/** `Authorization: Bearer <key>` for a civitai.com URL; {} for any other host or no key. */
export function civitaiAuthHeaders(url: string, apiKey?: string | null): Record<string, string> {
  const key = apiKey?.trim();
  if (!key || !isCivitaiUrl(url)) return {};
  return { Authorization: `Bearer ${key}` };
}

/**
 * The download URL with the key as `token=` (civitai.com only). The native background downloader
 * (WorkerDownload.kt buildRequest) sends no custom headers, so a gated file's key must ride in the
 * query; OkHttp then follows the redirect to the signed CDN URL, which does not carry it.
 */
export function withCivitaiDownloadToken(url: string, apiKey?: string | null): string {
  const key = apiKey?.trim();
  if (!key || !isCivitaiUrl(url)) return url;
  return `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(key)}`;
}

/** The URL with any `token=` query value replaced, safe for logs and persisted metadata. */
export function redactCivitaiToken(url: string): string {
  return url.replace(/([?&]token=)[^&#]*/i, '$1***');
}

function rec(value: unknown): Record<string, any> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, any>) : null;
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function str(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

/** Runnable = SafeTensor model weights (not VAE / training data / config), https civitai.com URL. */
function toRunnableFile(raw: unknown): CivitaiFile | null {
  const f = rec(raw);
  if (!f) return null;
  const id = num(f.id);
  const name = str(f.name);
  const downloadUrl = str(f.downloadUrl);
  const sizeKB = num(f.sizeKB);
  if (id === null || !name || !downloadUrl || sizeKB === null || sizeKB <= 0) return null;
  if (!isCivitaiUrl(downloadUrl)) return null;
  const meta = rec(f.metadata);
  const format = str(meta?.format)?.toLowerCase();
  const isSafetensorsName = name.toLowerCase().endsWith('.safetensors');
  // The name must be .safetensors (the import path keys on it) and a declared format must agree.
  if (!isSafetensorsName || (format && format !== 'safetensor')) return null;
  const fileType = str(f.type);
  if (fileType && fileType !== 'Model' && fileType !== 'Pruned Model') return null;
  const hashes = rec(f.hashes);
  const sha = str(hashes?.SHA256);
  return {
    id,
    name,
    sizeBytes: Math.round(sizeKB * 1024),
    fp: str(meta?.fp) ?? undefined,
    downloadUrl,
    sha256: sha && /^[0-9a-f]{64}$/i.test(sha) ? sha : undefined,
    primary: f.primary === true,
  };
}

/** The primary runnable file, else the smallest runnable one (pruned fp16 beats full fp32). */
export function pickRunnableFile(files: unknown): CivitaiFile | null {
  if (!Array.isArray(files)) return null;
  const runnable = files.map(toRunnableFile).filter((f): f is CivitaiFile => f !== null);
  if (runnable.length === 0) return null;
  return runnable.find(f => f.primary) ?? [...runnable].sort((a, b) => a.sizeBytes - b.sizeBytes)[0];
}

/** First still image (https, not a video). No content filter: every rating is shown. */
function pickPreview(images: unknown): string | undefined {
  if (!Array.isArray(images)) return undefined;
  for (const raw of images) {
    const img = rec(raw);
    const url = str(img?.url);
    if (!img || !url || !/^https:\/\//i.test(url)) continue;
    if (img.type === 'video' || /\.(mp4|webm)(\?|$)/i.test(url)) continue;
    return url;
  }
  return undefined;
}

/** One result per item: its newest version (first listed) of the requested family with a runnable file. */
export function parseCivitaiItem(raw: unknown, params: Pick<CivitaiSearchParams, 'type' | 'family'>): CivitaiResult | null {
  const item = rec(raw);
  if (!item) return null;
  const modelId = num(item.id);
  const name = str(item.name);
  if (modelId === null || !name) return null;
  if (item.type !== params.type) return null;
  const nsfw = item.nsfw === true;
  if (!Array.isArray(item.modelVersions)) return null;
  for (const rawVersion of item.modelVersions) {
    const v = rec(rawVersion);
    const versionId = num(v?.id);
    if (!v || versionId === null) continue;
    const family = civitaiBaseModelFamily(v.baseModel);
    if (family !== params.family) continue;
    const file = pickRunnableFile(v.files);
    if (!file) continue;
    const stats = rec(item.stats);
    return {
      key: `${modelId}:${versionId}:${file.id}`,
      modelId,
      versionId,
      name,
      versionName: str(v.name) ?? '',
      type: params.type,
      baseModel: str(v.baseModel) ?? '',
      family,
      creator: str(rec(item.creator)?.username) ?? undefined,
      nsfw,
      previewUrl: pickPreview(v.images),
      downloadCount: num(stats?.downloadCount) ?? undefined,
      file,
    };
  }
  return null;
}

function nextCursorOf(body: Record<string, any>): string | null {
  const meta = rec(body.metadata);
  if (!meta) return null;
  const cursor = meta.nextCursor;
  if (typeof cursor === 'string' && cursor) return cursor;
  if (typeof cursor === 'number' && Number.isFinite(cursor)) return String(cursor);
  const nextPage = str(meta.nextPage);
  const fromUrl = nextPage ? /[?&]cursor=([^&#]+)/.exec(nextPage)?.[1] : undefined;
  return fromUrl ? decodeURIComponent(fromUrl) : null;
}

export function buildCivitaiSearchUrl(params: CivitaiSearchParams, includeBaseModels = true): string {
  const q: string[] = [];
  const add = (k: string, v: string) => q.push(`${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
  add('limit', String(params.limit ?? DEFAULT_LIMIT));
  add('types', params.type);
  add('sort', params.sort ?? 'Most Downloaded');
  const query = params.query?.trim();
  if (query) add('query', query);
  // No `nsfw` param: Civitai then returns every rating (nsfw=false would hide NSFW models,
  // nsfw=true would return NSFW only). The app applies no content filter.
  if (includeBaseModels) for (const base of FAMILY_BASE_MODELS[params.family]) add('baseModels', base);
  if (params.cursor) add('cursor', params.cursor);
  return `${CIVITAI_API_BASE}?${q.join('&')}`;
}

async function getJson(url: string, apiKey?: string | null): Promise<{ status: number; ok: boolean; body: unknown }> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      headers: { Accept: 'application/json', ...civitaiAuthHeaders(url, apiKey) },
      signal: controller.signal,
    });
    if (!response.ok) return { status: response.status, ok: false, body: null };
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new Error('Civitai returned an unreadable response. Try again later.');
    }
    return { status: response.status, ok: true, body };
  } catch (error: any) {
    if (error?.name === 'AbortError') throw new Error('Civitai did not answer in time. Check your connection.');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function httpError(status: number): Error {
  if (status === 401 || status === 403) return new Error('Civitai refused the request. Check your Civitai API key.');
  if (status === 429) return new Error('Civitai is rate-limiting searches. Wait a minute and try again.');
  return new Error(`Civitai search failed (HTTP ${status}).`);
}

/** One page of runnable Civitai models. Throws a user-facing Error on network/HTTP/JSON failure. */
export async function searchCivitaiModels(params: CivitaiSearchParams): Promise<CivitaiSearchPage> {
  let res = await getJson(buildCivitaiSearchUrl(params, true), params.apiKey);
  // A `baseModels` label the API does not accept answers 400. Retry once without the filter:
  // parseCivitaiItem filters by family on the client anyway.
  if (!res.ok && res.status === 400) {
    logger.warn('[Civitai] search rejected with baseModels filter; retrying without it');
    res = await getJson(buildCivitaiSearchUrl(params, false), params.apiKey);
  }
  if (!res.ok) throw httpError(res.status);
  const body = rec(res.body);
  if (!body || !Array.isArray(body.items)) {
    throw new Error('Civitai returned an unexpected response. Try again later.');
  }
  const results: CivitaiResult[] = [];
  const seen = new Set<string>();
  let skipped = 0;
  for (const raw of body.items) {
    const parsed = parseCivitaiItem(raw, params);
    if (!parsed || seen.has(parsed.key)) {
      skipped++;
      continue;
    }
    seen.add(parsed.key);
    results.push(parsed);
  }
  return { results, nextCursor: nextCursorOf(body), skipped };
}
