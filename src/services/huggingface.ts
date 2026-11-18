import {
  HFModelSearchResult,
  ModelInfo,
  ModelFile,
  ModelCredibility,
} from '../types';
import {
  HF_API,
  LMSTUDIO_AUTHORS,
  OFFICIAL_MODEL_AUTHORS,
  VERIFIED_QUANTIZERS,
} from '../constants';
import { looksLikeVisionModel } from '../utils/visionModel';
import { huggingFaceRevisionPath } from '../utils/modelOrigin';
import { lookupQuantizationInfo, parseQuantization } from '../utils/quantization';
import { isMMProjFile, pickMmProjForDownload } from './mmproj';

interface TreeEntry {
  type: string;
  path: string;
  size?: number;
  lfs?: { size: number };
}

/** Upper bound on tree pages followed for one listing (HF serves ~1000 entries a page). */
const MAX_TREE_PAGES = 20;

function basenameOfPath(path: string): string {
  return path.substring(path.lastIndexOf('/') + 1);
}

/** The `rel="next"` URL of an HF paginated response, or null. Tolerates a response without headers. */
function nextPageUrl(response: Response): string | null {
  const link = response.headers?.get?.('link');
  if (!link) return null;
  for (const part of link.split(',')) {
    const m = /<([^>]+)>\s*;\s*rel="?next"?/i.exec(part);
    if (m) return m[1];
  }
  return null;
}

const SPLIT_PART_RE = /-(\d{5})-of-(\d{5})\.gguf$/i;

/**
 * Collapse split GGUF parts (`<name>-00001-of-00003.gguf`, …) into ONE row: part 1, carrying the
 * summed size of every listed part and `splitPartCount`. Listing each part as its own "model" let
 * the user download `-00002-of-00003` alone, a file llama.cpp cannot load. Parts 2..N are hidden;
 * a group whose part 1 is not listed is dropped (nothing loadable to offer).
 */
export function groupSplitParts(files: ModelFile[]): ModelFile[] {
  const groups = new Map<string, { first?: ModelFile; total: number; count: number; parts: number }>();
  const out: ModelFile[] = [];
  for (const file of files) {
    const m = SPLIT_PART_RE.exec(file.name);
    const parts = m ? Number.parseInt(m[2], 10) : 0;
    if (!m || parts <= 1) {
      out.push(file);
      continue;
    }
    const key = file.name.replace(SPLIT_PART_RE, '');
    const group = groups.get(key) ?? { total: 0, count: 0, parts };
    group.total += file.size || 0;
    group.count += 1;
    if (Number.parseInt(m[1], 10) === 1) group.first = file;
    groups.set(key, group);
  }
  for (const group of groups.values()) {
    if (!group.first) continue;
    out.push({ ...group.first, size: group.total, splitPartCount: group.parts });
  }
  return out;
}

class HuggingFaceService {
  private baseUrl = HF_API.baseUrl;
  private apiUrl = HF_API.apiUrl;

  private async fetchJson<T>(url: string): Promise<T> {
    const response = await fetch(url, {
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json() as Promise<T>;
  }

  async searchModels(
    query: string = '',
    options: {
      limit?: number;
      sort?: string;
      direction?: string;
      pipelineTag?: string;
    } = {},
  ): Promise<ModelInfo[]> {
    const {
      limit = 30,
      sort = 'downloads',
      direction = '-1',
      pipelineTag,
    } = options;
    const params = new URLSearchParams({
      filter: 'gguf',
      sort,
      direction,
      limit: limit.toString(),
    });
    if (query) params.append('search', query);
    if (pipelineTag) params.append('pipeline_tag', pipelineTag);
    const results = await this.fetchJson<HFModelSearchResult[]>(
      `${this.apiUrl}/models?${params.toString()}`,
    );
    return results.map(this.transformModelResult);
  }

  /**
   * Repos that publish a file of this name, each reduced to its file list with exact byte sizes.
   *
   * This is a CANDIDATE generator, not an identification: several repos publish the same file
   * name (`SmolVLM-500M-Instruct-GGUF` matches three, one an `i1` requantisation). The caller
   * decides which is ours by comparing sizes - see resolveVisionRepairSource.
   */
  async findReposPublishing(
    fileName: string,
    limit = 10,
  ): Promise<
    { repoId: string; files: { name: string; sizeBytes?: number }[] }[]
  > {
    // The repo is usually named after the model, so the file name minus its quant/extension is the
    // best query we have. `.gguf` and the trailing quant tag never appear in a repo id.
    const query = fileName
      .replace(/\.gguf$/i, '')
      .replace(/[._-]i1$/i, '')
      .replace(/[._-](?:Q[0-9]+(?:_[A-Z0-9]+)*|F(?:16|32))$/i, '');
    const results = await this.searchModels(query, { limit });
    const listings = await Promise.all(
      results.map(async result => ({
        repoId: result.id,
        files: await this.listRepoFileSizes(result.id),
      })),
    );
    return listings.filter(listing => listing.files.length > 0);
  }

  /** Every file in a repo with its exact size - the only field that identifies a build. */
  private async listRepoFileSizes(
    modelId: string,
  ): Promise<{ name: string; sizeBytes?: number }[]> {
    try {
      const result = await this.fetchJson<{
        siblings?: {
          rfilename: string;
          size?: number;
          lfs?: { size?: number };
        }[];
      }>(`${this.apiUrl}/models/${modelId}?blobs=true`);
      return (result.siblings ?? []).map(sibling => ({
        name: sibling.rfilename,
        sizeBytes: sibling.lfs?.size ?? sibling.size,
      }));
    } catch {
      // An unreachable or private repo is simply not a candidate. HF answers an unknown repo with
      // 401, so a throw here means "cannot confirm", never "this is the one".
      return [];
    }
  }

  async getModelDetails(modelId: string): Promise<ModelInfo> {
    const result = await this.fetchJson<HFModelSearchResult>(
      `${this.apiUrl}/models/${modelId}`,
    );
    return this.transformModelResult(result);
  }

  async getModelFiles(
    modelId: string,
    revision = 'main',
  ): Promise<ModelFile[]> {
    try {
      const files = await this.fetchTreeEntries(modelId, revision);
      if (!files) return this.getModelFilesFromSiblings(modelId, revision);
      const allGguf = files.filter(
        f => f.type === 'file' && f.path.toLowerCase().endsWith('.gguf'),
      );
      const mmProjFiles = allGguf.filter(f => this.isMMProjFile(f.path));
      const modelFiles = allGguf.filter(f => !this.isMMProjFile(f.path));
      return groupSplitParts(
        modelFiles.map(file => ({
          name: file.path,
          size: file.lfs?.size || file.size || 0,
          quantization: this.extractQuantization(file.path),
          downloadUrl: this.getDownloadUrl(modelId, file.path, revision),
          mmProjFile: this.findMatchingMMProj(file.path, mmProjFiles, {
            modelId,
            revision,
          }),
        })),
      ).sort((a, b) => a.size - b.size);
    } catch {
      return this.getModelFilesFromSiblings(modelId, revision);
    }
  }

  /**
   * Every entry of a repo revision, INCLUDING sub-folders (`?recursive=1`) and every page (HF pages
   * the tree endpoint and names the next page in a `Link: <url>; rel="next"` header). The bare call
   * listed the root folder's first page only, so a repo that keeps its quants in folders, or has
   * more files than one page, showed a partial list. Null when any page fails: the caller then uses
   * the siblings listing, which is recursive and unpaged, so both paths list the same files.
   */
  private async fetchTreeEntries(
    modelId: string,
    revision: string,
  ): Promise<TreeEntry[] | null> {
    const entries: TreeEntry[] = [];
    let url: string | null = `${this.apiUrl}/models/${modelId}/tree/${encodeURIComponent(revision)}?recursive=1`;
    for (let page = 0; url && page < MAX_TREE_PAGES; page++) {
      const response: Response = await fetch(url, {
        headers: { Accept: 'application/json' },
      });
      if (!response.ok) return null;
      const body: unknown = await response.json();
      if (!Array.isArray(body)) return null;
      entries.push(...(body as TreeEntry[]));
      url = nextPageUrl(response);
    }
    return entries;
  }

  private async getModelFilesFromSiblings(
    modelId: string,
    revision = 'main',
  ): Promise<ModelFile[]> {
    const revisionPath =
      revision === 'main' ? '' : `/revision/${encodeURIComponent(revision)}`;
    // `blobs=true` makes siblings carry sizes; without it every file read as 0 bytes here while the
    // tree path showed real sizes for the same repo.
    const result = await this.fetchJson<HFModelSearchResult>(
      `${this.apiUrl}/models/${modelId}${revisionPath}?blobs=true`,
    );
    if (!result.siblings) return [];
    const allGguf = result.siblings.filter(f =>
      f.rfilename.toLowerCase().endsWith('.gguf'),
    );
    const mmProjFiles = allGguf.filter(f => this.isMMProjFile(f.rfilename));
    const modelFiles = allGguf.filter(f => !this.isMMProjFile(f.rfilename));
    const mmProjForMatch = mmProjFiles.map(f => ({
      path: f.rfilename,
      size: f.size,
      lfs: f.lfs,
    }));
    return groupSplitParts(
      modelFiles.map(file => ({
        ...this.transformFileInfo(modelId, file, revision),
        mmProjFile: this.findMatchingMMProj(file.rfilename, mmProjForMatch, {
          modelId,
          revision,
        }),
      })),
    ).sort((a, b) => a.size - b.size);
  }

  getDownloadUrl(
    modelId: string,
    fileName: string,
    revision: string = 'main',
  ): string {
    return `${this.baseUrl}/${modelId}/resolve/${huggingFaceRevisionPath(
      revision,
    )}/${fileName}`;
  }

  private determineCredibility(author: string): ModelCredibility {
    if (LMSTUDIO_AUTHORS.includes(author))
      return {
        source: 'lmstudio',
        isOfficial: false,
        isVerifiedQuantizer: true,
        verifiedBy: 'LM Studio',
      };
    if (OFFICIAL_MODEL_AUTHORS[author])
      return {
        source: 'official',
        isOfficial: true,
        isVerifiedQuantizer: false,
        verifiedBy: OFFICIAL_MODEL_AUTHORS[author],
      };
    if (VERIFIED_QUANTIZERS[author])
      return {
        source: 'verified-quantizer',
        isOfficial: false,
        isVerifiedQuantizer: true,
        verifiedBy: VERIFIED_QUANTIZERS[author],
      };
    return {
      source: 'community',
      isOfficial: false,
      isVerifiedQuantizer: false,
    };
  }

  private transformModelResult = (result: HFModelSearchResult): ModelInfo => {
    const files =
      result.siblings
        ?.filter(file => file.rfilename.endsWith('.gguf'))
        .map(file => this.transformFileInfo(result.id, file)) || [];

    const author = result.author || result.id.split('/')[0] || 'Unknown';
    const credibility = this.determineCredibility(author);

    return {
      id: result.id,
      name: result.id.split('/').pop() || result.id,
      author,
      description: this.extractDescription(result),
      downloads: result.downloads || 0,
      likes: result.likes || 0,
      tags: result.tags || [],
      lastModified: result.lastModified,
      files,
      credibility,
    };
  };

  private transformFileInfo(
    modelId: string,
    file: {
      rfilename: string;
      size?: number;
      lfs?: { size: number; sha256: string };
    },
    revision = 'main',
  ): ModelFile {
    const fileName = file.rfilename;
    const size = file.lfs?.size || file.size || 0;
    const quantization = this.extractQuantization(fileName);

    return {
      name: fileName,
      size,
      quantization,
      downloadUrl: this.getDownloadUrl(modelId, fileName, revision),
      sha256: file.lfs?.sha256,
    };
  }

  // The shared label rule (src/utils/quantization.ts): IQ4_XS stays IQ4_XS, BF16 stays BF16,
  // UD-Q4_K_XL stays UD-Q4_K_XL. Read from the file's own name, never its folder.
  private extractQuantization(fileName: string): string {
    return parseQuantization(basenameOfPath(fileName));
  }

  // Delegates to the single source of truth (src/services/mmproj.ts) so "is this a projector" is defined once.
  private isMMProjFile(fileName: string): boolean {
    return isMMProjFile(fileName);
  }

  // Routes through the single projector-rule owner (src/services/mmproj.ts). Quant is NOT a matching
  // signal (one projector serves every quant of its model); a projector whose filename names a DIFFERENT
  // model+variant is the wrong architecture and is REFUSED, so the model downloads with its correct
  // projector or text-only rather than being mispaired (#510). See pickMmProjForDownload for the rule.
  private findMatchingMMProj(
    modelFileName: string,
    mmProjFiles: Array<{ path: string; size?: number; lfs?: { size: number } }>,
    source: string | { modelId: string; revision: string },
  ): { name: string; size: number; downloadUrl: string } | undefined {
    // Matched on BASE names: with the recursive listing a model in `Q4_K_M/` and its projector at the
    // repo root would otherwise differ by the folder token in their identity stems.
    const chosen = pickMmProjForDownload(
      basenameOfPath(modelFileName),
      mmProjFiles.map(f => basenameOfPath(f.path)),
    );
    if (!chosen) return undefined;

    const file = mmProjFiles.find(f => basenameOfPath(f.path) === chosen);
    if (!file) return undefined;
    const modelId = typeof source === 'string' ? source : source.modelId;
    const revision = typeof source === 'string' ? 'main' : source.revision;
    return {
      name: file.path,
      size: file.lfs?.size || file.size || 0,
      downloadUrl: this.getDownloadUrl(modelId, file.path, revision),
    };
  }

  private detectModelType(name: string, tags: string[]): string {
    if (
      tags.some(t => t.includes('code')) ||
      name.includes('code') ||
      name.includes('coder')
    )
      return 'Code generation';
    // Single source of truth (utils/visionModel) — was a 3-keyword subset that missed Pixtral/
    // Moondream/InternVL etc. (DR2).
    if (looksLikeVisionModel({ name, tags })) return 'Vision';
    return 'Text generation';
  }

  private extractDescription(result: HFModelSearchResult): string {
    const name = (result.id.split('/').pop() || '').toLowerCase();
    const tags = result.tags?.map(t => t.toLowerCase()) || [];
    const author = result.author || result.id.split('/')[0] || '';
    const type = this.detectModelType(name, tags);
    const paramMatch = name.match(/(\d+\.?\d*)\s*b(?:\b|-)/);
    const paramStr = paramMatch ? `${paramMatch[1]}B` : null;
    const license = result.cardData?.license;
    const licenseStr = license
      ? license.toUpperCase().replaceAll('-', ' ')
      : null;
    const parts: string[] = [type];
    if (paramStr) parts.push(paramStr);
    if (licenseStr) parts.push(licenseStr);
    if (author) parts.push(`by ${author}`);
    return parts.join(' · ');
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';

    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));

    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${units[i]}`;
  }

  getQuantizationInfo(quantization: string) {
    return (
      lookupQuantizationInfo(quantization) || {
        bitsPerWeight: 4.5,
        quality: 'Unknown',
        description: 'Unknown quantization level',
        recommended: false,
      }
    );
  }
}

export const huggingFaceService = new HuggingFaceService();
