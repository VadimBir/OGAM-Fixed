/**
 * Header reading for text LoRA import: GGUF (adapter detection, tensor types) and .safetensors
 * (key names, so a text LoRA / LLM in safetensors is refused before it is copied).
 *
 * GGUF layout: vendor ggml/include/gguf.h:1-30 (magic, u32 version, i64 tensor count, i64 KV count,
 * KV pairs, then per tensor: name, u32 n_dims, i64 dims, ggml_type, u64 offset). Value type ids:
 * gguf.h:54-66. The adapter keys llama.cpp checks are `general.type` == "adapter" and
 * `adapter.type` == "lora" (vendor src/llama-arch.cpp:166 / :417, checked in
 * src/llama-adapter.cpp:202-216).
 */
import { Buffer } from 'buffer';
import RNFS from 'react-native-fs';
import logger from '../utils/logger';

export type GgufScalar = string | number | boolean;

export interface GgufHeader {
  version: number;
  tensorCount: number;
  /** Scalar KVs only (arrays are skipped). */
  kv: Record<string, GgufScalar>;
  /** ggml_type of each tensor whose info fit in the bytes read. */
  tensorTypes: number[];
  /** True when the bytes ended before the KV section (and tensor infos) were fully read. */
  truncated: boolean;
}

/** ggml_type ids (vendor ggml/include/ggml.h:390-420). */
export const GGML_TYPE_F32 = 0;
export const GGML_TYPE_F16 = 1;
const GGML_TYPE_NAMES: Record<number, string> = { 0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 30: 'BF16' };
export const ggmlTypeName = (t: number): string => GGML_TYPE_NAMES[t] ?? `type ${t}`;

class OutOfBytes extends Error {}

/** GGUF value byte sizes per type id (gguf.h:54-66); 8 = string, 9 = array: variable. */
const FIXED_SIZE: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

class Reader {
  pos = 0;
  private view: DataView;
  constructor(private bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  need(n: number): void { if (this.pos + n > this.bytes.length) throw new OutOfBytes(); }
  u32(): number { this.need(4); const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  /** u64 / i64 as a Number (exact below 2^53, which every count here is). */
  u64(): number {
    this.need(8);
    const lo = this.view.getUint32(this.pos, true);
    const hi = this.view.getUint32(this.pos + 4, true);
    this.pos += 8;
    return hi * 4294967296 + lo;
  }
  str(): string {
    const len = this.u64();
    this.need(len);
    const s = Buffer.from(this.bytes.subarray(this.pos, this.pos + len)).toString('utf8');
    this.pos += len;
    return s;
  }
  scalar(type: number): GgufScalar {
    switch (type) {
      case 0: this.need(1); return this.bytes[this.pos++];
      case 1: this.need(1); return this.view.getInt8(this.pos++);
      case 2: { this.need(2); const v = this.view.getUint16(this.pos, true); this.pos += 2; return v; }
      case 3: { this.need(2); const v = this.view.getInt16(this.pos, true); this.pos += 2; return v; }
      case 4: return this.u32();
      case 5: { this.need(4); const v = this.view.getInt32(this.pos, true); this.pos += 4; return v; }
      case 6: { this.need(4); const v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; }
      case 7: this.need(1); return this.bytes[this.pos++] !== 0;
      case 8: return this.str();
      case 10: case 11: return this.u64();
      case 12: { this.need(8); const v = this.view.getFloat64(this.pos, true); this.pos += 8; return v; }
      default: throw new Error(`unknown GGUF value type ${type}`);
    }
  }
  skipArray(): void {
    const elemType = this.u32();
    const n = this.u64();
    if (elemType === 8) { for (let i = 0; i < n; i++) this.str(); return; }
    if (elemType === 9) throw new Error('nested GGUF arrays are not supported');
    const size = FIXED_SIZE[elemType];
    if (size === undefined) throw new Error(`unknown GGUF array type ${elemType}`);
    this.need(n * size);
    this.pos += n * size;
  }
}

/**
 * Parse a GGUF header prefix. Returns null when the bytes are not GGUF. A prefix that ends early is
 * not an error: what was read is returned with `truncated` set.
 */
export function parseGgufHeader(bytes: Uint8Array): GgufHeader | null {
  if (bytes.length < 4 || Buffer.from(bytes.subarray(0, 4)).toString('latin1') !== 'GGUF') return null;
  const r = new Reader(bytes);
  r.pos = 4;
  const header: GgufHeader = { version: 0, tensorCount: 0, kv: {}, tensorTypes: [], truncated: false };
  try {
    header.version = r.u32();
    header.tensorCount = r.u64();
    const kvCount = r.u64();
    for (let i = 0; i < kvCount; i++) {
      const key = r.str();
      const type = r.u32();
      if (type === 9) r.skipArray();
      else header.kv[key] = r.scalar(type);
    }
    for (let i = 0; i < header.tensorCount; i++) {
      r.str();
      const nDims = r.u32();
      r.need(nDims * 8);
      r.pos += nDims * 8;
      header.tensorTypes.push(r.u32());
      r.u64();
    }
  } catch (e) {
    if (!(e instanceof OutOfBytes)) throw e;
    header.truncated = true;
  }
  return header;
}

export type GgufAdapterKind =
  | { kind: 'text-lora'; architecture: string; alpha?: number }
  | { kind: 'other-adapter'; adapterType: string }
  | { kind: 'model' };

/**
 * What a GGUF is, by the keys llama.cpp itself checks (llama-adapter.cpp:202-216): general.type
 * "adapter" + adapter.type "lora" = a LoRA; general.type "adapter" with another adapter.type is an
 * adapter llama.rn's LoRA path refuses; anything else is left to the model import.
 */
export function classifyGgufAdapter(header: GgufHeader): GgufAdapterKind {
  if (header.kv['general.type'] !== 'adapter') return { kind: 'model' };
  const adapterType = String(header.kv['adapter.type'] ?? '');
  if (adapterType !== 'lora') return { kind: 'other-adapter', adapterType: adapterType || '(missing)' };
  const alpha = header.kv['adapter.lora.alpha'];
  return {
    kind: 'text-lora',
    architecture: String(header.kv['general.architecture'] ?? 'unknown'),
    ...(typeof alpha === 'number' ? { alpha } : {}),
  };
}

/**
 * Tensor types outside F32/F16. The pinned OpenCL backend takes F16/F32 MUL_MAT
 * (ggml-opencl.cpp:8795-8802) and HTP takes F16/F32 (ggml-hexagon.cpp:4945-4990, within its VTCM
 * budget); other LoRA tensor types are not proven to run there.
 */
export function nonAcceleratedLoraTypes(tensorTypes: number[]): string[] {
  const odd = new Set(tensorTypes.filter(t => t !== GGML_TYPE_F32 && t !== GGML_TYPE_F16));
  return [...odd].map(ggmlTypeName);
}

const READ_CHUNK = 256 * 1024;
const MAX_READ_CALLS = 64;

/**
 * Read up to `length` bytes from `position`. RNFS.read (Android: one InputStream.read per call,
 * @dr.pogodin/react-native-fs ReactNativeFsModule.kt:475-493, content:// via ContentResolver :891-901)
 * may return fewer bytes than asked, so this reads in chunks until `length` or end of file.
 */
export async function readFileBytes(path: string, length: number, position = 0): Promise<Uint8Array> {
  const parts: Uint8Array[] = [];
  let got = 0;
  // Bounded: a stream that keeps answering short reads cannot spin forever.
  for (let calls = 0; got < length && calls < MAX_READ_CALLS; calls++) {
    const want = Math.min(READ_CHUNK, length - got);
    const b64 = await RNFS.read(path, want, position + got, 'base64');
    const chunk = new Uint8Array(Buffer.from(b64, 'base64'));
    if (chunk.length === 0) break;
    parts.push(chunk);
    got += chunk.length;
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

/** Header bytes read for detection. LoRA GGUF headers are small; a full model's tokenizer arrays
 *  run past this, which only matters for files that are not adapters. */
export const GGUF_HEADER_PROBE_BYTES = 1024 * 1024;

export async function readGgufHeader(path: string, maxBytes = GGUF_HEADER_PROBE_BYTES): Promise<GgufHeader | null> {
  return parseGgufHeader(await readFileBytes(path, maxBytes));
}

/** Cap on the safetensors JSON header read (an LLM header is tens of KB to a few MB). */
const SAFETENSORS_HEADER_CAP = 4 * 1024 * 1024;

/**
 * Tensor names from a .safetensors header (u64 LE JSON length, then the JSON). A header beyond the
 * cap is read up to the cap and its names taken by pattern, which is enough to classify.
 */
export async function readSafetensorsKeys(path: string): Promise<string[]> {
  const lenBytes = await readFileBytes(path, 8, 0);
  if (lenBytes.length < 8) return [];
  const view = new DataView(lenBytes.buffer, lenBytes.byteOffset, 8);
  const headerLen = view.getUint32(4, true) * 4294967296 + view.getUint32(0, true);
  if (headerLen <= 0) return [];
  const text = Buffer.from(await readFileBytes(path, Math.min(headerLen, SAFETENSORS_HEADER_CAP), 8)).toString('utf8');
  return safetensorsKeysFromHeaderText(text);
}

/** Same cap as the native reader (SdCppImageModule.kt MAX_HEADER_BYTES). */
const SAFETENSORS_MAX_HEADER_LEN = 100 * 1024 * 1024;

/**
 * Whether the first 9 bytes are a safetensors file: u64 LE header length (sane, nonzero), then the
 * JSON header, which the format requires to start with `{` (0x7B).
 */
export function isSafetensorsPrefix(bytes: Uint8Array): boolean {
  if (bytes.length < 9) return false;
  const view = new DataView(bytes.buffer, bytes.byteOffset, 8);
  const headerLen = view.getUint32(4, true) * 4294967296 + view.getUint32(0, true);
  return headerLen >= 2 && headerLen <= SAFETENSORS_MAX_HEADER_LEN && bytes[8] === 0x7b;
}

/** Content sniff for a picked file whose name has no .safetensors extension (`.sft`, `.bin`, none). */
export async function looksLikeSafetensors(path: string): Promise<boolean> {
  try {
    return isSafetensorsPrefix(await readFileBytes(path, 9, 0));
  } catch (error) {
    logger.warn('[SDCPP-IMPORT] could not read the picked file to sniff its format', error);
    return false;
  }
}

export function safetensorsKeysFromHeaderText(text: string): string[] {
  const keys: string[] = [];
  const re = /"([^"\\]+)"\s*:\s*\{/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m[1] !== '__metadata__') keys.push(m[1]);
  }
  return keys;
}
