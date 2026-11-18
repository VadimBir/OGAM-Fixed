/**
 * Builds GGUF bytes per vendor ggml/include/gguf.h:1-30 (magic, u32 version, u64 tensor count,
 * u64 KV count, KVs, tensor infos) for header-detection tests. Values: string -> type 8,
 * { f32 } -> type 6, { u32 } -> type 4, { strArray } -> type 9 of type 8.
 */
import { Buffer } from 'buffer';

type KvValue = string | { f32: number } | { u32: number } | { strArray: string[] };

const u32 = (v: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(v); return b; };
const u64 = (v: number) => { const b = Buffer.alloc(8); b.writeUInt32LE(v % 4294967296); b.writeUInt32LE(Math.floor(v / 4294967296), 4); return b; };
const str = (s: string) => { const body = Buffer.from(s, 'utf8'); return Buffer.concat([u64(body.length), body]); };

function value(v: KvValue): Buffer {
  if (typeof v === 'string') return Buffer.concat([u32(8), str(v)]);
  if ('f32' in v) { const b = Buffer.alloc(4); b.writeFloatLE(v.f32); return Buffer.concat([u32(6), b]); }
  if ('u32' in v) return Buffer.concat([u32(4), u32(v.u32)]);
  return Buffer.concat([u32(9), u32(8), u64(v.strArray.length), ...v.strArray.map(str)]);
}

export function buildGguf(opts: {
  kv: Array<[string, KvValue]>;
  tensors?: Array<{ name: string; type: number; dims?: number[] }>;
}): Uint8Array {
  const tensors = opts.tensors ?? [];
  const parts: Buffer[] = [Buffer.from('GGUF', 'latin1'), u32(3), u64(tensors.length), u64(opts.kv.length)];
  for (const [key, v] of opts.kv) parts.push(str(key), value(v));
  for (const t of tensors) {
    const dims = t.dims ?? [64, 8];
    parts.push(str(t.name), u32(dims.length), ...dims.map(u64), u32(t.type), u64(0));
  }
  return new Uint8Array(Buffer.concat(parts));
}

/** A safetensors prefix: u64 LE JSON length + JSON header (no tensor data). */
export function buildSafetensorsHeader(keys: string[]): Uint8Array {
  const header: Record<string, unknown> = { __metadata__: { format: 'pt' } };
  keys.forEach((k, i) => { header[k] = { dtype: 'F16', shape: [8, 8], data_offsets: [i * 128, (i + 1) * 128] }; });
  const json = Buffer.from(JSON.stringify(header), 'utf8');
  return new Uint8Array(Buffer.concat([u64(json.length), json]));
}

/** RNFS.read mock (base64) serving `bytes` for (length, position). */
export const serveBytes = (bytes: Uint8Array) =>
  (_path: string, length: number, position: number) =>
    Promise.resolve(Buffer.from(bytes.subarray(position, position + length)).toString('base64'));
