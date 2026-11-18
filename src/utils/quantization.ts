/**
 * GGUF quantization label: the ONE rule that turns a file name into a quant label.
 *
 * Three copies used to exist and each was wrong somewhere: the catalog matched known keys by
 * substring and then fell back to /[QqFf]\d+[_]?[KkMmSs]*\/, so `IQ4_XS` read "Q4_", `BF16` read
 * "F16" and `UD-Q4_K_XL` read "Q4_K"; local import and the recovery scan used
 * /[_-](Q\d+[_\w]*|f16|f32)/, which missed IQ/BF16 and swallowed trailing words. A label is shown
 * to the user, gates the NPU/GPU badge (exact Q4_0 / Q8_0) and picks the onboarding file (exact
 * Q4_K_M), so a wrong label is a wrong decision, not a cosmetic slip.
 *
 * The rule: a quant token is a whole token (bounded by start/end or a non-alphanumeric char), from
 * the closed llama.cpp vocabulary below, optionally prefixed `UD-` (unsloth dynamic). When a name
 * holds several, the LONGEST wins (ties: the last one, where repos put the quant).
 */
import { QUANTIZATION_INFO } from '../constants/models';

export type QuantizationInfo = (typeof QUANTIZATION_INFO)[string];

// Longest alternatives first so `Q4_0_4_4` is not read as `Q4_0`, `IQ2_XXS` not as `IQ2_XS`.
const QUANT_TOKEN = [
  'MXFP4(?:_?MOE)?',
  'BF16', 'FP16', 'F16', 'FP32', 'F32',
  'TQ[12]_?0',
  'IQ[1-4]_?(?:XXS|XS|NL|S|M)',
  'Q4_?0_?[48]_?[48]',
  'Q[2-8]_?K(?:_?(?:XXL|XL|L|M|S))?',
  'Q[4-8]_?[01]',
].join('|');

const QUANT_TOKEN_RE = new RegExp(
  `(?:^|[^A-Za-z0-9])((?:UD[-_])?(?:${QUANT_TOKEN}))(?=$|[^A-Za-z0-9])`,
  'gi',
);

/** Canonical upper-case spelling with llama.cpp's underscores (`q4km` -> `Q4_K_M`). */
function canonicalize(raw: string): string {
  const upper = raw.toUpperCase();
  const ud = /^UD[-_]/.test(upper);
  const body = ud ? upper.slice(3) : upper;
  let out = body;
  let m: RegExpMatchArray | null;
  if ((m = /^MXFP4_?(MOE)?$/.exec(body))) out = m[1] ? 'MXFP4_MOE' : 'MXFP4';
  else if (body === 'FP16') out = 'F16';
  else if (body === 'FP32') out = 'F32';
  else if ((m = /^TQ([12])_?0$/.exec(body))) out = `TQ${m[1]}_0`;
  else if ((m = /^IQ([1-4])_?([A-Z]+)$/.exec(body))) out = `IQ${m[1]}_${m[2]}`;
  else if ((m = /^Q4_?0_?([48])_?([48])$/.exec(body))) out = `Q4_0_${m[1]}_${m[2]}`;
  else if ((m = /^Q(\d)_?K(?:_?([A-Z]+))?$/.exec(body))) out = `Q${m[1]}_K${m[2] ? `_${m[2]}` : ''}`;
  else if ((m = /^Q(\d)_?([01])$/.exec(body))) out = `Q${m[1]}_${m[2]}`;
  return ud ? `UD-${out}` : out;
}

/**
 * Where the quant token sits in `name` and its canonical label, or null when the name carries no
 * quant token. `index`/`length` address the raw token in `name` (for stripping it from a title).
 */
export function findQuantization(
  name: string,
): { label: string; index: number; length: number } | null {
  let best: { label: string; index: number; length: number } | null = null;
  QUANT_TOKEN_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = QUANT_TOKEN_RE.exec(name)) !== null) {
    const token = m[1];
    const index = m.index + m[0].length - token.length;
    if (!best || token.length >= best.length) {
      best = { label: canonicalize(token), index, length: token.length };
    }
    // The separator before the token is consumed; step back so an adjacent token is still seen.
    QUANT_TOKEN_RE.lastIndex = index + token.length;
  }
  return best;
}

/** The quant label of a file name, or 'Unknown'. */
export function parseQuantization(fileName: string): string {
  return findQuantization(fileName)?.label ?? 'Unknown';
}

/** A model title from a file name: extension and everything from the quant token on removed. */
export function stripQuantization(fileName: string): string {
  const base = fileName.replace(/\.(gguf|litertlm)$/i, '');
  const found = findQuantization(base);
  if (!found) return base;
  const head = base.slice(0, found.index).replace(/[-_.\s]+$/, '');
  return head || base;
}

/** The table key a label describes: `UD-` packaging and the `_MOE` role suffix carry no bits. */
function infoKey(label: string): string {
  return label.toUpperCase().replace(/^UD[-_]/, '').replace(/^MXFP4_MOE$/, 'MXFP4');
}

/**
 * Bits/quality for a label (or a raw file name). Unknown labels return undefined - callers decide
 * their own default rather than inheriting a guess.
 */
export function lookupQuantizationInfo(labelOrName: string | undefined | null): QuantizationInfo | undefined {
  if (!labelOrName) return undefined;
  const direct = QUANTIZATION_INFO[infoKey(labelOrName)];
  if (direct) return direct;
  const parsed = findQuantization(labelOrName);
  return parsed ? QUANTIZATION_INFO[infoKey(parsed.label)] : undefined;
}

/** Bits per weight for memory estimates; `fallback` when the label is unknown. */
export function quantizationBitsPerWeight(labelOrName: string | undefined | null, fallback = 4.5): number {
  return lookupQuantizationInfo(labelOrName)?.bitsPerWeight ?? fallback;
}
