/**
 * The sampling values the app decides for every text engine (local llama.rn, the user's own
 * llama-server / Ollama, third-party OpenAI-compatible hosts). ONE owner for the defaults and the
 * sanitising, so the local and remote builders cannot disagree about what "unset" means.
 *
 * Defaults = llama.cpp's own (node_modules/llama.rn/vendor/llama.cpp/common/common.h:225-252),
 * except top_k 40 which the app already sent (llmHelpers buildCompletionParams). Seed -1 =
 * LLAMA_DEFAULT_SEED, a random seed (llama.rn cpp/jsi/JSIParams.cpp:428 reads "seed" with -1 as
 * its fallback; vendor include/llama.h:37 + :1369 "seed == LLAMA_DEFAULT_SEED to use a random seed").
 */
import type { GenerationOptions } from './providers/types';

export interface SamplingSettings {
  topK?: number;
  minP?: number;
  typicalP?: number;
  penaltyLastN?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  dryMultiplier?: number;
  dryBase?: number;
  dryAllowedLength?: number;
  dryPenaltyLastN?: number;
  xtcProbability?: number;
  xtcThreshold?: number;
  /** 0 = off, 1 = Mirostat, 2 = Mirostat 2.0 */
  mirostat?: number;
  mirostatTau?: number;
  mirostatEta?: number;
  /** -1 = random */
  seed?: number;
  /** Extra stop strings, added to the model's own stops. */
  customStopStrings?: string[];
}

export type ResolvedSampling = Required<Omit<SamplingSettings, 'customStopStrings'>>;

export const SAMPLING_DEFAULTS: ResolvedSampling = {
  topK: 40,
  minP: 0.05,
  typicalP: 1,
  penaltyLastN: 64,
  presencePenalty: 0,
  frequencyPenalty: 0,
  dryMultiplier: 0,
  dryBase: 1.75,
  dryAllowedLength: 2,
  dryPenaltyLastN: 64,
  xtcProbability: 0,
  xtcThreshold: 0.1,
  mirostat: 0,
  mirostatTau: 5,
  mirostatEta: 0.1,
  seed: -1,
};

const INTEGER_KEYS = new Set<keyof ResolvedSampling>([
  'topK', 'penaltyLastN', 'dryAllowedLength', 'dryPenaltyLastN', 'mirostat', 'seed',
]);

/** Every sampling value, with a non-finite / missing one replaced by its default. */
export function resolveSampling(settings: SamplingSettings | undefined): ResolvedSampling {
  const out = { ...SAMPLING_DEFAULTS };
  for (const key of Object.keys(SAMPLING_DEFAULTS) as (keyof ResolvedSampling)[]) {
    const value = settings?.[key];
    if (typeof value !== 'number' || !Number.isFinite(value)) continue;
    out[key] = INTEGER_KEYS.has(key) ? Math.round(value) : value;
  }
  if (out.mirostat !== 1 && out.mirostat !== 2) out.mirostat = 0;
  if (out.seed < 0) out.seed = -1;
  // A negative window is NOT "whole context" in llama.rn: its sampler clamps it to 0 = off
  // (vendor src/llama-sampler.cpp:3209, :3640). Keep the stored value honest about that.
  if (out.penaltyLastN < 0) out.penaltyLastN = 0;
  if (out.dryPenaltyLastN < 0) out.dryPenaltyLastN = 0;
  if (out.topK < 0) out.topK = 0;
  return out;
}

/** The user's extra stop strings: non-empty, de-duplicated, in the order entered. */
export function customStopStrings(settings: SamplingSettings | undefined): string[] {
  const list = Array.isArray(settings?.customStopStrings) ? settings!.customStopStrings : [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const s of list) {
    if (typeof s !== 'string' || s.length === 0 || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out;
}

/**
 * Stop-string editor text <-> list. One string per line; the two-character escapes \n and \t
 * stand for a newline / tab so a stop on a line break can be typed.
 */
export function parseStopStringsText(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map(line => line.replace(/\\n/g, '\n').replace(/\\t/g, '\t'))
    .filter(line => line.length > 0);
}

export function formatStopStringsText(list: string[] | undefined): string {
  return (list ?? []).map(s => s.replace(/\n/g, '\\n').replace(/\t/g, '\\t')).join('\n');
}

/**
 * The app's sampling values as provider GenerationOptions. Every remote text path builds its
 * options here (generationRemoteHelpers, generationToolLoop remoteGenerateOnce), so a new value
 * cannot reach one path and miss the other. Each provider then sends what its server accepts.
 */
export function remoteSamplingOptions(
  settings: SamplingSettings & { temperature?: number; topP?: number; repeatPenalty?: number; maxTokens?: number },
): Pick<GenerationOptions,
  'temperature' | 'maxTokens' | 'topP' | 'repeatPenalty' | 'topK' | 'seed' | 'stopSequences' |
  'minP' | 'typicalP' | 'penaltyLastN' | 'presencePenalty' | 'frequencyPenalty' |
  'dryMultiplier' | 'dryBase' | 'dryAllowedLength' | 'dryPenaltyLastN' |
  'xtcProbability' | 'xtcThreshold' | 'mirostat' | 'mirostatTau' | 'mirostatEta'> {
  const s = resolveSampling(settings);
  return {
    temperature: settings.temperature,
    maxTokens: settings.maxTokens,
    topP: settings.topP,
    repeatPenalty: settings.repeatPenalty,
    topK: s.topK,
    seed: s.seed,
    stopSequences: customStopStrings(settings),
    minP: s.minP,
    typicalP: s.typicalP,
    penaltyLastN: s.penaltyLastN,
    presencePenalty: s.presencePenalty,
    frequencyPenalty: s.frequencyPenalty,
    dryMultiplier: s.dryMultiplier,
    dryBase: s.dryBase,
    dryAllowedLength: s.dryAllowedLength,
    dryPenaltyLastN: s.dryPenaltyLastN,
    xtcProbability: s.xtcProbability,
    xtcThreshold: s.xtcThreshold,
    mirostat: s.mirostat,
    mirostatTau: s.mirostatTau,
    mirostatEta: s.mirostatEta,
  };
}
