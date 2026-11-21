/**
 * Fast mode: a final image in 1, 2 or 4 steps.
 *
 * GPU (stable-diffusion.cpp): a few-step distillation LoRA per checkpoint family, downloaded in
 * the app on one tap, plus the sampler / schedule / CFG it was distilled for. NPU (LocalDream):
 * no runtime LoRA, so only checkpoints with the distillation baked in (name says DMD2 / Hyper /
 * LCM / Lightning / Turbo) get a preset.
 *
 * Pure data + pure functions (no store, no IO besides the RNFS document path).
 *
 * Sources (sizes and sha256 = the HF tree API, https://huggingface.co/api/models/<repo>/tree/main):
 *  - Hyper-SD (https://huggingface.co/ByteDance/Hyper-SD README): SD1.5/SDXL 2-step LoRA =
 *    DDIMScheduler timestep_spacing="trailing", guidance_scale=0; the "1step" file is the unified
 *    LoRA ("support 1 to 8 steps inference") run with TCDScheduler, eta=1.0, guidance_scale=0.
 *  - SDXL-Lightning (https://huggingface.co/ByteDance/SDXL-Lightning README): LoRA =
 *    EulerDiscreteScheduler timestep_spacing="trailing", guidance_scale=0; "Use LoRA only if you
 *    are using non-SDXL base models"; no 1-step LoRA ("The 1-step model is only experimental").
 *  - TAESD (https://huggingface.co/madebyollin/taesd, /taesdxl): "the SD and SDXL VAEs are
 *    incompatible"; sd.cpp docs/taesd.md loads diffusion_pytorch_model.safetensors with --taesd.
 *
 * Mapping to sd.cpp @ c92d73c (src/stable-diffusion.cpp sample_method_to_str / scheduler_to_str):
 *  - diffusers guidance_scale=0 (CFG off) = sd.cpp cfg 1: resolve_guidance (src/pipeline/request.cpp)
 *    runs no unconditional pass when txt_cfg == img_cfg == 1, and warns that cfg 0 is
 *    "unconditioned mode, images won't follow the prompt (use cfg-scale=1 for distilled models)".
 *  - "trailing" spacing = scheduler `simple`: SimpleScheduler (src/runtime/denoiser.hpp) picks
 *    timesteps 999 - i*1000/n, i.e. 999,499 (n=2) and 999,749,499,249 (n=4) = diffusers trailing.
 *  - DDIM = sampler `ddim_trailing` (euler_ancestral with eta 0, denoiser.hpp "DDIM is equivalent
 *    to Euler Ancestral with the Simple scheduler"; resolve_eta gives 0 for it).
 *  - TCD = sampler `tcd` + scheduler `lcm` (its default, request.cpp default_scheduler; n=1 ->
 *    timestep 999). sd.cpp's TCD eta defaults to 0 (resolve_eta) and the JNI does not expose eta;
 *    at 1 step eta changes nothing (sample_tcd: sigma_to = 0 -> timestep_s = 0, no re-noise).
 */
import RNFS from 'react-native-fs';

export type FastSteps = 1 | 2 | 4;
/** Checkpoint families a few-step LoRA exists for (sdCppModelDetection SdFamily values). */
export type FastLoraFamily = 'sd1' | 'sdxl';
/** TAESD weights: SD 1.x/2.x share one latent space, SDXL has its own. */
export type TaesdFamily = 'sd1' | 'sdxl';

/** Order shown in the UI. */
export const FAST_STEP_CHOICES: readonly FastSteps[] = [4, 2, 1];

export interface FastLoraPreset {
  family: FastLoraFamily;
  steps: FastSteps;
  repo: string;
  /** File name in the repo; also the name kept on the phone (the LoRA's display name). */
  fileName: string;
  url: string;
  sizeBytes: number;
  sha256: string;
  /** sd.cpp sampler / scheduler names (SDCPP_SAMPLERS / SDCPP_SCHEDULERS). */
  sampler: string;
  scheduler: string;
  cfg: number;
  loraWeight: number;
  license: string;
}

const hf = (repo: string, file: string) => `https://huggingface.co/${repo}/resolve/main/${file}`;

function hyper(family: FastLoraFamily, steps: FastSteps, fileName: string, sizeBytes: number, sha256: string): FastLoraPreset {
  const tcd = steps === 1;
  return {
    family, steps, repo: 'ByteDance/Hyper-SD', fileName, url: hf('ByteDance/Hyper-SD', fileName), sizeBytes, sha256,
    sampler: tcd ? 'tcd' : 'ddim_trailing',
    scheduler: tcd ? 'lcm' : 'simple',
    cfg: 1,
    loraWeight: 1,
    license: 'Hyper-SD LICENSE.md (Stability AI Community License section for SD-based files)',
  };
}

function lightning(steps: 2 | 4, sha256: string): FastLoraPreset {
  const fileName = `sdxl_lightning_${steps}step_lora.safetensors`;
  return {
    family: 'sdxl', steps, repo: 'ByteDance/SDXL-Lightning', fileName, url: hf('ByteDance/SDXL-Lightning', fileName),
    sizeBytes: 393_854_592, sha256, sampler: 'euler', scheduler: 'simple', cfg: 1, loraWeight: 1, license: 'openrail++',
  };
}

export const FAST_LORA_PRESETS: Record<FastLoraFamily, Record<FastSteps, FastLoraPreset>> = {
  sd1: {
    1: hyper('sd1', 1, 'Hyper-SD15-1step-lora.safetensors', 269_127_096, 'a04fd9a535c1e56d38f7590ee72a13fd5ca0409853b4fff021e5a9482cf1ca3b'),
    2: hyper('sd1', 2, 'Hyper-SD15-2steps-lora.safetensors', 269_127_064, '2f26dcc1d883feb07557a552315baae2ca2a04ac08556b08a355a244547e8c3a'),
    4: hyper('sd1', 4, 'Hyper-SD15-4steps-lora.safetensors', 269_127_064, 'c5dd058616461ed5053e2b14eec4dbe3fa0eea3b13688642f6d6c80ea2ba5958'),
  },
  sdxl: {
    1: hyper('sdxl', 1, 'Hyper-SDXL-1step-lora.safetensors', 787_359_648, 'c912df184c5116792d2c604d26c6bc2aa916685f4a793755255cda1c43a3c78a'),
    2: lightning(2, '04fafc778385b24144498a8247498ae4fb7a69f702ea0566bdce2845a31fcc43'),
    4: lightning(4, 'bf56cf2657efb15e465d81402ed481d1e11c4677e4bcce1bc11fe71ad8506b79'),
  },
};

export interface TaesdFile {
  family: TaesdFamily;
  repo: string;
  url: string;
  sizeBytes: number;
  sha256: string;
  /** Name kept on the phone (both repos call the file diffusion_pytorch_model.safetensors). */
  localFileName: string;
}

export const TAESD_FILES: Record<TaesdFamily, TaesdFile> = {
  sd1: {
    family: 'sd1', repo: 'madebyollin/taesd', url: hf('madebyollin/taesd', 'diffusion_pytorch_model.safetensors'),
    sizeBytes: 9_793_292, sha256: 'db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549',
    localFileName: 'taesd_sd1.safetensors',
  },
  sdxl: {
    family: 'sdxl', repo: 'madebyollin/taesdxl', url: hf('madebyollin/taesdxl', 'diffusion_pytorch_model.safetensors'),
    sizeBytes: 9_793_292, sha256: 'ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f',
    localFileName: 'taesdxl.safetensors',
  },
};

/** TAESD files live outside image_models/ (like image_loras/) so the orphan scan never lists them. */
export const TAESD_DIR = `${RNFS.DocumentDirectoryPath}/image_vae`;

export const taesdPathFor = (family: TaesdFamily) => `${TAESD_DIR}/${TAESD_FILES[family].localFileName}`;

export function fastLoraFamilyOf(sdFamily: string | undefined): FastLoraFamily | null {
  return sdFamily === 'sd1' || sdFamily === 'sdxl' ? sdFamily : null;
}

export function taesdFamilyOf(sdFamily: string | undefined): TaesdFamily | null {
  if (sdFamily === 'sd1' || sdFamily === 'sd2') return 'sd1';
  return sdFamily === 'sdxl' ? 'sdxl' : null;
}

// ---- NPU (LocalDream) ---------------------------------------------------------------------------

export interface NpuFastPreset {
  /** Which few-step family the name matched. */
  kind: 'dmd2' | 'lcm' | 'lightning' | 'hyper' | 'turbo';
  steps: FastSteps[];
  cfg: number;
  /** LocalDream sampler name (LOCALDREAM_SAMPLERS). */
  sampler: string;
}

const NPU_KINDS: ReadonlyArray<{ kind: NpuFastPreset['kind']; re: RegExp; steps: FastSteps[]; sampler: string }> = [
  // DMD2 README: LCMScheduler, 4 steps, guidance 0.
  { kind: 'dmd2', re: /dmd/i, steps: [4], sampler: 'lcm' },
  // LCM-LoRA README: LCMScheduler, 2-8 steps.
  { kind: 'lcm', re: /(^|[^a-z])lcm([^a-z]|$)/i, steps: [4, 2], sampler: 'lcm' },
  // Lightning: Euler, trailing spacing (LocalDream's Euler spacing: UNKNOWN).
  { kind: 'lightning', re: /lightning/i, steps: [4, 2], sampler: 'euler' },
  { kind: 'hyper', re: /hyper/i, steps: [4, 2], sampler: 'euler' },
  { kind: 'turbo', re: /turbo/i, steps: [4, 2, 1], sampler: 'euler_a' },
];

/**
 * Few-step preset for a LocalDream (NPU) model from its name, or null when the name marks no
 * few-step distillation. An explicit "<n>step(s)" in the name narrows the choice to n.
 */
export function npuFastPreset(name: string | undefined): NpuFastPreset | null {
  const text = name ?? '';
  const match = NPU_KINDS.find(k => k.re.test(text));
  if (!match) return null;
  const explicit = /(\d+)[\s_-]?steps?/i.exec(text);
  const n = explicit ? Number(explicit[1]) : null;
  let steps = match.steps;
  if (n === 1 || n === 2 || n === 4) steps = [n];
  return { kind: match.kind, steps, cfg: 1, sampler: match.sampler };
}

// ---- settings transitions ----------------------------------------------------------------------

/** What Fast mode changed, so turning it off restores exactly that. */
export interface FastModeState {
  steps: FastSteps;
  engine: 'sdcpp' | 'npu';
  /** sdcpp: family of the preset (a checkpoint of another family shows Fast mode as off). */
  family?: FastLoraFamily;
  /** sdcpp: the preset LoRA it enabled. */
  loraPath?: string;
  previous: {
    imageSteps: number;
    imageGuidanceScale: number;
    imageSampler?: string;
    imageScheduler?: string;
  };
}

export interface FastModeSettingsSlice {
  imageSteps: number;
  imageGuidanceScale: number;
  imageSampler?: string;
  imageScheduler?: string;
  imageLoras?: Array<{ path: string; enabled: boolean; weight: number }>;
  imageFastMode?: FastModeState | null;
}

/** Previous values: kept from the first activation, so 4 -> 2 -> Off returns to the user's own. */
function previousOf(s: FastModeSettingsSlice): FastModeState['previous'] {
  return s.imageFastMode?.previous ?? {
    imageSteps: s.imageSteps,
    imageGuidanceScale: s.imageGuidanceScale,
    imageSampler: s.imageSampler,
    imageScheduler: s.imageScheduler,
  };
}

/** Settings update that applies a GPU preset: enables its LoRA (and disables a previous preset LoRA). */
export function sdCppFastModeUpdate<T extends FastModeSettingsSlice>(
  s: T,
  preset: FastLoraPreset,
  loraPath: string,
): Partial<T> {
  const oldPath = s.imageFastMode?.loraPath;
  const imageLoras = (s.imageLoras ?? []).map(l => {
    if (l.path === loraPath) return { ...l, enabled: true, weight: preset.loraWeight };
    if (oldPath && l.path === oldPath) return { ...l, enabled: false };
    return l;
  });
  return {
    imageSteps: preset.steps,
    imageGuidanceScale: preset.cfg,
    imageSampler: preset.sampler,
    imageScheduler: preset.scheduler,
    imageLoras,
    imageFastMode: {
      steps: preset.steps, engine: 'sdcpp', family: preset.family, loraPath, previous: previousOf(s),
    },
  } as Partial<T>;
}

/** Settings update that applies an NPU preset (no LoRA; the schedule is LocalDream's sampler). */
export function npuFastModeUpdate<T extends FastModeSettingsSlice>(
  s: T,
  preset: NpuFastPreset,
  steps: FastSteps,
): Partial<T> {
  const oldPath = s.imageFastMode?.loraPath;
  const update: Partial<FastModeSettingsSlice> = {
    imageSteps: steps,
    imageGuidanceScale: preset.cfg,
    imageSampler: preset.sampler,
    imageFastMode: { steps, engine: 'npu', previous: previousOf(s) },
  };
  if (oldPath) update.imageLoras = (s.imageLoras ?? []).map(l => (l.path === oldPath ? { ...l, enabled: false } : l));
  return update as Partial<T>;
}

/** Settings update that turns Fast mode off: previous steps/cfg/sampler/scheduler, preset LoRA disabled. */
export function fastModeOffUpdate<T extends FastModeSettingsSlice>(s: T): Partial<T> {
  const fm = s.imageFastMode;
  if (!fm) return {};
  const update: Partial<FastModeSettingsSlice> = {
    imageSteps: fm.previous.imageSteps,
    imageGuidanceScale: fm.previous.imageGuidanceScale,
    imageSampler: fm.previous.imageSampler,
    imageScheduler: fm.previous.imageScheduler,
    imageFastMode: null,
  };
  if (fm.loraPath) {
    update.imageLoras = (s.imageLoras ?? []).map(l => (l.path === fm.loraPath ? { ...l, enabled: false } : l));
  }
  return update as Partial<T>;
}

/** The Fast mode choice shown for the active model (null = Off, also when it was set for another family). */
export function activeFastSteps(
  fm: FastModeState | null | undefined,
  model: { backend?: string; sdFamily?: string } | undefined,
): FastSteps | null {
  if (!fm || !model) return null;
  if (model.backend === 'sdcpp') {
    return fm.engine === 'sdcpp' && fm.family === fastLoraFamilyOf(model.sdFamily) ? fm.steps : null;
  }
  return fm.engine === 'npu' ? fm.steps : null;
}
