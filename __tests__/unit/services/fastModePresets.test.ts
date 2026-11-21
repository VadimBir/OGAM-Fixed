/**
 * Fast mode presets: every GPU preset names a sampler / schedule sd.cpp accepts and runs CFG-free
 * (cfg 1), apply keeps the user's own values for Off (also across 4 -> 2), Off restores them and
 * disables the preset LoRA, and the NPU name detection.
 */
jest.mock('react-native', () => ({
  Platform: { OS: 'android', select: (o: any) => o.android ?? o.default },
  NativeEventEmitter: jest.fn(),
  NativeModules: {},
}));

import { SDCPP_SAMPLERS, SDCPP_SCHEDULERS } from '../../../src/services/sdCppGenerator';
import {
  FAST_LORA_PRESETS,
  FAST_STEP_CHOICES,
  TAESD_FILES,
  activeFastSteps,
  fastLoraFamilyOf,
  fastModeOffUpdate,
  npuFastModeUpdate,
  npuFastPreset,
  sdCppFastModeUpdate,
  taesdFamilyOf,
  taesdPathFor,
  type FastModeSettingsSlice,
} from '../../../src/services/fastModePresets';
import { LOCALDREAM_SAMPLERS } from '../../../src/services/imageEngineParams';

const base = (): FastModeSettingsSlice => ({
  imageSteps: 20,
  imageGuidanceScale: 7.5,
  imageSampler: 'euler_a',
  imageScheduler: 'karras',
  imageLoras: [
    { path: '/l/style.safetensors', enabled: true, weight: 0.8 },
    { path: '/l/Hyper-SD15-4steps-lora.safetensors', enabled: false, weight: 1 },
    { path: '/l/Hyper-SD15-2steps-lora.safetensors', enabled: false, weight: 1 },
  ],
  imageFastMode: null,
});

describe('preset table', () => {
  it('every GPU preset uses a known sd.cpp sampler/scheduler, cfg 1, weight 1, HF resolve URL', () => {
    for (const family of ['sd1', 'sdxl'] as const) {
      for (const steps of FAST_STEP_CHOICES) {
        const p = FAST_LORA_PRESETS[family][steps];
        expect(p.steps).toBe(steps);
        expect(p.family).toBe(family);
        expect(SDCPP_SAMPLERS).toContain(p.sampler);
        expect(SDCPP_SCHEDULERS).toContain(p.scheduler);
        expect(p.cfg).toBe(1);
        expect(p.loraWeight).toBe(1);
        expect(p.url).toBe(`https://huggingface.co/${p.repo}/resolve/main/${p.fileName}`);
        expect(p.sha256).toMatch(/^[0-9a-f]{64}$/);
        expect(p.sizeBytes).toBeGreaterThan(100_000_000);
      }
    }
  });

  it('1 step = TCD with the lcm schedule; 2/4 steps = trailing (simple) schedule', () => {
    expect(FAST_LORA_PRESETS.sd1[1]).toMatchObject({ sampler: 'tcd', scheduler: 'lcm', fileName: 'Hyper-SD15-1step-lora.safetensors' });
    expect(FAST_LORA_PRESETS.sd1[2]).toMatchObject({ sampler: 'ddim_trailing', scheduler: 'simple' });
    expect(FAST_LORA_PRESETS.sdxl[1]).toMatchObject({ sampler: 'tcd', scheduler: 'lcm', fileName: 'Hyper-SDXL-1step-lora.safetensors' });
    expect(FAST_LORA_PRESETS.sdxl[4]).toMatchObject({ sampler: 'euler', scheduler: 'simple', fileName: 'sdxl_lightning_4step_lora.safetensors' });
  });

  it('families: LoRA for sd1/sdxl only; TAESD sd1+sd2 -> taesd, sdxl -> taesdxl', () => {
    expect(fastLoraFamilyOf('sd1')).toBe('sd1');
    expect(fastLoraFamilyOf('sdxl')).toBe('sdxl');
    expect(fastLoraFamilyOf('sd2')).toBeNull();
    expect(fastLoraFamilyOf(undefined)).toBeNull();
    expect(taesdFamilyOf('sd2')).toBe('sd1');
    expect(taesdFamilyOf('sdxl')).toBe('sdxl');
    expect(taesdFamilyOf('flux')).toBeNull();
    expect(taesdPathFor('sdxl')).toBe('/mock/documents/image_vae/taesdxl.safetensors');
    expect(TAESD_FILES.sd1.url).toBe('https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors');
  });
});

describe('apply / restore', () => {
  it('apply sets steps/cfg/sampler/scheduler, enables the preset LoRA, remembers the previous values', () => {
    const s = base();
    const p = FAST_LORA_PRESETS.sd1[4];
    const u = sdCppFastModeUpdate(s, p, '/l/Hyper-SD15-4steps-lora.safetensors');
    expect(u).toMatchObject({ imageSteps: 4, imageGuidanceScale: 1, imageSampler: 'ddim_trailing', imageScheduler: 'simple' });
    expect(u.imageLoras).toEqual([
      { path: '/l/style.safetensors', enabled: true, weight: 0.8 },
      { path: '/l/Hyper-SD15-4steps-lora.safetensors', enabled: true, weight: 1 },
      { path: '/l/Hyper-SD15-2steps-lora.safetensors', enabled: false, weight: 1 },
    ]);
    expect(u.imageFastMode).toEqual({
      steps: 4, engine: 'sdcpp', family: 'sd1', loraPath: '/l/Hyper-SD15-4steps-lora.safetensors',
      previous: { imageSteps: 20, imageGuidanceScale: 7.5, imageSampler: 'euler_a', imageScheduler: 'karras' },
    });
  });

  it('4 -> 2 swaps the LoRA and keeps the original previous values; Off restores them', () => {
    let s: FastModeSettingsSlice = base();
    s = { ...s, ...sdCppFastModeUpdate(s, FAST_LORA_PRESETS.sd1[4], '/l/Hyper-SD15-4steps-lora.safetensors') };
    s = { ...s, ...sdCppFastModeUpdate(s, FAST_LORA_PRESETS.sd1[2], '/l/Hyper-SD15-2steps-lora.safetensors') };
    expect(s.imageSteps).toBe(2);
    expect(s.imageLoras!.find(l => l.path.includes('4steps'))!.enabled).toBe(false);
    expect(s.imageLoras!.find(l => l.path.includes('2steps'))!.enabled).toBe(true);
    expect(s.imageFastMode!.previous.imageSteps).toBe(20);

    s = { ...s, ...fastModeOffUpdate(s) };
    expect(s).toMatchObject({ imageSteps: 20, imageGuidanceScale: 7.5, imageSampler: 'euler_a', imageScheduler: 'karras', imageFastMode: null });
    expect(s.imageLoras!.find(l => l.path.includes('2steps'))!.enabled).toBe(false);
    expect(s.imageLoras![0]).toEqual({ path: '/l/style.safetensors', enabled: true, weight: 0.8 });
  });

  it('Off when Fast mode is off changes nothing', () => {
    expect(fastModeOffUpdate(base())).toEqual({});
  });

  it('NPU preset: steps/cfg/sampler, no LoRA; Off restores', () => {
    let s: FastModeSettingsSlice = base();
    const preset = npuFastPreset('DMD2 SDXL 4step')!;
    s = { ...s, ...npuFastModeUpdate(s, preset, 4) };
    expect(s).toMatchObject({ imageSteps: 4, imageGuidanceScale: 1, imageSampler: 'lcm' });
    expect(s.imageFastMode).toMatchObject({ steps: 4, engine: 'npu' });
    expect(s.imageFastMode!.loraPath).toBeUndefined();
    s = { ...s, ...fastModeOffUpdate(s) };
    expect(s).toMatchObject({ imageSteps: 20, imageGuidanceScale: 7.5, imageSampler: 'euler_a', imageFastMode: null });
  });

  it('activeFastSteps: shown only for the engine/family it was set for', () => {
    const fm = sdCppFastModeUpdate(base(), FAST_LORA_PRESETS.sd1[2], '/l/x').imageFastMode!;
    expect(activeFastSteps(fm, { backend: 'sdcpp', sdFamily: 'sd1' })).toBe(2);
    expect(activeFastSteps(fm, { backend: 'sdcpp', sdFamily: 'sdxl' })).toBeNull();
    expect(activeFastSteps(fm, { backend: 'qnn' })).toBeNull();
    expect(activeFastSteps(null, { backend: 'sdcpp', sdFamily: 'sd1' })).toBeNull();
  });
});

describe('npuFastPreset (name detection)', () => {
  it.each([
    ['DMD2 SDXL', 'dmd2', [4], 'lcm'],
    ['HyperSpireV5', 'hyper', [4, 2], 'euler'],
    ['RealisticVisionHyper', 'hyper', [4, 2], 'euler'],
    ['LCM Dreamshaper v7', 'lcm', [4, 2], 'lcm'],
    ['Juggernaut Lightning', 'lightning', [4, 2], 'euler'],
    ['SD Turbo', 'turbo', [4, 2, 1], 'euler_a'],
    ['Hyper 1step', 'hyper', [1], 'euler'],
    ['lightning_2steps', 'lightning', [2], 'euler'],
  ])('%s -> %s %j %s', (name, kind, steps, sampler) => {
    expect(npuFastPreset(name)).toEqual({ kind, steps, cfg: 1, sampler });
    expect(LOCALDREAM_SAMPLERS).toContain(sampler);
  });

  it.each(['AnythingV5', 'Realistic Vision 5.1', 'ChilloutMix', '', undefined])('%s -> null', name => {
    expect(npuFastPreset(name)).toBeNull();
  });

  it('an explicit 8-step name keeps the family default choices', () => {
    expect(npuFastPreset('Hyper 8steps')!.steps).toEqual([4, 2]);
  });
});
