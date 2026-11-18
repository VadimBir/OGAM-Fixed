/**
 * Android live-memory residency: every Android load is gated on LIVE free RAM (minus the live
 * margin), the OS trim callbacks release idle residents in priority order, and an idle image
 * model is released after a generation when live RAM is below the margin. iOS is unchanged.
 */
import { Platform } from 'react-native';
import { modelResidencyManager } from '../../../src/services/modelResidency';
import {
  classifyTrim,
  trimReleaseTiers,
  TRIM_LEVEL,
} from '../../../src/services/modelResidency/memoryPressure';
import { hardwareService } from '../../../src/services/hardware';

const originalOS = Platform.OS;

function device(totalGB: number, availGB: number) {
  jest.spyOn(hardwareService, 'getTotalMemoryGB').mockReturnValue(totalGB);
  jest.spyOn(hardwareService, 'getAvailableMemoryGB').mockReturnValue(availGB);
  jest.spyOn(hardwareService, 'refreshMemoryInfo').mockResolvedValue(undefined as never);
}

beforeEach(() => {
  modelResidencyManager._reset();
  modelResidencyManager.setBudgetOverrideMB(null);
  modelResidencyManager.setLoadPolicy('balanced');
});
afterEach(() => {
  Platform.OS = originalOS;
  jest.restoreAllMocks();
});

describe('live residency gate — 12GB phone with 3GB free', () => {
  it('Android: refuses a clean 5000MB text model and reports needed 5000MB vs room 2048MB (3072 - 1024)', async () => {
    Platform.OS = 'android';
    device(12, 3);
    const r = await modelResidencyManager.makeRoomFor({ key: 'text', type: 'text', sizeMB: 5000 });
    expect(r.fits).toBe(false);
    expect(r.neededMB).toBe(5000);
    expect(r.availableMB).toBe(2048);
  });

  it('Android: admits a clean 1500MB text model (1500 <= 2048)', async () => {
    Platform.OS = 'android';
    device(12, 3);
    const r = await modelResidencyManager.makeRoomFor({ key: 'text', type: 'text', sizeMB: 1500 });
    expect(r.fits).toBe(true);
  });

  it('Android aggressive: room is 3072 - 512 = 2560MB', async () => {
    Platform.OS = 'android';
    device(12, 3);
    modelResidencyManager.setLoadPolicy('aggressive');
    const r = await modelResidencyManager.makeRoomFor({ key: 'text', type: 'text', sizeMB: 3000 });
    expect(r.fits).toBe(false);
    expect(r.availableMB).toBe(2560);
  });

  it('Android: credits RAM freed by evicting our own resident (image 2500MB at 1GB free → text 2000MB fits)', async () => {
    Platform.OS = 'android';
    device(12, 1);
    const unloadImage = jest.fn().mockResolvedValue(undefined);
    modelResidencyManager.register({ key: 'image', type: 'image', sizeMB: 2500, dirtyMemory: true }, unloadImage, 1);
    const r = await modelResidencyManager.makeRoomFor({ key: 'text', type: 'text', sizeMB: 2000 });
    expect(r.fits).toBe(true);
    expect(r.evicted).toEqual(['image']);
    expect(unloadImage).toHaveBeenCalledTimes(1);
  });

  it('Android: Load Anyway still forces the refused load', async () => {
    Platform.OS = 'android';
    device(12, 3);
    const r = await modelResidencyManager.makeRoomFor(
      { key: 'text', type: 'text', modelId: 'big', sizeMB: 5000 },
      { override: true },
    );
    expect(r.fits).toBe(true);
  });

  it('iOS unchanged: the same clean 5000MB model fits the physical budget', async () => {
    Platform.OS = 'ios';
    device(12, 3);
    const r = await modelResidencyManager.makeRoomFor({ key: 'text', type: 'text', sizeMB: 5000 });
    expect(r.fits).toBe(true);
  });

  it('Android: the preloader budget is live too (canLoadWithoutEviction)', () => {
    Platform.OS = 'android';
    device(12, 3);
    expect(modelResidencyManager.canLoadWithoutEviction({ key: 'text', sizeMB: 2048 })).toBe(true);
    expect(modelResidencyManager.canLoadWithoutEviction({ key: 'text', sizeMB: 2049 })).toBe(false);
  });
});

describe('classifyTrim', () => {
  it('maps ComponentCallbacks2 levels to severities', () => {
    expect(classifyTrim({ level: TRIM_LEVEL.RUNNING_MODERATE })).toBe('moderate');
    expect(classifyTrim({ level: TRIM_LEVEL.RUNNING_LOW })).toBe('moderate');
    expect(classifyTrim({ level: TRIM_LEVEL.RUNNING_CRITICAL })).toBe('critical');
    expect(classifyTrim({ level: TRIM_LEVEL.UI_HIDDEN })).toBe('none');
    expect(classifyTrim({ level: TRIM_LEVEL.BACKGROUND })).toBe('moderate');
    expect(classifyTrim({ level: TRIM_LEVEL.MODERATE })).toBe('moderate');
    expect(classifyTrim({ level: TRIM_LEVEL.COMPLETE })).toBe('critical');
  });

  it('onLowMemory and the system lowMemory flag are critical', () => {
    expect(classifyTrim({ level: -1, source: 'lowMemory' })).toBe('critical');
    expect(classifyTrim({ level: TRIM_LEVEL.UI_HIDDEN, lowMemory: true })).toBe('critical');
  });

  it('release tiers: sidecars, then image, then text (critical only)', () => {
    expect(trimReleaseTiers('none')).toEqual([]);
    expect(trimReleaseTiers('moderate')).toEqual([['whisper', 'tts', 'embedding'], ['image']]);
    expect(trimReleaseTiers('critical')).toEqual([['whisper', 'tts', 'embedding'], ['image'], ['text']]);
  });
});

describe('handleTrimMemory — release order', () => {
  function seed(order: string[], opts: { textBusy?: boolean; imageBusy?: boolean } = {}) {
    const unload = (key: string) => jest.fn(async () => { order.push(key); });
    // Registered text-first so the release order is NOT the registration order.
    modelResidencyManager.register(
      { key: 'text', type: 'text', sizeMB: 4000, isBusy: () => !!opts.textBusy },
      unload('text'), 1,
    );
    modelResidencyManager.register(
      { key: 'image', type: 'image', sizeMB: 2500, dirtyMemory: true, isBusy: async () => !!opts.imageBusy },
      unload('image'), 2,
    );
    modelResidencyManager.register({ key: 'embedding', type: 'embedding', sizeMB: 100 }, unload('embedding'), 3);
    modelResidencyManager.register({ key: 'classifier', type: 'classifier', sizeMB: 100, pinned: true }, unload('classifier'), 4);
  }

  it('critical, nothing in flight: sidecar → image → text; pinned kept', async () => {
    const order: string[] = [];
    seed(order);
    const released = await modelResidencyManager.handleTrimMemory('critical');
    expect(order).toEqual(['embedding', 'image', 'text']);
    expect(released).toEqual(['embedding', 'image', 'text']);
    expect(modelResidencyManager.isResident('classifier')).toBe(true);
  });

  it('moderate: sidecar → image; the text model stays', async () => {
    const order: string[] = [];
    seed(order);
    await modelResidencyManager.handleTrimMemory('moderate');
    expect(order).toEqual(['embedding', 'image']);
    expect(modelResidencyManager.isResident('text')).toBe(true);
  });

  it('critical with a text generation in flight: the text model stays', async () => {
    const order: string[] = [];
    seed(order, { textBusy: true });
    await modelResidencyManager.handleTrimMemory('critical');
    expect(order).toEqual(['embedding', 'image']);
    expect(modelResidencyManager.isResident('text')).toBe(true);
  });

  it('an image generation in flight keeps the image model', async () => {
    const order: string[] = [];
    seed(order, { imageBusy: true });
    await modelResidencyManager.handleTrimMemory('critical');
    expect(order).toEqual(['embedding', 'text']);
    expect(modelResidencyManager.isResident('image')).toBe(true);
  });

  it('none (UI hidden only): releases nothing', async () => {
    const order: string[] = [];
    seed(order);
    expect(await modelResidencyManager.handleTrimMemory('none')).toEqual([]);
    expect(order).toEqual([]);
  });

  it('a rejected native unload keeps the resident counted and continues', async () => {
    modelResidencyManager.register({ key: 'tts', type: 'tts', sizeMB: 300 }, jest.fn().mockRejectedValue(new Error('busy')), 1);
    modelResidencyManager.register({ key: 'image', type: 'image', sizeMB: 2500 }, jest.fn().mockResolvedValue(undefined), 2);
    const released = await modelResidencyManager.handleTrimMemory('moderate');
    expect(released).toEqual(['image']);
    expect(modelResidencyManager.isResident('tts')).toBe(true);
  });
});

describe('releaseIdleImageIfMemoryLow (after an image generation)', () => {
  function seedImage(busy = false) {
    const unload = jest.fn().mockResolvedValue(undefined);
    modelResidencyManager.register(
      { key: 'image', type: 'image', sizeMB: 2500, dirtyMemory: true, isBusy: () => busy },
      unload, 1,
    );
    return unload;
  }

  it('Android, 512MB free < 1024MB margin: releases the idle image model', async () => {
    Platform.OS = 'android';
    device(12, 0.5);
    const unload = seedImage();
    expect(await modelResidencyManager.releaseIdleImageIfMemoryLow()).toBe(true);
    expect(unload).toHaveBeenCalledTimes(1);
    expect(modelResidencyManager.isResident('image')).toBe(false);
  });

  it('Android, 3GB free: keeps it warm', async () => {
    Platform.OS = 'android';
    device(12, 3);
    const unload = seedImage();
    expect(await modelResidencyManager.releaseIdleImageIfMemoryLow()).toBe(false);
    expect(unload).not.toHaveBeenCalled();
  });

  it('Android, low RAM but still generating: keeps it', async () => {
    Platform.OS = 'android';
    device(12, 0.5);
    const unload = seedImage(true);
    expect(await modelResidencyManager.releaseIdleImageIfMemoryLow()).toBe(false);
    expect(unload).not.toHaveBeenCalled();
  });

  it('iOS: never (Android-only policy)', async () => {
    Platform.OS = 'ios';
    device(12, 0.5);
    const unload = seedImage();
    expect(await modelResidencyManager.releaseIdleImageIfMemoryLow()).toBe(false);
    expect(unload).not.toHaveBeenCalled();
  });
});
