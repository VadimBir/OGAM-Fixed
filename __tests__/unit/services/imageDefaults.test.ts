/**
 * Image defaults: the user's default negative prompt / fixed seed fill a request that names none,
 * and CLIP skip reaches the sd.cpp engine options (SdCppNative clipSkip slot).
 */
import { applyImageDefaults } from '../../../src/services/imageParameterPolicy';
import { clipSkipOf, resolveEngineImageRequest } from '../../../src/services/imageEngineParams';

describe('applyImageDefaults', () => {
  it('fills an empty request from settings', () => {
    expect(applyImageDefaults({ prompt: 'cat' }, { imageNegativePrompt: 'blurry', imageSeed: 42 }))
      .toEqual({ prompt: 'cat', negativePrompt: 'blurry', seed: 42 });
  });

  it('a request value wins over the defaults', () => {
    expect(applyImageDefaults({ prompt: 'cat', negativePrompt: 'dogs', seed: 7 }, { imageNegativePrompt: 'blurry', imageSeed: 42 }))
      .toEqual({ prompt: 'cat', negativePrompt: 'dogs', seed: 7 });
  });

  it('seed -1 leaves the seed out (each engine picks a random one)', () => {
    expect(applyImageDefaults({ prompt: 'cat' }, { imageNegativePrompt: '', imageSeed: -1 })).toEqual({ prompt: 'cat' });
  });

  it('a blank request negative prompt takes the default', () => {
    expect(applyImageDefaults({ prompt: 'cat', negativePrompt: '  ' }, { imageNegativePrompt: 'lowres' }).negativePrompt).toBe('lowres');
  });
});

describe('CLIP skip', () => {
  it('clipSkipOf: positive integers only', () => {
    expect(clipSkipOf({ imageClipSkip: 2 })).toBe(2);
    expect(clipSkipOf({ imageClipSkip: 0 })).toBeUndefined();
    expect(clipSkipOf({})).toBeUndefined();
  });

  it('sd.cpp engine options carry clipSkip when set, omit it at 0', () => {
    const on = resolveEngineImageRequest({ backend: 'sdcpp' }, { imageClipSkip: 2 }, {}, 512);
    expect(on.engineOptions.clipSkip).toBe(2);
    const off = resolveEngineImageRequest({ backend: 'sdcpp' }, { imageClipSkip: 0 }, {}, 512);
    expect(off.engineOptions).not.toHaveProperty('clipSkip');
  });

  it('LocalDream has no clip skip input: never sent there', () => {
    const r = resolveEngineImageRequest({ backend: 'qnn' }, { imageClipSkip: 2 }, {}, 512);
    expect(r.engineOptions).not.toHaveProperty('clipSkip');
  });
});
