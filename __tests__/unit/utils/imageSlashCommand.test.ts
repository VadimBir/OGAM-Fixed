import { parseImageSlashCommand } from '../../../src/utils/imageSlashCommand';

describe('parseImageSlashCommand', () => {
  it('returns null for text that is not the command', () => {
    expect(parseImageSlashCommand('draw a cat')).toBeNull();
    expect(parseImageSlashCommand('/imagine a cat')).toBeNull();
    expect(parseImageSlashCommand('/imgs a cat')).toBeNull();
    expect(parseImageSlashCommand('please /img a cat')).toBeNull();
    expect(parseImageSlashCommand('')).toBeNull();
  });

  it('accepts /img and /image, any case, with surrounding space', () => {
    expect(parseImageSlashCommand('/img a red fox')).toEqual({ prompt: 'a red fox', args: {} });
    expect(parseImageSlashCommand('  /IMAGE a red fox  ')).toEqual({ prompt: 'a red fox', args: {} });
  });

  it('returns an empty prompt for the bare command', () => {
    expect(parseImageSlashCommand('/img')).toEqual({ prompt: '', args: {} });
    expect(parseImageSlashCommand('/image --steps 20')).toEqual({ prompt: '', args: { steps: 20 } });
  });

  it('parses every flag and strips them from the prompt', () => {
    expect(
      parseImageSlashCommand('/img --w 256 a red fox --h 320 --steps 12 in snow --cfg 6.5 --seed 42 --neg "blurry, text"'),
    ).toEqual({
      prompt: 'a red fox in snow',
      args: { width: 256, height: 320, steps: 12, guidance: 6.5, seed: 42, negative_prompt: 'blurry, text' },
    });
  });

  it('accepts flags in any case and a single-quoted or bare negative prompt', () => {
    expect(parseImageSlashCommand("/img fox --NEG 'low quality' --W 128")).toEqual({
      prompt: 'fox',
      args: { negative_prompt: 'low quality', width: 128 },
    });
    expect(parseImageSlashCommand('/img fox --neg blurry')).toEqual({
      prompt: 'fox',
      args: { negative_prompt: 'blurry' },
    });
  });

  it('drops a numeric flag whose value is not a number and keeps that word in the prompt', () => {
    expect(parseImageSlashCommand('/img --steps many cats')).toEqual({ prompt: 'many cats', args: {} });
    expect(parseImageSlashCommand('/img cats --seed')).toEqual({ prompt: 'cats', args: {} });
  });

  it('keeps a negative seed (the bounding step turns it into random)', () => {
    expect(parseImageSlashCommand('/img cats --seed -1')).toEqual({ prompt: 'cats', args: { seed: -1 } });
  });

  it('does not take the next flag as the negative prompt', () => {
    expect(parseImageSlashCommand('/img cats --neg --w 256')).toEqual({ prompt: 'cats', args: { width: 256 } });
  });

  it('keeps unknown --words, quotes and apostrophes in the prompt', () => {
    expect(parseImageSlashCommand('/img a sign saying "hello world" --style ink, cat\'s toy')).toEqual({
      prompt: 'a sign saying "hello world" --style ink, cat\'s toy',
      args: {},
    });
  });

  it('does not treat prototype names as flags', () => {
    expect(parseImageSlashCommand('/img constructor toString 5')).toEqual({
      prompt: 'constructor toString 5',
      args: {},
    });
  });
});
