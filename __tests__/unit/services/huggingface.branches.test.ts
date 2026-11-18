declare const global: any;

/**
 * Branch-coverage tests for huggingface.ts.
 * Targets the extractQuantization underscore-stripped match branch (line 121),
 * and transformFileInfo's lfs.size fallbacks.
 */

import { huggingFaceService } from '../../../src/services/huggingface';

const service = huggingFaceService as any;

beforeEach(() => {
  jest.clearAllMocks();
});

// extractQuantization delegates to utils/quantization.parseQuantization (whole-token rule).
describe('extractQuantization (whole-token quant parser)', () => {
  it('reads an underscore-less token as its canonical label', () => {
    expect(service.extractQuantization('model-Q80.gguf')).toBe('Q8_0');
  });

  it('does not read a quant glued to the model name (substring match was the old bug)', () => {
    expect(service.extractQuantization('modelQ80.gguf')).toBe('Unknown');
  });

  it('returns Unknown for a Q pattern outside the llama.cpp vocabulary', () => {
    expect(service.extractQuantization('model-Q9_xyz.gguf')).toBe('Unknown');
  });

  it('uses only the file name of a nested path', () => {
    expect(service.extractQuantization('Q8_0/model-Q4_K_M.gguf')).toBe('Q4_K_M');
  });
});

describe('transformFileInfo size fallbacks', () => {
  it('uses lfs.size and carries sha256 through', () => {
    const out = service.transformFileInfo('org/model', {
      rfilename: 'm-Q4_K_M.gguf', size: 1, lfs: { size: 999, sha256: 'abc' },
    });
    expect(out.size).toBe(999);
    expect(out.sha256).toBe('abc');
  });

  it('falls back to plain size, then to 0 when both missing', () => {
    expect(service.transformFileInfo('org/model', { rfilename: 'm.gguf', size: 50 }).size).toBe(50);
    expect(service.transformFileInfo('org/model', { rfilename: 'm.gguf' }).size).toBe(0);
  });
});

describe('getModelDetails delegates to transformModelResult', () => {
  it('uses id split fallback for name when no slash present', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ id: 'soloname', downloads: 0, likes: 0, tags: [], siblings: [] }),
    });
    const result = await huggingFaceService.getModelDetails('soloname');
    expect(result.name).toBe('soloname');
    expect(result.author).toBe('soloname'); // id.split('/')[0]
  });
});
