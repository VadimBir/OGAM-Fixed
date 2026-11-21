import RNFS from 'react-native-fs';
import { validateModelFile, checkMemoryForModel, resolveSafeContext, safeCompletion } from '../../../src/services/llmSafetyChecks';
import { effectiveAvailableMB, liveRamMarginMB, textMemoryFootprint } from '../../../src/services/memoryBudget';
import { OverridableMemoryError } from '../../../src/utils/modelLoadErrors';
import { defaultNativeFileSystemBoundary } from '../../harness/nativeFileSystem';

jest.mock('react-native-fs', () => {
  const { defaultNativeFileSystemBoundary: boundary } = require('../../harness/nativeFileSystem');
  return { __esModule: true, default: boundary.module, ...boundary.module };
});

const mockedRNFS = RNFS as jest.Mocked<typeof RNFS>;

describe('validateModelFile', () => {
  beforeEach(() => {
    defaultNativeFileSystemBoundary.reset();
  });

  it('returns invalid when file is too small', async () => {
    defaultNativeFileSystemBoundary.seedFile('/models/tiny.gguf', 100);

    const result = await validateModelFile('/models/tiny.gguf');
    expect(result.valid).toBe(false);
    expect(result.reason).toContain('too small');
  });

  it('returns valid for a proper GGUF file', async () => {
    defaultNativeFileSystemBoundary.seedFile('/models/test.gguf', 1_000_000);

    const result = await validateModelFile('/models/test.gguf');
    expect(result).toEqual({ valid: true });
  });

  it('returns invalid when header is not GGUF', async () => {
    defaultNativeFileSystemBoundary.seedTextFile(
      '/models/test.bin',
      'NOPE',
      1_000_000,
    );

    const result = await validateModelFile('/models/test.bin');
    expect(result.valid).toBe(false);
    expect(result.reason).toContain('not a GGUF file');
  });

  it('returns valid when RNFS.read() throws (iOS bridging workaround)', async () => {
    defaultNativeFileSystemBoundary.seedFile('/models/test.gguf', 1_000_000);
    mockedRNFS.read.mockRejectedValueOnce(new Error('NSInteger bridge error'));

    const result = await validateModelFile('/models/test.gguf');
    expect(result).toEqual({ valid: true });
  });

  it('returns invalid when the safe directory lookup cannot find the file', async () => {
    const result = await validateModelFile('/models/missing.gguf');
    expect(result.valid).toBe(false);
    expect(result.reason).toContain('not found');
  });

  it('handles string file size from stat', async () => {
    defaultNativeFileSystemBoundary.seedFile('/models/test.gguf', 5_000_000);
    defaultNativeFileSystemBoundary.setReportedFileSize(
      '/models/test.gguf',
      '5000000',
    );

    const result = await validateModelFile('/models/test.gguf');
    expect(result).toEqual({ valid: true });
  });
});

describe('checkMemoryForModel', () => {
  const MB = 1024 * 1024;
  const mockGetMemory = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns safe when free RAM covers the required MB + margin', async () => {
    mockGetMemory.mockResolvedValue({ available: 4096 * MB, total: 8192 * MB });
    const result = await checkMemoryForModel({ requiredMB: 600, getAvailableMemory: mockGetMemory, safetyMarginMB: 1024 });
    expect(result.safe).toBe(true);
    expect(result.estimatedMB).toBe(600);
  });

  it('returns unsafe when not enough memory', async () => {
    mockGetMemory.mockResolvedValue({ available: 300 * MB, total: 4096 * MB });
    const result = await checkMemoryForModel({ requiredMB: 2200, getAvailableMemory: mockGetMemory });
    expect(result.safe).toBe(false);
    expect(result.reason).toContain('Not enough memory');
  });

  it('returns safe when memory check throws', async () => {
    mockGetMemory.mockRejectedValue(new Error('not supported'));
    const result = await checkMemoryForModel({ requiredMB: 500, getAvailableMemory: mockGetMemory });
    expect(result.safe).toBe(true);
  });

  it('the boundary: required + margin == free RAM is safe (same rule as the residency live gate)', async () => {
    mockGetMemory.mockResolvedValue({ available: 3072 * MB, total: 12288 * MB });
    expect((await checkMemoryForModel({ requiredMB: 2048, getAvailableMemory: mockGetMemory, safetyMarginMB: 1024 })).safe).toBe(true);
    expect((await checkMemoryForModel({ requiredMB: 2049, getAvailableMemory: mockGetMemory, safetyMarginMB: 1024 })).safe).toBe(false);
  });
});

describe('Android live gate arithmetic — 12GB phone with 3GB free (one estimate: textMemoryFootprint)', () => {
  const MB = 1024 * 1024;
  const GB = 1024 * MB;
  // Exactly what llm.ts feeds the check on Android: the live reading through effectiveAvailableMB
  // (uncapped) and the balanced live margin, which is also the residency gate's live headroom.
  const availMB = effectiveAvailableMB(3072);
  const marginMB = liveRamMarginMB('balanced');
  const getMem = async () => ({ available: availMB * MB, total: 12 * GB });
  // Fixed KV + compute so the numbers are exact: 256MB + 128MB.
  const anon = 256 * MB + 128 * MB;

  it('sees 3072MB live and keeps a 1024MB margin', () => {
    expect(availMB).toBe(3072);
    expect(marginMB).toBe(1024);
  });

  it('mmapped CPU weights are reclaimable: a 2GB model needs only (256 + 128) x 1.1 = 423MB live', async () => {
    const f = textMemoryFootprint({ weightsBytes: 2 * GB, anonBytes: anon }, true);
    expect(f.livePeakMB).toBe(423);
    const r = await checkMemoryForModel({ requiredMB: f.livePeakMB, getAvailableMemory: getMem, safetyMarginMB: marginMB });
    expect(r.safe).toBe(true);
  });

  it('weights in anonymous RAM (mmap off / GPU): (2048 + 384) x 1.1 = 2676MB + 1024 > 3072, and states both in MB', async () => {
    const f = textMemoryFootprint({ weightsBytes: 2 * GB, anonBytes: anon }, false);
    expect(f.livePeakMB).toBe(2676);
    const r = await checkMemoryForModel({ requiredMB: f.livePeakMB, getAvailableMemory: getMem, safetyMarginMB: marginMB });
    expect(r.safe).toBe(false);
    expect(r.reason).toContain('needs ~2676MB + 1024MB safety margin but only 3072MB available');
  });

  it('resolveSafeContext throws an OVERRIDABLE refusal with needed vs available MB; Load Anyway passes', async () => {
    const args = { requiredMB: 2676, requestedCtx: 4096, getAvailableMemory: getMem, safetyMarginMB: marginMB };
    await expect(resolveSafeContext(args)).rejects.toBeInstanceOf(OverridableMemoryError);
    await expect(resolveSafeContext(args)).rejects.toThrow('it needs ~2676MB + 1024MB safety margin but only 3072MB is available');
    await expect(resolveSafeContext({ ...args, override: true })).resolves.toMatchObject({ ctxLen: 4096 });
  });

  it('no margin clause in the wording when no margin applies', async () => {
    const r = await checkMemoryForModel({ requiredMB: 2676, getAvailableMemory: async () => ({ available: 1000 * MB, total: 12 * GB }) });
    expect(r.reason).toContain('needs ~2676MB but only 1000MB available');
  });
});

describe('safeCompletion', () => {
  it('returns result of completionFn on success', async () => {
    const mockContext = { clearCache: jest.fn() };
    const result = await safeCompletion(mockContext as any, async () => 'ok');
    expect(result).toBe('ok');
  });

  it('throws wrapped error and clears KV cache on native crash (ggml)', async () => {
    const mockContext = { clearCache: jest.fn().mockResolvedValue(undefined) };
    await expect(
      safeCompletion(mockContext as any, async () => {
        throw new Error('ggml alloc failed');
      }),
    ).rejects.toThrow('Model inference failed (native error)');
    expect(mockContext.clearCache).toHaveBeenCalledWith(true);
  });

  it('throws wrapped error even when clearCache also fails', async () => {
    const mockContext = { clearCache: jest.fn().mockRejectedValue(new Error('cache clear failed')) };
    await expect(
      safeCompletion(mockContext as any, async () => {
        throw new Error('abort detected');
      }),
    ).rejects.toThrow('Model inference failed (native error)');
  });

  it('re-throws non-native errors unchanged', async () => {
    const mockContext = { clearCache: jest.fn() };
    await expect(
      safeCompletion(mockContext as any, async () => {
        throw new Error('unknown error');
      }),
    ).rejects.toThrow('unknown error');
    expect(mockContext.clearCache).not.toHaveBeenCalled();
  });

  it('recognises OOM as native crash keyword', async () => {
    const mockContext = { clearCache: jest.fn().mockResolvedValue(undefined) };
    await expect(
      safeCompletion(mockContext as any, async () => {
        throw new Error('OOM: out of memory');
      }),
    ).rejects.toThrow('Model inference failed (native error)');
    expect(mockContext.clearCache).toHaveBeenCalled();
  });

  it('uses String(error) when thrown value has no message', async () => {
    const mockContext = { clearCache: jest.fn().mockResolvedValue(undefined) };
    await expect(
      safeCompletion(mockContext as any, async () => {
        throw new Error('tensor error string');
      }),
    ).rejects.toThrow('Model inference failed (native error)');
  });
});
