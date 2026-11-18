import RNFS from 'react-native-fs';
import { validateModelFile, checkMemoryForModel, resolveSafeContext, safeCompletion } from '../../../src/services/llmSafetyChecks';
import { effectiveAvailableMB, preloadSafetyMarginMB } from '../../../src/services/memoryBudget';
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
  const mockGetMemory = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns safe when enough memory is available', async () => {
    mockGetMemory.mockResolvedValue({
      available: 4 * 1024 * 1024 * 1024, // 4 GB
      total: 8 * 1024 * 1024 * 1024,
    });

    const result = await checkMemoryForModel({
      modelFileSize: 500 * 1024 * 1024, // 500 MB model
      contextLength: 2048,
      getAvailableMemory: mockGetMemory,
    });
    expect(result.safe).toBe(true);
  });

  it('returns unsafe when not enough memory', async () => {
    mockGetMemory.mockResolvedValue({
      available: 300 * 1024 * 1024, // 300 MB
      total: 4 * 1024 * 1024 * 1024,
    });

    const result = await checkMemoryForModel({
      modelFileSize: 2 * 1024 * 1024 * 1024, // 2 GB model
      contextLength: 4096,
      getAvailableMemory: mockGetMemory,
    });
    expect(result.safe).toBe(false);
    expect(result.reason).toContain('Not enough memory');
  });

  it('returns safe when memory check throws', async () => {
    mockGetMemory.mockRejectedValue(new Error('not supported'));

    const result = await checkMemoryForModel({ modelFileSize: 500 * 1024 * 1024, contextLength: 2048, getAvailableMemory: mockGetMemory });
    expect(result.safe).toBe(true);
  });

  it('scales KV cache with model size — a large model at high context is unsafe', async () => {
    // ~5 GB available; a 4 GB model at 8192 ctx. The old fixed ~2 MB KV estimate
    // wrongly called this safe; the size-scaled estimate flags it.
    mockGetMemory.mockResolvedValue({
      available: 5 * 1024 * 1024 * 1024,
      total: 8 * 1024 * 1024 * 1024,
    });
    const result = await checkMemoryForModel({ modelFileSize: 4 * 1024 * 1024 * 1024, contextLength: 8192, getAvailableMemory: mockGetMemory });
    expect(result.safe).toBe(false);
  });

  it('a quantized KV cache lowers the estimate enough to fit where f16 would not', async () => {
    const mem = { available: 3300 * 1024 * 1024, total: 6 * 1024 * 1024 * 1024 };
    mockGetMemory.mockResolvedValue(mem);
    const args = { modelFileSize: 2 * 1024 * 1024 * 1024, contextLength: 8192, getAvailableMemory: mockGetMemory };
    const f16 = await checkMemoryForModel({ ...args, quantizedCache: false });
    const quant = await checkMemoryForModel({ ...args, quantizedCache: true });
    expect(quant.estimatedMB).toBeLessThan(f16.estimatedMB);
    expect(f16.safe).toBe(false);
    expect(quant.safe).toBe(true);
  });
});

describe('Android live gate arithmetic — 12GB phone with 3GB free', () => {
  const MB = 1024 * 1024;
  const GB = 1024 * MB;
  // Exactly what llm.ts feeds the gate on Android: the live reading through effectiveAvailableMB
  // (no reclaim credit) plus the balanced live margin (1024MB) from preloadSafetyMarginMB.
  const availMB = effectiveAvailableMB(3072, 12288, { platform: 'android' });
  const marginMB = preloadSafetyMarginMB({ platform: 'android' });
  const getMem = async () => ({ available: availMB * MB, total: 12 * GB });
  // Fixed KV + compute so the numbers are exact: 256MB + 128MB.
  const archBytes = { kvBytes: 256 * MB, computeBytes: 128 * MB };

  it('sees 3072MB live and keeps a 1024MB margin', () => {
    expect(availMB).toBe(3072);
    expect(marginMB).toBe(1024);
  });

  it('admits a 1GB model: 1229 + 384 = 1613MB, + 200 + 1024 = 2837MB < 3072MB', async () => {
    const r = await checkMemoryForModel({ modelFileSize: 1 * GB, contextLength: 4096, getAvailableMemory: getMem, archBytes, safetyMarginMB: marginMB });
    expect(Math.round(r.estimatedMB)).toBe(1613);
    expect(r.safe).toBe(true);
  });

  it('refuses a 2GB model: 2458 + 384 = 2842MB, + 200 + 1024 = 4066MB > 3072MB, and states both in MB', async () => {
    const r = await checkMemoryForModel({ modelFileSize: 2 * GB, contextLength: 4096, getAvailableMemory: getMem, archBytes, safetyMarginMB: marginMB });
    expect(r.safe).toBe(false);
    expect(r.reason).toContain('needs ~2842MB + 1024MB safety margin but only 3072MB available');
  });

  it('the same 2GB model would pass without the margin (2842 + 200 = 3042 < 3072) - the margin is the deciding term', async () => {
    const r = await checkMemoryForModel({ modelFileSize: 2 * GB, contextLength: 4096, getAvailableMemory: getMem, archBytes });
    expect(r.safe).toBe(true);
  });

  it('resolveSafeContext throws an OVERRIDABLE refusal with needed vs available MB; Load Anyway passes', async () => {
    const args = { fileSize: 2 * GB, requestedCtx: 4096, quantizedCache: true, getAvailableMemory: getMem, archBytes, safetyMarginMB: marginMB };
    await expect(resolveSafeContext(args)).rejects.toBeInstanceOf(OverridableMemoryError);
    await expect(resolveSafeContext(args)).rejects.toThrow('it needs ~2842MB + 1024MB safety margin but only 3072MB is available');
    await expect(resolveSafeContext({ ...args, override: true })).resolves.toMatchObject({ ctxLen: 4096 });
  });

  it('iOS wording is unchanged when no margin applies', async () => {
    const r = await checkMemoryForModel({ modelFileSize: 2 * GB, contextLength: 4096, getAvailableMemory: async () => ({ available: 1000 * MB, total: 12 * GB }), archBytes });
    expect(r.reason).toContain('needs ~2842MB but only 1000MB available');
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
