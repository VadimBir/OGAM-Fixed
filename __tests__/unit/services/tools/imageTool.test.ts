/**
 * generate_image tool: argument bounding, the USED-values result, and failure as a real error.
 *
 * Falsify: return a plain string on failure (the old "could not start" text) -> the status:'error'
 * assertions go red; forward only prompt/steps -> the width/guidance/seed/sampler forwarding goes red.
 */
import { executeToolCall } from '../../../../src/services/tools/handlers';
import { normalizeToolResult } from '../../../../src/services/tools/toolResult';
import {
  resolveImageToolRequest,
  imageToolFailureMessage,
  imagePromptFromArgs,
  IMAGE_TOOL_NEXT_STEP,
} from '../../../../src/services/tools/imageTool';
import { availableToolIds } from '../../../../src/services/tools/availability';
import type { ToolCall } from '../../../../src/services/tools/types';

jest.mock('../../../../src/utils/logger', () => ({
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  debug: jest.fn(),
  log: jest.fn(),
}));

let mockEngine: any = null;
const mockGenerate = jest.fn();
jest.mock('../../../../src/services/imageGenerationService', () => ({
  imageGenerationService: {
    activeImageEngine: () => mockEngine,
    generateImageWithOutcome: (...args: any[]) => mockGenerate(...args),
  },
}));

const SDCPP = { kind: 'local', model: { id: 'sd', backend: 'sdcpp', sdFamily: 'sd1' } };
const LOCALDREAM = { kind: 'local', model: { id: 'ld', backend: 'mnn' } };

// null = no conversation (a default parameter would turn an explicit undefined back into 'conv-1').
function call(args: Record<string, any>, conversationId: string | null = 'conv-1'): ToolCall {
  return {
    id: 'tc-1',
    name: 'generate_image',
    arguments: args,
    ...(conversationId ? { context: { conversationId } } : {}),
  };
}

async function run(args: Record<string, any>, conversationId?: string | null) {
  const c = call(args, conversationId);
  return normalizeToolResult(c, await executeToolCall(c));
}

beforeEach(() => {
  jest.clearAllMocks();
  mockEngine = SDCPP;
});

describe('resolveImageToolRequest', () => {
  it('forwards width/height/steps/guidance/seed/sampler and omits what the model did not set', () => {
    const r = resolveImageToolRequest(
      { width: 256, height: 384, steps: 12, guidance: 6.5, seed: 42, sampler: 'Euler_A', negative_prompt: ' text ' },
      SDCPP as any,
    );
    expect(r.params).toEqual({
      width: 256, height: 384, steps: 12, guidanceScale: 6.5, seed: 42, sampler: 'euler_a', negativePrompt: 'text',
    });
    expect(r.notes).toEqual([]);
    expect(resolveImageToolRequest({}, SDCPP as any).params).toEqual({});
  });

  it('clamps out-of-range values to 64-512 / 1-50 / 1-20 and says so', () => {
    const r = resolveImageToolRequest({ width: 1024, height: 16, steps: 200, guidance: 40 }, SDCPP as any);
    expect(r.params).toMatchObject({ width: 512, height: 64, steps: 50, guidanceScale: 20 });
    expect(r.notes).toEqual([
      'width 1024 clamped to 512',
      'height 16 clamped to 64',
      'steps 200 clamped to 50',
      'guidance 40 clamped to 20',
    ]);
  });

  it('accepts numeric strings, drops non-numbers with a note, treats a negative seed as random', () => {
    const r = resolveImageToolRequest({ width: '320', steps: 'many', seed: -1 }, SDCPP as any);
    expect(r.params).toEqual({ width: 320 });
    expect(r.notes).toEqual(['steps "many" ignored (not a number)']);
  });

  // The description promises "multiple of 64"; 500 used to reach a non-snapping engine as 500.
  it('rounds a size that is not a multiple of 64 and says so', () => {
    const r = resolveImageToolRequest({ width: 500, height: 100, steps: 20 }, LOCALDREAM as any);
    expect(r.params).toMatchObject({ width: 512, height: 128 });
    expect(r.notes).toEqual(['width 500 rounded to 512', 'height 100 rounded to 128']);
    expect(resolveImageToolRequest({ width: 90 }, LOCALDREAM as any).params.width).toBe(64);
  });

  it('drops a sampler the engine does not accept so the user default stays', () => {
    const r = resolveImageToolRequest({ sampler: 'dpm++2m' }, LOCALDREAM as any);
    expect(r.params.sampler).toBeUndefined();
    expect(r.notes[0]).toContain('sampler "dpm++2m" not supported');
    expect(resolveImageToolRequest({ sampler: 'dpm++2m' }, SDCPP as any).params.sampler).toBe('dpm++2m');
  });
});

describe('generate_image handler', () => {
  it('returns compact JSON with the values actually used and notes what changed', async () => {
    mockEngine = LOCALDREAM;
    mockGenerate.mockResolvedValue({
      ok: true,
      image: {},
      used: { engine: 'local', width: 512, height: 512, steps: 20, guidance: 7.5, seed: 9, sampler: 'dpm' },
    });
    const result = await run({ prompt: 'a red fox', width: 256, height: 256, steps: 20 });
    expect(result.status).toBe('ok');
    expect(mockGenerate).toHaveBeenCalledWith({
      prompt: 'a red fox', conversationId: 'conv-1', width: 256, height: 256, steps: 20,
    });
    const body = JSON.parse(result.content);
    expect(body).toMatchObject({
      status: 'ok', shown_in_chat: true, width: 512, height: 512, steps: 20, guidance: 7.5, seed: 9, sampler: 'dpm',
    });
    expect(body.notes).toEqual(['width 256 -> 512 (engine limit)', 'height 256 -> 512 (engine limit)']);
    // The model is told what to do next: one short reply, no description, no re-call.
    expect(body.next).toBe(IMAGE_TOOL_NEXT_STEP);
    expect(body.next).toContain('one short sentence');
    expect(body.next).toContain('Do not describe the image');
    expect(body.next).toContain('Do not call generate_image again');
  });

  // Hunter H4: a small model puts the description under another key or as a list.
  it.each([
    [{ description: 'a red fox' }, 'a red fox'],
    [{ image_prompt: 'a red fox' }, 'a red fox'],
    [{ text: 'a red fox' }, 'a red fox'],
    [{ query: 'a red fox' }, 'a red fox'],
    [{ input: 'a red fox' }, 'a red fox'],
    [{ prompt: ['red fox', 'snow'] }, 'red fox, snow'],
    [{ prompt: '  ', description: 'a red fox' }, 'a red fox'],
  ])('accepts the prompt from %j', async (args, expected) => {
    mockGenerate.mockResolvedValue({ ok: true, image: {}, used: { engine: 'local', width: 512, height: 512 } });
    const result = await run(args);
    expect(result.status).toBe('ok');
    expect(mockGenerate.mock.calls[0][0].prompt).toBe(expected);
  });

  it('still fails with a missing-prompt error when no key holds a prompt', async () => {
    const result = await run({ width: 512 });
    expect(result.status).toBe('error');
    expect(result.error).toContain('prompt');
    expect(imagePromptFromArgs({ prompt: { a: 1 } })).toBeNull();
  });

  it('reports which arguments a remote server was not sent', async () => {
    mockEngine = { kind: 'remote', server: {} };
    mockGenerate.mockResolvedValue({ ok: true, image: {}, used: { engine: 'remote', width: 512, height: 512 } });
    const body = JSON.parse((await run({ prompt: 'x', steps: 10, seed: 3 })).content);
    expect(body.steps).toBeUndefined();
    expect(body.sampler).toBeUndefined();
    expect(body.notes).toEqual(['steps, seed not sent: this remote server does not take them']);
  });

  // Falsify: keep the old "remote takes prompt and size only" branch -> the sent values vanish.
  it('reports the steps / seed / guidance / sampler a remote server was sent', async () => {
    mockEngine = { kind: 'remote', server: {} };
    mockGenerate.mockResolvedValue({
      ok: true, image: {},
      used: { engine: 'remote', width: 384, height: 256, steps: 12, seed: 5, guidance: 6, sampler: 'euler' },
    });
    const body = JSON.parse((await run({ prompt: 'x', width: 384, height: 256, steps: 10, seed: 5, sampler: 'euler' })).content);
    expect(body).toMatchObject({ width: 384, height: 256, steps: 12, seed: 5, guidance: 6, sampler: 'euler' });
    expect(body.notes).toEqual(['steps 10 -> 12 (engine limit)']);
  });

  it.each([
    ['busy', 'Do not retry now'],
    ['out_of_memory', 'close other apps'],
    ['cancelled', 'cancelled'],
    ['load_failed', 'failed to load'],
    ['engine_error', 'engine failed'],
  ])('a %s failure is an error result that says what to do', async (reason, advice) => {
    mockGenerate.mockResolvedValue({ ok: false, reason, message: 'native detail' });
    const result = await run({ prompt: 'x' });
    expect(result.status).toBe('error');
    expect(result.error).toContain(`generate_image ${reason}:`);
    expect(result.error).toContain(advice);
  });

  it('fails with no_model, without calling the engine, when no image engine exists', async () => {
    mockEngine = null;
    const result = await run({ prompt: 'x' });
    expect(result.status).toBe('error');
    expect(result.error).toContain('no_model');
    expect(mockGenerate).not.toHaveBeenCalled();
  });

  it('fails (not succeeds) outside a conversation', async () => {
    const result = await run({ prompt: 'x' }, null);
    expect(result.status).toBe('error');
    expect(mockGenerate).not.toHaveBeenCalled();
  });

  it('adds the engine detail only for load/memory/engine failures, bounded', () => {
    expect(imageToolFailureMessage('busy', 'x')).not.toContain('Detail');
    const long = imageToolFailureMessage('engine_error', 'e'.repeat(500));
    expect(long).toContain('Detail: ');
    expect(long.length).toBeLessThan(400);
  });
});

describe('availableToolIds', () => {
  it('drops generate_image when no image engine is available and keeps everything else', () => {
    mockEngine = null;
    expect(availableToolIds(['web_search', 'generate_image', 'calculator'])).toEqual([
      'web_search',
      'calculator',
    ]);
    mockEngine = LOCALDREAM;
    expect(availableToolIds(['web_search', 'generate_image'])).toEqual(['web_search', 'generate_image']);
    mockEngine = { kind: 'remote', server: {} };
    expect(availableToolIds(['generate_image'])).toEqual(['generate_image']);
  });
});
