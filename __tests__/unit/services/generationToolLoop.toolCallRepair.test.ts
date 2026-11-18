/**
 * Tool-call recovery in the text tool loop:
 *  - <function_call> blocks are parsed (they used to be detected and dropped),
 *  - a call missing only its closing braces is completed; a call cut inside a value is not,
 *  - an unparseable call is retried ONCE with a repair instruction, then stripped (never raw markup),
 *  - a call written inside the reasoning channel runs when the visible answer is empty.
 * Mocks mirror generationToolLoop.branches.test.ts.
 */
import {
  runToolLoop,
  ToolLoopContext,
  parseToolCallsFromText,
  parseUntaggedToolCalls,
  completeTruncatedJson,
  toolCallRepairInstruction,
} from '../../../src/services/generationToolLoop';
import { llmService } from '../../../src/services/llm';
import { createMessage } from '../../utils/factories';
import logger from '../../../src/utils/logger';

const mockAddMessage = jest.fn();
const mockSetIsThinking = jest.fn();
const mockAppState: any = {
  downloadedModels: [],
  activeModelId: null,
  settings: { temperature: 0.7, maxTokens: 1024, topP: 0.9, repeatPenalty: 1.3, liteRTTemperature: 0.7, liteRTTopP: 0.9 },
};
const mockRemoteState: { activeServerId: string | null } = { activeServerId: null };

jest.mock('../../../src/stores', () => ({
  useChatStore: {
    getState: () => ({
      addMessage: mockAddMessage,
      setStreamingMessage: jest.fn(),
      setIsThinking: mockSetIsThinking,
    }),
  },
  useRemoteServerStore: { getState: () => mockRemoteState },
  useAppStore: { getState: () => mockAppState },
}));

jest.mock('../../../src/services/llm', () => ({
  llmService: {
    generateResponseWithTools: jest.fn(),
    supportsToolCalling: jest.fn(() => false),
    stopGeneration: jest.fn().mockResolvedValue(undefined),
    isModelLoaded: jest.fn(() => true),
    generateToolSelection: jest.fn(),
  },
}));

jest.mock('../../../src/services/litert', () => ({
  liteRTService: {
    isModelLoaded: jest.fn(() => false),
    prepareConversation: jest.fn(),
    generateRaw: jest.fn(),
    getLastBenchmarkStats: jest.fn(() => undefined),
  },
}));

jest.mock('../../../src/services/providers', () => ({
  providerRegistry: { hasProvider: jest.fn(() => false), getProvider: jest.fn(() => null) },
}));

const mockExecuteToolCall = jest.fn();
jest.mock('../../../src/services/tools', () => ({
  getToolsAsOpenAISchema: () => [{ type: 'function', function: { name: 'web_search' } }],
  executeToolCall: (call: Record<string, unknown>) => mockExecuteToolCall(call),
}));

jest.mock('../../../src/services/tools/extensions', () => ({ getToolExtensions: () => [] }));
jest.mock('../../../src/services/litertToolSelector', () => ({ selectRelevantTools: jest.fn() }));

const mockGenerate = llmService.generateResponseWithTools as jest.Mock;

function createContext(overrides: Partial<ToolLoopContext> = {}): ToolLoopContext {
  return {
    conversationId: 'conv-1',
    messages: [createMessage({ content: 'weather in Paris?' } as any)],
    enabledToolIds: ['web_search'],
    isAborted: () => false,
    onThinkingDone: jest.fn(),
    onFinalResponse: jest.fn(),
    onStream: jest.fn(),
    onStreamReset: jest.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  mockGenerate.mockReset();
  mockExecuteToolCall.mockReset();
  mockExecuteToolCall.mockResolvedValue({ name: 'web_search', content: 'sunny', durationMs: 1 });
});

describe('parseToolCallsFromText — <function_call> and repairable bodies', () => {
  it('parses a <function_call> JSON block and strips it from the text', () => {
    const r = parseToolCallsFromText(
      'Ok. <function_call>{"name": "web_search", "arguments": {"query": "paris"}}</function_call>',
    );
    expect(r.toolCalls).toHaveLength(1);
    expect(r.toolCalls[0]).toMatchObject({ name: 'web_search', arguments: { query: 'paris' } });
    expect(r.cleanText).toBe('Ok.');
  });

  it('parses the name({...}) body form inside <function_call>', () => {
    const r = parseToolCallsFromText('<function_call>calculator({"expression": "2+2"})</function_call>');
    expect(r.toolCalls[0]).toMatchObject({ name: 'calculator', arguments: { expression: '2+2' } });
  });

  it('completes a call missing only its closing brace', () => {
    const r = parseToolCallsFromText(
      '<tool_call>{"name": "web_search", "arguments": {"query": "paris"}</tool_call>',
    );
    expect(r.toolCalls[0]).toMatchObject({ name: 'web_search', arguments: { query: 'paris' } });
  });

  it('parses an unclosed second call that follows a complete first one', () => {
    const r = parseToolCallsFromText(
      '<tool_call>{"name":"a","arguments":{}}</tool_call> <tool_call>{"name":"b","arguments":{"q":"x"}}',
    );
    expect(r.toolCalls.map(c => c.name)).toEqual(['a', 'b']);
    expect(r.cleanText).toBe('');
  });
});

describe('completeTruncatedJson', () => {
  it('closes brackets when only closers are missing', () => {
    expect(completeTruncatedJson('{"name":"x","arguments":{"q":"y"')).toBe('{"name":"x","arguments":{"q":"y"}}');
    expect(completeTruncatedJson('{"a":["x","y"')).toBe('{"a":["x","y"]}');
    expect(completeTruncatedJson('{"a":{"b":true')).toBe('{"a":{"b":true}}');
  });

  it('refuses when completion would be a guess', () => {
    expect(completeTruncatedJson('{"name":"x","arguments":{"q":"half a val')).toBeNull(); // inside a string
    expect(completeTruncatedJson('{"name":"x","arguments":')).toBeNull(); // after ':'
    expect(completeTruncatedJson('{"name":"x","arguments":{')).toBeNull(); // arguments never written
    expect(completeTruncatedJson('{"a":1}}')).toBeNull(); // mismatched
    expect(completeTruncatedJson('{"a":1}')).toBeNull(); // nothing to complete
    expect(completeTruncatedJson('{"a":[1,2')).toBeNull(); // 2 may be the start of 25
    expect(completeTruncatedJson('{"name":"x","arguments":{"minutes": 3')).toBeNull(); // 3 of 30
    expect(completeTruncatedJson('{"a": -1.')).toBeNull();
  });
});

describe('runToolLoop — broken tool call recovery', () => {
  const CUT = 'Checking. <tool_call>{"name": "web_search", "arguments": {"query": "weather in Pa';

  it('retries once with a repair instruction, then runs the repaired call', async () => {
    mockGenerate
      .mockResolvedValueOnce({ fullResponse: CUT, toolCalls: [] })
      .mockResolvedValueOnce({
        fullResponse: '<tool_call>{"name": "web_search", "arguments": {"query": "paris weather"}}</tool_call>',
        toolCalls: [],
      })
      .mockResolvedValueOnce({ fullResponse: 'It is sunny.', toolCalls: [] });
    const ctx = createContext();

    await runToolLoop(ctx);

    expect(mockGenerate).toHaveBeenCalledTimes(3);
    const retryMessages = mockGenerate.mock.calls[1][0];
    expect(retryMessages[retryMessages.length - 1].content).toContain('cut off or malformed');
    // The broken output rides as the assistant turn before the nudge: the model sees what to repair,
    // and roles alternate (no user,user pair that strict-alternation chat templates reject).
    expect(retryMessages.slice(-3).map((m: any) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(retryMessages[retryMessages.length - 2].content).toBe(CUT);
    for (let i = 1; i < retryMessages.length; i++) {
      expect([retryMessages[i - 1].role, retryMessages[i].role]).not.toEqual(['user', 'user']);
    }
    // Neither repair message is stored in the chat, and the next round no longer carries them.
    expect(mockAddMessage.mock.calls.some(([, m]) => m.content === CUT)).toBe(false);
    const thirdCall = mockGenerate.mock.calls[2][0];
    expect(thirdCall.some((m: any) => String(m.content).includes('cut off or malformed'))).toBe(false);
    // The instruction is never stored in the chat.
    expect(mockAddMessage.mock.calls.some(([, m]) => String(m.content).includes('cut off or malformed'))).toBe(false);
    expect(mockExecuteToolCall).toHaveBeenCalledTimes(1);
    expect(mockExecuteToolCall.mock.calls[0][0].arguments).toEqual({ query: 'paris weather' });
    expect(ctx.onStreamReset).toHaveBeenCalled();
    expect(ctx.onFinalResponse).toHaveBeenCalledWith('It is sunny.');
  }, 10000);

  it('after a second broken call, ends with the markup stripped and runs nothing', async () => {
    mockGenerate.mockResolvedValue({ fullResponse: CUT, toolCalls: [] });
    const ctx = createContext();

    await runToolLoop(ctx);

    expect(mockGenerate).toHaveBeenCalledTimes(2);
    expect(mockExecuteToolCall).not.toHaveBeenCalled();
    expect(ctx.onFinalResponse).toHaveBeenCalledWith('Checking.');
  });

  it('shows a notice instead of raw markup when nothing else was written', async () => {
    mockGenerate.mockResolvedValue({ fullResponse: '<function_call>{"name": "web_search", "argu', toolCalls: [] });
    const ctx = createContext();

    await runToolLoop(ctx);

    const shown = (ctx.onFinalResponse as jest.Mock).mock.calls[0][0] as string;
    expect(shown).not.toMatch(/<function_call>|<tool_call>/);
    expect(shown).toContain('cut off');
  });
});

describe('runToolLoop — tool call in the reasoning channel', () => {
  it('runs a call written inside the reasoning when the visible answer is empty', async () => {
    mockGenerate
      .mockImplementationOnce(async (_m: any, opts: any) => {
        opts.onStream?.({
          reasoningContent:
            'I should search. <tool_call>{"name": "web_search", "arguments": {"query": "paris"}}</tool_call>',
        });
        return { fullResponse: '', toolCalls: [] };
      })
      .mockResolvedValueOnce({ fullResponse: 'Sunny in Paris.', toolCalls: [] });
    const ctx = createContext();

    await runToolLoop(ctx);

    expect(mockExecuteToolCall).toHaveBeenCalledTimes(1);
    expect(mockExecuteToolCall.mock.calls[0][0].arguments).toEqual({ query: 'paris' });
    // The stored tool-call message keeps the reasoning without the raw markup.
    const toolCallMsg = mockAddMessage.mock.calls.map(([, m]) => m).find(m => m.toolCalls);
    expect(toolCallMsg.reasoningContent).toBe('I should search.');
    expect(ctx.onFinalResponse).toHaveBeenCalledWith('Sunny in Paris.');
  }, 10000);

  // A 3B model writes the call while thinking, then a reply that only pretends the tool ran. The old
  // gate (visible answer must be empty) dropped the call and showed "Here is your image." with no image.
  it('runs a call from the reasoning even when a visible reply follows, if the reply holds no call', async () => {
    mockGenerate
      .mockImplementationOnce(async (_m: any, opts: any) => {
        opts.onStream?.({ reasoningContent: '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>' });
        opts.onStream?.({ content: 'Here are the results.' });
        return { fullResponse: 'Here are the results.', toolCalls: [] };
      })
      .mockResolvedValueOnce({ fullResponse: 'Sunny.', toolCalls: [] });
    const ctx = createContext();

    await runToolLoop(ctx);

    expect(mockExecuteToolCall).toHaveBeenCalledTimes(1);
    expect(mockExecuteToolCall.mock.calls[0][0].arguments).toEqual({ query: 'x' });
    const toolCallMsg = mockAddMessage.mock.calls.map(([, m]) => m).find(m => m.toolCalls);
    expect(toolCallMsg.content).toBe('Here are the results.');
    expect(ctx.onFinalResponse).toHaveBeenCalledWith('Sunny.');
  }, 10000);

  it('does not take the reasoning call when the reply itself holds (broken) call markup', async () => {
    mockGenerate.mockImplementation(async (_m: any, opts: any) => {
      opts.onStream?.({ reasoningContent: '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>' });
      return { fullResponse: '<tool_call>{"name": "web_search", "arguments": {"query": "half', toolCalls: [] };
    });
    const ctx = createContext();

    await runToolLoop(ctx);

    // The reply's own broken call gets the repair round, not the reasoning call.
    expect(mockGenerate).toHaveBeenCalledTimes(2);
    expect(mockExecuteToolCall).not.toHaveBeenCalled();
  });
});

describe('remoteGenerateOnce — options and malformed argument strings', () => {
  const { providerRegistry } = jest.requireMock('../../../src/services/providers');
  afterEach(() => {
    mockRemoteState.activeServerId = null;
    providerRegistry.getProvider.mockReturnValue(null);
  });

  it('passes repeatPenalty and runs a call whose argument string is malformed with {} instead of throwing', async () => {
    const generate = jest
      .fn()
      .mockImplementationOnce((_m: any, _o: any, cb: any) => {
        cb.onComplete({
          content: '',
          toolCalls: [{ id: 'c1', name: 'web_search', arguments: '{"query": "paris"' }],
        });
        return Promise.resolve();
      })
      .mockImplementationOnce((_m: any, _o: any, cb: any) => {
        cb.onComplete({ content: 'Sunny.' });
        return Promise.resolve();
      });
    mockRemoteState.activeServerId = 'srv-1';
    providerRegistry.getProvider.mockReturnValue({ capabilities: { supportsThinking: false }, generate });
    const ctx = createContext({ forceRemote: true });
    const warnSpy = jest.spyOn(logger, 'warn').mockImplementation(() => {});

    await runToolLoop(ctx);

    expect(generate.mock.calls[0][1].repeatPenalty).toBe(1.3);
    expect(mockExecuteToolCall).toHaveBeenCalledTimes(1);
    // Unparseable string -> {} (the web_search default then fills the query from the user turn).
    expect(mockExecuteToolCall.mock.calls[0][0].arguments).not.toHaveProperty('query', 'paris');
    expect(warnSpy.mock.calls.some(([m]) => String(m).includes('raw: {"query": "paris"'))).toBe(true);
    expect(ctx.onFinalResponse).toHaveBeenCalledWith('Sunny.');
    warnSpy.mockRestore();
  }, 10000);

  it('still parses a well-formed argument string', async () => {
    const generate = jest
      .fn()
      .mockImplementationOnce((_m: any, _o: any, cb: any) => {
        cb.onComplete({ content: '', toolCalls: [{ id: 'c1', name: 'web_search', arguments: '{query: "paris"}' }] });
        return Promise.resolve();
      })
      .mockImplementationOnce((_m: any, _o: any, cb: any) => {
        cb.onComplete({ content: 'Sunny.' });
        return Promise.resolve();
      });
    mockRemoteState.activeServerId = 'srv-1';
    providerRegistry.getProvider.mockReturnValue({ capabilities: { supportsThinking: false }, generate });

    await runToolLoop(createContext({ forceRemote: true }));

    expect(mockExecuteToolCall.mock.calls[0][0].arguments).toEqual({ query: 'paris' });
  }, 10000);
});

// Hunter H3 samples (Qwen-style 3B output). Falsify: drop parseUntaggedToolCalls / the body
// normalisation in parseToolCallBody -> these go red.
describe('parseToolCallsFromText / parseUntaggedToolCalls — forms small models emit', () => {
  const IMG = { name: 'generate_image', arguments: { prompt: 'a fox' } };
  const offered = new Set(['generate_image', 'web_search']);

  it.each([
    ['fence inside the tag', '<tool_call>\n```json\n{"name": "generate_image", "arguments": {"prompt": "a fox"}}\n```\n</tool_call>'],
    ['python dict', "<tool_call>{'name': 'generate_image', 'arguments': {'prompt': 'a fox'}}</tool_call>"],
    ['one extra closing brace', '<tool_call>{"name":"generate_image","arguments":{"prompt":"a fox"}}}</tool_call>'],
    ['args key', '<tool_call>{"name":"generate_image","args":{"prompt":"a fox"}}</tool_call>'],
    ['input key', '<tool_call>{"name":"generate_image","input":{"prompt":"a fox"}}</tool_call>'],
    ['stringified arguments', '<tool_call>{"name":"generate_image","arguments":"{\\"prompt\\": \\"a fox\\"}"}</tool_call>'],
    ['{"tool_call": {...}} wrapper', '<tool_call>{"tool_call": {"name": "generate_image", "arguments": {"prompt": "a fox"}}}</tool_call>'],
  ])('<tool_call> with %s', (_label, text) => {
    const r = parseToolCallsFromText(text);
    expect(r.toolCalls).toHaveLength(1);
    expect(r.toolCalls[0]).toMatchObject(IMG);
    expect(r.cleanText).toBe('');
  });

  it('python kwargs inside the tag, with typed values', () => {
    const r = parseToolCallsFromText('<tool_call>generate_image(prompt="a fox", width=512, seed=None)</tool_call>');
    expect(r.toolCalls[0]).toMatchObject({ name: 'generate_image', arguments: { prompt: 'a fox', width: 512 } });
    expect(r.toolCalls[0].arguments).not.toHaveProperty('seed');
  });

  it.each([
    ['bare JSON', '{"name":"generate_image","arguments":{"prompt":"a fox"}}', ''],
    ['```json fence', '```json\n{"name": "generate_image", "arguments": {"prompt": "a fox"}}\n```', ''],
    ['fence after text', 'Sure!\n```json\n{"name": "generate_image", "arguments": {"prompt": "a fox"}}\n```', 'Sure!'],
    ['{"tool_call": {...}}', '{"tool_call": {"name": "generate_image", "arguments": {"prompt": "a fox"}}}', ''],
    ['python kwargs', 'generate_image(prompt="a fox")', ''],
  ])('untagged %s naming an offered tool is a call', (_label, text, clean) => {
    const r = parseUntaggedToolCalls(text, offered);
    expect(r.toolCalls).toHaveLength(1);
    expect(r.toolCalls[0]).toMatchObject(IMG);
    expect(r.cleanText).toBe(clean);
  });

  it('untagged JSON / code that does not name an offered tool stays text', () => {
    for (const text of [
      '{"name": "Alice", "age": 3}',
      '```python\nprint(len("abc"))\n```',
      'calculator(expression="2+2")', // not offered here
      'The answer is generate_image(prompt="x") maybe.', // not the whole reply
    ]) {
      expect(parseUntaggedToolCalls(text, offered).toolCalls).toEqual([]);
    }
    expect(parseUntaggedToolCalls('{"name":"generate_image","arguments":{}}', new Set()).toolCalls).toEqual([]);
  });
});

describe('runToolLoop — untagged calls, empty replies, image prompt fallback', () => {
  it('runs an untagged JSON call and stores no raw JSON as the assistant text', async () => {
    const raw = '```json\n{"name": "web_search", "arguments": {"query": "paris"}}\n```';
    mockGenerate
      .mockImplementationOnce(async (_m: any, opts: any) => {
        opts.onStream?.({ content: raw });
        return { fullResponse: raw, toolCalls: [] };
      })
      .mockResolvedValueOnce({ fullResponse: 'Sunny.', toolCalls: [] });
    const ctx = createContext();

    await runToolLoop(ctx);

    expect(mockExecuteToolCall).toHaveBeenCalledTimes(1);
    expect(mockExecuteToolCall.mock.calls[0][0].arguments).toEqual({ query: 'paris' });
    const toolCallMsg = mockAddMessage.mock.calls.map(([, m]) => m).find(m => m.toolCalls);
    expect(toolCallMsg.content).toBe('');
    expect(ctx.onFinalResponse).toHaveBeenCalledWith('Sunny.');
  }, 10000);

  it('a cut-off untagged call gets the repair round, then a notice (never the raw JSON)', async () => {
    mockGenerate.mockResolvedValue({ fullResponse: '{"name": "web_search", "arguments": {"query": "par', toolCalls: [] });
    const ctx = createContext();

    await runToolLoop(ctx);

    expect(mockGenerate).toHaveBeenCalledTimes(2);
    const shown = (ctx.onFinalResponse as jest.Mock).mock.calls[0][0] as string;
    expect(shown).not.toContain('"name"');
    expect(shown).toContain('cut off');
  });

  it('text-format engine: the repair names the <tool_call> format; native: no syntax', () => {
    expect(toolCallRepairInstruction(false)).toContain('<tool_call>{"name": "TOOL_NAME"');
    expect(toolCallRepairInstruction(true)).not.toMatch(/<tool_call>|\{"name"/);
    expect(toolCallRepairInstruction(true)).toContain('cut off or malformed');
  });

  // Hunter H2: thinking ate n_predict -> empty reply -> the old fallback ran with tools=[], so the
  // tool could not be called this turn. Falsify: restore the immediate no-tools fallback.
  it('an empty reply is retried once WITH the tools and thinking off', async () => {
    mockGenerate
      .mockResolvedValueOnce({ fullResponse: '', toolCalls: [] })
      .mockResolvedValueOnce({ fullResponse: '', toolCalls: [{ id: 'c1', name: 'web_search', arguments: { query: 'paris' } }] })
      .mockResolvedValueOnce({ fullResponse: 'Sunny.', toolCalls: [] });
    const ctx = createContext();

    await runToolLoop(ctx);

    const second = mockGenerate.mock.calls[1][1];
    expect(second.tools.length).toBeGreaterThan(0);
    expect(second.disableThinking).toBe(true);
    expect(mockGenerate.mock.calls[0][1].disableThinking).toBeUndefined();
    expect(mockGenerate.mock.calls[2][1].disableThinking).toBeUndefined(); // one round only
    expect(mockExecuteToolCall).toHaveBeenCalledTimes(1);
    expect(ctx.onFinalResponse).toHaveBeenCalledWith('Sunny.');
  }, 10000);

  it('a second empty reply falls back to one no-tools answer', async () => {
    mockGenerate
      .mockResolvedValueOnce({ fullResponse: '', toolCalls: [] })
      .mockResolvedValueOnce({ fullResponse: '', toolCalls: [] })
      .mockResolvedValueOnce({ fullResponse: 'Plain answer.', toolCalls: [] });
    const ctx = createContext();

    await runToolLoop(ctx);

    expect(mockGenerate).toHaveBeenCalledTimes(3);
    expect(mockGenerate.mock.calls[2][1].tools).toEqual([]);
    expect(ctx.onFinalResponse).toHaveBeenCalledWith('Plain answer.');
  }, 10000);

  // Hunter H4: generate_image with the prompt missing / under no accepted key ran into
  // "Missing required parameter: prompt". The loop now fills it from the user's request.
  it('generate_image with no prompt gets the user message as the prompt', async () => {
    mockExecuteToolCall.mockResolvedValue({ name: 'generate_image', content: '{"status":"ok"}', durationMs: 1 });
    mockGenerate
      .mockResolvedValueOnce({ fullResponse: '', toolCalls: [{ id: 'c1', name: 'generate_image', arguments: { width: 512 } }] })
      .mockResolvedValueOnce({ fullResponse: 'Done.', toolCalls: [] });
    const ctx = createContext({ messages: [createMessage({ content: 'draw a red fox' } as any)] });

    await runToolLoop(ctx);

    expect(mockExecuteToolCall.mock.calls[0][0].arguments).toEqual({ width: 512, prompt: 'draw a red fox' });
  }, 10000);

  it('generate_image with the prompt under an alias key keeps it (no fallback)', async () => {
    mockExecuteToolCall.mockResolvedValue({ name: 'generate_image', content: '{"status":"ok"}', durationMs: 1 });
    mockGenerate
      .mockResolvedValueOnce({ fullResponse: '', toolCalls: [{ id: 'c1', name: 'generate_image', arguments: { description: 'a fox' } }] })
      .mockResolvedValueOnce({ fullResponse: 'Done.', toolCalls: [] });

    await runToolLoop(createContext());

    expect(mockExecuteToolCall.mock.calls[0][0].arguments).toEqual({ description: 'a fox' });
  }, 10000);
});
