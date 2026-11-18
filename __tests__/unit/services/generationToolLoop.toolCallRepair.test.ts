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
  completeTruncatedJson,
} from '../../../src/services/generationToolLoop';
import { llmService } from '../../../src/services/llm';
import { createMessage } from '../../utils/factories';

const mockAddMessage = jest.fn();
const mockSetIsThinking = jest.fn();
const mockAppState: any = {
  downloadedModels: [],
  activeModelId: null,
  settings: { temperature: 0.7, maxTokens: 1024, topP: 0.9, liteRTTemperature: 0.7, liteRTTopP: 0.9 },
};

jest.mock('../../../src/stores', () => ({
  useChatStore: {
    getState: () => ({
      addMessage: mockAddMessage,
      setStreamingMessage: jest.fn(),
      setIsThinking: mockSetIsThinking,
    }),
  },
  useRemoteServerStore: { getState: () => ({ activeServerId: null }) },
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
    expect(completeTruncatedJson('{"a":[1,2')).toBe('{"a":[1,2]}');
  });

  it('refuses when completion would be a guess', () => {
    expect(completeTruncatedJson('{"name":"x","arguments":{"q":"half a val')).toBeNull(); // inside a string
    expect(completeTruncatedJson('{"name":"x","arguments":')).toBeNull(); // after ':'
    expect(completeTruncatedJson('{"name":"x","arguments":{')).toBeNull(); // arguments never written
    expect(completeTruncatedJson('{"a":1}}')).toBeNull(); // mismatched
    expect(completeTruncatedJson('{"a":1}')).toBeNull(); // nothing to complete
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

  it('does not run a call from the reasoning when the model also gave a visible answer', async () => {
    mockGenerate.mockImplementationOnce(async (_m: any, opts: any) => {
      opts.onStream?.({ reasoningContent: '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>' });
      opts.onStream?.({ content: 'Paris is sunny today.' });
      return { fullResponse: 'Paris is sunny today.', toolCalls: [] };
    });
    const ctx = createContext();

    await runToolLoop(ctx);

    expect(mockExecuteToolCall).not.toHaveBeenCalled();
  });
});
