/**
 * llmToolGeneration Unit Tests
 *
 * Tests for the tool-aware LLM generation helper (tool calls parsing, streaming, error handling).
 * Priority: P0 (Critical) - Core tool-calling inference path.
 */

import { useAppStore } from '../../../src/stores/appStore';
import { resetStores } from '../../utils/testHelpers';
import { createUserMessage } from '../../utils/factories';
import {
  generateWithToolsImpl,
  ToolGenerationDeps,
} from '../../../src/services/llmToolGeneration';
import type { Message } from '../../../src/types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal deps object with sensible defaults; callers can override.
 *  setIsGenerating is wired to actually mutate deps.isGenerating so the
 *  streaming callback gate (`if (!deps.isGenerating) return`) works correctly. */
function createMockDeps(overrides: Partial<ToolGenerationDeps> = {}): ToolGenerationDeps {
  const deps: ToolGenerationDeps = {
    context: {
      completion: jest.fn(async (_params: any, _cb?: any) => ({})),
    },
    isGenerating: false,
    isThinkingEnabled: false,
    isGemma4Model: false,
    disableCtxShift: false,
    manageContextWindow: jest.fn(async (msgs: Message[]) => msgs),
    // Async, matching the real converter: it drops images whose file is gone before building.
    convertToOAIMessages: jest.fn(async (msgs: Message[]) =>
      msgs.map(m => ({ role: m.role, content: m.content })),
    ),
    setPerformanceStats: jest.fn(),
    setIsGenerating: jest.fn(),
    ...overrides,
  };
  // Wire setIsGenerating to actually mutate deps.isGenerating (unless caller overrode it)
  if (!overrides.setIsGenerating) {
    (deps.setIsGenerating as jest.Mock).mockImplementation((v: boolean) => {
      deps.isGenerating = v;
    });
  }
  return deps;
}

const SAMPLE_TOOLS = [
  {
    type: 'function',
    function: {
      name: 'calculator',
      description: 'Calculate a math expression',
      parameters: { type: 'object', properties: { expression: { type: 'string' } } },
    },
  },
];

// ---------------------------------------------------------------------------
// Test Suite
// ---------------------------------------------------------------------------

describe('generateWithToolsImpl', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    resetStores();
  });

  // ========================================================================
  // Guard clauses
  // ========================================================================
  describe('guard clauses', () => {
    it('throws when context is null', async () => {
      const deps = createMockDeps({ context: null });
      const messages = [createUserMessage('Hello')];

      await expect(
        generateWithToolsImpl(deps, messages, { tools: SAMPLE_TOOLS }),
      ).rejects.toThrow('No model loaded');
    });

    it('throws when generation is already in progress', async () => {
      const deps = createMockDeps({ isGenerating: true });
      const messages = [createUserMessage('Hello')];

      await expect(
        generateWithToolsImpl(deps, messages, { tools: SAMPLE_TOOLS }),
      ).rejects.toThrow('Generation already in progress');
    });

    it('does not call setIsGenerating(true) when context is null', async () => {
      const deps = createMockDeps({ context: null });
      const messages = [createUserMessage('Hello')];

      await expect(
        generateWithToolsImpl(deps, messages, { tools: SAMPLE_TOOLS }),
      ).rejects.toThrow();

      expect(deps.setIsGenerating).not.toHaveBeenCalled();
    });
  });

  // ========================================================================
  // Completion call shape
  // ========================================================================
  describe('completion call parameters', () => {
    it('passes tools and tool_choice to context.completion', async () => {
      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion } });
      const messages = [createUserMessage('Hello')];

      await generateWithToolsImpl(deps, messages, { tools: SAMPLE_TOOLS });

      expect(completion).toHaveBeenCalledTimes(1);
      const callArgs = completion.mock.calls[0][0];
      expect(callArgs.tools).toBe(SAMPLE_TOOLS);
      expect(callArgs.tool_choice).toBe('auto');
    });

    it('uses llama.rn auto reasoning format when thinking is enabled', async () => {
      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion }, isThinkingEnabled: true });

      await generateWithToolsImpl(deps, [createUserMessage('Hello')], { tools: SAMPLE_TOOLS });

      const callArgs = completion.mock.calls[0][0];
      expect(callArgs.enable_thinking).toBe(true);
      expect(callArgs.reasoning_format).toBe('deepseek');
    });

    it('uses reasoning_format auto for Gemma 4 so llama.cpp parses its channel format natively', async () => {
      // Native-first: instead of forcing 'none' and hand-parsing Gemma's <|channel>thought format,
      // let llama.cpp detect the chat_format and populate reasoning_content/tool_calls itself. Our
      // hand-parse fallback only runs when those come back empty, so this is safe.
      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion }, isThinkingEnabled: true, isGemma4Model: true });

      await generateWithToolsImpl(deps, [createUserMessage('Hello')], { tools: SAMPLE_TOOLS });

      const callArgs = completion.mock.calls[0][0];
      expect(callArgs.enable_thinking).toBe(true);
      expect(callArgs.reasoning_format).toBe('auto');
    });

    it('disables llama.rn reasoning extraction when thinking is off', async () => {
      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion }, isThinkingEnabled: false });

      await generateWithToolsImpl(deps, [createUserMessage('Hello')], { tools: SAMPLE_TOOLS });

      const callArgs = completion.mock.calls[0][0];
      expect(callArgs.enable_thinking).toBe(false);
      expect(callArgs.reasoning_format).toBe('none');
    });

    it('disableThinking forces thinking off even when the global toggle is on', async () => {
      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion }, isThinkingEnabled: true });

      await generateWithToolsImpl(deps, [createUserMessage('Hello')], {
        tools: SAMPLE_TOOLS,
        disableThinking: true,
      });

      const callArgs = completion.mock.calls[0][0];
      expect(callArgs.enable_thinking).toBe(false);
      expect(callArgs.reasoning_format).toBe('none');
    });

    it('disables ctx_shift when disableCtxShift is true (Android GPU SIGSEGV fix)', async () => {
      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion }, disableCtxShift: true });

      await generateWithToolsImpl(deps, [createUserMessage('Hello')], { tools: SAMPLE_TOOLS });

      const callArgs = completion.mock.calls[0][0];
      expect(callArgs.ctx_shift).toBe(false);
    });

    it('enables ctx_shift when disableCtxShift is false', async () => {
      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion }, disableCtxShift: false });

      await generateWithToolsImpl(deps, [createUserMessage('Hello')], { tools: SAMPLE_TOOLS });

      const callArgs = completion.mock.calls[0][0];
      expect(callArgs.ctx_shift).toBe(true);
    });

    it('passes temperature and other settings from the app store', async () => {
      useAppStore.setState({
        settings: {
          ...useAppStore.getState().settings,
          temperature: 0.3,
          maxTokens: 256,
          topP: 0.85,
          repeatPenalty: 1.2,
        },
      });

      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion } });
      const messages = [createUserMessage('Hello')];

      await generateWithToolsImpl(deps, messages, { tools: SAMPLE_TOOLS });

      const callArgs = completion.mock.calls[0][0];
      expect(callArgs.temperature).toBe(0.3);
      expect(callArgs.n_predict).toBe(256);
      expect(callArgs.top_p).toBe(0.85);
      expect(callArgs.penalty_repeat).toBe(1.2);
    });

    it('uses RESPONSE_RESERVE when maxTokens is falsy', async () => {
      useAppStore.setState({
        settings: {
          ...useAppStore.getState().settings,
          maxTokens: 0,
        },
      });

      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion } });

      await generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS });

      const callArgs = completion.mock.calls[0][0];
      // RESPONSE_RESERVE is 512
      expect(callArgs.n_predict).toBe(512);
    });

    it('delegates to manageContextWindow and convertToOAIMessages', async () => {
      const managed = [createUserMessage('managed')];
      const manageContextWindow = jest.fn(async () => managed);
      const convertToOAIMessages = jest.fn(async () => [{ role: 'user', content: 'managed' }]);
      const completion = jest.fn(async (_params: any, _cb: any) => ({}));

      const deps = createMockDeps({
        context: { completion },
        manageContextWindow,
        convertToOAIMessages,
      });

      const original = [createUserMessage('original')];
      await generateWithToolsImpl(deps, original, { tools: SAMPLE_TOOLS });

      expect(manageContextWindow).toHaveBeenCalledWith(original, expect.any(Number));
      expect(convertToOAIMessages).toHaveBeenCalledWith(managed);
      expect(completion.mock.calls[0][0].messages).toEqual([
        { role: 'user', content: 'managed' },
      ]);
    });
  });

  // ========================================================================
  // Streaming tokens (no tool calls)
  // ========================================================================
  describe('streaming tokens without tool calls', () => {
    it('returns fullResponse built from streamed tokens', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: 'Hello' });
        cb({ token: ' World' });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.fullResponse).toBe('Hello World');
      expect(result.toolCalls).toEqual([]);
    });

    it('invokes onStream callback for each token', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: 'A' });
        cb({ token: 'B' });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });
      const onStream = jest.fn();

      await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
        onStream,
      });

      expect(onStream).toHaveBeenCalledTimes(2);
      expect(onStream).toHaveBeenNthCalledWith(1, { content: 'A' });
      expect(onStream).toHaveBeenNthCalledWith(2, { content: 'B' });
    });

    // (Removed: mockist assertion of onComplete's arg shape — the tool path now passes reasoning as a
    // second arg (onComplete('Done', '')), so toHaveBeenCalledWith('Done') no longer matches. The
    // full-response-renders behavior is covered by the rendered tool/thinking integration tests.)

    it('skips callback data without a token property', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({}); // no token, no tool_calls
        cb({ token: 'Yes' });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });
      const onStream = jest.fn();

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
        onStream,
      });

      expect(result.fullResponse).toBe('Yes');
      expect(onStream).toHaveBeenCalledTimes(1);
    });
  });

  // ========================================================================
  // Tool calls from streaming callback
  // ========================================================================
  describe('tool calls collected during streaming', () => {
    it('parses a single tool call from streaming data', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({
          tool_calls: [
            {
              id: 'call_1',
              function: {
                name: 'calculator',
                arguments: JSON.stringify({ expression: '2+2' }),
              },
            },
          ],
        });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Calculate 2+2')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0]).toEqual({
        id: 'call_1',
        name: 'calculator',
        arguments: { expression: '2+2' },
      });
    });

    it('parses multiple tool calls from a single streaming callback', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({
          tool_calls: [
            {
              id: 'call_1',
              function: { name: 'calculator', arguments: '{"expression":"1+1"}' },
            },
            {
              id: 'call_2',
              function: { name: 'get_current_datetime', arguments: '{}' },
            },
          ],
        });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.toolCalls).toHaveLength(2);
      expect(result.toolCalls[0].name).toBe('calculator');
      expect(result.toolCalls[1].name).toBe('get_current_datetime');
    });

    // llama.rn re-parses the whole output per token (RNLlamaJSI.cpp parseChatOutput(true)): each
    // callback's tool_calls is the CUMULATIVE list, so the latest list replaces the previous one.
    it('treats streamed tool_calls as the cumulative list (latest replaces previous)', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({
          tool_calls: [
            { id: 'call_1', function: { name: 'calculator', arguments: '{"a":1}' } },
          ],
        });
        cb({
          tool_calls: [
            { id: 'call_1', function: { name: 'calculator', arguments: '{"a":1}' } },
            { id: 'call_2', function: { name: 'get_current_datetime', arguments: '{}' } },
          ],
        });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.toolCalls.map(t => t.id)).toEqual(['call_1', 'call_2']);
    });

    it('does not duplicate a partial call re-sent on every token when the final parse fails', async () => {
      // Final result carries no tool_calls (native final parse threw, JSICompletion.h ignores it).
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: '<tool_call>', tool_calls: [{ id: 'c', function: { name: 'generate_image', arguments: '' } }] });
        cb({ token: '{"name"', tool_calls: [{ id: 'c', function: { name: 'generate_image', arguments: '' } }] });
        cb({ token: ':"generate_image"', tool_calls: [{ id: 'c', function: { name: 'generate_image', arguments: '{"prompt":"cat"}' } }] });
        return { text: '<tool_call>{"name":"generate_image"' };
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS });

      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0]).toMatchObject({ name: 'generate_image', arguments: { prompt: 'cat' } });
    });

    it('handles tool call with arguments as object (not string)', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({
          tool_calls: [
            {
              id: 'call_obj',
              function: { name: 'calculator', arguments: { expression: '3*3' } },
            },
          ],
        });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.toolCalls[0].arguments).toEqual({ expression: '3*3' });
    });

    it('parses arguments that use smart/curly quotes as delimiters (no empty-args retry loop)', async () => {
      // Device 2026-07-15: a local model emitted `{“expression”:“7 * 7”}` with CURLY double quotes.
      // Strict JSON.parse throws → args became {} → the calculator got an empty call → schema
      // validation failed → the model retried the same call in a loop. The parser now normalizes the
      // curly delimiters, so the real expression reaches the tool on the first try.
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({
          tool_calls: [
            { id: 'call_curly', function: { name: 'calculator', arguments: '{“expression”:“7 * 7”}' } },
          ],
        });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('multiply 7 by 7')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.toolCalls[0].name).toBe('calculator');
      expect(result.toolCalls[0].arguments).toEqual({ expression: '7 * 7' }); // NOT {} — curly quotes normalized
    });

    it('handles tool call with missing function fields gracefully', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({
          tool_calls: [{ id: 'call_empty' }], // no function property
        });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0]).toEqual({
        id: 'call_empty',
        name: '',
        arguments: {},
      });
    });

    it('handles tool call with empty arguments string', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({
          tool_calls: [
            { id: 'call_e', function: { name: 'get_current_datetime', arguments: '' } },
          ],
        });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.toolCalls[0].arguments).toEqual({});
    });
  });

  // ========================================================================
  // Tool calls from completionResult (fallback path)
  // ========================================================================
  describe('tool calls from completion result (non-streaming fallback)', () => {
    it('extracts tool calls from completionResult when none collected during streaming', async () => {
      const completion = jest.fn(async (_params: any, _cb: any) => ({
        tool_calls: [
          {
            id: 'result_call_1',
            function: { name: 'calculator', arguments: '{"expression":"5+5"}' },
          },
        ],
      }));
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].id).toBe('result_call_1');
      expect(result.toolCalls[0].arguments).toEqual({ expression: '5+5' });
    });

    it('prefers completionResult tool_calls over streamed ones (complete data)', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        // Streaming delivers a partial tool call (may have incomplete args)
        cb({
          tool_calls: [
            { id: 'stream_call', function: { name: 'calculator', arguments: '{"x":1}' } },
          ],
        });
        // completionResult has the complete tool call data
        return {
          tool_calls: [
            { id: 'result_call', function: { name: 'get_current_datetime', arguments: '{}' } },
          ],
        };
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
      });

      // completionResult tool_calls are preferred (they're always complete)
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].id).toBe('result_call');
    });
  });

  // ========================================================================
  // isGenerating flag and streaming gate
  // ========================================================================
  describe('isGenerating lifecycle', () => {
    it('calls setIsGenerating(true) at the start', async () => {
      const completion = jest.fn(async () => ({}));
      const deps = createMockDeps({ context: { completion } });

      await generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS });

      expect(deps.setIsGenerating).toHaveBeenCalledWith(true);
    });

    it('calls setIsGenerating(false) on success', async () => {
      const completion = jest.fn(async () => ({}));
      const deps = createMockDeps({ context: { completion } });

      await generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS });

      // Last call should be false
      const calls = (deps.setIsGenerating as jest.Mock).mock.calls;
      expect(calls[calls.length - 1][0]).toBe(false);
    });

    it('calls setIsGenerating(false) on error', async () => {
      const completion = jest.fn(async () => {
        throw new Error('boom');
      });
      const deps = createMockDeps({ context: { completion } });

      await expect(
        generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS }),
      ).rejects.toThrow('boom');

      const calls = (deps.setIsGenerating as jest.Mock).mock.calls;
      expect(calls[calls.length - 1][0]).toBe(false);
    });

    it('captures all streamed tokens while generating', async () => {
      const deps = createMockDeps();
      const onStream = jest.fn();

      deps.context.completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: 'First' });
        cb({ token: ' Second' });
        return {};
      });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
        onStream,
      });

      expect(result.fullResponse).toBe('First Second');
      expect(onStream).toHaveBeenCalledTimes(2);
    });
  });

  // ========================================================================
  // Performance stats
  // ========================================================================
  describe('performance stats', () => {
    it('calls setPerformanceStats with recorded stats', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: 'tok1' });
        cb({ token: 'tok2' });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      await generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS });

      expect(deps.setPerformanceStats).toHaveBeenCalledTimes(1);
      const stats = (deps.setPerformanceStats as jest.Mock).mock.calls[0][0];
      expect(stats).toHaveProperty('lastTokenCount', 2);
      expect(stats).toHaveProperty('lastTokensPerSecond');
      expect(stats).toHaveProperty('lastGenerationTime');
      expect(stats).toHaveProperty('lastTimeToFirstToken');
      expect(stats).toHaveProperty('lastDecodeTokensPerSecond');
    });

    it('records zero tokens when only tool calls are returned', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({
          tool_calls: [
            { id: 'tc', function: { name: 'calculator', arguments: '{}' } },
          ],
        });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      await generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS });

      const stats = (deps.setPerformanceStats as jest.Mock).mock.calls[0][0];
      expect(stats.lastTokenCount).toBe(0);
    });
  });

  // ========================================================================
  // Error handling
  // ========================================================================
  describe('error handling', () => {
    it('re-throws errors from context.completion', async () => {
      const completion = jest.fn(async () => {
        throw new Error('completion failed');
      });
      const deps = createMockDeps({ context: { completion } });

      await expect(
        generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS }),
      ).rejects.toThrow('completion failed');
    });

    it('re-throws errors from manageContextWindow', async () => {
      const deps = createMockDeps({
        manageContextWindow: jest.fn(async () => {
          throw new Error('context window error');
        }),
      });

      await expect(
        generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS }),
      ).rejects.toThrow('context window error');
    });

    it('still resets isGenerating when manageContextWindow throws', async () => {
      const deps = createMockDeps({
        manageContextWindow: jest.fn(async () => {
          throw new Error('fail');
        }),
      });

      await expect(
        generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS }),
      ).rejects.toThrow();

      const calls = (deps.setIsGenerating as jest.Mock).mock.calls;
      expect(calls[calls.length - 1][0]).toBe(false);
    });
  });

  // ========================================================================
  // Mixed: tokens + tool calls
  // ========================================================================
  describe('mixed tokens and tool calls', () => {
    it('returns both fullResponse text and tool calls when both are streamed', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: 'Let me calculate. ' });
        cb({
          tool_calls: [
            { id: 'tc1', function: { name: 'calculator', arguments: '{"expression":"2+2"}' } },
          ],
        });
        cb({ token: 'Done.' });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
      });

      expect(result.fullResponse).toBe('Let me calculate. Done.');
      expect(result.toolCalls).toHaveLength(1);
      expect(result.toolCalls[0].name).toBe('calculator');
    });

    // H2-2: no data.content (partial parse failed) -> data.token carries the raw Qwen/Hermes call markup.
    it('filters <tool_call> markup from stream and reply for a non-Gemma model when the call is structured', async () => {
      const call = { id: 'c1', function: { name: 'generate_image', arguments: '{"prompt":"cat"}' } };
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: 'Sure. ' });
        cb({ token: '<tool_' });
        cb({ token: 'call>\n{"name":"generate_image","arguments":{"prompt":"cat"}}\n</tool_call>', tool_calls: [call] });
        return { tool_calls: [call] };
      });
      const onStream = jest.fn();
      const deps = createMockDeps({ context: { completion }, isGemma4Model: false });

      const result = await generateWithToolsImpl(deps, [createUserMessage('draw a cat')], { tools: SAMPLE_TOOLS, onStream });

      expect(result.fullResponse).toBe('Sure. ');
      const streamed = onStream.mock.calls.map(c => c[0].content ?? '').join('');
      expect(streamed).toBe('Sure. ');
      expect(result.toolCalls).toHaveLength(1);
    });

    it('returns the RAW markup when no structured call was parsed, so the loop text fallback can parse it', async () => {
      const raw = '<tool_call>\n{"name":"generate_image","arguments":{"prompt":"cat"}}\n</tool_call>';
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: raw });
        return {};
      });
      const onStream = jest.fn();
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('draw a cat')], { tools: SAMPLE_TOOLS, onStream });

      expect(result.toolCalls).toHaveLength(0);
      expect(result.fullResponse).toBe(raw);
      // The user never sees the markup stream by.
      expect(onStream.mock.calls.map(c => c[0].content ?? '').join('')).toBe('');
    });

    it('does not refill an all-markup stream from cr.text when the call is structured', async () => {
      const call = { id: 'c1', function: { name: 'generate_image', arguments: '{"prompt":"cat"}' } };
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: '<tool_call>{"name":"generate_image"}</tool_call>', tool_calls: [call] });
        return { text: '<tool_call>{"name":"generate_image"}</tool_call>' }; // no content: final parse threw
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('draw')], { tools: SAMPLE_TOOLS });

      expect(result.fullResponse).toBe('');
      expect(result.toolCalls).toHaveLength(1);
    });
  });

  // H2-4: Auto budget must still build llama.rn's reasoning-budget sampler (thinking_budget_tokens >= 0).
  describe('thinking budget on the tool path', () => {
    it('Auto budget + thinking on sends a finite thinking_budget_tokens below n_predict', async () => {
      useAppStore.getState().updateSettings({ maxTokens: 2048, reasoningBudget: -1 });
      const completion = jest.fn(async (_params: any, _cb: any) => ({}));
      const deps = createMockDeps({ context: { completion }, isThinkingEnabled: true });

      await generateWithToolsImpl(deps, [createUserMessage('Hi')], { tools: SAMPLE_TOOLS });

      const p = completion.mock.calls[0][0];
      expect(p.n_predict).toBe(2048);
      expect(p.thinking_budget_tokens).toBe(1536);
    });
  });

  // ========================================================================
  // Edge: optional callbacks not provided
  // ========================================================================
  describe('optional callbacks', () => {
    it('works without onStream or onComplete', async () => {
      const completion = jest.fn(async (_params: any, cb: any) => {
        cb({ token: 'Hi' });
        return {};
      });
      const deps = createMockDeps({ context: { completion } });

      const result = await generateWithToolsImpl(deps, [createUserMessage('Hi')], {
        tools: SAMPLE_TOOLS,
        // no onStream, no onComplete
      });

      expect(result.fullResponse).toBe('Hi');
    });
  });
});

// Drift guard: the [WIRE-TEMPLATED-PROMPT] survivor check must recognise the LIVE text tool hints.
// Fails if registry.ts hint wording changes in a way the check no longer matches (it matched
// "Tools available" after the hint became "# Tools").
describe('systemInstructionSurvived tracks the live tool hints', () => {
  const { systemInstructionSurvived } = require('../../../src/services/llmToolGeneration');
  const { buildToolSystemPromptHint, buildTextToolCallFormatHint } = require('../../../src/services/tools/registry');

  it.each([['buildToolSystemPromptHint', buildToolSystemPromptHint], ['buildTextToolCallFormatHint', buildTextToolCallFormatHint]])(
    'detects the %s output inside a templated prompt',
    (_name: string, build: (ids: string[]) => string) => {
      const hint = build(['generate_image']);
      expect(hint).not.toBe('');
      expect(systemInstructionSurvived(`<|im_start|>system\nYou are helpful.${hint}<|im_end|>`)).toBe(true);
    },
  );

  it('is false when no system instruction or hint survived', () => {
    expect(systemInstructionSurvived('<|im_start|>user\nhi<|im_end|>')).toBe(false);
  });
});
