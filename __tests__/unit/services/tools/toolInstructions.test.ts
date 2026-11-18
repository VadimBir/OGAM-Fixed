/**
 * Every instruction a model sees about tools, per path, plus schema validity and parse equivalence.
 *
 * Paths (how tools reach the model):
 *  - llama.rn with a Jinja tool template, LiteRT, remote OpenAI / Ollama: the `tools` JSON schema
 *    (getToolsAsOpenAISchema); the chat template renders its own native call syntax.
 *  - llama.rn WITHOUT native tool support: text only (buildToolSystemPromptHint and/or the
 *    format line augmentSystemPromptForTools adds), parsed by parseToolCallsFromText.
 *  - every path: TOOL_BEHAVIOR_GUIDANCE in the system prompt.
 *
 * Falsify: put a <tool_call> example back into a schema description, drop the generate_image example,
 * advertise `sampler` again, or break the text format -> these go red.
 */
import Ajv from 'ajv';
import {
  AVAILABLE_TOOLS,
  getToolsAsOpenAISchema,
  buildToolSystemPromptHint,
  buildTextToolCallFormatHint,
  TEXT_TOOL_CALL_FORMAT,
} from '../../../../src/services/tools/registry';
import {
  runToolLoop,
  parseToolCallsFromText,
  parseRemoteToolArguments,
  TOOL_BEHAVIOR_GUIDANCE,
  ToolLoopContext,
} from '../../../../src/services/generationToolLoop';
import { processDelta, ThinkTagParser } from '../../../../src/services/providers/openAICompatibleStream';
import { llmService } from '../../../../src/services/llm';
import { createMessage } from '../../../utils/factories';

const mockAppState: any = {
  downloadedModels: [],
  activeModelId: null,
  settings: { temperature: 0.7, maxTokens: 1024, topP: 0.9, repeatPenalty: 1.1, liteRTTemperature: 0.7, liteRTTopP: 0.9 },
};
const mockRemoteState: { activeServerId: string | null } = { activeServerId: null };

jest.mock('../../../../src/stores', () => ({
  useChatStore: { getState: () => ({ addMessage: jest.fn(), setStreamingMessage: jest.fn(), setIsThinking: jest.fn() }) },
  useRemoteServerStore: { getState: () => mockRemoteState },
  useAppStore: { getState: () => mockAppState },
}));
jest.mock('../../../../src/services/llm', () => ({
  llmService: {
    generateResponseWithTools: jest.fn(),
    supportsToolCalling: jest.fn(() => false),
    stopGeneration: jest.fn().mockResolvedValue(undefined),
    isModelLoaded: jest.fn(() => true),
    generateToolSelection: jest.fn(),
  },
}));
jest.mock('../../../../src/services/litert', () => ({
  liteRTService: { isModelLoaded: jest.fn(() => false), prepareConversation: jest.fn(), generateRaw: jest.fn() },
}));
jest.mock('../../../../src/services/providers', () => ({
  providerRegistry: { hasProvider: jest.fn(() => false), getProvider: jest.fn(() => null) },
}));
const mockExecuteToolCall = jest.fn();
jest.mock('../../../../src/services/tools/handlers', () => ({
  executeToolCall: (call: unknown) => mockExecuteToolCall(call),
}));
jest.mock('../../../../src/services/tools/extensions', () => ({ getToolExtensions: () => [] }));
jest.mock('../../../../src/services/litertToolSelector', () => ({ selectRelevantTools: jest.fn() }));
jest.mock('../../../../src/services/imageGenerationService', () => ({
  imageGenerationService: { activeImageEngine: () => null },
}));

const ALL_IDS = AVAILABLE_TOOLS.map(t => t.id);
const IMAGE = AVAILABLE_TOOLS.find(t => t.id === 'generate_image')!;
const mockGenerate = llmService.generateResponseWithTools as jest.Mock;
const mockSupportsTools = llmService.supportsToolCalling as jest.Mock;

beforeEach(() => {
  jest.clearAllMocks();
  mockSupportsTools.mockReturnValue(false);
  mockExecuteToolCall.mockResolvedValue({ name: 'generate_image', content: '{"status":"ok"}', durationMs: 1 });
});

describe('tool schemas (native paths: llama Jinja, LiteRT, OpenAI, Ollama)', () => {
  const ajv = new Ajv();
  const schemas = getToolsAsOpenAISchema(ALL_IDS);

  it.each(schemas.map(s => [s.function.name, s]))('%s parameters are a valid JSON Schema', (_n, s: any) => {
    expect(ajv.validateSchema(s.function.parameters)).toBe(true);
    expect(s.function.parameters.type).toBe('object');
    for (const key of s.function.parameters.required) {
      expect(s.function.parameters.properties).toHaveProperty(key);
    }
  });

  it.each(AVAILABLE_TOOLS.map(t => [t.id, t]))('%s example is valid against its own schema', (_id, tool: any) => {
    const schema = getToolsAsOpenAISchema([tool.id])[0].function.parameters;
    expect(tool.example).toBeDefined();
    expect(ajv.validate(schema, tool.example)).toBe(true);
    for (const key of Object.keys(tool.example)) expect(schema.properties).toHaveProperty(key);
  });

  it('each description = when to use + one format-neutral example (no call syntax of any template)', () => {
    for (const s of schemas) {
      const tool = AVAILABLE_TOOLS.find(t => t.name === s.function.name)!;
      expect(s.function.description.endsWith(` Example arguments: ${JSON.stringify(tool.example)}`)).toBe(true);
      expect(s.function.description).not.toMatch(/<\|?tool_call|<function|\[TOOL_CALLS\]|<invoke/);
      for (const p of Object.values<any>(s.function.parameters.properties)) {
        expect(p.description).not.toMatch(/<\|?tool_call|<function/);
      }
    }
  });

  it('generate_image: the exact schema the model receives', () => {
    expect(getToolsAsOpenAISchema(['generate_image'])).toEqual([
      {
        type: 'function',
        function: {
          name: 'generate_image',
          description:
            'Make an image and show it in the chat. Call it at once when the user asks to draw, paint, ' +
            'render or create a picture, photo, logo or artwork: do not ask first and do not describe the ' +
            'image in text instead. Only prompt is needed; leave the other arguments out unless the user ' +
            'asked for them (the app settings are used). After the result, reply with one short sentence.' +
            ' Example arguments: {"prompt":"a red fox in a snowy forest, golden hour"}',
          parameters: {
            type: 'object',
            properties: {
              prompt: { type: 'string', description: 'English description: subject, setting, style, lighting.' },
              width: { type: 'integer', description: 'Optional. Pixels, 64-512, multiple of 64. Default: app setting.' },
              height: { type: 'integer', description: 'Optional. Pixels, 64-512, multiple of 64. Default: app setting.' },
              steps: { type: 'integer', description: 'Optional. 1-50; more = slower, finer. Default: app setting.' },
              guidance: { type: 'number', description: 'Optional. How strictly to follow the prompt, 1-20. Default: app setting.' },
              seed: { type: 'integer', description: 'Optional. Same seed = same image. Default: random.' },
              negative_prompt: { type: 'string', description: 'Optional. Things to leave out, e.g. "blurry, text".' },
            },
            required: ['prompt'],
          },
        },
      },
    ]);
  });
});

describe('shared guidance (every path)', () => {
  it('is short, imperative and names no call syntax', () => {
    expect(TOOL_BEHAVIOR_GUIDANCE).not.toMatch(/<\|?tool_call|\{"name"/);
    expect(TOOL_BEHAVIOR_GUIDANCE).toContain('call it right away');
    expect(TOOL_BEHAVIOR_GUIDANCE).toContain('Do not ask permission');
    expect(TOOL_BEHAVIOR_GUIDANCE).toContain('do not repeat the same call');
    expect(TOOL_BEHAVIOR_GUIDANCE.length).toBeLessThan(700);
  });
});

describe('text path (llama without native tool support)', () => {
  it('the hint carries the one format, once, and a literal call per tool that the parser reads back', () => {
    const hint = buildToolSystemPromptHint(['generate_image', 'web_search']);
    expect(hint.split(TEXT_TOOL_CALL_FORMAT)).toHaveLength(2);
    expect(hint).toContain('prompt (string, required)');
    expect(hint).toContain('width (integer)');
    expect(hint).not.toContain('sampler');
    const parsed = parseToolCallsFromText(hint.slice(hint.indexOf('- generate_image')));
    expect(parsed.toolCalls[0]).toMatchObject({ name: 'generate_image', arguments: IMAGE.example });
  });

  it('the format line prefers generate_image as the example and parses back', () => {
    const line = buildTextToolCallFormatHint(['web_search', 'generate_image']);
    expect(line).toContain(TEXT_TOOL_CALL_FORMAT);
    const example = line.slice(line.indexOf('Example: ') + 'Example: '.length);
    expect(parseToolCallsFromText(example).toolCalls[0]).toMatchObject({ name: 'generate_image', arguments: IMAGE.example });
    expect(buildTextToolCallFormatHint([])).toBe('');
  });

  function ctx(): ToolLoopContext {
    return {
      conversationId: 'conv-1',
      messages: [
        createMessage({ role: 'system', content: 'You are helpful.' } as any),
        createMessage({ content: 'draw a red fox' } as any),
      ],
      enabledToolIds: ['generate_image'],
      isAborted: () => false,
      onThinkingDone: jest.fn(),
      onFinalResponse: jest.fn(),
    };
  }

  it('no native tool calling: the system prompt gets the guidance + the text format with the image example', async () => {
    mockGenerate.mockResolvedValueOnce({ fullResponse: 'Hi.', toolCalls: [] });
    await runToolLoop(ctx());
    const sys = mockGenerate.mock.calls[0][0].find((m: any) => m.role === 'system').content as string;
    expect(sys).toContain(TOOL_BEHAVIOR_GUIDANCE);
    expect(sys).toContain(TEXT_TOOL_CALL_FORMAT);
    expect(sys).toContain('<tool_call>{"name":"generate_image","arguments":{"prompt":"a red fox in a snowy forest, golden hour"}}</tool_call>');
  });

  it('native tool calling: guidance only, no second call syntax beside the template', async () => {
    mockSupportsTools.mockReturnValue(true);
    mockGenerate.mockResolvedValueOnce({ fullResponse: 'Hi.', toolCalls: [] });
    await runToolLoop(ctx());
    const sys = mockGenerate.mock.calls[0][0].find((m: any) => m.role === 'system').content as string;
    expect(sys).toContain(TOOL_BEHAVIOR_GUIDANCE);
    expect(sys).not.toContain('<tool_call>');
  });
});

describe('the same call from a Qwen text reply and from an OpenAI tool_calls delta', () => {
  const QWEN = '<tool_call>{"name":"generate_image","arguments":{"prompt":"a red fox","width":512,"height":384}}</tool_call>';
  const EXPECTED = { name: 'generate_image', arguments: { prompt: 'a red fox', width: 512, height: 384 } };

  function streamOpenAIDeltas() {
    const state: any = {
      fullContent: '', fullReasoningContent: '', reasoningDetails: [],
      toolCalls: [], currentToolCall: null, completeCalled: false, streamErrorOccurred: false,
    };
    const dctx = { thinkingEnabled: false, callbacks: { onToken: jest.fn(), onReasoning: jest.fn() } as any, thinkTagParser: new ThinkTagParser() };
    // Arguments split across fragments, id + name on the first one only (OpenAI streaming shape).
    processDelta({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'generate_image', arguments: '{"prompt":"a red' } }] }, state, dctx);
    processDelta({ tool_calls: [{ index: 0, function: { arguments: ' fox","width":512,' } }] }, state, dctx);
    processDelta({ tool_calls: [{ index: 0, function: { arguments: '"height":384}' } }] }, state, dctx);
    return state.toolCalls as Array<{ id: string; function: { name: string; arguments: string } }>;
  }

  it('parsers agree on name and arguments', () => {
    const fromText = parseToolCallsFromText(QWEN).toolCalls;
    const fromDelta = streamOpenAIDeltas().map(tc => ({
      name: tc.function.name,
      arguments: parseRemoteToolArguments(tc.function.name, tc.function.arguments),
    }));
    expect(fromText).toHaveLength(1);
    expect(fromDelta).toHaveLength(1);
    expect({ name: fromText[0].name, arguments: fromText[0].arguments }).toEqual(EXPECTED);
    expect(fromDelta[0]).toEqual(EXPECTED);
  });

  it('both paths execute generate_image with the same arguments', async () => {
    const base = (): ToolLoopContext => ({
      conversationId: 'conv-1',
      messages: [createMessage({ content: 'draw a red fox' } as any)],
      enabledToolIds: ['generate_image'],
      isAborted: () => false,
      onThinkingDone: jest.fn(),
      onFinalResponse: jest.fn(),
    });
    // llama text path
    mockGenerate
      .mockResolvedValueOnce({ fullResponse: QWEN, toolCalls: [] })
      .mockResolvedValueOnce({ fullResponse: 'Here it is.', toolCalls: [] });
    await runToolLoop(base());
    const textArgs = mockExecuteToolCall.mock.calls[0][0];

    // remote OpenAI path: the provider hands the accumulated delta calls to onComplete
    const calls = streamOpenAIDeltas();
    const generate = jest
      .fn()
      .mockImplementationOnce((_m: any, _o: any, cb: any) => {
        cb.onComplete({ content: '', toolCalls: calls.map(tc => ({ id: tc.id, name: tc.function.name, arguments: tc.function.arguments })) });
        return Promise.resolve();
      })
      .mockImplementationOnce((_m: any, _o: any, cb: any) => {
        cb.onComplete({ content: 'Here it is.' });
        return Promise.resolve();
      });
    const { providerRegistry } = jest.requireMock('../../../../src/services/providers');
    mockRemoteState.activeServerId = 'srv-1';
    providerRegistry.getProvider.mockReturnValue({ capabilities: { supportsThinking: false }, generate });
    await runToolLoop({ ...base(), forceRemote: true });
    mockRemoteState.activeServerId = null;
    const remoteArgs = mockExecuteToolCall.mock.calls[1][0];

    expect(textArgs.name).toBe('generate_image');
    expect(remoteArgs.name).toBe('generate_image');
    expect(remoteArgs.arguments).toEqual(textArgs.arguments);
    expect(textArgs.arguments).toEqual(EXPECTED.arguments);
  }, 10000);
});
