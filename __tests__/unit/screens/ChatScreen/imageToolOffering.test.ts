/**
 * generate_image offering + history fixes in useChatGenerationActions:
 *  - H1-B1 a local llama whose GGUF template reports NO native tool support is still offered the
 *    tools, via the text-hint path (previously the tool list was emptied, the hint unreachable)
 *  - H1-B4 the reason generate_image was dropped from a turn is named
 *  - H4-2 the app's "Generated image for:" caption is not fed back to the model as its own speech
 *  - H4-3 strict OpenAI order: assistant(tool_calls) is followed directly by its tool result
 */
jest.mock('../../../../src/services/modelResidency', () => ({
  modelResidencyManager: {
    reclaimSttForGeneration: jest.fn(() => Promise.resolve()),
    getResidents: jest.fn(() => []),
  },
}));
jest.mock('../../../../src/screens/ChatScreen/modelReadiness', () => ({
  ensureReadyOrAlert: jest.fn(() => Promise.resolve(true)),
}));
jest.mock('../../../../src/services/modelFailureHandler', () => ({
  reportModelFailure: jest.fn(),
}));

import {
  startGenerationFn,
  withoutImageCaptions,
  droppedImageToolReason,
  GenerationDeps,
} from '../../../../src/screens/ChatScreen/useChatGenerationActions';
import { generationService } from '../../../../src/services/generationService';
import { generationSession } from '../../../../src/services/generationSession';
import { llmService } from '../../../../src/services/llm';
import { imageGenerationService } from '../../../../src/services/imageGenerationService';
import { buildOpenAIMessagesImpl } from '../../../../src/services/providers/openAIMessageBuilder';
import { useChatStore } from '../../../../src/stores/chatStore';
import type { Message } from '../../../../src/types';
import { resetStores } from '../../../utils/testHelpers';

function makeDeps(overrides: Partial<GenerationDeps> = {}): GenerationDeps {
  const store = useChatStore.getState();
  return {
    activeModelId: 'txt',
    activeModel: { id: 'txt', name: 'Txt', engine: 'llama' } as any,
    activeModelInfo: { isRemote: false, model: null, modelId: 'txt', modelName: 'Txt' },
    hasActiveModel: true,
    hasTextModel: true,
    supportsToolCalling: true,
    activeConversationId: null,
    activeConversation: null,
    activeProject: null,
    activeImageModel: null,
    imageModelLoaded: false,
    isStreaming: false,
    isGeneratingImage: false,
    imageGenState: { isGenerating: false } as any,
    settings: {
      showGenerationDetails: false,
      imageGenerationMode: 'auto',
      autoDetectMethod: 'pattern',
      enabledTools: ['generate_image'],
    },
    downloadedModels: [],
    setAlertState: jest.fn(),
    setIsClassifying: jest.fn(),
    setAppImageGenerationStatus: jest.fn(),
    setAppIsGeneratingImage: jest.fn(),
    addMessage: (convId, msg) => store.addMessage(convId, msg),
    clearStreamingMessage: () => store.clearStreamingMessage(),
    deleteConversation: convId => store.deleteConversation(convId),
    setActiveConversation: convId => store.setActiveConversation(convId),
    removeImagesByConversationId: () => [],
    navigation: { navigate: jest.fn(), goBack: jest.fn() },
    ensureModelLoaded: jest.fn(() => Promise.resolve('ready' as any)),
    ensureTextModelForChat: jest.fn(() => Promise.resolve(true)),
    createConversation: (modelId, title, projectId) =>
      store.createConversation(modelId, title, projectId),
    ...overrides,
  } as GenerationDeps;
}

let toolsSpy: jest.SpyInstance;
let plainSpy: jest.SpyInstance;

beforeEach(() => {
  resetStores();
  generationSession._reset();
  jest.clearAllMocks();
  (generationService as any).abortRequested = false;
  toolsSpy = jest.spyOn(generationService, 'generateWithTools').mockResolvedValue(undefined as any);
  plainSpy = jest.spyOn(generationService, 'generateResponse').mockResolvedValue(undefined);
  jest.spyOn(llmService, 'getContextDebugInfo').mockRejectedValue(new Error('n/a'));
  jest.spyOn(llmService, 'getPerformanceSettings').mockReturnValue({ contextLength: 8192 } as any);
  // Template without tool support: the native capability is false.
  jest.spyOn(llmService, 'isModelLoaded').mockReturnValue(true);
  jest.spyOn(llmService, 'supportsToolCalling').mockReturnValue(false);
  jest.spyOn(imageGenerationService, 'activeImageEngine').mockReturnValue({ kind: 'local' } as any);
});

afterEach(() => {
  jest.restoreAllMocks();
});

async function send(deps: GenerationDeps) {
  const store = useChatStore.getState();
  const convId = store.createConversation('txt');
  store.addMessage(convId, { role: 'user', content: 'draw a cat' });
  await startGenerationFn({ ...deps, activeConversationId: convId }, {
    setDebugInfo: jest.fn(),
    targetConversationId: convId,
    messageText: 'draw a cat',
  });
  return convId;
}

describe('H1-B1: local llama without native tool support still gets generate_image', () => {
  it('offers the tool and injects the text hint into the system prompt', async () => {
    await send(makeDeps());
    expect(plainSpy).not.toHaveBeenCalled();
    expect(toolsSpy).toHaveBeenCalledTimes(1);
    const [, msgs, opts] = toolsSpy.mock.calls[0];
    expect(opts.enabledToolIds).toEqual(['generate_image']);
    expect(msgs[0].role).toBe('system');
    expect(msgs[0].content).toContain('generate_image');
  });

  it('still honours the UI gate (supportsToolCalling === false sends no tools)', async () => {
    await send(makeDeps({ supportsToolCalling: false }));
    expect(toolsSpy).not.toHaveBeenCalled();
    expect(plainSpy).toHaveBeenCalledTimes(1);
  });

  it('drops generate_image (and says why) when no image model is selected', async () => {
    (imageGenerationService.activeImageEngine as jest.Mock).mockReturnValue(null);
    await send(makeDeps());
    expect(toolsSpy).not.toHaveBeenCalled();
  });
});

describe('H1-B4: droppedImageToolReason', () => {
  const g = { uiGate: true, localToolSupport: false, isRemoteText: false, localLlama: true };
  it('null when not toggled or when sent', () => {
    expect(droppedImageToolReason([], [], g)).toBeNull();
    expect(droppedImageToolReason(['generate_image'], ['generate_image'], g)).toBeNull();
  });
  it('names the gate, the missing engine, or the missing image model', () => {
    expect(droppedImageToolReason(['generate_image'], [], { ...g, uiGate: false })).toMatch(/supportsToolCalling=false/);
    expect(droppedImageToolReason(['generate_image'], [], { ...g, localLlama: false })).toMatch(/no tool-capable text engine/);
    expect(droppedImageToolReason(['generate_image'], [], g)).toMatch(/no image model selected/);
  });
});

const user = (id: string, content: string) => ({ id, role: 'user', content, timestamp: 1 }) as Message;
const callMsg = { id: 'a1', role: 'assistant', content: '', timestamp: 2,
  toolCalls: [{ id: 'call-1', name: 'generate_image', arguments: { prompt: 'cat' } }] } as Message;
const caption = { id: 'a2', role: 'assistant', content: 'Generated image for: "cat"', timestamp: 3,
  attachments: [{ id: 'img', type: 'image', uri: 'file:///img.png' }] } as Message;
const toolResult = { id: 't1', role: 'tool', content: 'ok', toolCallId: 'call-1', toolName: 'generate_image', timestamp: 4 } as Message;
const answer = { id: 'a3', role: 'assistant', content: 'Here is your cat.', timestamp: 5 } as Message;

describe('H4-2: withoutImageCaptions', () => {
  it('drops the caption inside a tool turn, keeping tool call -> tool result adjacent', () => {
    const out = withoutImageCaptions([user('u1', 'draw'), callMsg, caption, toolResult, answer]);
    expect(out.map(m => m.id)).toEqual(['u1', 'a1', 't1', 'a3']);
  });
  it('empties a standalone caption (user -> caption) so role alternation survives', () => {
    const out = withoutImageCaptions([user('u1', 'draw'), caption, user('u2', 'again')]);
    expect(out.map(m => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(out[1].content).toBe('');
    expect(out[1].attachments).toBeUndefined();
  });
  it('leaves model speech alone', () => {
    const list = [user('u1', 'hi'), answer];
    expect(withoutImageCaptions(list)).toEqual(list);
  });
  it('send path: the caption never reaches the model', async () => {
    const store = useChatStore.getState();
    const convId = store.createConversation('txt');
    store.addMessage(convId, { role: 'user', content: 'draw a cat' });
    store.addMessage(convId, { role: 'assistant', content: 'Generated image for: "a cat"' });
    store.addMessage(convId, { role: 'user', content: 'now a dog' });
    await startGenerationFn({ ...makeDeps(), activeConversationId: convId }, {
      setDebugInfo: jest.fn(),
      targetConversationId: convId,
      messageText: 'now a dog',
    });
    const sent: Message[] = toolsSpy.mock.calls[0][1];
    expect(sent.some(m => /Generated image for:/.test(m.content))).toBe(false);
  });
});

describe('H4-3: OpenAI message order with an image caption inside a tool turn', () => {
  it('assistant(tool_calls) is followed directly by its tool message; the caption is dropped', async () => {
    const out = await buildOpenAIMessagesImpl(
      [user('u1', 'draw'), callMsg, caption, toolResult, answer],
      { systemPrompt: 'SYS' } as any,
      { supportsVision: false, supportsToolCalling: true } as any,
    );
    expect(out.map(m => m.role)).toEqual(['system', 'user', 'assistant', 'tool', 'assistant']);
    expect((out[2] as any).tool_calls).toHaveLength(1);
    expect(out.some(m => typeof m.content === 'string' && /Generated image for:/.test(m.content))).toBe(false);
    expect(out[4].content).toBe('Here is your cat.');
  });
  it('non-caption assistant text owed a tool result is moved after the result', async () => {
    const note = { id: 'n', role: 'assistant', content: 'working on it', timestamp: 3 } as Message;
    const out = await buildOpenAIMessagesImpl(
      [user('u1', 'draw'), callMsg, note, toolResult],
      { systemPrompt: 'SYS' } as any,
      { supportsVision: false, supportsToolCalling: true } as any,
    );
    expect(out.map(m => m.role)).toEqual(['system', 'user', 'assistant', 'tool', 'assistant']);
    expect(out[4].content).toBe('working on it');
  });
});
