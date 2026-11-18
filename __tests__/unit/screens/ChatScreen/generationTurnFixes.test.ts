/**
 * Send/resend turn fixes in useChatGenerationActions:
 *  - documents attached to EARLIER user turns stay in context (send + resend)
 *  - a compaction summary whose cutoff is gone is not sent
 *  - the KB prompt names search_knowledge_base only when that tool is sent
 *  - the empty-reply card ignores the isSystemInfo compaction notice
 *  - a throw before generation still ends the session and surfaces via the failure path
 *  - resend failures persist a failed turn like send
 *  - a superseded turn's late end() does not clear the newer turn's session
 *
 * Real chatStore + generationSession; generationService.generateResponse is the model boundary.
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
  regenerateResponseFn,
  injectRagContext,
  GenerationDeps,
} from '../../../../src/screens/ChatScreen/useChatGenerationActions';
import { generationService } from '../../../../src/services/generationService';
import { generationSession } from '../../../../src/services/generationSession';
import { llmService } from '../../../../src/services/llm';
import { ragService, retrievalService } from '../../../../src/services/rag';
import { embeddingService } from '../../../../src/services/rag/embedding';
import { reportModelFailure } from '../../../../src/services/modelFailureHandler';
import { ensureReadyOrAlert } from '../../../../src/screens/ChatScreen/modelReadiness';
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
    supportsToolCalling: false,
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
      enabledTools: [],
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

function messagesOf(convId: string): Message[] {
  return useChatStore.getState().getConversationMessages(convId);
}

/** Conversation: [user + document] [assistant] [user]. Returns ids. */
function seedDocConversation() {
  const store = useChatStore.getState();
  const convId = store.createConversation('txt');
  store.addMessage(convId, {
    role: 'user',
    content: 'summarize this',
    attachments: [
      { id: 'doc-1', type: 'document', uri: '/d.txt', fileName: 'notes.txt', textContent: 'SECRET-DOC-BODY' } as any,
    ],
  });
  store.addMessage(convId, { role: 'assistant', content: 'It is about X.' });
  store.addMessage(convId, { role: 'user', content: 'and what else?' });
  const msgs = messagesOf(convId);
  return { convId, firstUserId: msgs[0].id, lastUserId: msgs[2].id, msgs };
}

let generateSpy: jest.SpyInstance;

beforeEach(() => {
  resetStores();
  generationSession._reset();
  jest.clearAllMocks();
  (generationService as any).abortRequested = false;
  generateSpy = jest
    .spyOn(generationService, 'generateResponse')
    .mockResolvedValue(undefined);
  jest.spyOn(llmService, 'getContextDebugInfo').mockRejectedValue(new Error('n/a'));
  jest
    .spyOn(llmService, 'getPerformanceSettings')
    .mockReturnValue({ contextLength: 8192 } as any);
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe('documents stay in context after their turn', () => {
  it('send: an earlier user turn still carries its document text', async () => {
    const { convId, firstUserId } = seedDocConversation();
    await startGenerationFn(makeDeps({ activeConversationId: convId }), {
      setDebugInfo: jest.fn(),
      targetConversationId: convId,
      messageText: 'and what else?',
    });
    const sent: Message[] = generateSpy.mock.calls[0][1];
    const first = sent.find(m => m.id === firstUserId)!;
    expect(first.content).toContain('summarize this');
    expect(first.content).toContain('Attached Document: notes.txt');
    expect(first.content).toContain('SECRET-DOC-BODY');
    // the stored message is untouched (document text is only added to the request)
    expect(messagesOf(convId)[0].content).toBe('summarize this');
  });

  it('resend: an earlier user turn still carries its document text', async () => {
    const { convId, firstUserId, msgs } = seedDocConversation();
    await regenerateResponseFn(makeDeps({ activeConversationId: convId }), {
      setDebugInfo: jest.fn(),
      userMessage: msgs[2],
      recordedKind: 'text',
    });
    const sent: Message[] = generateSpy.mock.calls[0][1];
    expect(sent.find(m => m.id === firstUserId)!.content).toContain('SECRET-DOC-BODY');
  });
});

describe('stale compaction summary', () => {
  it('is not sent when its cutoff message is no longer in the history', async () => {
    const { convId } = seedDocConversation();
    useChatStore
      .getState()
      .updateCompactionState(convId, 'OLD-SUMMARY', 'deleted-message-id');
    await startGenerationFn(makeDeps({ activeConversationId: convId }), {
      setDebugInfo: jest.fn(),
      targetConversationId: convId,
      messageText: 'and what else?',
    });
    const sent: Message[] = generateSpy.mock.calls[0][1];
    expect(JSON.stringify(sent)).not.toContain('OLD-SUMMARY');
    expect(sent.filter(m => m.role !== 'system')).toHaveLength(3);
  });
});

describe('knowledge-base prompt', () => {
  beforeEach(() => {
    jest
      .spyOn(ragService, 'getDocumentsByProject')
      .mockResolvedValue([{ id: 1, name: 'manual.pdf', enabled: true } as any]);
    jest
      .spyOn(ragService, 'searchProject')
      .mockResolvedValue({ chunks: [{ content: 'CHUNK' }] } as any);
    jest.spyOn(retrievalService, 'formatForPrompt').mockReturnValue('RETRIEVED-CHUNKS');
    jest.spyOn(embeddingService, 'isLoaded').mockReturnValue(true);
  });

  it('names search_knowledge_base only when the tool is sent', async () => {
    const withTool = await injectRagContext('p1', 'q', 'BASE', ['search_knowledge_base']);
    expect(withTool).toContain('Use the search_knowledge_base tool');
    expect(withTool).toContain('RETRIEVED-CHUNKS');
  });

  it('injects only the documents + retrieved chunks when the tool is not sent', async () => {
    const withoutTool = await injectRagContext('p1', 'q', 'BASE', ['web_search']);
    expect(withoutTool).not.toContain('search_knowledge_base');
    expect(withoutTool).toContain('manual.pdf');
    expect(withoutTool).toContain('RETRIEVED-CHUNKS');
  });
});

describe('empty-reply card', () => {
  it('still shows when compaction appended an isSystemInfo notice after the user message', async () => {
    const { convId } = seedDocConversation();
    generateSpy.mockImplementation(async () => {
      useChatStore.getState().addMessage(convId, {
        role: 'assistant',
        content: 'Compacted conversation to make room for more messages.',
        isSystemInfo: true,
      });
    });
    await startGenerationFn(makeDeps({ activeConversationId: convId }), {
      setDebugInfo: jest.fn(),
      targetConversationId: convId,
      messageText: 'and what else?',
    });
    expect(reportModelFailure).toHaveBeenCalledWith(
      'text',
      'The model produced no output',
      expect.objectContaining({ title: 'No response' }),
    );
  });

  it('resend shows it too when the model returns nothing', async () => {
    const { convId, msgs } = seedDocConversation();
    await regenerateResponseFn(makeDeps({ activeConversationId: convId }), {
      setDebugInfo: jest.fn(),
      userMessage: msgs[2],
      recordedKind: 'text',
    });
    expect(reportModelFailure).toHaveBeenCalledTimes(1);
  });
});

describe('session always ends; failures surface like send', () => {
  it('send: a throw before generation ends the session and persists a failed turn', async () => {
    const { convId } = seedDocConversation();
    (ensureReadyOrAlert as jest.Mock).mockRejectedValueOnce(new Error('prep exploded'));
    const deps = makeDeps({ activeConversationId: convId });
    await startGenerationFn(deps, {
      setDebugInfo: jest.fn(),
      targetConversationId: convId,
      messageText: 'and what else?',
    });
    expect(generationSession.getConversationId()).toBeNull();
    const last = messagesOf(convId).at(-1)!;
    expect(last).toMatchObject({ role: 'assistant', turnStatus: 'failed' });
    expect(last.content).toContain('prep exploded');
  });

  it('resend: a generation error persists a failed turn and a generic alert, like send', async () => {
    const { convId, msgs } = seedDocConversation();
    generateSpy.mockRejectedValueOnce(new Error('native decode failed'));
    const deps = makeDeps({ activeConversationId: convId });
    await regenerateResponseFn(deps, {
      setDebugInfo: jest.fn(),
      userMessage: msgs[2],
      recordedKind: 'text',
    });
    const last = messagesOf(convId).at(-1)!;
    expect(last).toMatchObject({
      role: 'assistant',
      turnStatus: 'failed',
      content: 'native decode failed',
    });
    expect(deps.setAlertState).toHaveBeenCalledWith(
      expect.objectContaining({
        title: 'Generation Error',
        message: 'The model could not complete this response. The details are shown in the chat.',
      }),
    );
    expect(generationSession.getConversationId()).toBeNull();
  });

  it('resend: an error before generation still ends the session', async () => {
    const { convId, msgs } = seedDocConversation();
    (ensureReadyOrAlert as jest.Mock).mockRejectedValueOnce(new Error('ready check threw'));
    await regenerateResponseFn(makeDeps({ activeConversationId: convId }), {
      setDebugInfo: jest.fn(),
      userMessage: msgs[2],
      recordedKind: 'text',
    });
    expect(generationSession.getConversationId()).toBeNull();
    expect(messagesOf(convId).at(-1)).toMatchObject({ turnStatus: 'failed' });
  });
});

describe('stale end()', () => {
  it("a stopped turn finishing late does not clear the next turn's session", async () => {
    const { convId } = seedDocConversation();
    let release!: () => void;
    generateSpy.mockImplementationOnce(
      () => new Promise<void>(resolve => { release = resolve; }),
    );
    const stopped = startGenerationFn(makeDeps({ activeConversationId: convId }), {
      setDebugInfo: jest.fn(),
      targetConversationId: convId,
      messageText: 'and what else?',
    });
    await new Promise(r => setTimeout(r, 0));
    expect(generationSession.isGeneratingFor(convId)).toBe(true);
    // Stop ends the session; the queue drain then starts the NEXT turn on the same conversation.
    generationSession.end('stopped');
    generationSession.begin(convId);
    release();
    await stopped;
    expect(generationSession.isGeneratingFor(convId)).toBe(true);
  });
});
