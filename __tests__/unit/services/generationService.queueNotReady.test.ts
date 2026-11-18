/**
 * A queue processor that cannot dispatch yet (chat screen deps not wired) must not lose the user's
 * message: the service restores the batch at the queue head, does not spin re-draining, and the next
 * drain trigger dispatches it once the processor is ready.
 */
import {
  generationService,
  QueuedMessage,
  QUEUE_PROCESSOR_NOT_READY,
} from '../../../src/services/generationService';

const mockImageState: { isGenerating: boolean; conversationId: string | null } = {
  isGenerating: false,
  conversationId: null,
};
const mockImageListeners = new Set<(s: typeof mockImageState) => void>();

jest.mock('../../../src/services/imageGenerationService', () => ({
  imageGenerationService: {
    getState: () => ({ ...mockImageState }),
    subscribe: (listener: (s: typeof mockImageState) => void) => {
      mockImageListeners.add(listener);
      listener({ ...mockImageState });
      return () => mockImageListeners.delete(listener);
    },
  },
}));

function setImage(isGenerating: boolean, conversationId: string | null = null) {
  mockImageState.isGenerating = isGenerating;
  mockImageState.conversationId = conversationId;
  [...mockImageListeners].forEach(l => l({ ...mockImageState }));
}

jest.mock('../../../src/services/llm', () => ({
  llmService: {
    isModelLoaded: jest.fn(() => true),
    stopGeneration: jest.fn().mockResolvedValue(undefined),
  },
}));

jest.mock('../../../src/services/engines', () => ({
  getActiveEngineService: jest.fn(() => ({ stopGeneration: jest.fn().mockResolvedValue(undefined) })),
  prepareActiveConversation: jest.fn().mockResolvedValue(undefined),
  stopAllTextEngines: jest.fn().mockResolvedValue(undefined),
}));

jest.mock('../../../src/services/activeModelService', () => ({
  activeModelService: { getActiveModels: jest.fn(() => ({ text: null, image: null })) },
}));

jest.mock('../../../src/services/providers', () => ({
  providerRegistry: { getProvider: jest.fn(() => null), hasProvider: jest.fn(() => false) },
}));

jest.mock('../../../src/services/generationToolLoop', () => ({ runToolLoop: jest.fn() }));
jest.mock('../../../src/utils/sharePrompt', () => ({ maybeScheduleSharePrompt: jest.fn() }));

const q = (id: string, conversationId: string): QueuedMessage => ({
  id,
  conversationId,
  text: id,
  messageText: id,
});

const svc = generationService as any;

beforeEach(() => {
  jest.useFakeTimers();
  svc.state = {
    isGenerating: false,
    isThinking: false,
    conversationId: null,
    streamingContent: '',
    startTime: null,
    queuedMessages: [],
  };
  svc.listeners.clear();
  svc.abortRequested = false;
  svc.queueProcessor = null;
  svc.queueDispatchInFlight = false;
  svc.dispatchingConversationId = null;
  svc.turnConversationId = null;
  svc.imageWaitUnsubscribe?.();
  mockImageListeners.clear();
  mockImageState.isGenerating = false;
  mockImageState.conversationId = null;
});

afterEach(() => {
  jest.useRealTimers();
});

describe('queue processor not ready', () => {
  it('keeps the queued message (head position, original items) instead of dropping it', async () => {
    const processor = jest.fn().mockResolvedValue(QUEUE_PROCESSOR_NOT_READY);
    generationService.setQueueProcessor(processor);
    generationService.enqueueMessage(q('a1', 'A'));
    generationService.enqueueMessage(q('b1', 'B'));
    generationService.enqueueMessage(q('a2', 'A'));

    generationService.drainQueue();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(processor).toHaveBeenCalledTimes(1);
    expect(processor.mock.calls[0][0]).toMatchObject({ conversationId: 'A', text: 'a1\n\na2' });
    expect(generationService.getState().queuedMessages.map(m => m.id)).toEqual(['a1', 'a2', 'b1']);
  });

  it('does not spin: no automatic re-dispatch after a not-ready result', async () => {
    const processor = jest.fn().mockResolvedValue(QUEUE_PROCESSOR_NOT_READY);
    generationService.setQueueProcessor(processor);
    generationService.enqueueMessage(q('a1', 'A'));

    generationService.drainQueue();
    for (let i = 0; i < 10; i++) await Promise.resolve();
    jest.advanceTimersByTime(1000);
    for (let i = 0; i < 10; i++) await Promise.resolve();

    expect(processor).toHaveBeenCalledTimes(1);
    expect(svc.queueDispatchInFlight).toBe(false);
  });

  it('the next drain dispatches the kept message once the processor is ready', async () => {
    const processor = jest.fn()
      .mockResolvedValueOnce(QUEUE_PROCESSOR_NOT_READY)
      .mockResolvedValueOnce(undefined);
    generationService.setQueueProcessor(processor);
    generationService.enqueueMessage(q('a1', 'A'));

    generationService.drainQueue();
    for (let i = 0; i < 5; i++) await Promise.resolve();
    generationService.drainQueue();
    for (let i = 0; i < 5; i++) await Promise.resolve();

    expect(processor).toHaveBeenCalledTimes(2);
    expect(processor.mock.calls[1][0]).toMatchObject({ id: 'a1', conversationId: 'A' });
    expect(generationService.getState().queuedMessages).toEqual([]);
  });
});
