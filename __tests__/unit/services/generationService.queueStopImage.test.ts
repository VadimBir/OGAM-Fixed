/**
 * Queue dispatch rules in generationService:
 *  - a user Stop drops the stopped conversation's queued sends (no auto-dispatch after Stop),
 *    other conversations' sends stay queued and still drain,
 *  - a normal turn completion still drains the queue,
 *  - no queued turn starts while an image generation is in flight; it drains when the image ends.
 */
import { generationService, QueuedMessage } from '../../../src/services/generationService';

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

describe('Stop clears the stopped conversation queue', () => {
  it('drops queued sends of the stopped text turn and never dispatches them', async () => {
    const processor = jest.fn().mockResolvedValue(undefined);
    generationService.setQueueProcessor(processor);
    svc.state = { ...svc.state, isGenerating: true, conversationId: 'A' };
    generationService.enqueueMessage(q('a1', 'A'));
    generationService.enqueueMessage(q('a2', 'A'));

    await generationService.stopGeneration();

    expect(generationService.getState().queuedMessages).toEqual([]);
    jest.advanceTimersByTime(500);
    expect(processor).not.toHaveBeenCalled();
  });

  it('keeps other conversations queued; they still drain after the Stop', async () => {
    const processor = jest.fn().mockResolvedValue(undefined);
    generationService.setQueueProcessor(processor);
    svc.state = { ...svc.state, isGenerating: true, conversationId: 'A' };
    generationService.enqueueMessage(q('a1', 'A'));
    generationService.enqueueMessage(q('b1', 'B'));

    await generationService.stopGeneration();
    jest.advanceTimersByTime(100);

    expect(processor).toHaveBeenCalledTimes(1);
    expect(processor.mock.calls[0][0]).toMatchObject({ id: 'b1', conversationId: 'B' });
  });

  it('drops the queue of a batch that is still being dispatched (before isGenerating)', async () => {
    let finishBatch: () => void = () => {};
    const processor = jest
      .fn()
      .mockImplementationOnce(() => new Promise<void>(r => { finishBatch = r; }))
      .mockResolvedValue(undefined);
    generationService.setQueueProcessor(processor);
    generationService.enqueueMessage(q('a1', 'A'));
    generationService.drainQueue(); // a1 dispatched, promise pending
    generationService.enqueueMessage(q('a2', 'A'));

    await generationService.stopGeneration();
    finishBatch();
    await Promise.resolve();
    await Promise.resolve();
    jest.advanceTimersByTime(500);

    expect(processor).toHaveBeenCalledTimes(1);
    expect(generationService.getState().queuedMessages).toEqual([]);
  });

  it('drops the queue of the conversation whose image generation is stopped', async () => {
    const processor = jest.fn().mockResolvedValue(undefined);
    generationService.setQueueProcessor(processor);
    mockImageState.isGenerating = true;
    mockImageState.conversationId = 'A';
    generationService.enqueueMessage(q('a1', 'A'));

    await generationService.stopGeneration();
    setImage(false); // the image cancel lands
    generationService.drainQueue(); // the image path's own drain after generateImage resolves
    jest.advanceTimersByTime(500);

    expect(processor).not.toHaveBeenCalled();
    expect(generationService.getState().queuedMessages).toEqual([]);
  });
});

describe('normal completion still drains', () => {
  it('dispatches queued sends ~100 ms after the turn settles', () => {
    const processor = jest.fn().mockResolvedValue(undefined);
    generationService.setQueueProcessor(processor);
    svc.state = { ...svc.state, isGenerating: true, conversationId: 'A' };
    generationService.enqueueMessage(q('a1', 'A'));

    svc.resetState(); // the normal end-of-turn path
    expect(processor).not.toHaveBeenCalled();
    jest.advanceTimersByTime(100);

    expect(processor).toHaveBeenCalledTimes(1);
    expect(processor.mock.calls[0][0]).toMatchObject({ id: 'a1' });
  });
});

describe('queued dispatch waits for an in-flight image generation', () => {
  it('does not dispatch while the image runs, then drains once it ends', () => {
    const processor = jest.fn().mockResolvedValue(undefined);
    generationService.setQueueProcessor(processor);
    setImage(true, 'IMG');
    generationService.enqueueMessage(q('t1', 'A'));

    generationService.drainQueue();
    svc.resetState();
    jest.advanceTimersByTime(500);
    expect(processor).not.toHaveBeenCalled();
    expect(generationService.getState().queuedMessages).toHaveLength(1);
    expect(mockImageListeners.size).toBe(1); // one wait, not one per trigger

    setImage(false);

    expect(processor).toHaveBeenCalledTimes(1);
    expect(processor.mock.calls[0][0]).toMatchObject({ id: 't1' });
    expect(mockImageListeners.size).toBe(0);
  });

  it('dispatches at once when no image generation runs', () => {
    const processor = jest.fn().mockResolvedValue(undefined);
    generationService.setQueueProcessor(processor);
    generationService.enqueueMessage(q('t1', 'A'));

    generationService.drainQueue();

    expect(processor).toHaveBeenCalledTimes(1);
    expect(mockImageListeners.size).toBe(0);
  });
});
