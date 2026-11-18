/**
 * ChatScreen wiring fixes:
 *  - picking a character with an active conversation does not stash it for the next new chat
 *  - leaving a conversation for an empty chat resets the pending character to the route's
 *  - the Tools badge counts only tools a turn would actually send (availableToolIds)
 *  - mounting the screen (queue processor set again) drains messages queued while unmounted
 */
jest.mock('../../../../src/services/tools/availability', () => ({
  availableToolIds: jest.fn((ids: readonly string[]) => ids.filter(id => id !== 'generate_image')),
  isToolAvailable: jest.fn((id: string) => id !== 'generate_image'),
}));

import { renderHook } from '@testing-library/react-native';
import { useChatScreenActions } from '../../../../src/screens/ChatScreen/useChatScreenActions';
import {
  useChatConversationLifecycle,
  useChatRuntimeSubscriptions,
} from '../../../../src/screens/ChatScreen/useChatScreenLifecycle';
import { generationService, imageGenerationService } from '../../../../src/services';
import { QUEUE_PROCESSOR_NOT_READY } from '../../../../src/services/generationService';

function actionArgs(overrides: Record<string, any> = {}): any {
  return {
    generationDeps: {},
    generationDepsRef: { current: null },
    modelDeps: {},
    activeModelInfo: { modelId: null, isRemote: false },
    supportsToolCalling: true,
    activeModel: undefined,
    settings: { enabledTools: ['web_search', 'generate_image', 'read_url'] },
    loadedSettings: {},
    loadedTextModelId: null,
    isModelLoading: false,
    pendingMessageRef: { current: null },
    startGenerationRef: { current: null },
    setDebugInfo: jest.fn(),
    setAlertState: jest.fn(),
    activeConversationId: null,
    activeConversation: undefined,
    hasActiveModel: true,
    deleteMessagesAfter: jest.fn(),
    updateMessageContent: jest.fn(),
    setConversationProject: jest.fn(),
    setPendingProjectId: jest.fn(),
    setShowProjectSelector: jest.fn(),
    activeImageModel: null,
    viewerImageUri: null,
    setViewerImageUri: jest.fn(),
    ...overrides,
  };
}

const project = { id: 'char-1', name: 'Ava' } as any;

describe('handleSelectProject', () => {
  it('with an active conversation: files the character on that conversation only', () => {
    const args = actionArgs({ activeConversationId: 'conv-1' });
    const { result } = renderHook(() => useChatScreenActions(args));
    result.current.handleSelectProject(project);
    expect(args.setConversationProject).toHaveBeenCalledWith('conv-1', 'char-1');
    expect(args.setPendingProjectId).not.toHaveBeenCalled();
  });

  it('on an empty chat: stashes it as the pending character for the next conversation', () => {
    const args = actionArgs({ activeConversationId: null });
    const { result } = renderHook(() => useChatScreenActions(args));
    result.current.handleSelectProject(project);
    expect(args.setPendingProjectId).toHaveBeenCalledWith('char-1');
    expect(args.setConversationProject).not.toHaveBeenCalled();
    expect(args.setShowProjectSelector).toHaveBeenCalledWith(false);
  });
});

describe('Tools badge count', () => {
  it('counts only available tools', () => {
    const { result } = renderHook(() => useChatScreenActions(actionArgs()));
    expect(result.current.enabledTools).toEqual(['web_search', 'read_url']);
  });

  it('is empty when the model cannot call tools', () => {
    const { result } = renderHook(() =>
      useChatScreenActions(actionArgs({ supportsToolCalling: false })),
    );
    expect(result.current.enabledTools).toEqual([]);
  });
});

describe('pending character reset on leaving a conversation', () => {
  it('resets to the route character when the active conversation becomes null', () => {
    const setPendingProjectId = jest.fn();
    const setActiveConversation = jest.fn();
    const { rerender } = renderHook(
      (props: { active: string | null }) =>
        useChatConversationLifecycle({
          routeConversationId: undefined,
          routeProjectId: undefined,
          activeConversationId: props.active,
          setActiveConversation,
          setPendingProjectId,
        }),
      { initialProps: { active: 'conv-1' } },
    );
    setPendingProjectId.mockClear();
    rerender({ active: null });
    expect(setPendingProjectId).toHaveBeenCalledTimes(1);
    expect(setPendingProjectId).toHaveBeenCalledWith(undefined);
  });

  it('does not reset when moving into a conversation', () => {
    const setPendingProjectId = jest.fn();
    const { rerender } = renderHook(
      (props: { active: string | null }) =>
        useChatConversationLifecycle({
          routeConversationId: undefined,
          routeProjectId: 'route-proj',
          activeConversationId: props.active,
          setActiveConversation: jest.fn(),
          setPendingProjectId,
        }),
      { initialProps: { active: null } },
    );
    setPendingProjectId.mockClear();
    rerender({ active: 'conv-new' });
    expect(setPendingProjectId).not.toHaveBeenCalled();
  });
});

describe('queue drain when the screen mounts again', () => {
  afterEach(() => jest.restoreAllMocks());

  it('drains messages queued while no processor was set', () => {
    const drain = jest.spyOn(generationService, 'drainQueue').mockImplementation(() => {});
    jest
      .spyOn(imageGenerationService, 'getState')
      .mockReturnValue({ isGenerating: false } as any);
    renderHook(() =>
      useChatRuntimeSubscriptions({ current: null }, { current: null }),
    );
    expect(drain).toHaveBeenCalled();
  });

  it('drains during a running image generation too: drainQueue itself waits for the image', () => {
    const drain = jest.spyOn(generationService, 'drainQueue').mockImplementation(() => {});
    jest
      .spyOn(imageGenerationService, 'getState')
      .mockReturnValue({ isGenerating: true } as any);
    renderHook(() =>
      useChatRuntimeSubscriptions({ current: null }, { current: null }),
    );
    // The image gate lives in generationService.processNextInQueue (see generationService.queueStopImage.test).
    expect(drain).toHaveBeenCalled();
  });

  it('screen not ready: the registered processor reports NOT_READY (service keeps the message) instead of dropping it', async () => {
    jest.spyOn(generationService, 'drainQueue').mockImplementation(() => {});
    const setProcessor = jest.spyOn(generationService, 'setQueueProcessor');
    renderHook(() =>
      useChatRuntimeSubscriptions({ current: null }, { current: null }),
    );
    const processor = setProcessor.mock.calls.find(c => c[0] !== null)?.[0];
    expect(processor).toBeTruthy();
    await expect(
      processor!({ id: 'q1', conversationId: 'c1', text: 'hi', messageText: 'hi' }),
    ).resolves.toBe(QUEUE_PROCESSOR_NOT_READY);
  });
});
