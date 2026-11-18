import {
  MutableRefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import { AppState } from 'react-native';
import type { NavigationProp } from '@react-navigation/native';
import { callHook, HOOKS } from '../../bootstrap/hookRegistry';
import {
  contextCompactionService,
  generationService,
  imageGenerationService,
  ImageGenerationState,
  QueuedMessage,
} from '../../services';
import { generationSession } from '../../services/generationSession';
import type { RootStackParamList } from '../../navigation/types';
import logger from '../../utils/logger';
import {
  dispatchGenerationFn,
  GenerationDeps,
} from './useChatGenerationActions';

type StartGeneration = (
  conversationId: string,
  text: string,
) => Promise<void>;

/** A missing stream never belongs to a missing conversation. */
export function isStreamingActiveConversation(
  streamingConversationId: string | null,
  activeConversationId: string | null,
): boolean {
  return (
    streamingConversationId !== null &&
    streamingConversationId === activeConversationId
  );
}

export function useChatAudioLifecycle(
  navigation: Pick<NavigationProp<RootStackParamList>, 'addListener'>,
): void {
  useEffect(() => {
    // Leaving is unconditional. audioStop deliberately protects a warm-but-idle engine, which is
    // right at the start of a turn and wrong here: a reply that had not begun playing yet survived the
    // guard and started speaking into a screen the person had already left.
    const unsubscribeBlur = navigation.addListener('blur', () => {
      callHook(HOOKS.audioStopForExit);
    });
    const unsubscribeRemove = navigation.addListener('beforeRemove', () => {
      callHook(HOOKS.audioStopForExit);
    });
    const appStateSubscription = AppState.addEventListener(
      'change',
      nextState => {
        callHook(
          nextState === 'active'
            ? HOOKS.audioOnAppForeground
            : HOOKS.audioOnAppBackground,
        );
      },
    );
    return () => {
      unsubscribeBlur();
      unsubscribeRemove();
      appStateSubscription.remove();
    };
  }, [navigation]);
}

export function useChatRuntimeSubscriptions(
  generationDepsRef: MutableRefObject<GenerationDeps | null>,
  startGenerationRef: MutableRefObject<StartGeneration | null>,
): {
  imageGenState: ImageGenerationState;
  isCompacting: boolean;
  queueCount: number;
  queuedTexts: string[];
} {
  const [imageGenState, setImageGenState] = useState<ImageGenerationState>(
    imageGenerationService.getState(),
  );
  const [isCompacting, setIsCompacting] = useState(false);
  const [queueCount, setQueueCount] = useState(0);
  const [queuedTexts, setQueuedTexts] = useState<string[]>([]);

  useEffect(() => {
    const unsubscribeImage =
      imageGenerationService.subscribe(setImageGenState);
    const unsubscribeCompaction =
      contextCompactionService.subscribeCompacting(setIsCompacting);
    return () => {
      unsubscribeImage();
      unsubscribeCompaction();
    };
  }, []);

  useEffect(
    () =>
      generationService.subscribe(state => {
        setQueueCount(state.queuedMessages.length);
        setQueuedTexts(
          state.queuedMessages.map((message: QueuedMessage) => message.text),
        );
      }),
    [],
  );

  const handleQueuedSend = useCallback(
    async (item: QueuedMessage) => {
      if (!generationDepsRef.current || !startGenerationRef.current) {
        logger.error(
          `[GenerationService] queued message ${item.id} dropped: chat screen not ready`,
        );
        return;
      }
      await dispatchGenerationFn(
        generationDepsRef.current,
        {
          text: item.text,
          attachments: item.attachments,
          conversationId: item.conversationId,
          imageMode: item.imageMode,
          assistantEnabled: item.assistantEnabled,
        },
        startGenerationRef.current,
      );
    },
    [generationDepsRef, startGenerationRef],
  );

  useEffect(() => {
    generationService.setQueueProcessor(handleQueuedSend);
    // Messages queued while no ChatScreen was mounted (the processor is null between unmount and the
    // next mount) stay in the queue; release them now that a processor exists again. drainQueue is a
    // no-op while a text turn runs; an image turn drains on its own completion, so wait for it.
    if (!imageGenerationService.getState().isGenerating) {
      generationService.drainQueue();
    }
    return () => generationService.setQueueProcessor(null);
  }, [handleQueuedSend]);

  return { imageGenState, isCompacting, queueCount, queuedTexts };
}

interface ConversationLifecycleArgs {
  routeConversationId?: string;
  routeProjectId?: string;
  activeConversationId: string | null;
  setActiveConversation: (conversationId: string | null) => void;
  setPendingProjectId: (projectId?: string) => void;
}

export function useChatConversationLifecycle({
  routeConversationId,
  routeProjectId,
  activeConversationId,
  setActiveConversation,
  setPendingProjectId,
}: ConversationLifecycleArgs): void {
  useEffect(() => {
    setActiveConversation(routeConversationId ?? null);
    // The route ID is the owner of this transition; store callbacks are stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeConversationId]);

  useEffect(() => {
    setPendingProjectId(routeProjectId);
  }, [routeProjectId, setPendingProjectId]);

  // Leaving a conversation for an empty chat (new chat on the same screen, delete) must not carry that
  // conversation's character into the next one. Reset the pending character to what the ROUTE asked
  // for (undefined for a plain new chat), never a character picked for the previous conversation.
  const previousActiveConversationIdRef = useRef(activeConversationId);
  useEffect(() => {
    const previous = previousActiveConversationIdRef.current;
    previousActiveConversationIdRef.current = activeConversationId;
    if (previous && !activeConversationId) setPendingProjectId(routeProjectId);
    // Only the active-conversation transition triggers this; routeProjectId is read at that moment.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConversationId]);

  useEffect(() => {
    if (
      generationSession.getConversationId() &&
      !generationSession.isGeneratingFor(activeConversationId)
    ) {
      generationSession.end('conversation-switch');
    }
    // Native conversation isolation is awaited by generationService immediately
    // before a local turn starts. A navigation timer here raced the first Send and
    // could clear too late (context leak) or during prefill (no-op).
  }, [activeConversationId]);
}

export function useChatPresentationLifecycle(
  activeConversationId: string | null,
  messageCount: number,
  isStreamingForThisConversation: boolean,
): number {
  const [animateLastN, setAnimateLastN] = useState(0);
  const lastMessageCountRef = useRef(0);
  const previousStreamingRef = useRef(false);

  useEffect(() => {
    const previous = lastMessageCountRef.current;
    if (messageCount > previous && previous > 0) {
      setAnimateLastN(messageCount - previous);
    }
    lastMessageCountRef.current = messageCount;
  }, [messageCount]);

  useEffect(() => {
    lastMessageCountRef.current = 0;
    setAnimateLastN(0);
  }, [activeConversationId]);

  useEffect(() => {
    const wasStreaming = previousStreamingRef.current;
    previousStreamingRef.current = isStreamingForThisConversation;
    if (
      wasStreaming &&
      !isStreamingForThisConversation &&
      activeConversationId
    ) {
      callHook(HOOKS.audioOnStreamingEnd, activeConversationId);
    }
  }, [isStreamingForThisConversation]); // eslint-disable-line react-hooks/exhaustive-deps

  return animateLastN;
}
