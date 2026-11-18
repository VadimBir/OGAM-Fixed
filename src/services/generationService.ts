/** GenerationService - Handles LLM generation independently of UI lifecycle */
import { llmService } from './llm';
import { getActiveEngineService, prepareActiveConversation, stopAllTextEngines } from './engines';
import { activeModelService } from './activeModelService';
import { remoteServerManager } from './remoteServerManager';
import { useAppStore, useChatStore, useRemoteServerStore } from '../stores';
import { Message, GenerationMeta, MediaAttachment } from '../types';
import { runToolLoop } from './generationToolLoop';
import type { ToolResult } from './tools/types';
import { providerRegistry } from './providers';
import { contextCompactionService } from './contextCompaction';
import logger from '../utils/logger';
import { maybeScheduleSharePrompt } from '../utils/sharePrompt';
import { checkProPromptForText } from './proPrompt';
import {
  buildGenerationMetaImpl,
  buildToolLoopHandlersImpl,
  prepareGenerationImpl,
  generateResponseImpl,
  keepShownPartialOnError,
  type GenerationRequest,
  type GenerationWithToolsRequest,
} from './generationServiceHelpers';
import {
  generateRemoteResponseImpl,
  generateRemoteWithToolsImpl,
} from './generationRemoteHelpers';

const SHARE_PROMPT_DELAY_MS = 1500;
/** Lazy: imageGenerationService's import graph (native engines, residency) must not load with this
 *  module; same cycle-safe pattern as tools/imageTool.ts. */
function imageService(): typeof import('./imageGenerationService').imageGenerationService {
  return require('./imageGenerationService').imageGenerationService; // NOSONAR (cycle-safe lazy import)
}
type StreamChunk = string | { content?: string; reasoningContent?: string };

/**
 * A deterministic PROGRAMMING/native-binding error — one that will fail IDENTICALLY on every model,
 * so swapping to a fallback model is pointless and only produces the confusing "the model changed by
 * itself, then also failed" cascade the user sees. These are JS TypeError/ReferenceError-class faults
 * ("undefined is not a function", "x is not an object", "Cannot read property … of undefined"): a code
 * or native-bridge bug, not a model-specific failure like OOM / decode-error / incompatible-backend
 * (those legitimately differ across models and SHOULD fall back). We surface these immediately on the
 * user's chosen model instead of masking them behind a silent swap to a model that fails the same way.
 */
function isDeterministicProgramError(error: unknown): boolean {
  if (error instanceof TypeError || error instanceof ReferenceError) return true;
  const msg = (error instanceof Error ? error.message : String(error ?? '')).toLowerCase();
  return (
    msg.includes('is not a function') ||
    msg.includes('is not a constructor') ||
    msg.includes('is not an object') ||
    /cannot read propert(?:y|ies) .* of (?:undefined|null)/.test(msg) ||
    /(?:undefined|null) is not an object/.test(msg)
  );
}
type FallbackRoute =
  | { kind: 'remote'; serverId: string; id: string; name: string }
  | { kind: 'local'; id: string; name: string };

export interface QueuedMessage {
  id: string; conversationId: string; text: string;
  attachments?: MediaAttachment[]; messageText: string;
  /** The modality the user forced for THIS send (force/disabled/auto). Carried through the queue so a
   *  message the user explicitly forced to image mode is dispatched as image on drain — never re-decided
   *  at 'auto' by resolveTurnKind (#510: a queued force-image send generated as text). */
  imageMode?: 'auto' | 'force' | 'disabled';
  assistantEnabled?: boolean;
}

export interface GenerationState {
  isGenerating: boolean;
  isThinking: boolean;
  conversationId: string | null;
  streamingContent: string;
  startTime: number | null;
  queuedMessages: QueuedMessage[];
}

type GenerationListener = (state: GenerationState) => void;
/** A queue processor returns this when it cannot dispatch yet (chat screen deps not wired). The
 *  service puts the batch back at the queue head instead of losing the user's message, and does not
 *  re-drain on its own (that would spin); the next drain trigger (processor re-registration,
 *  resetState, image end, drainQueue) retries it. */
export const QUEUE_PROCESSOR_NOT_READY = 'not-ready' as const;
type QueueProcessor = (item: QueuedMessage) => Promise<void | typeof QUEUE_PROCESSOR_NOT_READY>;

class GenerationService {
  private state: GenerationState = {
    isGenerating: false, isThinking: false, conversationId: null,
    streamingContent: '', startTime: null, queuedMessages: [],
  };

  private listeners: Set<GenerationListener> = new Set();
  private abortRequested: boolean = false;
  /** Monotonic owner for async generation setup. Stop invalidates work that is between awaits. */
  private generationAttempt: number = 0;
  /** Whether the last/active generation was stopped by the user — lets callers skip a
   *  "no response" retry prompt when the empty result was an intentional abort. */
  wasAborted(): boolean { return this.abortRequested; }
  private pendingStop: Promise<void> | null = null;
  private queueProcessor: QueueProcessor | null = null;
  /** A queued batch has been handed to the processor and its promise has not settled yet. */
  private queueDispatchInFlight: boolean = false;
  /** Conversation of the queued batch whose processor promise is pending (setup before isGenerating). */
  private dispatchingConversationId: string | null = null;
  /** Conversation of the turn inside withModelFallback (set on entry, cleared on exit). Covers a
   *  Stop that lands between fallback attempts, while state.conversationId is already null. */
  private turnConversationId: string | null = null;
  /** Unsubscribe of the one pending "drain when the image generation ends" wait, if any. */
  private imageWaitUnsubscribe: (() => void) | null = null;
  private currentRemoteAbortController: AbortController | null = null;
  private remoteTimeToFirstToken: number | undefined;
  private contextUsage: Pick<GenerationMeta, 'contextPromptTokens' | 'contextWindowTokens' | 'contextEstimate'> | undefined;

  // Token batching — collect tokens and flush to UI at a controlled rate
  private tokenBuffer: string = '';
  private reasoningBuffer: string = '';
  private totalReasoningLength: number = 0;
  private flushTimer: ReturnType<typeof setTimeout> | null = null;

  /** Get the current provider (local or remote) */
  private getCurrentProvider() {
    const activeServerId = useRemoteServerStore.getState().activeServerId;
    if (activeServerId) {
      return providerRegistry.getProvider(activeServerId);
    }
    return providerRegistry.getProvider('local');
  }

  /** Check if using a remote provider */
  private isUsingRemoteProvider(): boolean {
    const { activeServerId } = useRemoteServerStore.getState();
    const hasProvider = activeServerId ? providerRegistry.hasProvider(activeServerId) : false;
    const localLoaded = llmService.isModelLoaded();
    logger.log(`[REMOTE-SM] isUsingRemoteProvider? activeServerId=${activeServerId ?? 'none'} hasProvider=${hasProvider} localLoaded=${localLoaded}`);
    // The explicit remote selection is authoritative. A local resident can still
    // exist briefly while an earlier load drains; it must never steal this turn.
    // A persisted server without its registered provider is not ready yet.
    return !!activeServerId && hasProvider;
  }

  private flushTokenBuffer(): void {
    const store = useChatStore.getState();
    if (this.tokenBuffer) {
      store.appendToStreamingMessage(this.tokenBuffer);
      this.tokenBuffer = '';
    }
    if (this.reasoningBuffer) {
      store.appendToStreamingReasoningContent(this.reasoningBuffer);
      this.reasoningBuffer = '';
    }
    this.flushTimer = null;
  }

  private forceFlushTokens(): void {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    this.flushTokenBuffer();
  }

  private normalizeStreamChunk(data: StreamChunk): { content?: string; reasoningContent?: string } {
    return typeof data === 'string' ? { content: data } : data;
  }

  getState(): GenerationState { return { ...this.state }; }

  isGeneratingFor(conversationId: string): boolean {
    return this.state.isGenerating && this.state.conversationId === conversationId;
  }

  subscribe(listener: GenerationListener): () => void {
    this.listeners.add(listener); listener(this.getState()); return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void { this.listeners.forEach(l => l(this.getState())); }

  private updateState(partial: Partial<GenerationState>): void {
    this.state = { ...this.state, ...partial };
    this.notifyListeners();
  }

  private checkSharePrompt(delayMs = SHARE_PROMPT_DELAY_MS): void {
    const s = useAppStore.getState();
    const count = s.incrementTextGenerationCount();
    maybeScheduleSharePrompt({ variant: 'text', count, hasEngaged: s.hasEngagedSharePrompt, delayMs });
    checkProPromptForText(delayMs);
  }

  private buildToolLoopHandlers() { return buildToolLoopHandlersImpl(this); }
  private buildGenerationMeta(): GenerationMeta { return buildGenerationMetaImpl(this); }
  private async prepareGeneration(conversationId: string): Promise<boolean> {
    return prepareGenerationImpl(this, conversationId);
  }

  /** Generate a response for a conversation. Runs independently of UI lifecycle. */
  // Keep the existing positional callback argument for callers outside ChatScreen.
  // eslint-disable-next-line max-params
  async generateResponse(
    conversationId: string,
    messages: Message[],
    onFirstToken?: () => void,
    contextUsage?: GenerationRequest['contextUsage'],
  ): Promise<void> {
    logger.log(`[REMOTE-SM] generateResponse entry conv=${conversationId} msgs=${messages.length}`);
    return this.withModelFallback(conversationId, messages, async (route, prepared) => {
      const request = { conversationId, messages, onFirstToken, contextUsage,
        prepared, preservePartialOnError: false };
      if (route.kind === 'remote') await generateRemoteResponseImpl(this, request);
      else await generateResponseImpl(this, request);
    });
  }

  private fallbackRoutes(messages: Message[]): FallbackRoute[] {
    const remote = useRemoteServerStore.getState();
    const local = useAppStore.getState();
    const startedRemote = this.isUsingRemoteProvider();
    const selectedId = startedRemote ? remote.activeRemoteTextModelId : local.activeModelId;
    const selectedName = startedRemote
      ? remote.getActiveRemoteTextModel()?.name || selectedId || 'Remote model'
      : local.downloadedModels.find(model => model.id === selectedId)?.name || 'Local model';
    const needsVision = messages.some(message =>
      message.attachments?.some(attachment => attachment.type === 'image'),
    );
    const remoteRoutes = startedRemote
      ? remote.servers.flatMap(server =>
          (remote.discoveredModels[server.id] || [])
            .filter(model =>
              (server.id !== remote.activeServerId || model.id !== selectedId) &&
              (!needsVision || model.capabilities.supportsVision),
            )
            .map(model => ({ kind: 'remote' as const, serverId: server.id, id: model.id, name: model.name })),
        )
      : [];
    const localRoutes = local.downloadedModels
      .filter(model =>
        model.id !== selectedId &&
        (!needsVision || (model.engine === 'litert' ? model.liteRTVision : model.isVisionModel)),
      )
      .sort((a, b) => a.fileSize - b.fileSize)
      .map(model => ({ kind: 'local' as const, id: model.id, name: model.name }));
    return [
      startedRemote
        ? { kind: 'remote', serverId: remote.activeServerId || '', id: selectedId || '', name: selectedName }
        : { kind: 'local', id: selectedId || '', name: selectedName },
      ...remoteRoutes,
      ...localRoutes,
    ];
  }

  private async withModelFallback<T>(
    conversationId: string,
    messages: Message[],
    run: (route: FallbackRoute, prepared: boolean) => Promise<T>,
    canRetry: () => boolean = () => true,
  ): Promise<T | void> {
    this.turnConversationId = conversationId;
    try {
      return await this.runModelFallbackRoutes(conversationId, messages, run, canRetry);
    } finally {
      if (this.turnConversationId === conversationId) this.turnConversationId = null;
    }
  }

  // eslint-disable-next-line max-params
  private async runModelFallbackRoutes<T>(
    conversationId: string,
    messages: Message[],
    run: (route: FallbackRoute, prepared: boolean) => Promise<T>,
    canRetry: () => boolean,
  ): Promise<T | void> {
    const routes = this.fallbackRoutes(messages);
    let failedName = routes[0].name;
    let lastError: unknown;
    let prepared = false;
    for (let index = 0; index < routes.length; index += 1) {
      const route = routes[index];
      if (index > 0) {
        try {
          if (route.kind === 'remote') {
            await remoteServerManager.setActiveRemoteTextModel(route.serverId, route.id);
          } else {
            await activeModelService.loadTextModel(route.id);
            // Loading a local fallback selects it. Clear the prior remote route
            // before an abort can return and leave both routes selected.
            remoteServerManager.clearActiveRemoteTextModel();
            if (this.abortRequested) return;
            await prepareActiveConversation(conversationId);
            if (this.abortRequested) return;
          }
        } catch (error) {
          lastError = error;
          continue;
        }
        if (this.abortRequested) return;
        if (this.state.isGenerating) {
          this.forceFlushTokens();
          useChatStore.getState().resetStreamingSegment();
          this.updateState({ streamingContent: '', isThinking: true, startTime: Date.now() });
        }
        this.tokenBuffer = '';
        this.reasoningBuffer = '';
        this.totalReasoningLength = 0;
        this.remoteTimeToFirstToken = undefined;
        useChatStore.getState().addMessage(conversationId, {
          role: 'tool', toolName: 'model_fallback',
          content: `${failedName} could not answer. Trying ${route.name}.`,
        });
        prepared = this.state.isGenerating;
      }
      try {
        return await run(route, prepared);
      } catch (error) {
        if (this.abortRequested) return;
        if (contextCompactionService.isContextFullError(error)) {
          keepShownPartialOnError(this, conversationId);
          throw error;
        }
        lastError = error;
        failedName = route.name;
        // A deterministic programming/native-binding fault fails identically on every model, so a
        // fallback swap only produces the confusing "model changed by itself, then also failed"
        // cascade and buries the real error behind the LAST route's copy of it. Surface it now, on
        // the user's chosen model, with no swap.
        if (isDeterministicProgramError(error)) {
          logger.error(
            `[GenerationService] ${route.name} hit a deterministic error — surfacing without model fallback: ${error instanceof Error ? error.message : String(error)}`,
          );
          break;
        }
        logger.warn(
          `[GenerationService] ${route.name} failed before model fallback: ${error instanceof Error ? error.message : String(error)}`,
        );
        if (!canRetry()) break;
      }
    }
    if (this.state.isGenerating) keepShownPartialOnError(this, conversationId);
    throw lastError;
  }

  /** Generate a response with tool calling support (LLM → tools → repeat, max 5 iterations). */
  async generateWithTools(
    conversationId: string,
    messages: Message[],
    options: {
      enabledToolIds: string[];
      projectId?: string;
      onToolCallStart?: (name: string, args: Record<string, any>) => void;
      onToolCallComplete?: (name: string, result: ToolResult) => void;
      onFirstToken?: () => void;
      contextUsage?: GenerationRequest['contextUsage'];
      assistantEnabled?: boolean;
    },
  ): Promise<import('./generationToolLoop').ToolLoopOutcome | void> {
    let toolStarted = false;
    const trackedOptions = {
      ...options,
      onToolCallStart: (name: string, args: Record<string, any>) => {
        toolStarted = true;
        options.onToolCallStart?.(name, args);
      },
      onToolCallComplete: (name: string, result: ToolResult) => {
        toolStarted = true;
        options.onToolCallComplete?.(name, result);
      },
    };
    return this.withModelFallback(conversationId, messages, async (route, prepared) => {
      if (route.kind === 'remote') {
        return generateRemoteWithToolsImpl(this, {
          conversationId, messages,
          options: { ...trackedOptions, prepared, preservePartialOnError: false },
        });
      }
      const { enabledToolIds, projectId, contextUsage, assistantEnabled, ...callbacks } = trackedOptions;
      if (!prepared && !(await this.prepareGeneration(conversationId))) return;
      this.contextUsage = contextUsage;
      try {
      const outcome = await runToolLoop({
        conversationId,
        messages,
        enabledToolIds,
        projectId,
        assistantEnabled,
        callbacks,
        ...this.buildToolLoopHandlers(),
      });

      // If aborted, stopGeneration() already handled cleanup.
      logger.log(`[GenService][ToolLoop] runToolLoop done — aborted=${this.abortRequested}, streamingContent=${this.state.streamingContent?.length ?? 0}ch, tokenBuffer=${this.tokenBuffer?.length ?? 0}ch`);
      if (!this.abortRequested) {
        this.forceFlushTokens();
        const store = useChatStore.getState();
        logger.log(`[GenService][ToolLoop] pre-finalize — streamingForConvId=${store.streamingForConversationId}, targetConvId=${conversationId}, streamingMsg=${store.streamingMessage?.length ?? 0}ch`);
        const generationTime = this.state.startTime ? Date.now() - this.state.startTime : undefined;
        store.finalizeStreamingMessage(conversationId, generationTime, this.buildGenerationMeta());
        logger.log(`[GenService][ToolLoop] finalizeStreamingMessage called — convId=${conversationId}`);
        this.checkSharePrompt();
        this.resetState();
      }
      return outcome;
    } catch (error) {
      if (this.abortRequested) return;
      logger.error('[GenerationService] Tool generation error:', error);
      throw error;
      }
    }, () => !toolStarted);
  }

  /**
   * Keep whatever is ALREADY on screen. The source of truth is the store's streamingMessage (what the user
   * sees) — NOT generationService.state.streamingContent, which can be empty for LiteRT or after a state
   * reset while a partial is still rendered. If there's shown content, finalize it (an interrupted partial
   * is still the model's output); only clear when the stream is genuinely empty. Once tokens are shown, they
   * are never discarded (device 2026-07-14: Stop dropped the partial because the decision read the wrong source).
   */
  private keepShownPartialOrClear(generationTimeMs?: number): void {
    this.forceFlushTokens(); // flush any batched tokens to the store so a partial isn't lost to a pending timer
    const store = useChatStore.getState();
    const convId = store.streamingForConversationId;
    // There is NO case to discard shown output. finalizeStreamingMessage persists whatever streamed —
    // content OR reasoning (its own guard drops a genuinely-empty stream) — and resets the streaming state
    // either way, so it's a strict superset of clear. ALWAYS finalize when a conversation is streaming; only
    // clear when there is no streaming conversation at all (nothing was ever shown). The old check looked at
    // streamingMessage ONLY, so a reasoning-only partial (LiteRT still THINKING at stop) was wrongly cleared.
    const shownLen = (store.streamingMessage + store.streamingReasoningContent).trim().length;
    logger.log(`[STOP-SM] keepShownPartialOrClear convId=${convId ?? 'null'} shown=${shownLen}ch → ${convId ? 'finalize' : 'clear'}`);
    if (convId) {
      store.finalizeStreamingMessage(
        convId,
        generationTimeMs,
        this.buildGenerationMeta(),
        'cancelled',
      );
    } else {
      store.clearStreamingMessage();
    }
  }

  /** Stop the current generation. Returns partial content if any was generated. */
  async stopGeneration(): Promise<string> {
    // Set this even between native attempts. Conversation compaction runs after the
    // failed completion has reset `isGenerating`; Stop/Eject must still prevent its retry.
    this.abortRequested = true;
    this.generationAttempt += 1;
    // "I CANNOT STOP IT": a user Stop ends the stopped conversation's queued sends too. Before this, they
    // auto-dispatched ~100 ms later (resetState's timer, the image path's drainQueue, the batch's settle).
    // Cleared BEFORE resetState so its timer finds nothing of this conversation to dispatch.
    this.clearQueueForStoppedConversations();
    if (!this.state.isGenerating) {
      // Settle the visible reply before native stop callbacks can finalize it as a normal completion.
      this.keepShownPartialOrClear();
      // Stop generation on every engine through the registry — no engine enumeration leaked into the caller.
      await stopAllTextEngines();
      const provider = this.getCurrentProvider();
      if (provider) provider.stopGeneration().catch(() => { });
      if (this.currentRemoteAbortController) {
        this.currentRemoteAbortController.abort();
        this.currentRemoteAbortController = null;
      }
      return '';
    }

    this.forceFlushTokens();

    const { startTime } = this.state;
    const generationTime = startTime ? Date.now() - startTime : undefined;
    // Capture the return value BEFORE resetState clears it (prefer the shown text; fall back to state).
    const partialContent = useChatStore.getState().streamingMessage || this.state.streamingContent;

    // Keep whatever is shown (based on the store, not this.state.streamingContent which LiteRT may not fill).
    const hadShownPartial = !!useChatStore.getState().streamingMessage.trim();
    this.keepShownPartialOrClear(generationTime);
    if (hadShownPartial) this.checkSharePrompt();

    this.resetState();

    // Stop both local and remote
    if (this.isUsingRemoteProvider()) {
      // Abort the provider's XHR so the server connection is closed immediately
      const provider = this.getCurrentProvider();
      if (provider) provider.stopGeneration().catch(() => { });
      if (this.currentRemoteAbortController) {
        this.currentRemoteAbortController.abort();
        this.currentRemoteAbortController = null;
      }
      return partialContent;
    }

    // Stop the native completion after we've already updated UI state,
    // so the user sees immediate feedback. Store the promise so new
    // generations can drain it before starting.
    const engine = getActiveEngineService();
    this.pendingStop = (engine?.stopGeneration() ?? Promise.resolve())
      .catch(() => { })
      .finally(() => { this.pendingStop = null; });

    return partialContent;
  }

  /** Generate a response using a remote provider */
  // eslint-disable-next-line max-params
  async generateRemoteResponse(
    conversationId: string,
    messages: Message[],
    onFirstToken?: () => void,
    contextUsage?: GenerationRequest['contextUsage'],
  ): Promise<void> {
    return generateRemoteResponseImpl(this, { conversationId, messages, onFirstToken, contextUsage });
  }

  /** Generate a response with tools using a remote provider */
  async generateRemoteWithTools(
    conversationId: string,
    messages: Message[],
    options: GenerationWithToolsRequest['options'],
  ): Promise<void> {
    return generateRemoteWithToolsImpl(this, { conversationId, messages, options });
  }

  enqueueMessage(entry: QueuedMessage): void {
    this.state = { ...this.state, queuedMessages: [...this.state.queuedMessages, entry] };
    this.notifyListeners();
  }

  removeFromQueue(id: string): void {
    this.state = { ...this.state, queuedMessages: this.state.queuedMessages.filter(m => m.id !== id) };
    this.notifyListeners();
  }

  clearQueue(): void { this.state = { ...this.state, queuedMessages: [] }; this.notifyListeners(); }

  setQueueProcessor(processor: QueueProcessor | null): void { this.queueProcessor = processor; }

  /**
   * Process queued messages now. Text generation drains its own queue on
   * completion, but image generation finishes outside this service, so the
   * image path calls this to release messages that queued behind it. No-op if a
   * text generation is currently running.
   */
  drainQueue(): void {
    if (this.state.isGenerating) return;
    this.processNextInQueue();
  }

  /**
   * Dispatch the next batch: the head item plus every queued item for the SAME conversation (coalesced
   * into one turn). Items for other conversations stay queued, in order, for the next drain: merging
   * them would post their text into the head item's conversation. Without a processor (no ChatScreen
   * mounted) nothing is taken off the queue; the next setQueueProcessor owner drains it.
   * One batch at a time: while a batch's processor promise is pending, later triggers (resetState's
   * timer, an image turn's drainQueue) are deferred to that promise settling, so two batches can never
   * start concurrently; on settle, whatever is still queued drains if no text turn is running.
   */
  private processNextInQueue(): void {
    if (this.state.queuedMessages.length === 0 || !this.queueProcessor) return;
    if (this.queueDispatchInFlight) return;
    // Never start a queued turn while an image generation holds the RAM (text model load + diffusion on a
    // 12 GB phone = OOM). Wait for it to end, then drain. drainQueue/resetState's timer/the batch settle
    // all funnel here, so this is the one gate.
    if (imageService().getState().isGenerating) {
      this.drainWhenImageIdle();
      return;
    }
    const queued = this.state.queuedMessages;
    const headConversationId = queued[0].conversationId;
    const all = queued.filter(m => m.conversationId === headConversationId);
    const rest = queued.filter(m => m.conversationId !== headConversationId);
    this.state = { ...this.state, queuedMessages: rest };
    this.notifyListeners();
    const combined: QueuedMessage = all.length === 1 ? all[0] : {
      id: all[0].id, conversationId: headConversationId,
      text: all.map(m => m.text).join('\n\n'),
      attachments: all.flatMap(m => m.attachments || []),
      messageText: all.map(m => m.messageText).join('\n\n'),
      // If ANY coalesced send forced image mode, the combined dispatch must force image too — the
      // user's explicit force must never be dropped by the merge (mirror of the single-message carry).
      imageMode: all.some(m => m.imageMode === 'force') ? 'force' : all[0].imageMode,
      assistantEnabled: all.some(m => m.assistantEnabled),
    };
    this.queueDispatchInFlight = true;
    this.dispatchingConversationId = headConversationId;
    let requeued = false;
    this.queueProcessor(combined)
      .then(result => {
        if (result !== QUEUE_PROCESSOR_NOT_READY) return;
        // Not dispatched: restore the original items at the head (still visible in the queue UI).
        requeued = true;
        this.state = { ...this.state, queuedMessages: [...all, ...this.state.queuedMessages] };
        this.notifyListeners();
        logger.warn(`[GenerationService] queue processor not ready; kept ${all.length} queued message(s) for ${headConversationId}`);
      })
      .catch(e => { logger.error('[GenerationService] Queue processor error:', e); })
      .finally(() => {
        this.queueDispatchInFlight = false;
        this.dispatchingConversationId = null;
        // Items left for other conversations (or queued during this batch) must not wait for an
        // unrelated future turn: a batch that ended without a text turn never reaches resetState.
        // A not-ready batch does NOT re-drain here: the same processor would refuse again at once.
        if (!requeued && this.state.queuedMessages.length > 0) this.drainQueue();
      });
  }

  /** Subscribe once; drain on the first image state that is no longer in flight. */
  private drainWhenImageIdle(): void {
    if (this.imageWaitUnsubscribe) return;
    let settled = false;
    let unsubscribe: (() => void) | null = null;
    const release = () => {
      settled = true;
      unsubscribe?.();
      this.imageWaitUnsubscribe = null;
    };
    unsubscribe = imageService().subscribe(state => {
      if (settled || state.isGenerating) return;
      release();
      this.drainQueue();
    });
    // subscribe() reports the current state synchronously; the image may have ended in between.
    if (settled) unsubscribe();
    else this.imageWaitUnsubscribe = release;
  }

  /** Drop queued sends of every conversation this Stop ends: the text turn, the queued batch being
   *  dispatched, the turn between fallback attempts, and the in-flight image generation. */
  private clearQueueForStoppedConversations(): void {
    if (this.state.queuedMessages.length === 0) return;
    const image = imageService().getState();
    const stopped = new Set(
      [
        this.state.conversationId,
        this.dispatchingConversationId,
        this.turnConversationId,
        image.isGenerating ? image.conversationId : null,
      ].filter((id): id is string => !!id),
    );
    if (stopped.size === 0) return;
    const kept = this.state.queuedMessages.filter(m => !stopped.has(m.conversationId));
    if (kept.length === this.state.queuedMessages.length) return;
    logger.log(
      `[GenerationService] stop: dropped ${this.state.queuedMessages.length - kept.length} queued message(s) for [${[...stopped].join(',')}]`,
    );
    this.state = { ...this.state, queuedMessages: kept };
    this.notifyListeners();
  }

  private resetState(): void {
    const hasQueuedItems = this.state.queuedMessages.length > 0;
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    this.tokenBuffer = '';
    this.reasoningBuffer = '';
    this.totalReasoningLength = 0;
    this.remoteTimeToFirstToken = undefined;
    this.contextUsage = undefined;
    this.updateState({
      isGenerating: false,
      isThinking: false,
      conversationId: null,
      streamingContent: '',
      startTime: null,
    });
    if (hasQueuedItems) {
      setTimeout(() => this.processNextInQueue(), 100);
    }
  }
}

export const generationService = new GenerationService();
