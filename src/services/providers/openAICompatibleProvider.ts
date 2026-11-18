/**
 * OpenAI-Compatible Provider
 *
 * Provider implementation for OpenAI-compatible servers (Ollama, LM Studio, etc.)
 * Handles model discovery, streaming generation, vision, and tool calling.
 */

import { Message } from '../../types';
import { openRouterReasoningPayload, reasoningBudgetPayload, REASONING_BUDGET_AUTO } from '@offgrid/models';
import type {
  LLMProvider,
  ProviderType,
  ProviderCapabilities,
  GenerationOptions,
  StreamCallbacks,
} from './types';
import { createStreamingRequest, parseOpenAIMessage } from '../httpClient';
import { ThinkTagParser, processDelta, generateOllamaChatImpl } from './openAICompatibleStream';
import { buildOpenAIMessagesImpl } from './openAIMessageBuilder';
import logger from '../../utils/logger';
import type {
  OpenAIChatMessage,
  OpenAIConfig,
  OpenAIStreamState,
} from './openAICompatibleTypes';
import { remoteApiUrl, remoteAuthorizationHeaders } from '../remoteTransportPolicy';

export type { OpenAIChatMessage, OpenAIConfig } from './openAICompatibleTypes';

/** True when the endpoint's port is exactly Ollama's 11434 (not 114340, not a path or query hit). */
export function isOllamaEndpoint(endpoint: string): boolean {
  try {
    return new URL(endpoint).port === '11434';
  } catch {
    return /^[a-z][a-z0-9+.-]*:\/\/[^/?#]*:11434(?:[/?#]|$)/i.test(endpoint.trim());
  }
}

/**
 * Sampling fields the app decides and the server must obey (llama-server / OpenAI-compatible names).
 * An unset option is not sent, so the server default applies only where the app has no value.
 */
export function openAISamplingParams(options: GenerationOptions): Record<string, unknown> {
  const stop = options.stopSequences?.filter(sequence => sequence.length > 0) ?? [];
  return {
    ...(options.temperature !== undefined && { temperature: options.temperature }),
    ...(options.topP !== undefined && { top_p: options.topP }),
    ...(options.topK !== undefined && { top_k: options.topK }),
    ...(options.repeatPenalty !== undefined && { repeat_penalty: options.repeatPenalty }),
    ...(options.seed !== undefined && { seed: options.seed }),
    ...(stop.length > 0 && { stop }),
    // The app's max-tokens setting is the output cap. Sent only when it is a positive number.
    ...(typeof options.maxTokens === 'number' && options.maxTokens > 0 && { max_tokens: options.maxTokens }),
  };
}

/**
 * OpenAI-Compatible Provider Implementation
 */
export class OpenAICompatibleProvider implements LLMProvider {
  readonly type: ProviderType = 'openai-compatible';

  private config: OpenAIConfig;
  private abortController: AbortController | null = null;
  private modelCapabilities: ProviderCapabilities;

  constructor(
    public readonly id: string,
    config: OpenAIConfig
  ) {
    this.config = config;
    this.modelCapabilities = {
      supportsVision: false,
      supportsToolCalling: true, // Assume true for OpenAI-compatible
      supportsThinking: false,
      acceptsThinkingKwarg: false, // set from discovery (/props or LM Studio probe)
    };
  }

  get capabilities(): ProviderCapabilities {
    return this.modelCapabilities;
  }

  updateConfig(config: Partial<OpenAIConfig>): void {
    this.config = { ...this.config, ...config };
  }

  async loadModel(modelId: string): Promise<void> {
    this.config.modelId = modelId;
    // Capabilities are set via updateCapabilities() after discovery results are applied
  }

  /**
   * Apply authoritative capabilities from server discovery results
   */
  updateCapabilities(capabilities: Partial<ProviderCapabilities>): void {
    this.modelCapabilities = { ...this.modelCapabilities, ...capabilities };
  }

  async unloadModel(): Promise<void> {
    this.config.modelId = '';
    this.abortController = null;
  }

  isModelLoaded(): boolean { return !!this.config.modelId; }

  getLoadedModelId(): string | null { return this.config.modelId || null; }

  /**
   * Build the request body for the /v1/chat/completions endpoint.
   */
  private buildRequestBody(
    openaiMessages: OpenAIChatMessage[],
    options: GenerationOptions,
    thinkingEnabled: boolean
  ): Record<string, unknown> {
    const isOpenRouter = new URL(this.config.endpoint).hostname === 'openrouter.ai';
    return {
      model: this.config.modelId,
      messages: openaiMessages,
      stream: true,
      // The app owns every sampling parameter, max_tokens included (settings.maxTokens).
      ...openAISamplingParams(options),
      // llama-server: reuse the KV cache of the shared prompt prefix across turns.
      ...(!isOpenRouter && { cache_prompt: true }),
      ...(this.modelCapabilities.supportsToolCalling && options.tools && options.tools.length > 0 && {
        tools: options.tools,
        tool_choice: 'auto',
      }),
      // Control Qwen3 thinking per-request via chat_template_kwargs, but only for
      // servers that advertised (at discovery) that they honor it. Gating on a
      // discovered capability — not the endpoint's port — keeps the "which server
      // accepts this?" decision in one place and free of implementation coupling.
      ...(this.modelCapabilities.acceptsThinkingKwarg && { chat_template_kwargs: { enable_thinking: thinkingEnabled } }),
      // Budget goes to every server that advertised the kwarg at discovery, not only 'remote-vision:' ids.
      ...(this.modelCapabilities.acceptsThinkingKwarg
        ? reasoningBudgetPayload(thinkingEnabled, options.reasoningBudget ?? REASONING_BUDGET_AUTO)
        : {}),
      ...(isOpenRouter && this.modelCapabilities.supportsThinking
        ? thinkingEnabled
          ? openRouterReasoningPayload(true, options.reasoningBudget ?? REASONING_BUDGET_AUTO)
          : { reasoning: { effort: 'none' } }
        : {}),
    };
  }

  async generate(
    messages: Message[],
    options: GenerationOptions,
    callbacks: StreamCallbacks
  ): Promise<void> {
    if (!this.config.modelId) {
      callbacks.onError(new Error('No model selected'));
      return;
    }

    const controller = new AbortController();
    this.abortController = controller;
    // Capture signal in closure so abort checks remain valid even after
    // this.abortController is nulled by stopGeneration().
    const { signal } = controller;
    let streamError: Error | null = null;

    try {
      const openaiMessages = await this.buildOpenAIMessages(messages, options);
      const thinkingEnabled =
        this.modelCapabilities.thinkingLevelsOnly ||
        options.enableThinking !== false;

      logger.log(`[Provider] generate — model=${this.config.modelId}, isOllama=${isOllamaEndpoint(this.config.endpoint)}, thinking=${thinkingEnabled}, tools=${options.tools?.length || 0}, messages=${openaiMessages.length}`);

      // Route Ollama through its native /api/chat which supports think: true/false
      if (isOllamaEndpoint(this.config.endpoint)) {
        return generateOllamaChatImpl(openaiMessages, {
          options, callbacks, signal,
          endpoint: this.config.endpoint,
          modelId: this.config.modelId,
          thinkingLevelsOnly: this.modelCapabilities.thinkingLevelsOnly,
          // Abort THIS turn's controller, never a newer turn's.
          abort: () => controller.abort(),
          headers: remoteAuthorizationHeaders(this.config.endpoint, this.config.apiKey),
        });
      }

      const requestBody = this.buildRequestBody(openaiMessages, options, thinkingEnabled);
      logger.log(`[Provider][DEBUG] OpenAI request — hasTools=${!!requestBody.tools}, toolChoice=${typeof requestBody.tool_choice === 'string' ? requestBody.tool_choice : JSON.stringify(requestBody.tool_choice) || 'none'}`);
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      };
      Object.assign(headers, remoteAuthorizationHeaders(this.config.endpoint, this.config.apiKey));

      // One normalizer: an endpoint saved as `.../v1` must not become `/v1/v1/chat/completions`.
      const url = remoteApiUrl(this.config.endpoint, '/v1/chat/completions');

      const state: OpenAIStreamState = {
        fullContent: '', fullReasoningContent: '',
        reasoningDetails: [],
        toolCalls: [], currentToolCall: null,
        completeCalled: false, streamErrorOccurred: false,
      };
      const thinkTagParser = new ThinkTagParser();

      await createStreamingRequest(url, { body: requestBody, headers, signal }, (event) => {
        if (signal.aborted) return;
        const message = parseOpenAIMessage(event);
        if (!message) return;

        if (message.error) {
          logger.error(`[Provider][DEBUG] Stream error: ${JSON.stringify(message.error)}`);
          state.streamErrorOccurred = true;
          streamError = new Error(message.error.message || 'API error');
          controller.abort();
          return;
        }
        if (message.object === 'done') return;

        if (message.choices && message.choices.length > 0) {
          const choice = message.choices[0];
          if (choice.delta) {
            processDelta(choice.delta, state, { thinkingEnabled, callbacks, thinkTagParser });
          }
          if (choice.finish_reason) {
            logger.log(`[Provider][DEBUG] finish_reason=${choice.finish_reason}, fullContent=${state.fullContent.length}, reasoning=${state.fullReasoningContent.length}, toolCalls=${state.toolCalls.length}`);
          }
          if (
            !state.completeCalled &&
            (choice.finish_reason === 'stop' ||
              choice.finish_reason === 'tool_calls')
          ) {
            state.completeCalled = true;
            const completedCalls = state.toolCalls.filter(tc => tc.function?.name);
            logger.log(`[Provider][DEBUG] Completing — content=${state.fullContent.length} chars, reasoning=${state.fullReasoningContent.length} chars, completedCalls=${completedCalls.length}`);
            callbacks.onComplete({
              content: state.fullContent,
              reasoningContent: state.fullReasoningContent || undefined,
              reasoningDetails: state.reasoningDetails?.length ? state.reasoningDetails : undefined,
              meta: { gpu: false, gpuBackend: 'Remote' },
              toolCalls: completedCalls.length > 0 ? completedCalls.map(tc => ({
                id: tc.id, name: tc.function.name, arguments: tc.function.arguments,
              })) : undefined,
            });
          }
        }
      });

      if (streamError) throw streamError;

      // A user stop resolves the request. End the turn with the same empty completion the
      // rejected-abort path gives, never the fallback below: that would finalize a partial turn and
      // hand half-streamed tool calls to the tool loop. A callback must still fire, because
      // generationToolLoop's remoteGenerateOnce waits on onComplete/onError, not on this promise.
      if (signal.aborted) {
        if (!state.completeCalled) callbacks.onComplete({ content: '', meta: { gpu: false } });
        return;
      }

      // Fallback: if stream ended without a recognised finish_reason (e.g. 'length',
      // 'content_filter', null), ensure the generation is finalised.
      if (!state.completeCalled && !state.streamErrorOccurred) {
        logger.log(`[Provider][DEBUG] Fallback complete (no finish_reason) — content=${state.fullContent.length}, reasoning=${state.fullReasoningContent.length}, toolCalls=${state.toolCalls.length}`);
        const completedCalls = state.toolCalls.filter(tc => tc.function?.name);
        callbacks.onComplete({
          content: state.fullContent,
          reasoningContent: state.fullReasoningContent || undefined,
          reasoningDetails: state.reasoningDetails?.length ? state.reasoningDetails : undefined,
          meta: { gpu: false, gpuBackend: 'Remote' },
          toolCalls: completedCalls.length > 0 ? completedCalls.map(tc => ({
            id: tc.id, name: tc.function.name, arguments: tc.function.arguments,
          })) : undefined,
        });
      }
    } catch (error) {
      if (streamError) {
        callbacks.onError(streamError);
        return;
      }
      if (signal.aborted) {
        callbacks.onComplete({ content: '', meta: { gpu: false } });
        return;
      }
      callbacks.onError(error instanceof Error ? error : new Error(String(error)));
    } finally {
      // A newer generate() may already own this.abortController; only clear our own.
      if (this.abortController === controller) this.abortController = null;
    }
  }

  private buildOpenAIMessages(
    messages: Message[],
    options: GenerationOptions
  ): Promise<OpenAIChatMessage[]> {
    return buildOpenAIMessagesImpl(messages, options, this.modelCapabilities);
  }

  async stopGeneration(): Promise<void> {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  async getTokenCount(text: string): Promise<number> {
    // Approximate token count for remote providers
    // Most models use ~4 characters per token
    return Math.ceil(text.length / 4);
  }

  async isReady(): Promise<boolean> {
    return !!this.config.modelId && !!this.config.endpoint;
  }

  async dispose(): Promise<void> {
    await this.stopGeneration();
    this.config.modelId = '';
  }
}

/**
 * Factory to create an OpenAI-compatible provider
 */
export function createOpenAIProvider(
  serverId: string,
  endpoint: string,
  opts?: { apiKey?: string; modelId?: string }
): OpenAICompatibleProvider {
  return new OpenAICompatibleProvider(serverId, {
    endpoint,
    apiKey: opts?.apiKey,
    modelId: opts?.modelId || '',
  });
}
