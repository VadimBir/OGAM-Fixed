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
import {
  acceptsServerExtensions,
  remoteApiUrl,
  remoteAuthorizationHeaders,
} from '../remoteTransportPolicy';

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
 * `extensions` = the server takes non-OpenAI fields (top_k, repeat_penalty); a strict server 400s on them.
 * `llamaServer` = the user's own server (acceptsServerExtensions): every value is sent, plus the
 * llama.cpp-only fields (min_p, typical_p, repeat_last_n, DRY, XTC, mirostat).
 * max_tokens is never sent: settings.maxTokens is the on-device cap (default 2048) and cut reasoning
 * models short on a remote server, so the remote server owns the output limit.
 */
export function openAISamplingParams(
  options: GenerationOptions,
  extensions: boolean,
  llamaServer: boolean = false,
): Record<string, unknown> {
  const allStops = options.stopSequences?.filter(sequence => sequence.length > 0) ?? [];
  // The OpenAI API takes at most 4 stop sequences; a strict host 400s on more. The user's own
  // llama-server takes the whole list.
  const stop = llamaServer ? allStops : allStops.slice(0, OPENAI_MAX_STOP_SEQUENCES);
  // Third-party hosts: presence / frequency are standard OpenAI fields, sent only when they do
  // something (0 = the OpenAI default = off), and a seed only when fixed (unset = random there).
  // The user's own server gets every value, so a server-side default never decides one.
  const nonZero = (v: number | undefined): v is number => v !== undefined && v !== 0;
  return {
    ...(options.temperature !== undefined && { temperature: options.temperature }),
    ...(options.topP !== undefined && { top_p: options.topP }),
    ...(extensions && options.topK !== undefined && { top_k: options.topK }),
    ...(extensions && options.repeatPenalty !== undefined && { repeat_penalty: options.repeatPenalty }),
    ...((llamaServer ? options.presencePenalty !== undefined : nonZero(options.presencePenalty)) && { presence_penalty: options.presencePenalty }),
    ...((llamaServer ? options.frequencyPenalty !== undefined : nonZero(options.frequencyPenalty)) && { frequency_penalty: options.frequencyPenalty }),
    // llama-server: seed -1 = random (tools/server/README.md "Default: `-1`, which is a random seed").
    ...(options.seed !== undefined && (llamaServer || options.seed >= 0) && { seed: options.seed }),
    ...(llamaServer && llamaServerSamplingExtras(options)),
    ...(stop.length > 0 && { stop }),
  };
}

const OPENAI_MAX_STOP_SEQUENCES = 4;

/**
 * llama-server sampling fields beyond the OpenAI set (llama.cpp tools/server/README.md, POST
 * /completion options; /v1/chat/completions takes the same llama.cpp-specific fields). Sent only to
 * the user's own server (acceptsServerExtensions): a third-party host 400s on unknown fields.
 */
export function llamaServerSamplingExtras(options: GenerationOptions): Record<string, unknown> {
  const pick: Array<[string, number | undefined]> = [
    ['min_p', options.minP],
    ['typical_p', options.typicalP],
    ['repeat_last_n', options.penaltyLastN],
    ['dry_multiplier', options.dryMultiplier],
    ['dry_base', options.dryBase],
    ['dry_allowed_length', options.dryAllowedLength],
    ['dry_penalty_last_n', options.dryPenaltyLastN],
    ['xtc_probability', options.xtcProbability],
    ['xtc_threshold', options.xtcThreshold],
    ['mirostat', options.mirostat],
    ['mirostat_tau', options.mirostatTau],
    ['mirostat_eta', options.mirostatEta],
  ];
  const out: Record<string, unknown> = {};
  for (const [key, value] of pick) if (value !== undefined) out[key] = value;
  return out;
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
    // A server the user runs (LAN / Tailscale) takes the llama.cpp fields; a public host gets
    // standard OpenAI fields only. OpenRouter keeps top_k / repeat_penalty, as before.
    const localServer = acceptsServerExtensions(this.config.endpoint);
    return {
      model: this.config.modelId,
      messages: openaiMessages,
      stream: true,
      // The app owns every sampling parameter; the output cap stays with the server (no max_tokens).
      ...openAISamplingParams(options, localServer || isOpenRouter, localServer),
      // llama-server: reuse the KV cache of the shared prompt prefix across turns.
      ...(localServer && { cache_prompt: true }),
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

      // Route Ollama through its native /api/chat which supports think: true/false.
      // Awaited: a bare `return` ran `finally` at once, nulled this.abortController, and
      // stopGeneration() could no longer abort the Ollama stream.
      if (isOllamaEndpoint(this.config.endpoint)) {
        await generateOllamaChatImpl(openaiMessages, {
          options, callbacks, signal,
          endpoint: this.config.endpoint,
          modelId: this.config.modelId,
          thinkingLevelsOnly: this.modelCapabilities.thinkingLevelsOnly,
          // Abort THIS turn's controller, never a newer turn's.
          abort: () => controller.abort(),
          headers: remoteAuthorizationHeaders(this.config.endpoint, this.config.apiKey),
        });
        return;
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
