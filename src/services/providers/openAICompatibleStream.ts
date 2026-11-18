/**
 * OpenAI-Compatible Provider — streaming utilities
 * ThinkTagParser, processDelta, generateOllamaChatImpl
 */
import { createNDJSONStreamingRequest } from '../httpClient';
import { remoteApiUrl } from '../remoteTransportPolicy';
import logger from '../../utils/logger';
import { REASONING_DELIMITERS, partialTagSuffix, maxPartialTagSuffix } from '../../utils/messageContent';
import type { GenerationOptions, StreamCallbacks } from './types';
import type {
  OpenAIChatMessage,
  OpenAIToolCall,
  OpenAIStreamState,
  OllamaChatRequest,
} from './openAICompatibleTypes';

/**
 * Streaming parser for <think>...</think> tags embedded in delta.content.
 * Routes thinking content to onReasoning and regular content to onToken.
 * Handles tags split across multiple streaming chunks.
 */
export class ThinkTagParser {
  private inThinkBlock = false;
  private buffer = '';
  // The close tag for the reasoning format currently open — set when an opener is matched,
  // so a block opened as Gemma closes on `<channel|>` and one opened as <think> on </think>.
  private activeClose = '';
  // A single `\n` immediately after an opener is a SYNTACTIC separator (`<|channel>thought\n…`,
  // `<think>\n…`), not reasoning — consume it once on block entry so reasoning has no leading
  // newline. Mirrors the complete parser's `\n?` after the Gemma opener; without it the bare
  // `<|channel>thought` opener would emit the separating newline as reasoning content.
  private stripLeadingNewlineOnEntry = false;

  process(content: string, onToken: (t: string) => void, onReasoning: (t: string) => void): void {
    this.buffer += content;
    this.flush(onToken, onReasoning);
  }

  /**
   * Handle one iteration of the while loop when we are outside a reasoning block.
   * Scans for the EARLIEST opener across every reasoning format (the shared grammar), so
   * remote-streamed Gemma/Qwen channel reasoning is recognised, not just <think> (DR1).
   * Returns true if the while loop should break (buffer needs more data).
   */
  private handleOutsideThink(onToken: (t: string) => void): boolean {
    let bestIdx = -1;
    let bestOpen = '';
    let bestClose = '';
    for (const d of REASONING_DELIMITERS) {
      const idx = this.buffer.indexOf(d.open);
      if (idx !== -1 && (bestIdx === -1 || idx < bestIdx)) {
        bestIdx = idx;
        bestOpen = d.open;
        bestClose = d.close;
      }
    }
    if (bestIdx === -1) {
      // No complete opener. Hold back only a suffix that could still become one of the openers.
      const partial = maxPartialTagSuffix(this.buffer, REASONING_DELIMITERS.map((d) => d.open));
      if (partial > 0) {
        onToken(this.buffer.slice(0, this.buffer.length - partial));
        this.buffer = this.buffer.slice(this.buffer.length - partial);
        return true;
      }
      onToken(this.buffer);
      this.buffer = '';
      return true;
    }
    if (bestIdx > 0) onToken(this.buffer.slice(0, bestIdx));
    this.buffer = this.buffer.slice(bestIdx + bestOpen.length);
    this.inThinkBlock = true;
    this.activeClose = bestClose;
    this.stripLeadingNewlineOnEntry = true;
    return false;
  }

  /**
   * Handle one iteration of the while loop when we are inside a reasoning block, matching the
   * close tag for the format that opened it (this.activeClose).
   * Returns true if the while loop should break (buffer needs more data).
   */
  private handleInsideThink(onReasoning: (t: string) => void): boolean {
    // Consume the one optional `\n` separator that follows the opener before any reasoning is
    // emitted. Char-by-char safe: if the buffer is still empty, wait for the next char; if it
    // starts with a non-newline, there was no separator — clear the flag and fall through.
    if (this.stripLeadingNewlineOnEntry) {
      if (this.buffer.length === 0) return true;
      if (this.buffer[0] === '\n') this.buffer = this.buffer.slice(1);
      this.stripLeadingNewlineOnEntry = false;
    }
    const closeTag = this.activeClose;
    const idx = this.buffer.indexOf(closeTag);
    if (idx === -1) {
      const partial = partialTagSuffix(this.buffer, closeTag);
      if (partial > 0) {
        onReasoning(this.buffer.slice(0, this.buffer.length - partial));
        this.buffer = this.buffer.slice(this.buffer.length - partial);
        return true;
      }
      onReasoning(this.buffer);
      this.buffer = '';
      return true;
    }
    if (idx > 0) onReasoning(this.buffer.slice(0, idx));
    this.buffer = this.buffer.slice(idx + closeTag.length);
    this.inThinkBlock = false;
    this.activeClose = '';
    return false;
  }

  private flush(onToken: (t: string) => void, onReasoning: (t: string) => void): void {
    while (this.buffer.length > 0) {
      const shouldBreak = this.inThinkBlock
        ? this.handleInsideThink(onReasoning)
        : this.handleOutsideThink(onToken);
      if (shouldBreak) break;
    }
  }
}

/** Context passed to processDelta */
interface DeltaCtx {
  thinkingEnabled: boolean;
  callbacks: StreamCallbacks;
  thinkTagParser: ThinkTagParser;
}

type DeltaShape = {
  content?: string;
  reasoning_content?: string;
  reasoning?: string;
  thinking?: string;
  reasoning_details?: Array<Record<string, unknown>>;
  tool_calls?: Array<{
    index?: number; id?: string; type?: string;
    function?: { name?: string; arguments?: string };
  }>;
};

/** Per-stream map from the server's tool_calls[].index to the call being accumulated. */
const toolCallsByIndex = new WeakMap<OpenAIStreamState, Map<number, OpenAIToolCall>>();

function startToolCall(state: OpenAIStreamState, id: string | undefined): OpenAIToolCall {
  // Index-suffixed fallback when the server sends no id: unique within this turn's calls.
  const call: OpenAIToolCall = {
    id: id || `call_${state.toolCalls.length}`,
    type: 'function',
    function: { name: '', arguments: '' },
  };
  state.toolCalls.push(call);
  state.currentToolCall = call;
  return call;
}

/**
 * Accumulate one streamed tool-call fragment. OpenAI-compatible servers key fragments by `index`
 * and send `id` on the first fragment only (some repeat it on every fragment, some never send it),
 * so `index` decides which call a fragment extends. A server without `index` falls back to the id.
 */
function processToolCallChunk(
  tc: NonNullable<DeltaShape['tool_calls']>[0],
  state: OpenAIStreamState,
): void {
  let call: OpenAIToolCall;
  if (typeof tc.index === 'number') {
    let calls = toolCallsByIndex.get(state);
    if (!calls) {
      calls = new Map();
      toolCallsByIndex.set(state, calls);
    }
    const existing = calls.get(tc.index);
    // A NEW id at an index whose call already has a name is a second call reusing that index.
    const reused = !!existing && !!tc.id && tc.id !== existing.id && !!existing.function.name;
    if (!existing || reused) {
      call = startToolCall(state, tc.id);
      calls.set(tc.index, call);
    } else {
      call = existing;
      state.currentToolCall = call;
    }
  } else {
    const current = state.currentToolCall as OpenAIToolCall | null;
    call = !current || (tc.id && tc.id !== current.id) ? startToolCall(state, tc.id) : current;
  }
  if (tc.function?.name) call.function.name = tc.function.name;
  if (tc.function?.arguments) call.function.arguments += tc.function.arguments;
}

/**
 * Process a streaming delta — extracted to reduce complexity of the SSE event handler.
 * Mutates state.fullContent, state.fullReasoningContent, state.toolCalls, state.currentToolCall.
 */
export function processDelta(
  delta: DeltaShape,
  state: OpenAIStreamState,
  ctx: DeltaCtx,
): void {
  // Shape only: chat content and tool arguments are private and must not reach the log file.
  logger.log(`[WIRE-REMOTE] delta keys=${Object.keys(delta).join(',')} content=${delta.content?.length ?? 0} toolCalls=${delta.tool_calls?.length ?? 0}`);
  if (delta.content) {
    ctx.thinkTagParser.process(
      delta.content,
      (text) => { state.fullContent += text; ctx.callbacks.onToken(text); },
      (reasoning) => {
        if (ctx.thinkingEnabled) {
          state.fullReasoningContent += reasoning;
          ctx.callbacks.onReasoning?.(reasoning);
        }
      },
    );
  }

  // Reasoning content — check all known field names across providers:
  // - delta.reasoning_content (LM Studio)
  // - delta.reasoning         (Ollama /v1/chat/completions)
  // - delta.thinking          (kept as fallback)
  if (delta.reasoning_details?.length) {
    (state.reasoningDetails ??= []).push(...delta.reasoning_details);
  }
  const structuredReasoning = (delta.reasoning_details || [])
    .map(detail => typeof detail.text === 'string'
      ? detail.text
      : typeof detail.summary === 'string' ? detail.summary : '')
    .filter(Boolean)
    .join('');
  const reasoningDelta = delta.reasoning_content || delta.reasoning || delta.thinking || structuredReasoning;
  // A DEDICATED reasoning field means the remote model chose to emit reasoning — surface
  // it regardless of the local thinkingEnabled toggle. Remote has no thinking toggle, so
  // gating on it dropped reasoning LM Studio actually sent (B16); providers that CAN be
  // told not to think (Ollama's `think` param) simply won't send this field when off.
  if (reasoningDelta) {
    state.fullReasoningContent += reasoningDelta;
    ctx.callbacks.onReasoning?.(reasoningDelta);
  }

  if (delta.tool_calls) {
    for (const tc of delta.tool_calls) {
      processToolCallChunk(tc, state);
    }
  }
}

/** Build the completion result from accumulated Ollama tool calls */
function buildOllamaCompletion(
  fullContent: string,
  fullReasoningContent: string,
  accumulatedToolCalls: OpenAIToolCall[],
): { content: string; reasoningContent?: string; meta: { gpu: boolean; gpuBackend: string }; toolCalls?: Array<{ id: string; name: string; arguments: string }> } {
  return {
    content: fullContent,
    reasoningContent: fullReasoningContent || undefined,
    meta: { gpu: false, gpuBackend: 'Remote' },
    toolCalls: accumulatedToolCalls.length > 0 ? accumulatedToolCalls.map((tc, index) => ({
      // Ollama's native tool_calls often carry no id; give each call a distinct one.
      id: tc.id || `call_${index}`, name: tc.function.name,
      arguments: typeof tc.function.arguments === 'string' ? tc.function.arguments : JSON.stringify(tc.function.arguments),
    })) : undefined,
  };
}

/** NDJSON line handler state for Ollama streaming */
interface OllamaStreamState {
  fullContent: string;
  fullReasoningContent: string;
  completeCalled: boolean;
  streamErrorOccurred: boolean;
  /** Server error line, reported once the request has settled (never from inside the XHR handler). */
  streamError?: Error;
  accumulatedToolCalls: OpenAIToolCall[];
}

/** Process a single NDJSON line from Ollama's /api/chat streaming response */
function handleOllamaChatLine(
  line: Record<string, unknown>,
  streamState: OllamaStreamState,
  req: { callbacks: StreamCallbacks; signal: AbortSignal; abort: () => void },
): void {
  if (req.signal.aborted) return;
  // Shape only: chat content is private and must not reach the log file.
  logger.log(`[WIRE-OLLAMA] line keys=${Object.keys(line).join(',')} done=${line.done === true}`);

  if (line.error) {
    // Record and stop. Reported after the request settles: calling onError here and then aborting
    // fired onError AND the abort path's empty onComplete for one turn.
    streamState.streamErrorOccurred = true;
    streamState.streamError = new Error(typeof line.error === 'string' ? line.error : JSON.stringify(line.error));
    req.abort();
    return;
  }

  const msg = line.message as {
    role?: string; content?: string; thinking?: string; tool_calls?: OpenAIToolCall[]
  } | undefined;
  if (msg) {
    if (msg.thinking) { streamState.fullReasoningContent += msg.thinking; req.callbacks.onReasoning?.(msg.thinking); }
    if (msg.content) { streamState.fullContent += msg.content; req.callbacks.onToken(msg.content); }
    // Collect tool_calls from every message (they arrive in done:false chunks)
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        if (tc.function?.name) {
          logger.log(`[OllamaChat] tool_call: ${tc.function.name}`);
          streamState.accumulatedToolCalls.push(tc);
        }
      }
    }
  }

  if (line.done) {
    logger.log(`[OllamaChat] stream done — content=${streamState.fullContent.length}, reasoning=${streamState.fullReasoningContent.length}, toolCalls=${streamState.accumulatedToolCalls.length}`);
    streamState.completeCalled = true;
    req.callbacks.onComplete(buildOllamaCompletion(streamState.fullContent, streamState.fullReasoningContent, streamState.accumulatedToolCalls));
  }
}

/**
 * Generate using Ollama's native /api/chat endpoint (NDJSON streaming).
 * Supports think: true/false for reasoning control.
 */
/**
 * Ollama /api/chat `options`: the same app-decided sampling values, under Ollama's names.
 * num_predict is never sent: settings.maxTokens is the on-device cap (default 2048) and cut
 * reasoning models short; the remote server owns the output limit.
 * Names = Ollama api/types.go Options json tags (top_k, min_p, repeat_last_n, repeat_penalty,
 * presence_penalty, frequency_penalty, seed, stop; DefaultOptions Seed -1). Ollama has no DRY / XTC /
 * mirostat option, and typical_p is deprecated there ("still accepted per request with a warning"),
 * so those are not sent.
 */
export function ollamaSamplingOptions(options: GenerationOptions): Record<string, unknown> {
  const stop = options.stopSequences?.filter(sequence => sequence.length > 0) ?? [];
  return {
    ...(options.temperature !== undefined && { temperature: options.temperature }),
    ...(options.topP !== undefined && { top_p: options.topP }),
    ...(options.topK !== undefined && { top_k: options.topK }),
    ...(options.minP !== undefined && { min_p: options.minP }),
    ...(options.repeatPenalty !== undefined && { repeat_penalty: options.repeatPenalty }),
    ...(options.penaltyLastN !== undefined && { repeat_last_n: options.penaltyLastN }),
    ...(options.presencePenalty !== undefined && { presence_penalty: options.presencePenalty }),
    ...(options.frequencyPenalty !== undefined && { frequency_penalty: options.frequencyPenalty }),
    ...(options.seed !== undefined && { seed: options.seed }),
    ...(stop.length > 0 && { stop }),
  };
}

export async function generateOllamaChatImpl(
  openaiMessages: OpenAIChatMessage[],
  req: OllamaChatRequest & { headers?: Record<string, string> },
): Promise<void> {
  const { options, callbacks, signal, endpoint, modelId, abort, headers = {} } = req;
  const thinkingEnabled = options.enableThinking !== false;
  const think = req.thinkingLevelsOnly
    ? options.reasoningBudget == null || options.reasoningBudget < 0 ? 'medium'
      : options.reasoningBudget <= 1024 ? 'low'
        : options.reasoningBudget <= 4096 ? 'medium' : 'high'
    : thinkingEnabled;

  // Convert to Ollama message format
  // Convert tool_calls arguments from JSON strings to objects for Ollama's native format
  const convertToolCalls = (tcs: OpenAIToolCall[] | undefined) => {
    if (!tcs) return undefined;
    return tcs.map(tc => ({
      ...tc,
      function: {
        ...tc.function,
        arguments: typeof tc.function.arguments === 'string'
          ? (() => { try { return JSON.parse(tc.function.arguments); } catch { return tc.function.arguments; } })()
          : tc.function.arguments,
      },
    }));
  };

  // Images go in a top-level `images` array as raw base64 (strip data:...;base64, prefix)
  const ollamaMessages = openaiMessages.map(m => {
    const toolCalls = convertToolCalls(m.tool_calls);
    if (typeof m.content === 'string') {
      return {
        role: m.role, content: m.content,
        ...(toolCalls && { tool_calls: toolCalls }),
        ...(m.tool_call_id && { tool_call_id: m.tool_call_id }),
      };
    }
    const parts = m.content as Array<{ type: string; text?: string; image_url?: { url: string } }>;
    const text = parts.find(p => p.type === 'text')?.text ?? '';
    const images = parts
      .filter(p => p.type === 'image_url')
      .map(p => {
        const url = p.image_url?.url ?? '';
        // Strip data:image/...;base64, prefix — Ollama expects raw base64
        const b64Match = /^data:[^;]+;base64,(.+)$/.exec(url);
        return b64Match ? b64Match[1] : url;
      });
    return {
      role: m.role, content: text,
      ...(images.length > 0 && { images }),
      ...(toolCalls && { tool_calls: toolCalls }),
      ...(m.tool_call_id && { tool_call_id: m.tool_call_id }),
    };
  });

  const requestBody: Record<string, unknown> = {
    model: modelId, messages: ollamaMessages, stream: true, think,
    ...(options.tools && options.tools.length > 0 && { tools: options.tools }),
    // The app owns every sampling value; the output cap stays with the server (no num_predict).
    options: ollamaSamplingOptions(options),
  };

  // /api/chat lives at the server root, also when the endpoint was saved as `.../v1`.
  const url = remoteApiUrl(endpoint, '/api/chat');

  const streamState: OllamaStreamState = {
    fullContent: '',
    fullReasoningContent: '',
    completeCalled: false,
    streamErrorOccurred: false,
    accumulatedToolCalls: [],
  };

  try {
    await createNDJSONStreamingRequest(url, { body: requestBody, headers, signal }, (line) => {
      handleOllamaChatLine(line, streamState, { callbacks, signal, abort });
    });
  } catch (error) {
    if (streamState.streamError) { callbacks.onError(streamState.streamError); return; }
    if (signal.aborted) {
      if (!streamState.completeCalled) callbacks.onComplete({ content: '', meta: { gpu: false } });
      return;
    }
    callbacks.onError(error instanceof Error ? error : new Error(String(error)));
    return;
  }
  if (streamState.streamError) { callbacks.onError(streamState.streamError); return; }
  if (streamState.completeCalled) return;
  // A user stop resolves the request: end the turn empty, never finalize the partial stream or hand
  // half-received tool calls on. The tool loop still needs this one terminal callback.
  if (signal.aborted) { callbacks.onComplete({ content: '', meta: { gpu: false } }); return; }
  callbacks.onComplete(buildOllamaCompletion(streamState.fullContent, streamState.fullReasoningContent, streamState.accumulatedToolCalls));
}
