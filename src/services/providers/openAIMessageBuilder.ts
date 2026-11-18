/**
 * OpenAI-Compatible Provider — message builder
 * Converts app Message[] to OpenAI chat message format
 */
import type { Message } from '../../types';
import type { GenerationOptions, ProviderCapabilities } from './types';
import type { OpenAIChatMessage, OpenAIContentPart, OpenAIToolCall } from './openAICompatibleTypes';
import { imageToBase64DataUrl } from '../httpClient';
import { useAppStore } from '../../stores/appStore';
import logger from '../../utils/logger';
import { generateId } from '../../utils/generateId';

/** Build multimodal content array for a vision-capable user message */
async function buildVisionContent(
  msg: Message,
  capabilities: ProviderCapabilities,
): Promise<OpenAIContentPart[]> {
  const content: OpenAIContentPart[] = [{ type: 'text', text: msg.content }];

  if (!capabilities.supportsVision) return content;

  for (const attachment of msg.attachments || []) {
    if (attachment.type === 'image') {
      try {
        const dataUrl = await imageToBase64DataUrl(attachment.uri);
        content.push({ type: 'image_url', image_url: { url: dataUrl } });
      } catch (error) {
        logger.warn('[OpenAIProvider] Failed to encode image:', error);
      }
    }
  }
  return content;
}

/** Build an assistant message with tool calls */
function buildAssistantToolCallMessage(msg: Message): OpenAIChatMessage {
  return {
    role: 'assistant',
    content: msg.content || '',
    ...(msg.reasoningDetails?.length ? { reasoning_details: msg.reasoningDetails } : {}),
    tool_calls: (msg.toolCalls || []).map(tc => ({
      id: tc.id || `call_${generateId()}`,
      type: 'function' as const,
      function: { name: tc.name, arguments: tc.arguments },
    })) as OpenAIToolCall[],
  };
}

/**
 * Build OpenAI chat messages from app messages.
 * Handles system, tool, user (with optional vision), and assistant messages.
 */
export async function buildOpenAIMessagesImpl(
  messages: Message[],
  options: GenerationOptions,
  capabilities: ProviderCapabilities,
): Promise<OpenAIChatMessage[]> {
  const openaiMessages: OpenAIChatMessage[] = [];
  const hasSystemMessage = messages.some(m => m.role === 'system');

  // Add system prompt if provided and no system message exists in messages
  const systemPrompt = options.systemPrompt || useAppStore.getState().settings.systemPrompt;
  if (systemPrompt && !hasSystemMessage) {
    openaiMessages.push({ role: 'system', content: [{ type: 'text', text: systemPrompt }] });
  }

  // Strict OpenAI-compatible servers require every assistant tool_calls message to be followed
  // DIRECTLY by its tool results. The store writes the image record (app caption) between the call and
  // its result (generationToolLoop tool-call record -> imageGenerationResult caption -> tool result), so
  // assistant text arriving while results are still owed is deferred until they are in, and the app's
  // image caption (not model speech) is dropped.
  let owedToolResults = 0;
  let deferred: Message[] = [];
  const flushDeferred = () => {
    for (const d of deferred) openaiMessages.push({ role: 'assistant', content: d.content });
    deferred = [];
  };

  for (const msg of messages) {
    if (msg.role === 'system') {
      openaiMessages.push({ role: 'system', content: [{ type: 'text', text: msg.content }] });
      continue;
    }

    if (msg.role === 'tool') {
      // Tool result — wrap as array so models with strict Jinja templates (e.g. qwen3.5)
      // that iterate over message['content'] don't fail on plain strings
      openaiMessages.push({
        role: 'tool',
        content: [{ type: 'text', text: msg.content }],
        tool_call_id: msg.toolCallId || '',
      });
      if (owedToolResults > 0) owedToolResults -= 1;
      if (owedToolResults === 0) flushDeferred();
      continue;
    }

    if (owedToolResults > 0 && msg.role === 'assistant' && !msg.toolCalls?.length) {
      if (!isAppImageCaption(msg)) deferred.push(msg);
      continue;
    }
    if (owedToolResults > 0) {
      // A user turn / new tool-call record before all results arrived: results are missing for good.
      owedToolResults = 0;
      flushDeferred();
    }
    if (msg.role === 'assistant' && msg.toolCalls?.length) owedToolResults = msg.toolCalls.length;

    const hasImages = msg.attachments?.some(a => a.type === 'image');

    if (msg.role === 'user' && hasImages && capabilities.supportsVision) {
      const content = await buildVisionContent(msg, capabilities);
      openaiMessages.push({ role: 'user', content });
    } else if (msg.role === 'assistant' && msg.toolCalls && msg.toolCalls.length > 0) {
      openaiMessages.push(buildAssistantToolCallMessage(msg));
    } else if (msg.role === 'user') {
      // Wrap user content as array — some model templates (e.g. qwen3.5) require
      // message['content'] to be iterable, not a plain string
      openaiMessages.push({ role: 'user', content: [{ type: 'text', text: msg.content }] });
    } else {
      // Assistant text message
      openaiMessages.push({ role: 'assistant', content: msg.content });
    }
  }
  flushDeferred();

  return openaiMessages;
}

/** App-written caption under a generated image (imageGenerationResult.ts), not model speech. */
function isAppImageCaption(msg: Message): boolean {
  return Boolean(msg.generationMeta?.resolution) || /^Generated image for:\s*["“]/i.test(msg.content.trim());
}
