import { RNLlamaOAICompatibleMessage, RNLlamaMessagePart } from 'llama.rn';
import { Message, MediaAttachment } from '../types';
import { modelImageAttachments } from './llmImageInput';

/**
 * PRODUCT RULE: every voice note is transcribed (whisper) and ONLY its transcript is sent to the
 * model — the audio attachment is display/playback ONLY, never model input. So the llama message
 * builders NEVER attach audio as media. Sending the audio broke turns two ways:
 *  - the transcript is already in message.content, so the audio is redundant; and
 *  - the audio path is an absolute iOS container path that goes stale on reinstall/rebuild (new
 *    container UUID) → "File does not exist or cannot be opened", hard-failing every voice-mode turn
 *    in a persisted conversation (B9); a non-audio mmproj also throws "Failed to load media" (B5).
 * Returning [] unconditionally enforces the transcript-only contract for every model + state
 * (including a not-yet-transcribed note — which must not reach the model as raw audio either). */
function modelAudioAttachments(_attachments: MediaAttachment[] | undefined): MediaAttachment[] {
  return [];
}

export function formatLlamaMessages(messages: Message[], supportsVision: boolean, supportsAudio = false): string {
  let prompt = '';
  for (const message of messages.filter(m => !m.isSystemInfo)) {
    if (message.role === 'system') {
      prompt += `<|im_start|>system\n${message.content}<|im_end|>\n`;
    } else if (message.role === 'user') {
      let content = message.content;
      if (message.attachments && message.attachments.length > 0) {
        const imageMarkers = supportsVision
          ? modelImageAttachments(message.attachments).map(() => '<__media__>').join('')
          : '';
        const audioMarkers = supportsAudio
          ? modelAudioAttachments(message.attachments).map(() => '<__media__>').join('')
          : '';
        content = imageMarkers + audioMarkers + content;
      }
      prompt += `<|im_start|>user\n${content}<|im_end|>\n`;
    } else if (message.role === 'assistant') {
      prompt += `<|im_start|>assistant\n${message.content}<|im_end|>\n`;
    }
  }
  prompt += '<|im_start|>assistant\n';
  return prompt;
}

export function extractImageUris(messages: Message[]): string[] {
  const uris: string[] = [];
  for (const message of messages) {
    for (const attachment of modelImageAttachments(message.attachments)) {
      uris.push(attachment.uri);
    }
  }
  return uris;
}

/**
 * Format a tool call as plain text for the assistant message.
 * Avoids structured tool_calls which cause Jinja template errors
 * (C++ wants arguments as string, Jinja wants dict — can't satisfy both).
 */
function formatToolCallAsText(tc: { name: string; arguments: string }): string {
  const escapedName = JSON.stringify(tc.name);
  return `<tool_call>{"name":${escapedName},"arguments":${tc.arguments}}</tool_call>`;
}

function toFileUrl(uri: string, requireFilePrefix = false): string {
  if (requireFilePrefix) return uri.startsWith('file://') ? uri : `file://${uri}`;
  return uri.startsWith('file://') || uri.startsWith('http') ? uri : `file://${uri}`;
}

function buildMediaParts(message: Message, supportsAudio: boolean): RNLlamaMessagePart[] {
  const parts: RNLlamaMessagePart[] = [];
  for (const a of modelImageAttachments(message.attachments)) {
    parts.push({ type: 'image_url', image_url: { url: toFileUrl(a.uri) } });
  }
  if (supportsAudio) {
    for (const a of modelAudioAttachments(message.attachments)) {
      parts.push({ type: 'input_audio', input_audio: { format: a.audioFormat ?? 'wav', url: toFileUrl(a.uri, true) } });
    }
  }
  if (message.content) parts.push({ type: 'text', text: message.content });
  return parts;
}

/**
 * A tool result as text, in the <tool_response> wrapper Qwen's chat template itself renders for a
 * role:'tool' message (and the text-hint prompt names). The Qwen3 template also treats a user message
 * that starts with <tool_response> and ends with </tool_response> as a tool turn, not a new user query.
 */
export function toolResponseText(toolName: string | undefined, content: string): string {
  return `<tool_response>\n[${toolName || 'tool'}] ${content}\n</tool_response>`;
}

export interface OAIMessageOptions {
  /** Send tool results as role:'tool' (tool_call_id + name) for the chat template to render natively.
   *  Only for templates that render a tool role; default false = user-role <tool_response> text, which
   *  every template accepts. */
  toolRole?: boolean;
}

type OAIOutMessage = RNLlamaOAICompatibleMessage & { tool_call_id?: string; name?: string };

export function buildOAIMessages(
  messages: Message[],
  supportsAudio = false,
  opts: OAIMessageOptions = {},
): RNLlamaOAICompatibleMessage[] {
  const out: OAIOutMessage[] = [];
  // Index in `out` of the user message holding the current run of tool results (user-role fallback).
  let toolRunIdx = -1;
  for (const message of messages.filter(m => !m.isSystemInfo)) {
    if (message.role === 'tool') {
      if (opts.toolRole) {
        out.push({
          role: 'tool',
          content: message.content,
          ...(message.toolCallId ? { tool_call_id: message.toolCallId } : {}),
          ...(message.toolName ? { name: message.toolName } : {}),
        });
        continue;
      }
      const text = toolResponseText(message.toolName, message.content);
      // Several results of one round go in ONE user message: two user turns in a row are rejected by
      // strict-alternation templates (Gemma, Mistral).
      if (toolRunIdx >= 0) out[toolRunIdx].content = `${out[toolRunIdx].content as string}\n${text}`;
      else {
        out.push({ role: 'user', content: text });
        toolRunIdx = out.length - 1;
      }
      continue;
    }
    toolRunIdx = -1;
    out.push(convertMessage(message, supportsAudio));
  }
  return out;
}

function convertMessage(message: Message, supportsAudio: boolean): RNLlamaOAICompatibleMessage {
  if (message.role === 'assistant' && message.toolCalls?.length) {
    const toolCallText = message.toolCalls.map(formatToolCallAsText).join('\n');
    return { role: 'assistant' as const, content: message.content ? `${message.content}\n${toolCallText}` : toolCallText };
  }
  const hasImage =
    message.role === 'user' && modelImageAttachments(message.attachments).length > 0;
  const hasAudio = supportsAudio && message.role === 'user' && modelAudioAttachments(message.attachments).length > 0;
  if (!hasImage && !hasAudio) return { role: message.role, content: message.content };
  return { role: message.role, content: buildMediaParts(message, supportsAudio) };
}
