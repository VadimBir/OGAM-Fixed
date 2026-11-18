'use strict';
/*
 * @offgrid/sync - LOCAL RECONSTRUCTION (private off-grid-ai/shared is absent).
 *
 * REAL (on the local chat path, derived from call sites + contract tests):
 *   chat grammar (REASONING_DELIMITERS, TOOL_CALL_OPENERS/CLOSERS, XML markers, partial-tag suffix),
 *   parseChatThinkingContent / parseChatModelOutput / stripChatControlTokens / splitInlineReasoning,
 *   preprocessChatMarkdown, safeChatExternalUrl, chatListPreviewLine, isRuntimeOnlyMessage,
 *   isSupportingChatContext, prompt-enhancement constants, hasProAccess, chat stream tools/previews,
 *   serializeSyncedMessageContext, ogamModelTransferBlocker, projectPersonalMeshActivationFailure.
 * STUB (multi-device mesh / pairing / transfer; not reachable from the free app entry):
 *   engine, transports, discovery, pairing, blob, checksum, oplog, file transfer classes and helpers.
 *   Stubs never throw at import; instances answer every method with a resolved no-op.
 */
Object.defineProperty(exports, '__esModule', { value: true });

// =============================================================================================
// Chat grammar
// =============================================================================================
/** Ordered longest-open-first (src/utils/messageContent.ts:88-97 comment). */
var REASONING_DELIMITERS = [
  { open: '<|channel|>analysis<|message|>', close: '<|channel|>final<|message|>' },
  { open: '<|channel>thought', close: '<channel|>' },
  { open: '<think>', close: '</think>' },
];
/** Gemma-native tool-call grammar (src/services/llmToolGeneration.ts:17-21, generationToolLoop.ts:254). */
var TOOL_CALL_OPENERS = ['<|tool_call>', '<tool_call:'];
var TOOL_CALL_CLOSERS = ['<tool_call|>', '</tool_call>'];
/** `\w+` after `=` is the tool/param name (src/utils/messageContent.ts:25-32); group 1 = name. */
var XML_TOOL_CALL_FUNCTION_MARKER = '<function=(\\w+)>';
var XML_TOOL_CALL_PARAMETER_MARKER = '<parameter=(\\w+)>';

function _esc(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Longest suffix of `text` that is a proper prefix of `tag`. */
function partialTagSuffix(text, tag) {
  if (!text || !tag) return 0;
  var max = Math.min(text.length, tag.length - 1);
  for (var k = max; k > 0; k--) {
    if (text.slice(text.length - k) === tag.slice(0, k)) return k;
  }
  return 0;
}
function maxPartialTagSuffix(text, tags) {
  var best = 0;
  var list = Array.isArray(tags) ? tags : [];
  for (var i = 0; i < list.length; i++) {
    var k = partialTagSuffix(text, list[i]);
    if (k > best) best = k;
  }
  return best;
}

var LABEL_PATTERN = /^\s*__LABEL:([\s\S]*?)__\s*\n?/;
function _withLabel(result, thinking) {
  if (thinking == null) return result;
  var m = LABEL_PATTERN.exec(thinking);
  if (m) {
    result.thinkingLabel = m[1];
    result.thinking = thinking.slice(m[0].length).trim();
  } else {
    result.thinking = thinking.trim();
  }
  return result;
}
function _indexCI(haystack, needle, from) {
  return haystack.toLowerCase().indexOf(needle.toLowerCase(), from || 0);
}

/**
 * Complete-string reasoning split. Contract: __tests__/unit/components/ChatMessage/utils.test.ts and
 * __tests__/unit/utils/reasoningGrammar.contract.test.ts.
 */
function parseChatThinkingContent(content) {
  var text = typeof content === 'string' ? content : '';
  // 1. Channel format first (checked before <think>).
  var chOpen = REASONING_DELIMITERS[0].open;
  var chClose = REASONING_DELIMITERS[0].close;
  var a = _indexCI(text, chOpen);
  if (a !== -1) {
    var f = _indexCI(text, chClose);
    if (f !== -1 && f > a) {
      return _withLabel(
        { thinking: null, response: text.slice(f + chClose.length).trim(), isThinkingComplete: true },
        text.slice(a + chOpen.length, f),
      );
    }
    return _withLabel(
      { thinking: null, response: '', isThinkingComplete: false },
      text.slice(a + chOpen.length),
    );
  }
  // 2. Gemma 4 channel, opener followed by an optional newline.
  var gOpen = REASONING_DELIMITERS[1].open;
  var gClose = REASONING_DELIMITERS[1].close;
  var g = _indexCI(text, gOpen);
  if (g !== -1) {
    var bodyStart = g + gOpen.length;
    if (text.charAt(bodyStart) === '\n') bodyStart += 1;
    var gc = _indexCI(text, gClose, bodyStart);
    var before = text.slice(0, g);
    if (gc !== -1) {
      return _withLabel(
        { thinking: null, response: (before + text.slice(gc + gClose.length)).trim(), isThinkingComplete: true },
        text.slice(bodyStart, gc),
      );
    }
    return _withLabel(
      { thinking: null, response: before.trim(), isThinkingComplete: false },
      text.slice(bodyStart),
    );
  }
  // 3. <think> ... </think> (case-insensitive).
  var tOpen = REASONING_DELIMITERS[2].open;
  var tClose = REASONING_DELIMITERS[2].close;
  var t = _indexCI(text, tOpen);
  if (t !== -1) {
    var tb = t + tOpen.length;
    var tc = _indexCI(text, tClose, tb);
    var pre = text.slice(0, t);
    if (tc !== -1) {
      return _withLabel(
        { thinking: null, response: (pre + text.slice(tc + tClose.length)).trim(), isThinkingComplete: true },
        text.slice(tb, tc),
      );
    }
    return _withLabel(
      { thinking: null, response: pre.trim(), isThinkingComplete: false },
      text.slice(tb),
    );
  }
  // 4. Orphan </think> (template forced the opener into the prompt).
  var oc = _indexCI(text, tClose);
  if (oc !== -1) {
    var orphan = text.slice(0, oc).trim();
    if (orphan) {
      return _withLabel(
        { thinking: null, response: text.slice(oc + tClose.length).trim(), isThinkingComplete: true },
        orphan,
      );
    }
  }
  return { thinking: null, response: text, isThinkingComplete: true };
}

var CONTROL_TOKEN_PATTERNS = [
  /<\|im_start\|>\s*(?:system|assistant|user|tool)?\s*\n?/gi,
  /<\|im_end\|>\s*\n?/gi,
  /<\|end\|>/gi,
  /<\|eot_id\|>/gi,
  /<\|endoftext\|>/gi,
  /<end_of_turn>/gi,
  /<start_of_turn>(?:model|user)?\s*\n?/gi,
  /<\/s>/gi,
  /<\|">/g,
];
var _closersAlt = TOOL_CALL_CLOSERS.map(_esc).join('|');
var _openersAlt = TOOL_CALL_OPENERS.map(_esc).join('|');
var TOOL_MARKUP_PATTERNS = [
  // Gemma-native blocks: any opener to the NEAREST closer.
  new RegExp('(?:' + _openersAlt + ')[\\s\\S]*?(?:' + _closersAlt + ')\\s*', 'g'),
  // Hermes / Qwen text tool calls.
  /<tool_call>[\s\S]*?<\/tool_call>\s*/gi,
  /<function_call>[\s\S]*?<\/function_call>\s*/gi,
  // XML-style <function=NAME>...</function>.
  new RegExp(XML_TOOL_CALL_FUNCTION_MARKER + '[\\s\\S]*?<\\/function>\\s*', 'gi'),
  // Anthropic-style <invoke ...>...</invoke>, optionally wrapped.
  /<(?:[\w-]+:)?function_calls>[\s\S]*?<\/(?:[\w-]+:)?function_calls>\s*/gi,
  /<invoke\b[\s\S]*?<\/invoke>\s*/gi,
  // Namespaced wrapper blocks: ns:tool_call ... </ns:tool_call>
  /<?[\w]+:tool_call[\s\S]*?<\/[\w]+:tool_call>\s*/g,
];
// Unclosed tool markup at end of stored content (EOS mid-call): drop to the end.
var TOOL_MARKUP_TAIL = new RegExp(
  '(?:' + _openersAlt + '|<tool_call>|<function_call>|<function=|<invoke\\b)[\\s\\S]*$',
  'i',
);
// Stray single tags left behind by partial blocks.
var STRAY_TOOL_TAGS = /<\/?(?:tool_call|function_call|function|parameter|invoke)(?:=[^>]*)?>|<tool_call\|>|<parameter=\w+>/gi;

function _stripToolMarkup(text) {
  var out = text;
  for (var i = 0; i < TOOL_MARKUP_PATTERNS.length; i++) out = out.replace(TOOL_MARKUP_PATTERNS[i], '');
  out = out.replace(TOOL_MARKUP_TAIL, '');
  out = out.replace(STRAY_TOOL_TAGS, '');
  return out;
}
function _stripReasoningMarkup(text) {
  var out = text;
  for (var i = 0; i < REASONING_DELIMITERS.length; i++) {
    var d = REASONING_DELIMITERS[i];
    out = out.replace(new RegExp(_esc(d.open) + '[\\s\\S]*?' + _esc(d.close) + '\\s*', 'gi'), '');
  }
  // Channel "final" marker with no analysis block: keep what follows it.
  out = out.replace(new RegExp(_esc(REASONING_DELIMITERS[0].close), 'gi'), '');
  // Unclosed opener: everything after it is reasoning, never answer.
  for (var j = 0; j < REASONING_DELIMITERS.length; j++) {
    out = out.replace(new RegExp(_esc(REASONING_DELIMITERS[j].open) + '[\\s\\S]*$', 'i'), '');
  }
  out = out.replace(/<\/think>|<channel\|>|<\|channel\|>|<\|message\|>/gi, '');
  return out;
}
/** Strip all control tokens, reasoning delimiters and tool-call markup from stored content. */
function stripChatControlTokens(content) {
  var out = typeof content === 'string' ? content : '';
  out = _stripReasoningMarkup(out);
  for (var i = 0; i < CONTROL_TOKEN_PATTERNS.length; i++) out = out.replace(CONTROL_TOKEN_PATTERNS[i], '');
  out = _stripToolMarkup(out);
  return out.trim();
}

/**
 * THE display parse: reasoning + clean answer. Contract:
 * __tests__/unit/components/ChatMessage/parseModelOutput.contract.test.ts
 */
function parseChatModelOutput(content, reasoningContent) {
  var parsed = parseChatThinkingContent(typeof content === 'string' ? content : '');
  var answer = stripChatControlTokens(parsed.response);
  var reasoning = parsed.thinking;
  var label = parsed.thinkingLabel;
  var complete = parsed.isThinkingComplete;
  if (typeof reasoningContent === 'string' && reasoningContent.trim()) {
    var sep = _withLabel({ thinking: null }, reasoningContent);
    reasoning = sep.thinking;
    if (!label && sep.thinkingLabel) label = sep.thinkingLabel;
    // Separate reasoning channel (litert / remote): no closing tag ever arrives in content, so the
    // reasoning is complete only once the answer has started. Consumers gate this on isStreaming.
    if (parsed.thinking == null) complete = answer.trim().length > 0;
  }
  if (typeof reasoning === 'string') reasoning = _stripToolMarkup(reasoning).trim();
  var out = {
    reasoning: reasoning == null ? null : reasoning,
    answer: answer,
    isReasoningComplete: complete,
  };
  if (label) out.reasoningLabel = label;
  return out;
}
/** Inline-only split (no separate reasoning channel). */
function splitInlineReasoning(content) {
  var p = parseChatModelOutput(content, null);
  var out = { answer: p.answer, reasoning: p.reasoning == null ? undefined : p.reasoning, isReasoningComplete: p.isReasoningComplete };
  if (p.reasoningLabel) out.reasoningLabel = p.reasoningLabel;
  return out;
}

// =============================================================================================
// Chat presentation helpers
// =============================================================================================
/** Escape digit*digit so markdown-it does not read multiplication as emphasis. */
function preprocessChatMarkdown(text) {
  if (typeof text !== 'string') return '';
  return text.replace(/(\d)\*(?=\d)/g, '$1\\*');
}
var SAFE_URL_PROTOCOLS = ['http:', 'https:', 'mailto:'];
/** Only http(s)/mailto links leave the app; anything else (javascript:, file:, intent:) is refused. */
function safeChatExternalUrl(url) {
  if (typeof url !== 'string') return null;
  var trimmed = url.trim();
  var m = /^([a-z][a-z0-9+.-]*:)/i.exec(trimmed);
  if (!m) return null;
  if (SAFE_URL_PROTOCOLS.indexOf(m[1].toLowerCase()) === -1) return null;
  if (/[\s<>"]/.test(trimmed)) return null;
  return trimmed;
}
var PREVIEW_MAX_CHARS = 100;
function chatListPreviewLine(role, content) {
  if (typeof content !== 'string' || !content) return '';
  var body = role === 'assistant' ? parseChatModelOutput(content, null).answer : stripChatControlTokens(content);
  body = body
    .replace(/```[\s\S]*?(```|$)/g, ' [code] ')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' [image] ')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[*_`#>]+/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (body.length > PREVIEW_MAX_CHARS) body = body.slice(0, PREVIEW_MAX_CHARS - 1).trimEnd() + '…';
  return role === 'user' ? 'You: ' + body : body;
}

/** imageGenerationHelpers.ts:95 (`<think>__LABEL:Enhanced prompt__`), ThinkingBlock.tsx:29 ('Enhanced'). */
var PROMPT_ENHANCEMENT_REASONING_LABEL = 'Enhanced prompt';
/** src/services/sync/mutation.ts:197 comment ("Enhancing your prompt..."). */
var PROMPT_ENHANCEMENT_STATUS = 'Enhancing your prompt...';

var RUNTIME_NOTICE_PATTERN = /^_?\s*(?:model (?:loaded|unloaded|load failed)|loading model|switched to)\b/i;
/** Device state written by the app, not conversation (heuristic for unflagged rows: RECONSTRUCTED). */
function isRuntimeOnlyMessage(input) {
  if (!input) return false;
  if (input.notice === true) return true;
  if (input.role !== 'assistant' && input.role !== 'system') return false;
  return typeof input.content === 'string' && RUNTIME_NOTICE_PATTERN.test(input.content.trim());
}
/** A labelled, answer-less reasoning card that supports the next message (prompt enhancement). */
function isSupportingChatContext(input) {
  if (!input || !input.reasoningLabel) return false;
  var answer = typeof input.answer === 'string' ? input.answer.trim() : '';
  return input.reasoningLabel === PROMPT_ENHANCEMENT_REASONING_LABEL && !answer;
}

// =============================================================================================
// Pro access rule
// =============================================================================================
/** admission: 'unknown' | 'active' | 'inactive'. Unknown is not revoked (proAccessSlice.ts:15-25). */
function hasProAccess(input) {
  if (!input || !input.hasCredential) return false;
  return input.admission !== 'inactive';
}
var _MESH_FAILURES = {
  invalid_key: { title: 'Invalid license key', description: 'Check the key and try again.' },
  invalid_credential: { title: 'License not accepted', description: 'That license key is invalid or revoked.' },
  expired: { title: 'License expired', description: 'This license key has expired.' },
  device_cap_reached: { title: 'Device limit reached', description: 'Deactivate another device to use Pro here.' },
  network: { title: 'No connection', description: 'Connect to the internet and try again.' },
  registration_failed: { title: 'Activation failed', description: 'Something went wrong. Please try again.' },
};
function projectPersonalMeshActivationFailure(reason) {
  var hit = _MESH_FAILURES[reason] || _MESH_FAILURES.registration_failed;
  return { reason: reason, title: hit.title, description: hit.description };
}

// =============================================================================================
// Live chat stream previews (paired devices; empty in the free build)
// =============================================================================================
var _STATUS_PHASES = ['waiting', 'loading', 'loading_model', 'loading_image_model', 'tool', 'tools', 'searching', 'generating_image', 'enhancing'];
var _PHASE_LABELS = {
  waiting: 'Waiting...',
  loading: 'Loading model...',
  loading_model: 'Loading model...',
  loading_image_model: 'Loading image model...',
  thinking: 'Thinking...',
  tool: 'Using tools...',
  tools: 'Using tools...',
  searching: 'Searching...',
  generating_image: 'Generating image...',
  enhancing: PROMPT_ENHANCEMENT_STATUS,
};
function chatStreamPhaseIsStatus(phase) {
  return _STATUS_PHASES.indexOf(phase) !== -1;
}
function chatStreamPhaseLabel(phase, progress) {
  var label = _PHASE_LABELS[phase];
  if (!label) return undefined;
  var done = progress && (typeof progress.current === 'number' ? progress.current : progress.step);
  if (typeof done === 'number' && typeof progress.total === 'number' && progress.total > 0) {
    return label + ' ' + done + '/' + progress.total;
  }
  return label;
}
function chatStreamPreviewHasMessageBody(preview) {
  return !!(preview && typeof preview.content === 'string' && /[^\s\p{P}]/u.test(preview.content));
}
function chatStreamPreviewRows(previews, activeConversationId) {
  var list = Array.isArray(previews) ? previews : previews && typeof previews === 'object' ? Object.values(previews) : [];
  var rows = [];
  for (var i = 0; i < list.length; i++) {
    var p = list[i];
    if (!p || (activeConversationId && p.conversationId !== activeConversationId)) continue;
    var row = Object.assign({}, p);
    row.id = 'remote-stream:' + (p.deviceId || 'peer') + ':' + (p.messageId || p.generationId || i);
    row.content = typeof p.content === 'string' ? p.content : '';
    rows.push(row);
  }
  return rows;
}
function startChatStreamTool(tools, name) {
  return (Array.isArray(tools) ? tools : []).concat([{ name: name, status: 'running' }]);
}
function completeChatStreamTool(tools, name, result) {
  var list = (Array.isArray(tools) ? tools : []).slice();
  for (var i = 0; i < list.length; i++) {
    if (list[i].name === name && list[i].status === 'running') {
      list[i] = Object.assign({}, list[i], { status: 'completed', result: result });
      return list;
    }
  }
  list.push({ name: name, status: 'completed', result: result });
  return list;
}

// =============================================================================================
// Message context (peer-safe serialization)
// =============================================================================================
function _prune(value) {
  if (Array.isArray(value)) {
    var arr = value.map(_prune).filter(function (v) { return v !== undefined; });
    return arr.length ? arr : undefined;
  }
  if (value && typeof value === 'object') {
    var out = {};
    var any = false;
    for (var k in value) {
      if (!Object.prototype.hasOwnProperty.call(value, k)) continue;
      var v = _prune(value[k]);
      if (v !== undefined) {
        out[k] = v;
        any = true;
      }
    }
    return any ? out : undefined;
  }
  if (value === null || value === undefined || value === '') return undefined;
  if (typeof value === 'number' && !Number.isFinite(value)) return undefined;
  return value;
}
function serializeSyncedMessageContext(input) {
  var pruned = _prune(input || {});
  if (!pruned) return null;
  try {
    return JSON.stringify(pruned);
  } catch (e) {
    return null;
  }
}
/** STUB-ish projection of a peer row (only pro consumes it). */
function projectSyncedMessageTurn(input) {
  if (!input || typeof input.content !== 'string') return null;
  var ctx = null;
  if (typeof input.context === 'string' && input.context) {
    try {
      ctx = JSON.parse(input.context);
    } catch (e) {
      ctx = null;
    }
  } else if (input.context && typeof input.context === 'object') ctx = input.context;
  var out = { role: input.role, content: input.content };
  if (ctx) {
    if (ctx.reasoning) out.reasoningContent = ctx.reasoning;
    if (ctx.timeline) out.timeline = ctx.timeline;
    if (ctx.toolCalls) out.toolArtifacts = ctx.toolCalls;
    if (ctx.notice) out.isSystemInfo = true;
    if (ctx.status) out.turnStatus = ctx.status;
  }
  return out;
}

// =============================================================================================
// Model transfer admission
// =============================================================================================
var _MODEL_KINDS = ['text', 'vision', 'image'];
function _safeFileName(name) {
  return typeof name === 'string' && name.length > 0 && !/[\\/]/.test(name) && name !== '.' && name !== '..';
}
/** null = admissible; otherwise a reason string. */
function ogamModelTransferBlocker(manifest) {
  if (!manifest || typeof manifest !== 'object') return 'missing_manifest';
  if (typeof manifest.id !== 'string' || !manifest.id) return 'missing_id';
  if (typeof manifest.name !== 'string' || !manifest.name) return 'missing_name';
  if (_MODEL_KINDS.indexOf(manifest.kind) === -1) return 'unsupported_kind';
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) return 'missing_files';
  for (var i = 0; i < manifest.files.length; i++) {
    var f = manifest.files[i];
    if (!f || !_safeFileName(f.name)) return 'unsafe_file_name';
    if (typeof f.sizeBytes !== 'number' || !Number.isFinite(f.sizeBytes) || f.sizeBytes <= 0) return 'invalid_file_size';
  }
  return null;
}

// =============================================================================================
// Entities and wire constants (names RECONSTRUCTED to match the snake_case 'model_setting' family)
// =============================================================================================
var KNOWLEDGE_DOCUMENT_ENTITY = 'knowledge_document';
var SHARED_FILE_ENTITY = 'shared_file';
var TASK_LAUNCH_ENTITY = 'task_launch';
var TASK_RUN_ENTITY = 'task_run';
var TASK_CONTROL_ENTITY = 'task_control';
var TASK_VISUAL_STEP_ENTITY = 'task_visual_step';
var CLIPBOARD_CHANNEL = 'clipboard';
var TASK_GUIDANCE_CHANNEL = 'task_guidance';
var AMBIENT_SHARE_ANY_DESTINATION = '*';
var SHARED_FILE_MIME = 'application/x-offgrid-shared-file';
var MODEL_TRANSFER_MIME = 'application/x-offgrid-model';
var OFFGRID_SYNC_PORT = 47800;
var CHUNK_SIZE = 1024 * 1024;
var BLOB_FRAME_BYTES = 64 * 1024;
var BLOB_TOKEN_TTL_MS = 60 * 1000;
var MAX_CLIPBOARD_TEXT_BYTES = 64 * 1024;
var MAX_SHARED_FILE_BYTES = 2 * 1024 * 1024 * 1024;
var MAX_SYNCED_TASK_VISUAL_STEPS = 50;
var DEFAULT_COMPLETED_TRANSFER_HISTORY_LIMIT = 50;
var DEFAULT_RECEIVE_POLICY = 'ask';
var PAIRING_CODE_ALPHABET = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ';
var PAIRING_CODE_LENGTH = 6;
var PAIRING_QR_VALIDITY_MS = 5 * 60 * 1000;
var PERSONAL_MESH_DEVICE_CAP = 5;
var PERSONAL_MESH_ENTITLEMENT_REVALIDATION_INTERVAL_MS = 24 * 60 * 60 * 1000;

function _snake(obj) {
  var out = {};
  if (!obj || typeof obj !== 'object') return out;
  for (var k in obj) {
    if (!Object.prototype.hasOwnProperty.call(obj, k)) continue;
    var v = obj[k];
    if (v === undefined || typeof v === 'function') continue;
    out[k.replace(/[A-Z]/g, function (c) { return '_' + c.toLowerCase(); })] = v;
  }
  return out;
}
function createKnowledgeDocumentStateFields(doc) {
  return _snake(doc);
}
function createSharedFileStateFields(file) {
  return _snake(file);
}
function createSharedFileTransferMetadata(file) {
  return _snake(file);
}
function sharedFileActivityId(syncId) {
  return 'shared-file:' + syncId;
}
function syncFileCompletionNotificationId(fact) {
  return 'sync-file:' + (fact && (fact.syncId || fact.transferId || fact.id));
}
function taskVisualStepId(runId, index) {
  return 'task-step:' + runId + ':' + index;
}
function projectSharedFilePreview(file) {
  return file ? { name: file.name, mimeType: file.mimeType, sizeBytes: file.sizeBytes } : null;
}
function isClipboardAckMessage(msg) {
  return !!msg && typeof msg === 'object' && msg.type === 'clipboard_ack';
}
function isDialableAddress(address) {
  return typeof address === 'string' && address.length > 0 && !/^(0\.0\.0\.0|::)$/.test(address);
}
function parsePairingCode(input) {
  if (typeof input !== 'string') return null;
  var code = input.toUpperCase().replace(/[^0-9A-Z]/g, '');
  if (code.length !== PAIRING_CODE_LENGTH) return null;
  for (var i = 0; i < code.length; i++) if (PAIRING_CODE_ALPHABET.indexOf(code[i]) === -1) return null;
  return code;
}
function encodePairingQrPayload(payload) {
  return 'offgrid-pair:' + JSON.stringify(payload || {});
}
function parsePairingQrPayload(text) {
  if (typeof text !== 'string' || text.indexOf('offgrid-pair:') !== 0) return null;
  try {
    return JSON.parse(text.slice('offgrid-pair:'.length));
  } catch (e) {
    return null;
  }
}
function createTxtRecord(info) {
  return _snake(info);
}

// =============================================================================================
// STUBS: mesh engine, transports, discovery, transfer, blobs, checksums
// =============================================================================================
function _noop() {
  return Promise.resolve(undefined);
}
var _SYNC_NOOP_METHODS = {
  on: function () { return function () {}; },
  off: function () {},
  subscribe: function () { return function () {}; },
  addListener: function () { return { remove: function () {} }; },
  removeAllListeners: function () {},
};
function _inertClass(name) {
  function Inert() {
    var self = this;
    if (typeof Proxy !== 'function') return;
    return new Proxy(self, {
      get: function (target, prop) {
        if (prop in target) return target[prop];
        if (typeof prop === 'symbol') return undefined;
        if (Object.prototype.hasOwnProperty.call(_SYNC_NOOP_METHODS, prop)) return _SYNC_NOOP_METHODS[prop];
        if (prop === 'then') return undefined;
        return _noop;
      },
    });
  }
  Inert.prototype.__offgridStub = name;
  Inert.prototype.start = _noop;
  Inert.prototype.stop = _noop;
  Inert.prototype.close = _noop;
  Inert.prototype.dispose = function () {};
  Inert.prototype.on = _SYNC_NOOP_METHODS.on;
  Inert.prototype.off = _SYNC_NOOP_METHODS.off;
  Inert.prototype.subscribe = _SYNC_NOOP_METHODS.subscribe;
  return Inert;
}
var SyncEngine = _inertClass('SyncEngine');
var MultiTransportBridge = _inertClass('MultiTransportBridge');
var DiscoveryOrchestrator = _inertClass('DiscoveryOrchestrator');
var CompositeDiscoveryService = _inertClass('CompositeDiscoveryService');
var FileTransferManager = _inertClass('FileTransferManager');
var OpLog = _inertClass('OpLog');
var StateSync = _inertClass('StateSync');

function IncrementalChecksum() {
  this._bytes = 0;
}
IncrementalChecksum.prototype.update = function (chunk) {
  this._bytes += chunk && chunk.length ? chunk.length : 0;
  return this;
};
IncrementalChecksum.prototype.digest = function () {
  return null; // STUB: no hash implementation in this reconstruction.
};
function checksumFromSha512Hex(hex) {
  return typeof hex === 'string' && hex ? 'sha512:' + hex.toLowerCase() : null;
}
function blobKeyBase64() {
  return null; // STUB
}
function createBlobMaterial() {
  return null; // STUB
}

function PersonalMeshEntitlementError(message, reason) {
  var err = new Error(message || 'Personal mesh entitlement error');
  Object.setPrototypeOf(err, PersonalMeshEntitlementError.prototype);
  err.name = 'PersonalMeshEntitlementError';
  err.reason = reason;
  return err;
}
PersonalMeshEntitlementError.prototype = Object.create(Error.prototype);
PersonalMeshEntitlementError.prototype.constructor = PersonalMeshEntitlementError;

// =============================================================================================
exports.REASONING_DELIMITERS = REASONING_DELIMITERS;
exports.TOOL_CALL_OPENERS = TOOL_CALL_OPENERS;
exports.TOOL_CALL_CLOSERS = TOOL_CALL_CLOSERS;
exports.XML_TOOL_CALL_FUNCTION_MARKER = XML_TOOL_CALL_FUNCTION_MARKER;
exports.XML_TOOL_CALL_PARAMETER_MARKER = XML_TOOL_CALL_PARAMETER_MARKER;
exports.partialTagSuffix = partialTagSuffix;
exports.maxPartialTagSuffix = maxPartialTagSuffix;
exports.parseChatThinkingContent = parseChatThinkingContent;
exports.parseChatModelOutput = parseChatModelOutput;
exports.stripChatControlTokens = stripChatControlTokens;
exports.splitInlineReasoning = splitInlineReasoning;
exports.preprocessChatMarkdown = preprocessChatMarkdown;
exports.safeChatExternalUrl = safeChatExternalUrl;
exports.chatListPreviewLine = chatListPreviewLine;
exports.PROMPT_ENHANCEMENT_REASONING_LABEL = PROMPT_ENHANCEMENT_REASONING_LABEL;
exports.PROMPT_ENHANCEMENT_STATUS = PROMPT_ENHANCEMENT_STATUS;
exports.isRuntimeOnlyMessage = isRuntimeOnlyMessage;
exports.isSupportingChatContext = isSupportingChatContext;
exports.hasProAccess = hasProAccess;
exports.projectPersonalMeshActivationFailure = projectPersonalMeshActivationFailure;
exports.chatStreamPhaseIsStatus = chatStreamPhaseIsStatus;
exports.chatStreamPhaseLabel = chatStreamPhaseLabel;
exports.chatStreamPreviewHasMessageBody = chatStreamPreviewHasMessageBody;
exports.chatStreamPreviewRows = chatStreamPreviewRows;
exports.startChatStreamTool = startChatStreamTool;
exports.completeChatStreamTool = completeChatStreamTool;
exports.serializeSyncedMessageContext = serializeSyncedMessageContext;
exports.projectSyncedMessageTurn = projectSyncedMessageTurn;
exports.ogamModelTransferBlocker = ogamModelTransferBlocker;
exports.KNOWLEDGE_DOCUMENT_ENTITY = KNOWLEDGE_DOCUMENT_ENTITY;
exports.SHARED_FILE_ENTITY = SHARED_FILE_ENTITY;
exports.TASK_LAUNCH_ENTITY = TASK_LAUNCH_ENTITY;
exports.TASK_RUN_ENTITY = TASK_RUN_ENTITY;
exports.TASK_CONTROL_ENTITY = TASK_CONTROL_ENTITY;
exports.TASK_VISUAL_STEP_ENTITY = TASK_VISUAL_STEP_ENTITY;
exports.CLIPBOARD_CHANNEL = CLIPBOARD_CHANNEL;
exports.TASK_GUIDANCE_CHANNEL = TASK_GUIDANCE_CHANNEL;
exports.AMBIENT_SHARE_ANY_DESTINATION = AMBIENT_SHARE_ANY_DESTINATION;
exports.SHARED_FILE_MIME = SHARED_FILE_MIME;
exports.MODEL_TRANSFER_MIME = MODEL_TRANSFER_MIME;
exports.OFFGRID_SYNC_PORT = OFFGRID_SYNC_PORT;
exports.CHUNK_SIZE = CHUNK_SIZE;
exports.BLOB_FRAME_BYTES = BLOB_FRAME_BYTES;
exports.BLOB_TOKEN_TTL_MS = BLOB_TOKEN_TTL_MS;
exports.MAX_CLIPBOARD_TEXT_BYTES = MAX_CLIPBOARD_TEXT_BYTES;
exports.MAX_SHARED_FILE_BYTES = MAX_SHARED_FILE_BYTES;
exports.MAX_SYNCED_TASK_VISUAL_STEPS = MAX_SYNCED_TASK_VISUAL_STEPS;
exports.DEFAULT_COMPLETED_TRANSFER_HISTORY_LIMIT = DEFAULT_COMPLETED_TRANSFER_HISTORY_LIMIT;
exports.DEFAULT_RECEIVE_POLICY = DEFAULT_RECEIVE_POLICY;
exports.PAIRING_CODE_ALPHABET = PAIRING_CODE_ALPHABET;
exports.PAIRING_CODE_LENGTH = PAIRING_CODE_LENGTH;
exports.PAIRING_QR_VALIDITY_MS = PAIRING_QR_VALIDITY_MS;
exports.PERSONAL_MESH_DEVICE_CAP = PERSONAL_MESH_DEVICE_CAP;
exports.PERSONAL_MESH_ENTITLEMENT_REVALIDATION_INTERVAL_MS = PERSONAL_MESH_ENTITLEMENT_REVALIDATION_INTERVAL_MS;
exports.createKnowledgeDocumentStateFields = createKnowledgeDocumentStateFields;
exports.createSharedFileStateFields = createSharedFileStateFields;
exports.createSharedFileTransferMetadata = createSharedFileTransferMetadata;
exports.sharedFileActivityId = sharedFileActivityId;
exports.syncFileCompletionNotificationId = syncFileCompletionNotificationId;
exports.taskVisualStepId = taskVisualStepId;
exports.projectSharedFilePreview = projectSharedFilePreview;
exports.isClipboardAckMessage = isClipboardAckMessage;
exports.isDialableAddress = isDialableAddress;
exports.parsePairingCode = parsePairingCode;
exports.encodePairingQrPayload = encodePairingQrPayload;
exports.parsePairingQrPayload = parsePairingQrPayload;
exports.createTxtRecord = createTxtRecord;
exports.SyncEngine = SyncEngine;
exports.MultiTransportBridge = MultiTransportBridge;
exports.DiscoveryOrchestrator = DiscoveryOrchestrator;
exports.CompositeDiscoveryService = CompositeDiscoveryService;
exports.FileTransferManager = FileTransferManager;
exports.OpLog = OpLog;
exports.StateSync = StateSync;
exports.IncrementalChecksum = IncrementalChecksum;
exports.checksumFromSha512Hex = checksumFromSha512Hex;
exports.blobKeyBase64 = blobKeyBase64;
exports.createBlobMaterial = createBlobMaterial;
exports.PersonalMeshEntitlementError = PersonalMeshEntitlementError;
exports.__inertClass = _inertClass;
