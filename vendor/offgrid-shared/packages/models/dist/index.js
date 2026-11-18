'use strict';
/*
 * @offgrid/models - LOCAL RECONSTRUCTION (private off-grid-ai/shared is absent).
 * Every shape below is derived from the mobile call sites; see
 * OGAM-Fixed/docs/refactor-map/SHARED_RECONSTRUCTION_FIXES.md for Fix 1-3.
 */
Object.defineProperty(exports, '__esModule', { value: true });

function _finite(v) {
  return typeof v === 'number' && Number.isFinite(v);
}
function _positive(v) {
  return _finite(v) && v > 0 ? v : undefined;
}

// ---- Thinking / reasoning budget -------------------------------------------------------------
// REASONING_BUDGET_AUTO = -1 (src/stores/appStore.ts:136, __tests__/unit/services/thinkingBudget.test.ts:6)
var REASONING_BUDGET_AUTO = -1;
// Fix 1: number[] (numeric compare / indexOf at the slider consumer).
var REASONING_BUDGET_OPTIONS = [256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768];
function reasoningBudgetLabel(v) {
  return v == null || v === -1 ? 'Auto' : v <= 0 ? 'Off' : String(v) + ' tokens';
}
function _positiveBudget(enable, budget) {
  if (!enable) return undefined;
  if (!_finite(budget) || budget <= 0) return undefined;
  return Math.round(budget);
}
// Fix 2: (enableThinking, budget). llama.rn fragment: { thinking_budget_tokens?: number }
function thinkingBudgetPayload(enable, budget) {
  var b = _positiveBudget(enable, budget);
  return b === undefined ? {} : { thinking_budget_tokens: b };
}
// Remote llama-server / OpenAI-compatible: { thinking: { type, budget_tokens? } }
function reasoningBudgetPayload(enable, budget) {
  var b = _positiveBudget(enable, budget);
  var thinking = { type: enable ? 'enabled' : 'disabled' };
  if (b !== undefined) thinking.budget_tokens = b;
  return { thinking: thinking };
}
// OpenRouter: { reasoning: { max_tokens? } }
function openRouterReasoningPayload(enable, budget) {
  var b = _positiveBudget(enable, budget);
  return { reasoning: b === undefined ? {} : { max_tokens: b } };
}

// ---- Tool loop limits ------------------------------------------------------------------------
// Default 25: __tests__/unit/stores/appStore.test.ts:289. Range 1..100: src/services/sync/mutation.ts
// (maxToolCalls accepts integerInRange(1, 100)).
var MIN_MAX_TOOL_CALLS = 1;
var MAX_MAX_TOOL_CALLS = 100;
var DEFAULT_MAX_TOOL_CALLS = 25;
function normalizeMaxToolCalls(value) {
  var n = typeof value === 'string' ? Number(value) : value;
  if (!_finite(n)) return DEFAULT_MAX_TOOL_CALLS;
  return Math.min(MAX_MAX_TOOL_CALLS, Math.max(MIN_MAX_TOOL_CALLS, Math.round(n)));
}
function toolLimitFinalAnswerInstruction(maximum) {
  var max = normalizeMaxToolCalls(maximum);
  return (
    'You have reached the limit of ' + max + ' tool call' + (max === 1 ? '' : 's') +
    ' for this response. Do not call any more tools. Answer the user now using only the ' +
    'information you already have, and say plainly if something could not be completed.'
  );
}

// ~24k chars ~= 6k tokens (src/services/tools/toolResult.ts comment).
var MAX_TOOL_RESULT_CHARS = 24000;
var _CHARS_PER_TOKEN = 4;
var _MIN_TOOL_RESULT_CHARS = 1000;
function boundToolResult(name, content, maxChars) {
  var text = typeof content === 'string' ? content : content == null ? '' : String(content);
  var cap = _finite(maxChars) && maxChars > 0 ? Math.floor(maxChars) : MAX_TOOL_RESULT_CHARS;
  if (text.length <= cap) return text;
  return (
    text.slice(0, cap) +
    '\n\n[Result of tool "' + String(name) + '" truncated: showing the first ' + cap + ' of ' +
    text.length + ' characters.]'
  );
}
function _charsOf(value) {
  if (value == null) return 0;
  if (typeof value === 'string') return value.length;
  try {
    return JSON.stringify(value).length;
  } catch (e) {
    return 0;
  }
}
function toolPromptChars(messages, tools) {
  var total = 0;
  var list = Array.isArray(messages) ? messages : [];
  for (var i = 0; i < list.length; i++) {
    var m = list[i];
    if (m == null) continue;
    if (typeof m === 'string') {
      total += m.length;
      continue;
    }
    total += _charsOf(m.content);
    if (typeof m.reasoningContent === 'string') total += m.reasoningContent.length;
    if (m.toolCalls) total += _charsOf(m.toolCalls);
  }
  if (Array.isArray(tools) && tools.length > 0) total += _charsOf(tools);
  return total;
}
function toolResultCharBudget(input) {
  var ctx = input && _positive(input.contextLength);
  if (!ctx) return MAX_TOOL_RESULT_CHARS;
  var promptTokens = Math.ceil((_finite(input.promptChars) ? Math.max(0, input.promptChars) : 0) / _CHARS_PER_TOKEN);
  var reserve = _finite(input.replyReserveTokens) ? Math.max(0, input.replyReserveTokens) : 0;
  var freeChars = (ctx - promptTokens - reserve) * _CHARS_PER_TOKEN;
  return Math.max(_MIN_TOOL_RESULT_CHARS, Math.min(MAX_TOOL_RESULT_CHARS, Math.floor(freeChars)));
}
function finalResponseFromToolResults(prefix, results) {
  var parts = [];
  if (typeof prefix === 'string' && prefix.trim()) parts.push(prefix.trim());
  var list = Array.isArray(results) ? results : [];
  for (var i = 0; i < list.length; i++) {
    if (typeof list[i] === 'string' && list[i].trim()) parts.push(list[i].trim());
  }
  return parts.join('\n\n');
}

// ---- Image parameter policy (Fix 3) ---------------------------------------------------------
// SD1.5 defaults: steps 20, cfg 7.5, size 512.
var DEFAULT_IMAGE_STEPS = 20;
var DEFAULT_IMAGE_CFG_SCALE = 7.5;
var DEFAULT_IMAGE_SIZE = 512;
function resolveImageParameters(model, store) {
  var id = model && model.id;
  var entry = (store && id != null && store[id]) || {};
  return {
    steps: _positive(entry.steps) || DEFAULT_IMAGE_STEPS,
    cfgScale: _positive(entry.cfgScale) || DEFAULT_IMAGE_CFG_SCALE,
    size: _positive(entry.size) || DEFAULT_IMAGE_SIZE,
  };
}
function effectiveImageParameter(requestValue, fallback) {
  return _positive(requestValue) !== undefined ? requestValue : fallback;
}

exports.REASONING_BUDGET_AUTO = REASONING_BUDGET_AUTO;
exports.REASONING_BUDGET_OPTIONS = REASONING_BUDGET_OPTIONS;
exports.reasoningBudgetLabel = reasoningBudgetLabel;
exports.thinkingBudgetPayload = thinkingBudgetPayload;
exports.reasoningBudgetPayload = reasoningBudgetPayload;
exports.openRouterReasoningPayload = openRouterReasoningPayload;
exports.MIN_MAX_TOOL_CALLS = MIN_MAX_TOOL_CALLS;
exports.MAX_MAX_TOOL_CALLS = MAX_MAX_TOOL_CALLS;
exports.DEFAULT_MAX_TOOL_CALLS = DEFAULT_MAX_TOOL_CALLS;
exports.normalizeMaxToolCalls = normalizeMaxToolCalls;
exports.toolLimitFinalAnswerInstruction = toolLimitFinalAnswerInstruction;
exports.MAX_TOOL_RESULT_CHARS = MAX_TOOL_RESULT_CHARS;
exports.boundToolResult = boundToolResult;
exports.toolPromptChars = toolPromptChars;
exports.toolResultCharBudget = toolResultCharBudget;
exports.finalResponseFromToolResults = finalResponseFromToolResults;
exports.DEFAULT_IMAGE_STEPS = DEFAULT_IMAGE_STEPS;
exports.DEFAULT_IMAGE_CFG_SCALE = DEFAULT_IMAGE_CFG_SCALE;
exports.DEFAULT_IMAGE_SIZE = DEFAULT_IMAGE_SIZE;
exports.resolveImageParameters = resolveImageParameters;
exports.effectiveImageParameter = effectiveImageParameter;
