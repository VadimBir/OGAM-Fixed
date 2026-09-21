# 03 — Tool-Calling, Skills, and System-Prompt Construction

Covers requirement item 6:
> (A) model can tool-call on demand; (B) an OPEN, user-viewable system prompt; (C) a skills
> system — model already has partial skills but NO "generate image" skill; user wants a skill
> that lets the model call an image-gen tool whose result shows up in chat.

## 6A — Tool-calling infrastructure (EXISTS, robust)

### Built-in tool registry — `src/services/tools/registry.ts`
`AVAILABLE_TOOLS` `:3-89` (6 tools): `web_search` `:4`, `calculator` `:19`,
`get_current_datetime` `:33`, `get_device_info` `:46`, `search_knowledge_base` `:60`,
`read_url` `:74`. **No image-generation tool** (this is the 6C gap).
- `getToolsAsOpenAISchema(enabledToolIds)` `:91-117` — filters to enabled ids, emits OpenAI
  function schemas.
- `buildToolSystemPromptHint(enabledToolIds)` `:119-125` — text list of tools for models without
  native FC.

### Handlers — `src/services/tools/handlers.ts`
`executeToolCall` `:15` → `dispatchTool` switch `:26-52` (one case per built-in tool). Notable:
`handleWebSearch` `:54-84` scrapes Brave HTML on-device; `handleReadUrl` `:343-374` fetches+SSRF-guards;
`handleSearchKnowledgeBase` `:376-384` calls `ragService`.

### Extension seam ("skills"-like) — `src/services/tools/extensions.ts`
`ToolExtension` interface `:3-27` (`getSystemPromptHint`, `getOpenAISchemas`, `getToolDefinitions`,
`parseToolCalls`, `execute`, `canHandle`, `enabledToolCount`, `enableAssistantTools`).
`registerToolExtension` `:32-46` / `getToolExtensions` `:54-56`. **This is how Pro/MCP add tools at
runtime** — the closest thing to a pluggable "skills" registry (see 6C).

### The loop — `src/services/generationToolLoop.ts` (1665 lines)
- `runToolLoop(ctx)` `:1396` — call LLM → parse tool calls → execute → re-inject results → repeat
  until a final answer or `maxToolCalls` (`:38-41`).
- Native tool calling: `nativeToolCalling` (`:1077-1080`) true for LiteRT+conversation, remote
  providers, or llama with a Jinja tool template (`llmService.supportsToolCalling()`).
- Text-parse fallback for small models: `parseToolCallsFromText` `:280`, Gemma native
  (`parseGemmaNativeToolCalls` `:217`), `<invoke>` blocks `:255`, XML/JSON bodies.
- Per-engine dispatch: `callLiteRTForLoop` `:794`, `callRemoteLLMWithTools` `:556`,
  `callLocalWithRetry` `:612`.
- On-device tool routing to keep prefill small: `selectEffectiveSchemas` `:1281` (embedding router
  + LiteRT/llama-iOS selection).
- Tool result → chat: `executeToolCalls` `:403-460` pushes a `role:'tool'` message into
  `chatStore` `:456`; the LiteRT handler `buildLiteRTToolCallHandler` `:710-792` does the same
  (`:781-782`).

**Status 6A: fully present.** "On demand" (the model chooses when) already works via native FC
and text-parse. Gap: tools must be enabled (`ctx.enabledToolIds`) and there is no image tool.

## 6B — System prompt (PARTIALLY open)

- Source of truth: `settings.systemPrompt` (`appStore.ts:50`, default
  `APP_CONFIG.defaultSystemPrompt` at `src/constants/index.ts:110`).
- **User-editable UI EXISTS:** `ModelSettingsScreen/SystemPromptSection.tsx:18-26` — a plain
  `TextInput` bound to `updateSettings({ systemPrompt })`. Help text `:15-17`: "Used when chatting
  without a project selected."
- Per-persona override: each Project carries its own `systemPrompt` (`projectStore.ts`; see 04).
- Injected into the request: `providers/openAIMessageBuilder.ts:62-64`
  (`options.systemPrompt || settings.systemPrompt`); LiteRT reads the system message
  (`generationToolLoop.ts:810-812`).
- **What is NOT user-visible (the "open" gap):** at send time
  `augmentSystemPromptForTools` (`generationToolLoop.ts:966-1036`) appends, invisibly to the user:
  `TOOL_BEHAVIOR_GUIDANCE` (`:908-909`), a date-context block (`buildDateContext` `:946-951`),
  extension/MCP hints (`getSystemPromptHint` `:990`), and (for calendar tools) an exact-time note
  on the latest user message (`:960-963`). The DebugSheet shows only the base
  (`components/DebugSheet.tsx:106`).

**Status 6B:** base prompt is editable and viewable; the runtime-augmented final prompt is not
exposed. An "open, user-viewable system prompt" goal means surfacing the assembled prompt
(base + persona + tool guidance + date + hints), and letting the user own/disable the augmentation.

## 6C — Skills system (NOT a named concept; nearest = tools + extensions + MCP)

Search method: `grep -rniE "\bskill" src` → only prose matches (e.g. a project template that says
"conversation skills", `projectStore.ts:55`). There is **no module, store, or type named "skill".**
What the user calls "partial skills" maps to:
- **Built-in tools** (`registry.ts`, 6 of them) — always-available capabilities.
- **Tool extensions** (`extensions.ts`) — runtime-registered capability packs (Pro tools, MCP).
- **The "assistant"** — `ASSISTANT_TOOL_NAMES = {web_use, computer_use}`
  (`generationToolLoop.ts:357`), gated by `assistantEnabled` (`:380-382`).
- **MCP** — `mcpContextBoost.ts`, `toolEmbeddingRouter.ts`, routed in `selectEffectiveSchemas`.

**The missing "generate image" skill.** Today image generation is triggered by a *heuristic
intent classifier*, NOT by the model deciding to call a tool: `intentClassifier.ts` (IMAGE_PATTERNS
regex `:20-41`, e.g. avatar/wallpaper/logo phrases) routes a user message to
`imageGenerationService.generateImage()`. The model never sees an `generate_image` tool. To meet
6C, add a `generate_image` tool to `AVAILABLE_TOOLS` (registry.ts) + a handler in `handlers.ts`
that calls `imageGenerationService.generateImage({ prompt, conversationId })`, so the result is
added to chat via the existing `messageId`/`conversationId` seam
(`imageGenerationService.ts:367,378-390`; result save `imageGenerationResult.ts`). The tool result
already flows back into chat as a `role:'tool'` message via the loop
(`generationToolLoop.ts:446-456`).

**Status 6C:** infra to add the skill exists (registry + handler + extension seam + in-chat image
render). The specific "generate image" tool is absent; wiring it is a small, well-seamed change.

## Key wiring facts for the expansion
- Enabled tools come from `ctx.enabledToolIds` (managed in `ToolsScreen` + `settings`); a new tool
  must be added to that enable list to be offered.
- Native-FC models are handed schemas structurally; text-only small models get the text hint —
  `augmentSystemPromptForTools` suppresses the hint when `nativeToolCalling` (`:987-992`).
