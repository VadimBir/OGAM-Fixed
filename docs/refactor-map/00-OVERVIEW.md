# 00 — Architecture Overview (OGAM / Off Grid AI Mobile)

React Native app. On-device LLM (llama.rn + LiteRT), on-device image gen (Android
local-dream subprocess / iOS Core ML), STT (whisper.rn), remote OpenAI-compatible
providers, and a Pro submodule (`pro/`) wired through registration seams.

Method note: searched with Grep/Glob for each area below; where a concept was absent I
tried synonyms (see the per-item notes in files 03/04). File:line refs throughout.

## src/ module map (1-line purpose each)

### services/ (business logic; the core of the app)
- `llm.ts` — llama.rn text-inference service (load/generate/unload); GPU/KV state. (498 lines)
- `llmHelpers.ts` — builds llama.rn context params (`buildModelParams`), init-with-CPU-fallback, KV-cache coercion.
- `providers/localProvider.ts` + `providers/openAICompatible*.ts` — LLMProvider abstraction (local llama vs remote server).
- `generationService.ts` + `generationToolLoop.ts` — orchestrates a chat turn; the tool-calling loop (1665 lines).
- `tools/` — built-in tool registry (`registry.ts`), handlers (`handlers.ts`), extension seam (`extensions.ts`), types.
- `intentClassifier.ts` — regex/model heuristic that routes a user message to text vs image generation.
- `imageGenerationService.ts` — image-gen state machine (idle→enhancing→loading→generating); routes local vs remote.
- `localDreamGenerator.ts` — **LIVE** image generator; talks to `LocalDreamModule` (Android) / `CoreMLDiffusionModule` (iOS).
- `imageGenerator.ts` — **DEAD/legacy** image generator; talks to nonexistent `ImageGeneratorModule` (see 02/CP-02).
- `remoteImageGeneration.ts` + `remoteMediaRuntime.ts` — remote image gen over a remote server.
- `imageParameterPolicy.ts`, `imageGenerationHelpers.ts`, `imageGenerationResult.ts`, `imagePromptEnhancement.ts` — image param resolution, phase text, result save, prompt enhance.
- `litert.ts` + `litertToolSelector.ts` + `liteRTCompaction.ts` — Google LiteRT engine (Android GPU/CPU/NPU) + its native tool loop.
- `engines.ts` — which text engine is active (llama vs LiteRT vs remote); `unloadAllTextEngines`.
- `activeModelService/` — load/unload/resolve the active text & image models; residency-aware (`loaders.ts` is the native call site).
- `modelResidency/` — memory budget: what stays loaded, eviction (`makeRoomFor`).
- `hardware.ts`, `memoryBudget.ts`, `cpuTopologyReader.ts`, `llmDeviceLimits.ts` — device RAM/CPU/GPU/NPU capability probing.
- `rag/` — knowledge-base embedding + retrieval (SQLite vector store) for the `search_knowledge_base` tool.
- `whisperService.ts` + `voiceService.ts` + `voiceSession.ts` — STT + voice mode.
- `sync/` — Personal Mesh peer sync (blob channel, discovery, mutation log); Pro feature.
- `modelDownloadService/`, `modelManager/`, `backgroundDownloadService.ts`, `huggingface.ts` — model catalog, download, import, scan.
- `mcp*` (`mcpContextBoost.ts`, `mcpOAuthNativeAdapters.ts`) + `toolEmbeddingRouter.ts` — MCP tool routing (extensions).

### stores/ (Zustand, persisted)
- `appStore.ts` — global settings incl. `systemPrompt`, `inferenceBackend`, `cacheType`, `gpuLayers`, `imageUseOpenCL`, model lists.
- `chatStore.ts` (+ `chatPersistence.ts`, `chatMessageMutationActions.ts`) — conversations & messages.
- `projectStore.ts` — **Projects = de-facto personas**: name + description + `systemPrompt` + `icon` (color). Nearest thing to character cards (04).
- `remoteServerStore.ts`, `whisperStore.ts`, `downloadStore.ts`, `proAccessSlice.ts`, `uiModeStore.ts`, `modelFailureStore.ts`, `debugLogsStore.ts` — as named.

### screens/
- `HomeScreen/` — landing; models summary, new chat, gallery card, Pro crown, Desktop promo (see 04, item 4).
- `ChatScreen/` — the chat surface; generation actions, image previews, message rendering.
- `ModelsScreen/` — text/image/transcription model browsing & download.
- `ModelSettingsScreen/` — per-model text-gen + image-gen + **SystemPrompt editor** (`SystemPromptSection.tsx`, item 6B).
- `SettingsScreen.tsx` + `SettingsCommunitySections.tsx` — global settings; section registry driven.
- `ProjectsScreen.tsx` / `ProjectDetailScreen.tsx` / `ProjectEditScreen.tsx` — persona/project CRUD (04).
- `ProDetailScreen/` — Pro upsell/unlock (crown target).
- `GalleryScreen/` — generated-image gallery.
- `ToolsScreen/` — tool enable/disable + Pro tools.

### components/
- `GenerationSettingsModal/` + `settings/textGenAdvancedSections.tsx` — the shared advanced text/image controls (BackendSelector, KvCacheTypeToggle, FlashAttentionToggle). Item 3 UI lives here.
- `ChatMessage/` — message rendering incl. tool messages, image attachments, thinking blocks.
- `ChatInput/` — composer, voice, image-mode toggle.
- `ModelSelectorModal/`, `models/` — model pickers/sheets.

### navigation/
- `AppNavigator.tsx` — bottom tabs `HomeTab, ChatsTab, ProjectsTab, ModelsTab, SettingsTab` + a RootStack (Chat, ProjectDetail, ModelSettings, ProDetail, Tools, Gallery …). Pro screens injected via `screenRegistry.ts`.
- `screenRegistry.ts` / `slotRegistry.ts` / `sectionRegistry.ts` — runtime registration seams the Pro submodule fills.

### hooks/, utils/, theme/, config/, bootstrap/
- `hooks/useTextGenerationAdvanced.ts` — derives cache/gpu UI state (item 3).
- `config/featureFlags.ts` — `HTP_ENABLED` etc.
- `bootstrap/loadProFeatures.ts` — registers Pro screens/slots/extensions.

## android native (android/app/src/main/java/ai/offgridmobile/)
- `localdream/LocalDreamModule.kt` — **image gen**: spawns `libstable_diffusion_core.so` HTTP server on :18081; POST /generate SSE; decodes payload → PNG. **Rainbow bug lives here** (01).
- `litert/LiteRTModule.kt` — LiteRT text/vision engine.
- `download/`, `sync/`, `clipboard/`, `screenshot/`, `directory/`, `devicememory/`, `pdf/` — supporting native modules.
- No `ImageGeneratorModule.kt` exists (the legacy `imageGenerator.ts` module is unregistered → dead; see 02).

## ios native (ios/)
- `CoreMLDiffusionModule.swift` (+ `.m`) — **image gen**: Apple ml-stable-diffusion pipeline; writes real PNG via `UIImage.pngData()`. **Not affected by rainbow bug.** (01)
- llama.rn / whisper.rn are pods (not in-repo).
- `sync/`, `download/`, `pdf/`, `devicememory/` — mirrors of Android modules; no iOS `ImageGeneratorModule`.

## Prior planning docs (context, not source of truth)
`docs/refactor-plan/CP-01-image-decode.md`, `CP-02-dedup.md`, `CP-03-inference-wiring.md`,
`PLAN.md` — earlier stubs that point at these very map files. `docs/GAPS_BACKLOG.md`
(IM2/IM4/IM5) and `docs/standards/CODEBASE_GUIDE.md` corroborate several findings below.
