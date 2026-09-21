# 04 — Personas / Characters, Web-Search, Settings/Tabs (SillyTavern-style card goal)

Covers requirement item 7 (verbatim):
> End goal = an Oobabooga/SillyTavern-like CHARACTER CARD system: create character cards; cards
> have an avatar; import SillyTavern PNG cards (which embed character description metadata)
> auto-filling the description; model can call image-gen; user controls prompts exactly; existing
> features (web search, etc.) should ALSO become editable "skill cards" the user can edit/create;
> editing lives in settings or a separate bottom tab.

## Current persona/character model = "Projects"

The nearest existing concept to a character card is a **Project**.
- Type: `Project` (`src/types/index.ts:449-459`): `{ id, provenance?, name, description,
  systemPrompt, icon?, createdAt, updatedAt }`. **`icon` is a hex COLOR string, not an image
  avatar.** No avatar image field, no character metadata, no PNG import.
- Store: `src/stores/projectStore.ts`. `DEFAULT_PROJECTS` `:39-103` ship 4 built-in personas
  (General Assistant, Spanish Learning, Code Review, Writing Helper), each just a
  `name`+`description`+`systemPrompt`+color. CRUD: `createProject` `:112`, `updateProject`,
  `deleteProject`, `duplicateProject` (`:24-33`). Projects are **synced** across the Personal Mesh
  (`projectPutMutation`, `emitSyncMutation` `:14-18`).
- The global (project-less) persona is `settings.systemPrompt` (`appStore.ts:50,233`), same default
  text as `default-assistant` — comment at `appStore.ts:230-233` / `projectStore.ts:44-46` notes
  they deliberately share ONE default persona.
- UI: `ProjectsScreen.tsx`, `ProjectDetailScreen.tsx`, `ProjectEditScreen.tsx`; project selection
  in `components/ProjectSelectorSheet.tsx`. Editing a persona's prompt today = editing a Project's
  `systemPrompt` or the global `SystemPromptSection` (`ModelSettingsScreen/SystemPromptSection.tsx`).

**What a SillyTavern card system would touch/extend here:**
1. Extend `Project` (or a new `Character` type) with an **avatar image path** and character-card
   metadata (personality, scenario, first message, example dialogue) — today only
   `name/description/systemPrompt/icon`.
2. **PNG card import**: no importer exists (searched `png|tEXt|chara|charcard|silly` — no match).
   SillyTavern embeds base64 JSON in a PNG `tEXt`/`iTXt` chunk keyed `chara`. New code must read
   that chunk and map it → the character's description/systemPrompt fields. Nearest existing file
   ingest: `services/pdfExtractor.ts` / `documentService.ts` (documents, not PNG metadata) — no
   PNG chunk parser present.
3. Avatar display would flow through `components/ChatMessage/` and the project/character list rows.

## Web-search "skill" (item 7: make features editable skill-cards)

- `web_search` is a **built-in, hardcoded** tool: definition `services/tools/registry.ts:4-18`,
  implementation `services/tools/handlers.ts:54-84` (Brave HTML scrape). Its description/enabled
  state is NOT user-editable content — it is code. `read_url` (`registry.ts:74`, `handlers.ts:343`)
  is its companion.
- Enabled/disabled via `ToolsScreen/` + `settings` (`ctx.enabledToolIds`), but the prompt/description
  and behavior are fixed in source.
- To make web-search (and the other 5 tools) into user-editable "skill cards," the tool
  definition (name, description, when-to-use prompt, params) would move from the hardcoded
  `AVAILABLE_TOOLS` array into a persisted, editable store (mirroring `projectStore`), and
  `getToolsAsOpenAISchema` / `buildToolSystemPromptHint` would read from that store. The
  `ToolExtension` seam (`extensions.ts`) already proves runtime-pluggable tools are supported.

## Settings / tabs — where editing would live

Bottom tabs (`src/navigation/AppNavigator.tsx:143-181`):
`HomeTab`, `ChatsTab`, `ProjectsTab`, `ModelsTab`, `SettingsTab`.
RootStack screens (`:214-256`) include `ProjectDetail`, `ProjectEdit`, `ModelSettings`, `Tools`,
`ProDetail`, `Gallery`, plus **Pro screens injected at runtime** via `screenRegistry.ts`
(`useRegisteredScreens` `:59-61`, rendered at `AppNavigator.tsx:255-256`).

- A "character cards" surface would most naturally either (a) **repurpose/rename `ProjectsTab`**
  (Projects already ARE personas) or (b) add a **new bottom tab** (`Tab.Screen` in
  `AppNavigator.tsx`, plus `MainTabParamList` in `navigation/types.ts` and `TAB_ICON_MAP`).
- A "skill cards" editor would live under `SettingsTab` (section registry:
  `components/settings/sectionRegistry.ts`) or its own tab, alongside the existing `Tools` screen.
- Runtime registration seams (`screenRegistry.ts`, `slotRegistry.ts`, `sectionRegistry.ts`) mean
  new screens/sections can be added without touching the navigator wholesale.

## Summary of gaps for item 7
| Need | Exists today | Gap |
|---|---|---|
| Create character cards | Projects (name/desc/systemPrompt/color) | no avatar image, no char metadata |
| Avatar | `icon` = color only | no image field/upload |
| Import SillyTavern PNG | — | no PNG tEXt/`chara` chunk parser anywhere |
| Model calls image-gen | image-gen via intent classifier | no `generate_image` tool (see 03/6C) |
| Exact prompt control | `settings.systemPrompt` + per-Project prompt (editable) | runtime augmentation hidden (see 03/6B) |
| Web-search as editable card | hardcoded built-in tool | tool defs not persisted/editable |
| Editing location | SettingsTab / ProjectsTab / Tools screen | pick tab vs settings; registries ready |
