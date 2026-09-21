# CP-06 — Tool-calling, viewable system prompt, skills (+ image-gen skill)

Read first: `docs/refactor-map/03-tools-skills-sysprompt.md`.

## Sub-goals
- 6A. Model can tool-call on demand.
- 6B. Open, user-VIEWABLE (and per CP-07, editable) system prompt.
- 6C. Skills system: partial today; ADD a "generate image" skill — model calls an
  image-gen tool and the resulting image shows up in the chat.

## Current infra (confirm via map)
- `src/services/tools/*`, `src/services/generationToolLoop.ts` — tool loop.
- System-prompt construction (grep `systemPrompt`, `buildSystemPrompt`, `proPrompt.ts`).
- Existing skills (web search etc.).

## Design (draft — needs user sign-off)
- Image-gen tool: a tool schema `generate_image({prompt, negative_prompt?, steps?, size?})`
  routed to `localDreamGeneratorService.generateImage`; on completion, insert an assistant
  message with an image attachment (reuse `saveImageGenerationResult` /
  `imageGenerationResult.ts` path) so it renders in chat exactly like a user-triggered gen.
- Selectable steps: expose steps in the tool args + a default from image settings.
- Viewable system prompt: a settings screen (or card, CP-07) rendering the resolved system
  prompt read-only first, then editable.

## Verification
- On device: ask model to "draw a cat"; confirm it calls the tool, image appears in chat.
- Unit: tool-loop test that image tool produces an image attachment message.

## Open questions for user
- Tool-calling: only for models that support native tool/function calling, or also a
  prompt-based fallback for models that don't?
- Should the image-gen tool auto-trigger on intent, or only when the skill card is enabled?
