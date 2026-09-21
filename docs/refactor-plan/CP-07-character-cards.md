# CP-07 — Character card system (SillyTavern / Oobabooga style)

Read first: `docs/refactor-map/04-cards-personas.md`. Depends on CP-06 (tools/skills).

## Requirements (verbatim intent)
- Create character cards.
- Cards have an avatar.
- Import SillyTavern PNG cards → auto-include the embedded description. (ST embeds a
  `chara` tEXt/zTXt chunk = base64 JSON, spec V2 `chara_card_v2`.)
- Model can call image-gen (via CP-06 skill).
- User controls the prompts EXACTLY as written.
- Existing features (web search + other skills) become editable "skill cards" the user can
  edit AND create new ones. If a card "says no" (disabled), user can edit it.
- Editing lives in Settings or a new bottom tab.

## Scope map (fill from mapping agent)
- Current personas/characters store + UI (grep persona/character/card).
- Web-search tool definition (to convert into an editable skill card).
- Bottom tab navigator (`src/navigation/*`) — where to add a "Cards"/"Skills" tab.
- Settings screen structure.

## Design (draft — NEEDS user sign-off before build; large)
- Data model: `CharacterCard { id, name, avatarUri, systemPrompt, description,
  greeting?, exampleDialogue?, tags?, enabledSkills: string[] }`. Persist in appStore
  (+ migration). SillyTavern V2 import maps: name, description, personality, scenario,
  first_mes, mes_example → our fields; avatar = the PNG itself.
- Skill cards: `SkillCard { id, name, description, enabled, editable, toolSchema,
  promptTemplate }`. Built-ins (web search, image gen) seeded as editable cards.
- UI: new bottom tab "Cards" with two sections (Characters, Skills); create/edit/import.
- PNG import: read tEXt `chara` chunk (need a PNG chunk reader — pure JS or native).

## Verification
- Import a real ST card PNG → fields populate, avatar shows. Create a card, chat uses its
  system prompt verbatim. Toggle a skill card off → tool not offered; edit → takes effect.

## Open questions for user (blocking design)
- One system prompt per character, or global base + per-character overlay?
- SillyTavern spec version to support (V2 `chara_card_v2` is the common one; V3?).
- New bottom tab vs nested under Settings? (User leaned "separate tab at the bottom".)
