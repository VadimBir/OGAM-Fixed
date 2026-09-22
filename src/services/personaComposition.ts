/**
 * Character + user-persona prompt composition (SillyTavern-style).
 *
 * Two card concepts:
 *  - CHARACTER = who the AI is ({char}). Today a `Project` (name + systemPrompt + card metadata).
 *  - PERSONA   = who the USER is ({user}). Stored on `AppSettings` (personaName + personaPrompt).
 *
 * The model FOLLOWS the character card while CONSIDERING the user persona. This module is PURE
 * (strings in → string out) so it is unit-testable with no store/native dependency; the store glue
 * lives at the single generation composition site (useChatGenerationActions).
 */

export interface PersonaContext {
  /** The character's name → replaces {char}/{{char}}. Falls back to a neutral noun. */
  characterName?: string;
  /** The user's persona name → replaces {user}/{{user}}. Falls back to a neutral noun. */
  userName?: string;
  /** Free-form user-persona description. When present, a "who the user is" block is appended. */
  personaDescription?: string;
}

const CHAR_FALLBACK = 'the assistant';
const USER_FALLBACK = 'the user';

/**
 * Replace {char}/{{char}} and {user}/{{user}} (case-insensitive, optional inner spaces) with the
 * resolved names. Mirrors SillyTavern's {{char}}/{{user}} macros AND the single-brace form the
 * user writes. Leaves every other brace sequence untouched.
 */
export function substituteCardPlaceholders(
  text: string,
  charName: string,
  userName: string,
): string {
  if (!text) return text;
  return text
    .replace(/\{\{?\s*char\s*\}?\}/gi, charName)
    .replace(/\{\{?\s*user\s*\}?\}/gi, userName);
}

/**
 * Compose the final base system prompt from the raw character prompt + the user persona.
 * - Substitutes {char}/{user} in the character prompt.
 * - Appends a persona block describing WHO THE USER IS, with an explicit instruction to stay in
 *   character as {char} while treating that block as the user's identity.
 * Returns the raw prompt unchanged (only substituted) when no persona is configured.
 */
export function composeCharacterPersonaPrompt(
  rawPrompt: string,
  ctx: PersonaContext,
): string {
  const char = (ctx.characterName ?? '').trim() || CHAR_FALLBACK;
  const user = (ctx.userName ?? '').trim() || USER_FALLBACK;

  const base = substituteCardPlaceholders(rawPrompt ?? '', char, user);

  const persona = (ctx.personaDescription ?? '').trim();
  const hasNamedUser = !!(ctx.userName ?? '').trim();

  if (!persona && !hasNamedUser) return base;

  const personaBody = persona
    ? `${substituteCardPlaceholders(persona, char, user)}`
    : '';

  const block = persona
    ? `\n\nAbout the person you are talking to (${user}):\n${personaBody}\n` +
      `Stay in character as ${char}. Treat the description above as who ${user} is, and address them accordingly — never adopt ${user}'s identity yourself.`
    : `\n\nYou are talking to ${user}. Stay in character as ${char}.`;

  return `${base}${block}`;
}
