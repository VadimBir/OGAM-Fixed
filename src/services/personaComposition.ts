/**
 * Character + user-persona prompt composition (SillyTavern-style), assembled ONCE into the system
 * instruction — not injected per message.
 *
 * Two card concepts, each with a separate INSTRUCTION and CAVEATS field:
 *  - CHARACTER = who the AI is ({char}). Instruction + caveats live on a `Project`.
 *  - PERSONA   = who the USER is ({user}). Instruction + caveats live on `AppSettings`.
 *
 * The output is a single role-tagged system instruction: a short framing preamble stating the
 * purpose (who is who, follow the character, honor caveats, consider the persona) followed by the
 * character and user-persona sections. The model FOLLOWS the character while CONSIDERING the user
 * persona. Pure (strings in → string out) so it is unit-testable with no store/native dependency.
 */

export interface PersonaContext {
  /** Character name → replaces {char}/{{char}}. */
  characterName?: string;
  /** Character caveats (hard constraints / what NOT to do). */
  characterCaveats?: string;
  /** User-persona name → replaces {user}/{{user}}. */
  userName?: string;
  /** User-persona instruction (who the user is). */
  personaDescription?: string;
  /** User-persona caveats. */
  personaCaveats?: string;
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
 * Compose the single system instruction from the character instruction + its caveats and the user
 * persona + its caveats. `characterInstruction` is the character's main prompt ({char}).
 *
 * When neither a persona nor character caveats are configured, returns the character instruction
 * with only macro substitution applied (behaviour-neutral for a plain, persona-less setup).
 */
export function composeCharacterPersonaPrompt(
  characterInstruction: string,
  ctx: PersonaContext,
): string {
  const char = (ctx.characterName ?? '').trim() || CHAR_FALLBACK;
  const user = (ctx.userName ?? '').trim() || USER_FALLBACK;
  const sub = (t?: string) => substituteCardPlaceholders((t ?? '').trim(), char, user);

  const charInstr = sub(characterInstruction);
  const charCaveats = sub(ctx.characterCaveats);
  const personaInstr = sub(ctx.personaDescription);
  const personaCaveats = sub(ctx.personaCaveats);

  const hasPersona = !!(personaInstr || (ctx.userName ?? '').trim() || personaCaveats);
  const hasCharCaveats = !!charCaveats;

  // No caveats and no persona → keep the plain (substituted) instruction, unchanged in shape.
  if (!hasPersona && !hasCharCaveats) return charInstr;

  const preamble =
    `You are ${char}. Follow your CHARACTER instruction below and always honor your CHARACTER ` +
    `caveats.` +
    (hasPersona
      ? ` You are speaking with ${user}; the USER PERSONA below describes who they are — consider ` +
        `it and address them accordingly, but never adopt ${user}'s identity or break character.`
      : '');

  const parts: string[] = [preamble];

  parts.push(`<character name="${char}">\n${charInstr}\n</character>`);
  if (hasCharCaveats) {
    parts.push(`<character_caveats>\n${charCaveats}\n</character_caveats>`);
  }
  if (hasPersona) {
    parts.push(
      `<user_persona name="${user}">\n${personaInstr || `${user} (no further description provided).`}\n</user_persona>`,
    );
    if (personaCaveats) {
      parts.push(`<user_persona_caveats>\n${personaCaveats}\n</user_persona_caveats>`);
    }
  }

  return parts.join('\n\n');
}
