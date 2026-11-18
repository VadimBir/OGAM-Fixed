/**
 * substituteCardPlaceholders inserts names VERBATIM: String.replace with a replacement STRING would
 * interpret `$&`, `$1`, `$$` inside a user-chosen name.
 */
import {
  substituteCardPlaceholders,
  composeCharacterPersonaPrompt,
} from '../../../src/services/personaComposition';

describe('substituteCardPlaceholders', () => {
  it('replaces single- and double-brace macros', () => {
    expect(substituteCardPlaceholders('{char} meets {{ user }}', 'Ava', 'Sam')).toBe('Ava meets Sam');
  });

  it('inserts names containing $ replacement patterns verbatim', () => {
    expect(substituteCardPlaceholders('Hi {user}', 'Ava', 'A$&B')).toBe('Hi A$&B');
    expect(substituteCardPlaceholders('{char}!', 'Cash$$', 'Sam')).toBe('Cash$$!');
    expect(substituteCardPlaceholders('{char}', '$1$`', 'Sam')).toBe('$1$`');
  });

  it('composed prompt keeps a $-name verbatim', () => {
    const out = composeCharacterPersonaPrompt('I am {char}, talking to {user}.', {
      characterName: 'Mr$&',
      userName: 'Me$$',
    });
    expect(out).toContain('I am Mr$&, talking to Me$$.');
  });
});
