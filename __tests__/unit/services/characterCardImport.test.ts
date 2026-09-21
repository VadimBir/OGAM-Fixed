import { Buffer } from 'buffer';
import {
  parseCharacterCardFromPngBase64,
  mapCharaJsonToCard,
  assembleCharacterSystemPrompt,
  cardToProjectFields,
  CharacterCardParseError,
} from '../../../src/services/characterCardImport';

const PNG_SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** Build a minimal PNG carrying one tEXt chunk (CRC left zero — the reader ignores CRC). */
function makePngWithText(keyword: string, text: string): string {
  const body = Buffer.concat([
    Buffer.from(keyword, 'latin1'),
    Buffer.from([0]),
    Buffer.from(text, 'latin1'),
  ]);
  const len = Buffer.alloc(4);
  len.writeUInt32BE(body.length, 0);
  const chunk = Buffer.concat([len, Buffer.from('tEXt', 'latin1'), body, Buffer.alloc(4)]);
  const iend = Buffer.concat([Buffer.alloc(4), Buffer.from('IEND', 'latin1'), Buffer.alloc(4)]);
  return Buffer.concat([PNG_SIG, chunk, iend]).toString('base64');
}

describe('characterCardImport', () => {
  it('parses a V2 chara_card from a tEXt chunk (base64 JSON under `data`)', () => {
    const card = {
      spec: 'chara_card_v2',
      data: {
        name: 'Aria',
        description: 'A curious explorer.',
        personality: 'brave, witty',
        scenario: 'aboard a starship',
        first_mes: 'Hello, traveler!',
        mes_example: '{{user}}: hi\n{{char}}: greetings',
      },
    };
    const chara = Buffer.from(JSON.stringify(card), 'utf8').toString('base64');
    const png = makePngWithText('chara', chara);

    const parsed = parseCharacterCardFromPngBase64(png);
    expect(parsed.name).toBe('Aria');
    expect(parsed.description).toBe('A curious explorer.');
    expect(parsed.personality).toBe('brave, witty');
    expect(parsed.scenario).toBe('aboard a starship');
    expect(parsed.firstMessage).toBe('Hello, traveler!');
    expect(parsed.exampleDialogue).toContain('greetings');
  });

  it('parses a flat V1 card (no `data` nesting)', () => {
    const parsed = mapCharaJsonToCard({ name: 'Bob', description: 'plain' });
    expect(parsed.name).toBe('Bob');
    expect(parsed.description).toBe('plain');
    expect(parsed.personality).toBeUndefined();
  });

  it('throws a precise error for a PNG with no character chunk', () => {
    const png = makePngWithText('Comment', 'just an image');
    expect(() => parseCharacterCardFromPngBase64(png)).toThrow(CharacterCardParseError);
  });

  it('throws for a non-PNG buffer', () => {
    const notPng = Buffer.from('hello world').toString('base64');
    expect(() => parseCharacterCardFromPngBase64(notPng)).toThrow(/PNG/);
  });

  it('assembles a deterministic in-character system prompt', () => {
    const prompt = assembleCharacterSystemPrompt({
      name: 'Aria',
      description: 'A curious explorer.',
      personality: 'brave',
    });
    expect(prompt.startsWith('You are Aria. Stay in character')).toBe(true);
    expect(prompt).toContain('Description:');
    expect(prompt).toContain('Personality:');
  });

  it('maps parsed card + avatar into Project fields', () => {
    const fields = cardToProjectFields(
      { name: 'Aria', description: 'x', firstMessage: 'hi' },
      'file:///avatar.png',
    );
    expect(fields.avatarUri).toBe('file:///avatar.png');
    expect(fields.firstMessage).toBe('hi');
    expect(fields.systemPrompt).toContain('You are Aria');
  });
});
