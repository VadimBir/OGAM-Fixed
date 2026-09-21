import { Buffer } from 'buffer';
import { Project } from '../types';

/**
 * SillyTavern / Oobabooga character-card import.
 *
 * A SillyTavern card is a PNG whose metadata carries a base64-encoded JSON character definition
 * in a text chunk keyed `chara` (spec V1/V2 `chara_card_v2`) or `ccv3` (spec V3). This module reads
 * that chunk and maps the card fields onto our `Project` (character) shape.
 *
 * The parsing here is PURE (base64 PNG string in → fields out) so it is unit-testable without any
 * native file I/O; the RN file/picker glue lives in `importCharacterCardFromFile`.
 */

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

export interface ParsedCharacterCard {
  name: string;
  description: string;
  personality?: string;
  scenario?: string;
  firstMessage?: string;
  exampleDialogue?: string;
}

/** Thrown for a malformed / non-card PNG so the UI can show a precise reason. */
export class CharacterCardParseError extends Error {}

interface PngTextChunk {
  keyword: string;
  /** UTF-8 text value of the chunk (already inflated is NOT supported — see note). */
  text: string;
}

/** Read all uncompressed text chunks (tEXt, iTXt) from a PNG byte buffer. */
function readPngTextChunks(bytes: Uint8Array): PngTextChunk[] {
  for (let i = 0; i < PNG_SIGNATURE.length; i++) {
    if (bytes[i] !== PNG_SIGNATURE[i]) {
      throw new CharacterCardParseError('Not a PNG file (bad signature).');
    }
  }

  const chunks: PngTextChunk[] = [];
  let offset = 8; // past the signature
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  while (offset + 8 <= bytes.length) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(
      bytes[offset + 4],
      bytes[offset + 5],
      bytes[offset + 6],
      bytes[offset + 7],
    );
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > bytes.length) break; // truncated

    if (type === 'tEXt') {
      // keyword \0 text  (Latin-1)
      const data = bytes.subarray(dataStart, dataEnd);
      const nul = data.indexOf(0);
      if (nul >= 0) {
        const keyword = latin1(data.subarray(0, nul));
        const text = latin1(data.subarray(nul + 1));
        chunks.push({ keyword, text });
      }
    } else if (type === 'iTXt') {
      // keyword \0 compressionFlag(1) compressionMethod(1) lang \0 translatedKeyword \0 text
      const data = bytes.subarray(dataStart, dataEnd);
      const nul = data.indexOf(0);
      if (nul >= 0) {
        const keyword = latin1(data.subarray(0, nul));
        const compressionFlag = data[nul + 1];
        // Skip lang tag and translated keyword (two more NUL-terminated fields).
        let p = nul + 3;
        const langEnd = data.indexOf(0, p);
        p = langEnd >= 0 ? langEnd + 1 : p;
        const transEnd = data.indexOf(0, p);
        p = transEnd >= 0 ? transEnd + 1 : p;
        if (compressionFlag === 0) {
          const text = utf8(data.subarray(p));
          chunks.push({ keyword, text });
        }
        // compressionFlag === 1 (zlib) is unsupported here (no inflate dependency).
      }
    } else if (type === 'IEND') {
      break;
    }
    offset = dataEnd + 4; // skip CRC
  }
  return chunks;
}

function latin1(b: Uint8Array): string {
  let s = '';
  for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
  return s;
}

function utf8(b: Uint8Array): string {
  return Buffer.from(b).toString('utf8');
}

function str(v: unknown): string | undefined {
  if (typeof v !== 'string') return undefined;
  const t = v.trim();
  return t.length ? t : undefined;
}

/**
 * Parse a SillyTavern card from a base64-encoded PNG.
 * @throws CharacterCardParseError when the PNG carries no readable `chara`/`ccv3` card.
 */
export function parseCharacterCardFromPngBase64(pngBase64: string): ParsedCharacterCard {
  const bytes = new Uint8Array(Buffer.from(pngBase64, 'base64'));
  const chunks = readPngTextChunks(bytes);

  // Prefer V3 (`ccv3`) when present, else V1/V2 (`chara`).
  const chunk =
    chunks.find(c => c.keyword.toLowerCase() === 'ccv3') ??
    chunks.find(c => c.keyword.toLowerCase() === 'chara');
  if (!chunk) {
    throw new CharacterCardParseError(
      'No character data found in this PNG. It may be a plain image, or its card data is zlib-compressed (zTXt), which is not supported.',
    );
  }

  let json: any;
  try {
    // The chunk text is base64-encoded JSON.
    json = JSON.parse(Buffer.from(chunk.text.trim(), 'base64').toString('utf8'));
  } catch {
    throw new CharacterCardParseError('Character data is present but is not valid base64 JSON.');
  }

  return mapCharaJsonToCard(json);
}

/** Map a chara_card V1/V2/V3 JSON object to our card fields. V2/V3 nest under `data`. */
export function mapCharaJsonToCard(json: any): ParsedCharacterCard {
  const d = json && typeof json === 'object' && json.data && typeof json.data === 'object'
    ? json.data
    : json ?? {};

  const name = str(d.name) ?? str(json?.name) ?? 'Imported Character';
  const description = str(d.description) ?? '';
  return {
    name,
    description,
    personality: str(d.personality),
    scenario: str(d.scenario),
    firstMessage: str(d.first_mes) ?? str(d.firstMes),
    exampleDialogue: str(d.mes_example) ?? str(d.exampleDialogue),
  };
}

/**
 * Fold the character-card fields into a single system prompt so the model stays in character.
 * Kept deterministic and label-based (matches SillyTavern's own conventions closely enough that
 * users can predict the text). The user can still edit the result verbatim afterwards.
 */
export function assembleCharacterSystemPrompt(card: ParsedCharacterCard): string {
  const parts: string[] = [];
  if (card.description) parts.push(`Description:\n${card.description}`);
  if (card.personality) parts.push(`Personality:\n${card.personality}`);
  if (card.scenario) parts.push(`Scenario:\n${card.scenario}`);
  if (card.exampleDialogue) parts.push(`Example dialogue:\n${card.exampleDialogue}`);
  const header = `You are ${card.name}. Stay in character at all times.`;
  return parts.length ? `${header}\n\n${parts.join('\n\n')}` : header;
}

/** Build the `Project` fields (minus id/timestamps) from a parsed card + its avatar. */
export function cardToProjectFields(
  card: ParsedCharacterCard,
  avatarUri?: string,
): Omit<Project, 'id' | 'createdAt' | 'updatedAt'> {
  return {
    name: card.name,
    description: card.description || `Imported character: ${card.name}`,
    systemPrompt: assembleCharacterSystemPrompt(card),
    avatarUri,
    personality: card.personality,
    scenario: card.scenario,
    firstMessage: card.firstMessage,
    exampleDialogue: card.exampleDialogue,
  };
}
