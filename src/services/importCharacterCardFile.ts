import { Platform } from 'react-native';
import RNFS from 'react-native-fs';
import { pick } from '@react-native-documents/picker';
import { resolvePickedFileUri } from '../utils/resolvePickedFileUri';
import { generateId } from '../utils/generateId';
import {
  parseCharacterCardFromPngBase64,
  cardToProjectFields,
  type ParsedCharacterCard,
} from './characterCardImport';
import { useProjectStore } from '../stores/projectStore';
import { Project } from '../types';
import logger from '../utils/logger';

/**
 * Pick a SillyTavern PNG, parse its embedded card, and persist the PNG as an avatar. Returns the
 * parsed fields + avatar URI, or null if the user cancelled. Does NOT persist a Project — callers
 * decide what to do with the fields (create a character, or fill the user persona).
 *
 * @throws CharacterCardParseError when the chosen PNG has no readable card.
 */
export async function pickAndParseSillyTavernCard(): Promise<
  { parsed: ParsedCharacterCard; avatarUri?: string } | null
> {
  const picked =
    Platform.OS === 'android'
      ? await pick({ mode: 'open', allowMultiSelection: false })
      : await pick({ mode: 'import', allowMultiSelection: false });
  const file = picked[0];
  if (!file) return null;

  const fileName = file.name ?? 'character.png';
  const localPath = await resolvePickedFileUri(file.uri, fileName);
  const base64 = await RNFS.readFile(localPath, 'base64');

  // Parse first — if this throws, we never create a half-formed avatar.
  const parsed = parseCharacterCardFromPngBase64(base64);

  let avatarUri: string | undefined;
  try {
    const avatarDir = `${RNFS.DocumentDirectoryPath}/character_avatars`;
    await RNFS.mkdir(avatarDir);
    const avatarPath = `${avatarDir}/${generateId()}.png`;
    await RNFS.writeFile(avatarPath, base64, 'base64');
    avatarUri = `file://${avatarPath}`;
  } catch (e) {
    logger.warn('[ST-import] failed to persist avatar; importing without image', e);
  }
  return { parsed, avatarUri };
}

/**
 * RN glue for SillyTavern PNG import: pick a PNG, read its bytes, parse the embedded card
 * (pure logic in `characterCardImport`), persist the PNG as the avatar, and create a character
 * Project. Returns the created character, or null if the user cancelled the picker.
 *
 * Throws `CharacterCardParseError` (from the parser) when the chosen PNG has no readable card, so
 * the caller can surface a precise message.
 */
export async function importSillyTavernCardFromFile(): Promise<Project | null> {
  const result = await pickAndParseSillyTavernCard();
  if (!result) return null;
  const fields = cardToProjectFields(result.parsed, result.avatarUri);
  return useProjectStore.getState().createProject(fields);
}
