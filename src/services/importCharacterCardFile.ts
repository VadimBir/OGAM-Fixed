import { Platform } from 'react-native';
import RNFS from 'react-native-fs';
import { pick } from '@react-native-documents/picker';
import { resolvePickedFileUri } from '../utils/resolvePickedFileUri';
import { generateId } from '../utils/generateId';
import {
  parseCharacterCardFromPngBase64,
  cardToProjectFields,
} from './characterCardImport';
import { useProjectStore } from '../stores/projectStore';
import { Project } from '../types';
import logger from '../utils/logger';

/**
 * RN glue for SillyTavern PNG import: pick a PNG, read its bytes, parse the embedded card
 * (pure logic in `characterCardImport`), persist the PNG as the avatar, and create a character
 * Project. Returns the created character, or null if the user cancelled the picker.
 *
 * Throws `CharacterCardParseError` (from the parser) when the chosen PNG has no readable card, so
 * the caller can surface a precise message.
 */
export async function importSillyTavernCardFromFile(): Promise<Project | null> {
  // SillyTavern cards are PNGs with an embedded `chara` text chunk. We don't constrain the
  // picker MIME here (option shape varies by picker version); the parser rejects non-card PNGs
  // and non-PNG files with a precise error.
  const picked =
    Platform.OS === 'android'
      ? await pick({ mode: 'open', allowMultiSelection: false })
      : await pick({ mode: 'import', allowMultiSelection: false });
  const file = picked[0];
  if (!file) return null;

  const fileName = file.name ?? 'character.png';
  const localPath = await resolvePickedFileUri(file.uri, fileName);
  const base64 = await RNFS.readFile(localPath, 'base64');

  // Parse first — if this throws, we never create a half-formed avatar/character.
  const parsed = parseCharacterCardFromPngBase64(base64);

  // Persist the source PNG as the character's avatar in a stable owned directory.
  let avatarUri: string | undefined;
  try {
    const avatarDir = `${RNFS.DocumentDirectoryPath}/character_avatars`;
    await RNFS.mkdir(avatarDir);
    const avatarPath = `${avatarDir}/${generateId()}.png`;
    await RNFS.writeFile(avatarPath, base64, 'base64');
    avatarUri = `file://${avatarPath}`;
  } catch (e) {
    // A missing avatar is non-fatal — the character still imports with its text.
    logger.warn('[ST-import] failed to persist avatar; importing without image', e);
  }

  const fields = cardToProjectFields(parsed, avatarUri);
  return useProjectStore.getState().createProject(fields);
}
