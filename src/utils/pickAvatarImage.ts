import { Platform } from 'react-native';
import RNFS from 'react-native-fs';
import { pick, types } from '@react-native-documents/picker';
import { resolvePickedFileUri } from './resolvePickedFileUri';
import { generateId } from './generateId';
import logger from './logger';

/**
 * Pick an image from the device and persist a stable owned copy for use as a character/persona
 * avatar. Returns a `file://` URI, or null if the user cancelled the picker.
 *
 * The copy lives under DocumentDirectory/<subdir> so the avatar survives the source file being
 * moved/deleted (a picked content:// URI is not durable). Mirrors the persistence the SillyTavern
 * PNG importer already does for card avatars, kept in ONE place so both callers behave the same.
 */
export async function pickAvatarImage(
  subdir: 'character_avatars' | 'persona_avatars' = 'character_avatars',
): Promise<string | null> {
  const picked =
    Platform.OS === 'android'
      ? await pick({ mode: 'open', allowMultiSelection: false, type: [types.images] })
      : await pick({ mode: 'import', allowMultiSelection: false, type: [types.images] });
  const file = picked[0];
  if (!file) return null;

  const fileName = file.name ?? 'avatar.img';
  const localPath = await resolvePickedFileUri(file.uri, fileName);
  const base64 = await RNFS.readFile(localPath, 'base64');

  const ext = /\.(png|jpe?g|webp|gif)$/i.exec(fileName)?.[1]?.toLowerCase() ?? 'png';
  const dir = `${RNFS.DocumentDirectoryPath}/${subdir}`;
  await RNFS.mkdir(dir);
  const dest = `${dir}/${generateId()}.${ext}`;
  await RNFS.writeFile(dest, base64, 'base64');
  logger.log(`[avatar] persisted ${subdir} avatar -> ${dest}`);
  return `file://${dest}`;
}
