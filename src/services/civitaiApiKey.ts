/**
 * The Civitai API key, held in the Keychain / Android Keystore like remote-server keys
 * (remoteServerManagerUtils.ts). Never in AsyncStorage or the persisted Zustand store. Search works
 * without it; most model downloads 401 without it (Civitai CLI guide), so downloads send it as
 * `?token=` (civitaiBrowser.ts withCivitaiDownloadToken). That URL, key included, is kept in the
 * app-private native download row (DownloadManagerModule.kt startDownload) so WorkManager can resume
 * after app exit, and in AsyncStorage while a start waits for a slot (queuedDownloadPersistence.ts).
 * Never log it.
 */
import * as Keychain from 'react-native-keychain';
import logger from '../utils/logger';

const SERVICE = 'ai.offgridmobile.civitai';
const ACCOUNT = 'civitai_api_key';

export async function getCivitaiApiKey(): Promise<string | null> {
  try {
    const credentials = await Keychain.getGenericPassword({ service: SERVICE });
    return credentials && credentials.password ? credentials.password : null;
  } catch (error) {
    logger.error('[Civitai] could not read the API key from the Keychain:', error);
    return null;
  }
}

/** Store the key (trimmed). An empty key removes it. */
export async function setCivitaiApiKey(apiKey: string): Promise<void> {
  const key = apiKey.trim();
  if (!key) {
    await removeCivitaiApiKey();
    return;
  }
  await Keychain.setGenericPassword(ACCOUNT, key, {
    service: SERVICE,
    accessible: Keychain.ACCESSIBLE.WHEN_UNLOCKED,
  });
}

export async function removeCivitaiApiKey(): Promise<void> {
  try {
    await Keychain.resetGenericPassword({ service: SERVICE });
  } catch (error) {
    logger.error('[Civitai] could not remove the API key from the Keychain:', error);
  }
}
