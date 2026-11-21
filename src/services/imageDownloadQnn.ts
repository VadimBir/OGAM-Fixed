import { hideAlert, showAlert } from '../utils/alertState';
import { ImageModelDescriptor, ImageDownloadDeps } from './imageModelDownloadTypes';
import RNFS from 'react-native-fs';
import logger from '../utils/logger';

/** Room kept free beyond the download (same 256 MB as sdCppFileRegistration's import check, which
 *  is not imported here: it pulls the sd.cpp engine into the download module graph). */
const FREE_SPACE_HEADROOM_BYTES = 256 * 1024 * 1024;
const formatGB = (bytes: number) => `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;

/** Bytes still missing for `needed` (plus headroom), or null when it fits or free space is unknown. */
async function missingFreeSpace(needed: number): Promise<number | null> {
  if (!needed || needed <= 0) return null;
  try {
    const { freeSpace } = await RNFS.getFSInfo();
    const total = needed + FREE_SPACE_HEADROOM_BYTES;
    return freeSpace >= total ? null : total - freeSpace;
  } catch (error) {
    logger.warn('[ImageDownload] free space unavailable, downloading without the check:', error);
    return null;
  }
}

/**
 * Bytes a zip image model needs free before it starts: the zip itself plus its extracted files,
 * which sit side by side until the zip is deleted after the unzip (imageDownloadActions
 * proceedWithDownload). Model zips barely compress (weights), so the extraction is about the zip
 * size again: a 3.8 GB SDXL zip needs about 7.6 GB. The native worker only checks the zip itself.
 */
export function zipExtractionBytes(zipBytes: number): number {
  return zipBytes > 0 ? zipBytes * 2 : 0;
}

/**
 * True (and an alert shown) when free storage cannot hold a zip download plus its extraction.
 * Multi-file and Core ML downloads write each file once, so they are not checked here. An unknown
 * size or free space is not blocked (missingFreeSpace).
 */
export async function refuseZipWithoutExtractionSpace(
  modelInfo: ImageModelDescriptor,
  deps: ImageDownloadDeps,
): Promise<boolean> {
  if (modelInfo.huggingFaceFiles || (modelInfo.coremlFiles && modelInfo.coremlFiles.length > 0)) return false;
  const needed = zipExtractionBytes(modelInfo.size);
  const shortBy = await missingFreeSpace(needed);
  if (shortBy == null) return false;
  deps.setAlertState(showAlert(
    'Not enough storage',
    `${modelInfo.name} needs about ${formatGB(needed)} free while it downloads and unzips ` +
      `(the zip plus its extracted files). Free up ${formatGB(shortBy)} and try again.`,
  ));
  return true;
}

export function getQnnWarningMessage(
  modelInfo: ImageModelDescriptor,
  socInfo: { hasNPU: boolean; qnnVariant?: string },
): string | null {
  if (!socInfo.hasNPU) {
    return 'NPU models require a Qualcomm Snapdragon processor. ' +
      'Your device does not have a compatible NPU and this model will not work. ' +
      'Consider downloading a CPU model instead.';
  }
  if (!modelInfo.variant || !socInfo.qnnVariant) return null;

  const deviceVariant = socInfo.qnnVariant;
  const modelVariant = modelInfo.variant;
  const compatible =
    modelVariant === deviceVariant || deviceVariant === '8gen2' ||
    (deviceVariant === '8gen1' && modelVariant !== '8gen2');
  if (compatible) return null;

  return `This model is built for ${modelVariant === '8gen2' ? 'flagship' : modelVariant} Snapdragon chips. ` +
    `Your device uses a ${deviceVariant === 'min' ? 'non-flagship' : deviceVariant} chip and this model will likely crash. ` +
    `Download the non-flagship variant instead.`;
}

export function showQnnWarningAlert(
  opts: {
    warningMessage: string;
    hasNPU: boolean;
    modelInfo: ImageModelDescriptor;
    onDownloadAnyway: () => void;
  },
  deps: ImageDownloadDeps,
): void {
  const { warningMessage, hasNPU, onDownloadAnyway } = opts;
  if (hasNPU) {
    deps.setAlertState(showAlert('Incompatible Model', warningMessage, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Download Anyway',
        style: 'destructive',
        onPress: () => {
    deps.setAlertState(hideAlert());
          onDownloadAnyway();
        },
      },
    ]));
    return;
  }

  deps.setAlertState(showAlert('Incompatible Model', warningMessage, [
    { text: 'OK', style: 'cancel' },
  ]));
}
