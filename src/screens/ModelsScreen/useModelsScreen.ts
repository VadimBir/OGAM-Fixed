import { useState, useEffect, useCallback, useRef } from 'react';
import { Platform } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import RNFS from 'react-native-fs';
import { unzip } from 'react-native-zip-archive';
import { pick, types, isErrorWithCode, errorCodes } from '@react-native-documents/picker';
import { showAlert, AlertState, initialAlertState } from '../../components/CustomAlert';
import { useFocusTrigger } from '../../hooks/useFocusTrigger';
import { useAppStore } from '../../stores';
import { useDownloadStore, isActiveStatus, isFailedStatus } from '../../stores/downloadStore';
import { modelManager } from '../../services';
import { isLiteRTAvailable } from '../../services/engines';
import { resolveCoreMLModelDir } from '../../utils/coreMLModelUtils';
import { ONNXImageModel } from '../../types';
import { ModelTab, NavigationProp } from './types';
import { initialFilterState } from './constants';
import { getDirectorySize } from './utils';
import { useTextModels } from './useTextModels';
import { useImageModels } from './useImageModels';
import { importGgufFiles, getErrorMessage } from './importHelpers';
import { importSdCheckpointOrLora } from './importSdCheckpoint';
import { isSdCppCandidateFileName } from '../../services/sdCppModelDetection';
import { isPickerStuck } from '../../utils/pickerErrorUtils';
import { isLiteRTFileName } from '../../utils/modelHelpers';
import logger from '../../utils/logger';

type ZipImportDeps = {
  addDownloadedImageModel: (model: ONNXImageModel) => void;
  activeImageModelId: string | null;
  setActiveImageModelId: (id: string | null) => void;
  setImportProgress: (p: { fraction: number; fileName: string } | null) => void;
  setAlertState: (s: AlertState) => void;
};

type ReadDirItems = Awaited<ReturnType<typeof RNFS.readDir>>;

/** A CoreML bundle in the extracted tree (top level or one folder down). */
async function containsMLModelC(dirContents: ReadDirItems): Promise<boolean> {
  if (dirContents.some(f => f.name.endsWith('.mlmodelc'))) return true;
  for (const sub of dirContents.filter(f => f.isDirectory())) {
    let inner: ReadDirItems = [];
    try {
      inner = await RNFS.readDir(sub.path);
    } catch (error) {
      logger.warn(`[ImageZipImport] could not read ${sub.path}:`, error);
    }
    if (inner.some(f => f.name.endsWith('.mlmodelc'))) return true;
  }
  return false;
}

/**
 * The image backend an extracted zip is for, or null when it is none this device can run.
 * CoreML only on iOS and only for a real `.mlmodelc`: "any folder in the zip" used to mean CoreML,
 * so an Android MNN/QNN zip that unpacks into a folder registered as an iOS-only model that could
 * never load, and a zip of nothing recognisable registered with `backend: undefined`.
 */
async function detectZipBackend(
  modelDir: string,
  dirContents: ReadDirItems,
): Promise<{ backend: 'mnn' | 'qnn' | 'coreml'; resolvedModelDir: string } | null> {
  if (Platform.OS === 'ios' && (await containsMLModelC(dirContents))) {
    return { backend: 'coreml', resolvedModelDir: await resolveCoreMLModelDir(modelDir) };
  }
  // A zip that wraps everything in one folder: look one level down for the model files.
  const dirs = dirContents.filter(f => f.isDirectory());
  const files = dirs.length === 1 && dirs.length === dirContents.length
    ? await RNFS.readDir(dirs[0].path)
    : dirContents;
  if (files.some(f => f.name.endsWith('.mnn'))) return { backend: 'mnn', resolvedModelDir: modelDir };
  if (files.some(f => f.name.endsWith('.bin') || f.name.includes('qnn'))) {
    return { backend: 'qnn', resolvedModelDir: modelDir };
  }
  return null;
}

async function removeIfPresent(path: string): Promise<void> {
  try {
    if (await RNFS.exists(path)) await RNFS.unlink(path);
  } catch (error) {
    logger.warn(`[ImageZipImport] could not remove ${path}:`, error);
  }
}

async function importImageModelZip(sourceUri: string, fileName: string, deps: ZipImportDeps): Promise<void> {
  const { addDownloadedImageModel, activeImageModelId, setActiveImageModelId, setImportProgress, setAlertState } = deps;
  const imageModelsDir = modelManager.getImageModelsDirectory();
  const modelId = `local_${fileName.replaceAll(/\.zip$/gi, '').replaceAll(/[^a-zA-Z0-9_-]/g, '_')}_${Date.now()}`;
  const modelDir = `${imageModelsDir}/${modelId}`;
  const zipPath = `${imageModelsDir}/${modelId}.zip`;
  try {
    if (!(await RNFS.exists(imageModelsDir))) await RNFS.mkdir(imageModelsDir);
    setImportProgress({ fraction: 0.1, fileName });
    if (Platform.OS === 'ios') await RNFS.moveFile(sourceUri, zipPath);
    else await RNFS.copyFile(sourceUri, zipPath);
    setImportProgress({ fraction: 0.5, fileName });
    if (!(await RNFS.exists(modelDir))) await RNFS.mkdir(modelDir);
    setImportProgress({ fraction: 0.6, fileName });
    await unzip(zipPath, modelDir);
    setImportProgress({ fraction: 0.85, fileName });
    const dirContents = await RNFS.readDir(modelDir);
    const detected = await detectZipBackend(modelDir, dirContents);
    if (!detected) {
      throw new Error(Platform.OS === 'ios'
        ? 'This zip holds no image model this device can run (expected a CoreML .mlmodelc, or MNN/QNN files).'
        : 'This zip holds no image model this device can run (expected MNN .mnn or QNN files).');
    }
    const { backend, resolvedModelDir } = detected;
    await removeIfPresent(zipPath);
    const totalSize = await getDirectorySize(resolvedModelDir);
    setImportProgress({ fraction: 0.95, fileName });
    const modelName = fileName.replaceAll(/\.zip$/gi, '').replaceAll(/[_-]/g, ' ');
    const imageModel: ONNXImageModel = {
      id: modelId, name: modelName, description: 'Locally imported image model',
      modelPath: resolvedModelDir, downloadedAt: new Date().toISOString(), size: totalSize, backend,
    };
    await modelManager.addDownloadedImageModel(imageModel);
    addDownloadedImageModel(imageModel);
    if (!activeImageModelId) setActiveImageModelId(imageModel.id);
    setImportProgress({ fraction: 1, fileName });
    setAlertState(showAlert('Success', `${modelName} imported successfully!`));
  } catch (error) {
    // Nothing half-imported may stay behind: a leftover dir is adopted by the next scan as a
    // `recovered_` model, and a leftover zip is dead weight in image_models/.
    logger.error(`[ImageZipImport] import of ${fileName} failed:`, error);
    await removeIfPresent(zipPath);
    await removeIfPresent(modelDir);
    setAlertState(showAlert('Import Failed', getErrorMessage(error)));
  }
}


export function useModelsScreen() {
  const navigation = useNavigation<NavigationProp>();
  const focusTrigger = useFocusTrigger();
  const [activeTab, setActiveTabState] = useState<ModelTab>('text');
  const [alertState, setAlertState] = useState<AlertState>(initialAlertState);
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState<{ fraction: number; fileName: string } | null>(null);

  const { addDownloadedModel, activeImageModelId, setActiveImageModelId, addDownloadedImageModel } = useAppStore();

  const text = useTextModels(setAlertState);
  const image = useImageModels(setAlertState);

  useEffect(() => {
    if (activeTab === 'image' && image.availableHFModels.length === 0 && !image.hfModelsLoading) {
      image.loadHFModels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  const setActiveTab = (tab: ModelTab) => {
    setActiveTabState(tab);
    text.setFilterState(initialFilterState);
    text.setTextFiltersVisible(false);
    image.setImageFiltersVisible(false);
  };

  const handleRefresh = async () => {
    text.setIsRefreshing(true);
    await text.loadDownloadedModels();
    await image.loadDownloadedImageModels();
    if (text.hasSearched && text.searchQuery.trim()) await text.handleSearch();
    if (activeTab === 'image') await image.loadHFModels(true);
    text.setIsRefreshing(false);
  };

  const handleImportImageModelZip = (sourceUri: string, fileName: string) =>
    importImageModelZip(sourceUri, fileName, { addDownloadedImageModel, activeImageModelId, setActiveImageModelId, setImportProgress, setAlertState });

  const isPickingRef = useRef(false);

  const validateImportFiles = (resolvedFiles: Array<{ name: string; uri: string }>): string | null => {
    const singleLitert = resolvedFiles.length === 1 && isLiteRTFileName(resolvedFiles[0].name);
    if (singleLitert && !isLiteRTAvailable()) {
      return 'litert_unsupported';
    }
    const allGguf = resolvedFiles.every(f => f.name.toLowerCase().endsWith('.gguf'));
    const singleZip = resolvedFiles.length === 1 && resolvedFiles[0].name.toLowerCase().endsWith('.zip');
    const singleSd = resolvedFiles.length === 1 && isSdCppCandidateFileName(resolvedFiles[0].name);
    if (!allGguf && !singleZip && !singleLitert && !singleSd) return 'invalid_format';
    if (resolvedFiles.length > 2) return 'too_many';
    return null;
  };

  const handleImportLocalModel = async () => {
    if (isImporting || isPickingRef.current) return;
    isPickingRef.current = true;
    setIsImporting(true);
    try {
      const result = await pick({ type: [types.allFiles], allowMultiSelection: true });

      if (!result || result.length === 0) return;

      const resolvedFiles = result.map(f => ({
        ...f,
        name: (f.name?.trim() || decodeURIComponent(f.uri.split('/').pop() ?? '') || 'unknown').split('/').pop() || 'unknown',
      }));

      const validationError = validateImportFiles(resolvedFiles);
      if (validationError === 'litert_unsupported') {
        setAlertState(showAlert('Not Supported', 'LiteRT models are only supported on Android.'));
        return;
      }
      if (validationError === 'invalid_format') {
        setAlertState(showAlert(
          'Invalid File',
          resolvedFiles.length > 1
            ? 'When selecting multiple files, all must be .gguf files (main model + mmproj projector).'
            : 'Supported formats: .gguf (text models), .litertlm (LiteRT models), .zip (image models), and .safetensors/.ckpt (Stable Diffusion checkpoints or LoRAs).',
        ));
        return;
      }
      if (validationError === 'too_many') {
        setAlertState(showAlert('Too Many Files', 'Select 1 file (text/zip/litertlm) or 2 .gguf files (vision model + mmproj projector).'));
        return;
      }

      const firstUri = resolvedFiles[0].uri;
      const firstFileName = resolvedFiles[0].name;
      setImportProgress({ fraction: 0, fileName: firstFileName });

      const singleZip = resolvedFiles.length === 1 && resolvedFiles[0].name.toLowerCase().endsWith('.zip');
      if (singleZip) {
        await handleImportImageModelZip(firstUri, firstFileName);
        return;
      }
      if (resolvedFiles.length === 1 && isSdCppCandidateFileName(firstFileName)) {
        await importSdCheckpointOrLora(
          { uri: firstUri, name: firstFileName, size: resolvedFiles[0].size },
          { addDownloadedImageModel, activeImageModelId, setActiveImageModelId, setImportProgress, setAlertState },
        );
        return;
      }

      await importGgufFiles(resolvedFiles.slice(0, 2), { setAlertState, setImportProgress, addDownloadedModel });
    } catch (error: unknown) {
      if (isErrorWithCode(error) && error.code === errorCodes.OPERATION_CANCELED) return;
      if (isPickerStuck(error)) {
        setAlertState(showAlert(
          'File Picker Unavailable',
          "The file picker isn't responding. Please close and reopen the app, then try again.",
        ));
        return;
      }
      setAlertState(showAlert('Import Failed', getErrorMessage(error)));
    } finally {
      isPickingRef.current = false;
      setIsImporting(false);
      setImportProgress(null);
    }
  };

  const activeDownloadCount = useDownloadStore(state =>
    Object.values(state.downloads).filter(
      d => isActiveStatus(d.status),
    ).length,
  );
  // The icon badge answers "is there download work outstanding?" — so it counts active AND
  // failed/retriable (a failed download needs a retry or remove and must not be invisible).
  const downloadBadgeCount = useDownloadStore(state =>
    Object.values(state.downloads).filter(
      d => isActiveStatus(d.status) || isFailedStatus(d.status) || d.status === 'paused',
    ).length,
  );
  const totalModelCount =
    text.downloadedModels.length +
    image.downloadedImageModels.length +
    activeDownloadCount;

  // No caller-side "too many downloads" gate: backgroundDownloadService caps real
  // concurrency at MAX_CONCURRENT_DOWNLOADS and FIFO-queues the rest, so extra starts
  // just queue (shown as "Queued") instead of hurting performance. The old
  // "Starting more can affect performance / Start Anyway" alert was obsolete friction
  // (and its threshold of 2 didn't even match the cap of 3).
  const handleDownload = useCallback(
    (...args: Parameters<typeof text.handleDownload>) => {
      text.handleDownload(...args);
    },
    [text],
  );

  const handleDownloadImageModel = useCallback(
    (...args: Parameters<typeof image.handleDownloadImageModel>) => {
      image.handleDownloadImageModel(...args);
    },
    [image],
  );

  return {
    navigation,
    focusTrigger,
    activeTab,
    setActiveTab,
    alertState,
    setAlertState,
    isImporting,
    importProgress,
    totalModelCount,
    activeDownloadCount,
    downloadBadgeCount,
    handleImportLocalModel,
    handleRefresh,
    // text model state & handlers
    searchQuery: text.searchQuery,
    setSearchQuery: text.setSearchQuery,
    isLoading: text.isLoading,
    isRefreshing: text.isRefreshing,
    hasSearched: text.hasSearched,
    selectedModel: text.selectedModel,
    setSelectedModel: text.setSelectedModel,
    modelFiles: text.modelFiles,
    setModelFiles: text.setModelFiles,
    isLoadingFiles: text.isLoadingFiles,
    filterState: text.filterState,
    setFilterState: text.setFilterState,
    textFiltersVisible: text.textFiltersVisible,
    setTextFiltersVisible: text.setTextFiltersVisible,
    downloadedModels: text.downloadedModels,
    hasActiveFilters: text.hasActiveFilters,
    ramGB: text.ramGB,
    deviceRecommendation: text.deviceRecommendation,
    filteredResults: text.filteredResults,
    recommendedAsModelInfo: text.recommendedAsModelInfo,
    trendingAsModelInfo: text.trendingAsModelInfo,
    handleSearch: text.handleSearch,
    handleSelectModel: text.handleSelectModel,
    handleDownload,
    handleRepairMmProj: text.handleRepairMmProj,
    handleCancelDownload: text.handleCancelDownload,
    handleDeleteModel: text.handleDeleteModel,
    clearFilters: text.clearFilters,
    toggleFilterDimension: text.toggleFilterDimension,
    toggleOrg: text.toggleOrg,
    setTypeFilter: text.setTypeFilter,
    setSourceFilter: text.setSourceFilter,
    setSizeFilter: text.setSizeFilter,
    setQuantFilter: text.setQuantFilter,
    setSortOption: text.setSortOption,
    isModelDownloaded: text.isModelDownloaded,
    getDownloadedModel: text.getDownloadedModel,
    isRepairingVisionModel: text.isRepairingVisionModel,
    // image model state & handlers
    availableHFModels: image.availableHFModels,
    hfModelsLoading: image.hfModelsLoading,
    hfModelsError: image.hfModelsError,
    backendFilter: image.backendFilter,
    setBackendFilter: image.setBackendFilter,
    styleFilter: image.styleFilter,
    setStyleFilter: image.setStyleFilter,
    sdVersionFilter: image.sdVersionFilter,
    setSdVersionFilter: image.setSdVersionFilter,
    imageFilterExpanded: image.imageFilterExpanded,
    setImageFilterExpanded: image.setImageFilterExpanded,
    imageSearchQuery: image.imageSearchQuery,
    setImageSearchQuery: image.setImageSearchQuery,
    imageFiltersVisible: image.imageFiltersVisible,
    setImageFiltersVisible: image.setImageFiltersVisible,
    imageRec: image.imageRec,
    showRecommendedOnly: image.showRecommendedOnly,
    setShowRecommendedOnly: image.setShowRecommendedOnly,
    showRecHint: image.showRecHint,
    setShowRecHint: image.setShowRecHint,
    downloadedImageModels: image.downloadedImageModels,
    hasActiveImageFilters: image.hasActiveImageFilters,
    filteredHFModels: image.filteredHFModels,
    downloadedSearchMatches: image.downloadedSearchMatches,
    imageRecommendation: image.imageRecommendation,
    loadHFModels: image.loadHFModels,
    clearImageFilters: image.clearImageFilters,
    isRecommendedModel: image.isRecommendedModel,
    handleDownloadImageModel,
    handleCancelImageDownload: image.handleCancelImageDownload,
    setUserChangedBackendFilter: image.setUserChangedBackendFilter,
  };
}

export type ModelsScreenViewModel = ReturnType<typeof useModelsScreen>;
