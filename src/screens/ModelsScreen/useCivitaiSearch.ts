import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { showAlert, type AlertState } from '../../components/CustomAlert';
import { useAppStore } from '../../stores';
import { useDownloadStore } from '../../stores/downloadStore';
import { makeImageModelKey } from '../../utils/modelKey';
import { backgroundDownloadService, hardwareService } from '../../services';
import {
  searchCivitaiModels,
  type CivitaiFamily,
  type CivitaiModelType,
  type CivitaiResult,
} from '../../services/civitaiBrowser';
import { getCivitaiApiKey, setCivitaiApiKey, removeCivitaiApiKey } from '../../services/civitaiApiKey';
import {
  civitaiImageModelId,
  civitaiLocalFileName,
  startCivitaiPhoneDownload,
} from '../../services/civitaiDownload';
import { estimateSdCppModelMemory } from '../../services/imageModelMemory';
import { modelMemoryBudgetMB } from '../../services/memoryBudget';
import logger from '../../utils/logger';

/**
 * Peak RAM budget for one model on this phone (the same budget the load gate uses), in MB.
 * 0 when the device RAM is unknown: then nothing is hidden for size.
 */
function deviceBudgetMB(): number {
  const totalGB = hardwareService.getTotalMemoryGB();
  return totalGB > 0 ? modelMemoryBudgetMB(totalGB * 1024) : 0;
}

/** False when a checkpoint's sd.cpp peak RAM (current weight-type / size settings) exceeds the budget. */
export function civitaiResultFits(result: CivitaiResult, budgetMB: number): boolean {
  if (result.type === 'LORA' || budgetMB <= 0) return true;
  return estimateSdCppModelMemory({ size: result.file.sizeBytes, sdFamily: result.family }).totalMB <= budgetMB;
}

export function useCivitaiSearch(setAlertState: (s: AlertState) => void) {
  const [query, setQuery] = useState('');
  const [type, setType] = useState<CivitaiModelType>('Checkpoint');
  const [family, setFamily] = useState<CivitaiFamily>('sd1');
  const [results, setResults] = useState<CivitaiResult[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hiddenCount, setHiddenCount] = useState(0);
  const [hasSearched, setHasSearched] = useState(false);
  const [apiKey, setApiKeyState] = useState<string | null>(null);
  const seqRef = useRef(0);

  const {
    settings, updateSettings, downloadedImageModels,
    addDownloadedImageModel, activeImageModelId, setActiveImageModelId,
  } = useAppStore();
  const showNsfw = settings?.civitaiShowNsfw === true;
  const imageLoras = settings?.imageLoras;
  const budgetMB = useMemo(deviceBudgetMB, []);

  useEffect(() => {
    let cancelled = false;
    getCivitaiApiKey().then(key => { if (!cancelled) setApiKeyState(key); });
    return () => { cancelled = true; };
  }, []);

  const runSearch = useCallback(async (cursor: string | null) => {
    const seq = ++seqRef.current;
    const append = cursor !== null;
    if (append) setLoadingMore(true); else { setLoading(true); setError(null); }
    try {
      const page = await searchCivitaiModels({ query, type, family, showNsfw, cursor, apiKey });
      if (seq !== seqRef.current) return;
      const fitting = page.results.filter(r => civitaiResultFits(r, budgetMB));
      const tooBig = page.results.length - fitting.length;
      setResults(prev => {
        if (!append) return fitting;
        const keys = new Set(prev.map(r => r.key));
        return [...prev, ...fitting.filter(r => !keys.has(r.key))];
      });
      setHiddenCount(prev => (append ? prev : 0) + page.skipped + tooBig);
      setNextCursor(page.nextCursor);
      setHasSearched(true);
    } catch (e: any) {
      if (seq !== seqRef.current) return;
      logger.warn('[Civitai] search failed:', e?.message);
      setError(e?.message || 'Civitai search failed.');
      if (!append) { setResults([]); setNextCursor(null); }
    } finally {
      if (seq === seqRef.current) { setLoading(false); setLoadingMore(false); }
    }
  }, [query, type, family, showNsfw, apiKey, budgetMB]);

  const search = useCallback(() => runSearch(null), [runSearch]);
  const loadMore = useCallback(() => {
    if (nextCursor && !loading && !loadingMore) runSearch(nextCursor);
  }, [nextCursor, loading, loadingMore, runSearch]);

  // A filter change (type / family / NSFW) re-runs the current search; the first run happens on open.
  useEffect(() => {
    runSearch(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [type, family, showNsfw]);

  const setShowNsfw = useCallback((value: boolean) => {
    updateSettings({ civitaiShowNsfw: value });
  }, [updateSettings]);

  const saveApiKey = useCallback(async (key: string) => {
    try {
      await setCivitaiApiKey(key);
      setApiKeyState(key.trim() || null);
    } catch (e: any) {
      setAlertState(showAlert('Could not save key', e?.message || 'Keychain write failed.'));
    }
  }, [setAlertState]);

  const clearApiKey = useCallback(async () => {
    await removeCivitaiApiKey();
    setApiKeyState(null);
  }, []);

  const isOnPhone = useCallback((result: CivitaiResult): boolean => {
    if (result.type === 'LORA') {
      const name = `/${civitaiLocalFileName(result)}`;
      return (imageLoras ?? []).some(l => l.path.endsWith(name));
    }
    const id = civitaiImageModelId(result);
    return downloadedImageModels.some(m => m.id === id);
  }, [downloadedImageModels, imageLoras]);

  const downloadToPhone = useCallback((result: CivitaiResult) => {
    startCivitaiPhoneDownload(result, {
      addDownloadedImageModel, activeImageModelId, setActiveImageModelId, setAlertState,
    }, apiKey).catch(e => logger.error('[Civitai] download start failed:', e?.message));
  }, [addDownloadedImageModel, activeImageModelId, setActiveImageModelId, setAlertState, apiKey]);

  const cancelDownload = useCallback(async (result: CivitaiResult) => {
    const modelKey = makeImageModelKey(civitaiImageModelId(result));
    const entry = useDownloadStore.getState().downloads[modelKey];
    if (!entry) return;
    useDownloadStore.getState().remove(modelKey);
    if (entry.downloadId) await backgroundDownloadService.cancelDownload(entry.downloadId).catch(() => {});
  }, []);

  return {
    query, setQuery, type, setType, family, setFamily,
    results, nextCursor, loading, loadingMore, error, hiddenCount, hasSearched,
    showNsfw, setShowNsfw,
    hasApiKey: !!apiKey, saveApiKey, clearApiKey,
    search, loadMore, isOnPhone, downloadToPhone, cancelDownload,
  };
}

export type CivitaiSearchViewModel = ReturnType<typeof useCivitaiSearch>;
