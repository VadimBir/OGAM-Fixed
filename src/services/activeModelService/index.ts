// ActiveModelService — THE ONLY PLACE models should be loaded/unloaded from.
import { llmService } from '../llm';
import { liteRTService } from '../litert';
import { loadedTextEngines } from '../engines';
import { imageEngineRouter as onnxImageGeneratorService } from '../imageEngineRouter';
import { sdCppLoadConfigFromSettings } from '../sdCppGenerator';
import { estimateImageModelRamBytes, estimateSdCppModelMemory, imageModelResidencyMB } from '../imageModelMemory';
import { awaitMemoryReclaim, INCOMING_PEAK_FACTOR } from '../memoryBudget';
import { hardwareService } from '../hardware';
import { modelResidencyManager } from '../modelResidency';
import { OverridableMemoryError, ImageModelIncompleteError } from '../modelLoadErrors';
import { validateImageModelDir } from '../../utils/imageModelIntegrity';
import { remoteServerManager } from '../remoteServerManager';
import { useAppStore, useRemoteServerStore } from '../../stores';
import logger from '../../utils/logger';
import {
  createSelectedTextModelResolver,
  selectedTextModelIdOf,
} from './selectedTextModel';
import { activeModelSnapshot } from './snapshot';
import type { DownloadedModel, ONNXImageModel } from '../../types';
import type {
  ActiveModelInfo,
  ResourceUsage,
  ModelChangeListener,
} from './types';
import { estimateTextModelMemory, textFootprintOf } from './memory';
import { doLoadTextModel, doLoadImageModel, checkImageModelCanLoad } from './loaders';
import {
  getResourceUsage as _getResourceUsage,
  syncWithNativeState as _syncWithNativeState,
  memoryRefusalDetail,
  isTextGenerationInFlight,
  isImageGenerationInFlight,
} from './utils';
;
class ActiveModelService {
  private readonly listeners: Set<ModelChangeListener> = new Set();
  private readonly loadingState = { text: false, image: false };
  private loadedTextModelId: string | null = null;
  private loadedImageModelId: string | null = null;
  private loadedImageModelThreads: number | null = null;
  /** True once a local generation completed on the loaded image model: sd.cpp allocates its weights
   *  on first use (no eager_load in sdcpp_jni.cpp), so after that only the compute part is still to
   *  come. Reset on every image load/unload. */
  private imageWeightsMaterialized = false;
  private textLoadPromise: Promise<void> | null = null;
  /** Resolves the selection against the downloaded list, tolerating a rebuilt id (see resolveModel). */
  private readonly selectedTextModel = createSelectedTextModelResolver({
    read: () => {
      const store = useAppStore.getState();
      return { models: store.downloadedModels, selectedId: store.activeModelId };
    },
    warn: message => logger.warn(message),
  });
  private imageLoadPromise: Promise<void> | null = null;
  /** The SINGLE writer for the loaded-text-model id: keeps the private field and the reactive store
   *  projection (loadedTextModelId) in lockstep, so every surface reads one truth for "currently loaded". */
  private setLoadedText(id: string | null): void {
    this.loadedTextModelId = id;
    useAppStore.getState().setLoadedTextModelId(id);
  }
  /**
   * The selected text model, resolved. The ONE answer to "which text model is chosen", so no surface
   * repeats the lookup and none of them can disagree about it.
   */
  resolveSelectedTextModel(): DownloadedModel | null {
    return this.selectedTextModel();
  }

  /** The id to load for a turn (see selectedTextModelIdOf for why the order matters). */
  selectedTextModelId(): string | null {
    return selectedTextModelIdOf(useAppStore.getState());
  }

  getActiveModels(): ActiveModelInfo {
    return activeModelSnapshot({
      textModel: this.resolveSelectedTextModel(),
      imageModel:
        useAppStore
          .getState()
          .downloadedImageModels.find(
            m => m.id === useAppStore.getState().activeImageModelId,
          ) ?? null,
      textIsLoaded: llmService.isModelLoaded() || liteRTService.isModelLoaded(),
      imageIsLoaded: this.loadedImageModelId != null,
      loading: this.loadingState,
    });
  }

  hasAnyModelLoaded(): boolean {
    const info = this.getActiveModels();
    return info.text.isLoaded || info.image.isLoaded;
  }
  /**
   * Whether the currently-active text model accepts audio input directly (no
   * Whisper STT needed). Engine-aware dispatch lives here so UI/hooks never
   * branch on engine type: LiteRT reports via its loaded model's audio flag,
   * llama.cpp via the multimodal projector's reported audio support.
   */
  supportsAudioInput(): boolean {
    const store = useAppStore.getState();
    const model = store.downloadedModels.find(m => m.id === store.activeModelId);
    if (!model) return false;
    if (model.engine === 'litert') {
      return liteRTService.supportsAudio();
    }
    return llmService.isModelLoaded() && !!llmService.getMultimodalSupport()?.audio;
  }
  getLoadedModelIds(): {
    textModelId: string | null;
    imageModelId: string | null;
  } {
    return {
      textModelId: this.loadedTextModelId,
      imageModelId: this.loadedImageModelId,
    };
  }
  getPerformanceStats() {
    return llmService.getPerformanceStats();
  }
  /**
   * Whether `modelId` is the currently-loaded text model. Engine-aware: LiteRT
   * models live in liteRTService, llama/GGUF models in llmService. Checking only
   * llmService reported a loaded LiteRT model as not-loaded, so the fast path
   * below missed and re-loaded it (unload+load) every time the chat called
   * loadTextModel — the "second loader" seen only for LiteRT models.
   */
  private isTextModelCurrent(modelId: string): boolean {
    if (this.loadedTextModelId !== modelId) return false;
    const model = useAppStore.getState().downloadedModels.find(m => m.id === modelId);
    return model?.engine === 'litert'
      ? liteRTService.isModelLoaded()
      : llmService.isModelLoaded();
  }

  async loadTextModel(
    modelId: string,
    timeoutMs: number = 120000,
    opts?: { override?: boolean },
  ): Promise<void> {
    // Fast path — model already loaded (no lock; just sync the store).
    if (this.isTextModelCurrent(modelId)) {
      const store = useAppStore.getState();
      if (store.activeModelId !== modelId) {
        store.setActiveModelId(modelId);
      }
      return;
    }
    // Everything else goes through the residency manager's global lock so no two
    // model operations ever touch memory at once (the single load gateway).
    await modelResidencyManager.runExclusive(`load:text:${modelId}`, () =>
      this.doLoadTextModelLocked(modelId, timeoutMs, opts),
    );
  }
  private async doLoadTextModelLocked(
    modelId: string,
    timeoutMs: number,
    opts?: { override?: boolean },
  ): Promise<void> {
    // Re-check after acquiring — a queued call may have loaded it already.
    if (this.isTextModelCurrent(modelId)) {
      const store = useAppStore.getState();
      if (store.activeModelId !== modelId) {
        store.setActiveModelId(modelId);
      }
      return;
    }
    const store = useAppStore.getState();
    const model = store.downloadedModels.find(m => m.id === modelId);
    if (!model) {
      throw new Error('Model not found');
    }
    // One estimate (estimateLlamaMemory for GGUF; file-size heuristic otherwise) for the gate and
    // the register. The load is charged its peak (x1.1); once loaded it is charged its resident
    // bytes. Reclaimable mmapped weights are left out of the live free-RAM charge.
    const textEstimate = await estimateTextModelMemory(model);
    const incoming = textFootprintOf(textEstimate, textEstimate.predictedReclaimable);
    // LiteRT weights + KV are dirty/accelerator memory → gated on REAL free RAM (mmap GGUF
    // stays clean/physical-cap). Derived once so makeRoomFor and register agree.
    const textIsDirty = model.engine === 'litert';
    // Residency manager is authoritative, and it makes the free-RAM decision BEFORE evicting: a
    // load it approves is not refused afterwards (llm.ts gets memoryGateApproved), so a resident
    // image model is never evicted for a text load that then fails its memory check.
    // Evicted unloads are the non-locking internal variants (we hold the lock), so no deadlock.
    const room = await modelResidencyManager.makeRoomFor(
      {
        key: 'text', type: 'text', modelId, sizeMB: incoming.peakMB,
        // GGUF: always state the live part, so the gate checks free RAM on every platform (iOS too,
        // where it is otherwise physical-only for clean loads) before evicting anything.
        liveMB: model.engine === 'llama' ? incoming.livePeakMB : undefined, dirtyMemory: textIsDirty,
      },
      { override: opts?.override },
    );
    // First refusal is OVERRIDABLE ("Load Anyway"); a refusal UNDER override is a hard limit → plain Error.
    if (!room.fits) {
      const detail = memoryRefusalDetail(room);
      throw opts?.override
        ? new Error(`Not enough free memory to load this model, even after freeing other models${detail}. Close other apps or choose a smaller model.`)
        : new OverridableMemoryError(`Not enough free memory to load this model${detail}. Close other apps or choose a smaller model.`);
    }
    this.loadingState.text = true;
    this.notifyListeners();
    this.textLoadPromise = doLoadTextModel({
      model,
      modelId,
      store,
      timeoutMs,
      override: !!opts?.override,
      loadedTextModelId: this.loadedTextModelId,
      onLoaded: id => {
        this.setLoadedText(id);
        useAppStore.getState().setTextModelEvicted(false); // loaded → clear any prior eviction
        // A GGUF loaded with mmap OFF (shouldDisableMmap / mmapMode 'off') holds its weights in
        // anonymous RAM: register it as dirty so the dirty ceiling counts it (it was always 'clean').
        const residentDirty = textIsDirty || llmService.weightsAreMmapped?.() === false;
        // Charged at its resident bytes, with the weights' actual mmap/offload state.
        const resident = textFootprintOf(textEstimate, !textIsDirty && !!llmService.weightsAreReclaimable?.());
        modelResidencyManager.register(
          // isBusy: the Android trim release never drops the text model mid-generation.
          {
            key: 'text', type: 'text', modelId, sizeMB: resident.residentMB,
            liveMB: model.engine === 'llama' ? resident.liveResidentMB : undefined,
            dirtyMemory: residentDirty, isBusy: isTextGenerationInFlight,
          },
          () => this.doUnloadTextModelLocked(true), // eviction keeps the selection
        );
      },
      onError: () => {
        this.setLoadedText(null);
      },
      onFinally: () => {
        this.loadingState.text = false;
        this.textLoadPromise = null;
        this.notifyListeners();
      },
    });
    await this.textLoadPromise;
  }
  /**
   * The ONE owner of the "active text model" write. Selecting a model MARKS it active (the load is
   * deferred to the first message); this is the single place a selection is recorded, so the View
   * dispatches this intent instead of poking the store — presentation holds no authoritative state.
   * activeModelId is then set from exactly three places, all here in the service: select (this),
   * load-success (loaders), and cleared on load-failure/unload — so it can never drift from reality.
   */
  selectTextModel(modelId: string): void {
    const store = useAppStore.getState();
    store.setActiveModelId(modelId);
    store.setLastTextModelId(modelId); // remembered choice → routing reloads it on demand after eviction
  }
  async unloadTextModel(keepSelection = false): Promise<void> {
    await modelResidencyManager.runExclusive('unload:text', () =>
      this.doUnloadTextModelLocked(keepSelection),
    );
  }
  /**
   * Non-locking unload core. Safe to call from inside a held lock (eviction).
   * `keepSelection` is true for residency EVICTION: free the RAM but keep the
   * model SELECTED (activeModelId) so the UI still shows it and it reloads on
   * demand — clearing the selection here is what made chat fall back to "Load a
   * model" and thrash. A user-initiated unload passes false to fully deselect.
   */
  private async doUnloadTextModelLocked(keepSelection = false): Promise<void> {
    if (this.textLoadPromise !== null) {
      await this.textLoadPromise;
    }
    const storeActiveModelId = useAppStore.getState().activeModelId;
    // Unload every engine that HOLDS a model, not the selected model's engine: the selection can name
    // the other engine (a LiteRT model picked while a GGUF is loaded) or resolve to nothing, and the
    // loaded model then stayed in RAM while residency forgot it.
    const engines = loadedTextEngines();
    const isNativeLoaded = engines.length > 0;
    if (!storeActiveModelId && !this.loadedTextModelId && !isNativeLoaded) {
      return;
    }
    this.loadingState.text = true;
    this.notifyListeners();
    try {
      for (const engine of engines) await engine.unloadModel();
      this.setLoadedText(null);
      // Eviction (keepSelection) keeps the selection & flags "tap to continue"; user unload clears both.
      if (keepSelection) { if (isNativeLoaded) useAppStore.getState().setTextModelEvicted(true); }
      else { useAppStore.getState().setActiveModelId(null); useAppStore.getState().setTextModelEvicted(false); }
      modelResidencyManager.release('text');
    } finally {
      this.loadingState.text = false;
      this.notifyListeners();
    }
  }
  async loadImageModel(
    modelId: string,
    timeoutMs: number = 180000,
    opts?: { override?: boolean },
  ): Promise<void> {
    await modelResidencyManager.runExclusive(`load:image:${modelId}`, () =>
      this.doLoadImageModelLocked(modelId, timeoutMs, opts),
    );
  }
  private async doLoadImageModelLocked(
    modelId: string,
    timeoutMs: number,
    opts?: { override?: boolean },
  ): Promise<void> {
    // Hydrate real device RAM BEFORE the compute-path decision. preferGpuForImageGen()
    // and estimateImageModelRam() read getTotalMemoryGB(), which returns a 4GB fallback
    // until the cache is populated — on a cold start that misclassifies an 8GB iOS 26
    // device as ANE (the path that fails outright there). Awaiting hydration first makes
    // the GPU/ANE choice + RAM estimate use the true total.
    await hardwareService.getDeviceInfo();
    const store = useAppStore.getState();
    const imageThreads = store.settings?.imageThreads ?? 4;
    // A thread change or an sd.cpp load-time option change (weight type / flash attention —
    // weights are converted at load) both need a fresh native context.
    const needsThreadReload =
      this.loadedImageModelId === modelId &&
      (this.loadedImageModelThreads !== imageThreads ||
        onnxImageGeneratorService.sdCppConfigDiffers(sdCppLoadConfigFromSettings(store.settings ?? {})));
    if (this.loadedImageModelId === modelId) {
      const isLoaded = await onnxImageGeneratorService.isModelLoaded();
      if (isLoaded && !needsThreadReload) {
        if (store.activeImageModelId !== modelId) {
          store.setActiveImageModelId(modelId);
        }
        return;
      }
    }
    const model = store.downloadedImageModels.find(m => m.id === modelId);
    if (!model) {
      throw new Error('Model not found');
    }
    // Integrity gate (B): a partial extraction crashes the native server as a misleading "backend unsupported" — throw typed error so the UI says "re-download".
    if (model.backend === 'mnn' || model.backend === 'qnn') {
      const integ = await validateImageModelDir(model.modelPath, model.backend);
      if (!integ.complete) throw new ImageModelIncompleteError(integ.missing);
    }
    const check = await checkImageModelCanLoad(modelId, model, opts);
    if (!check.canLoad) {
      throw check.overridable
        ? new OverridableMemoryError(check.error ?? 'Not enough memory to load this model.')
        : new Error(check.error);
    }
    this.loadingState.image = true;
    this.notifyListeners();
    this.imageLoadPromise = doLoadImageModel({
      model,
      modelId,
      imageThreads,
      needsThreadReload,
      cpuOnly: false,
      preferGpu: hardwareService.preferGpuForImageGen(),
      store,
      timeoutMs,
      loadedImageModelId: this.loadedImageModelId,
      onLoaded: (id, threads) => {
        this.loadedImageModelId = id;
        this.loadedImageModelThreads = threads;
        this.imageWeightsMaterialized = false;
        modelResidencyManager.register(
          // dirtyMemory: a resident CoreML/ONNX image model's working set can't be paged
          // out like clean mmap weights — its presence must gate other loads on live
          // os_proc RAM (capacityForSpec dirty-pressure), or a sidecar stacks onto its
          // generation spike and jetsams the app. Without this flag the resident-dirty
          // arm of the gate is a no-op.
          // isBusy: the trim / post-generation release never drops it mid-generation.
          // sizeMB: resident bytes (no x1.1, no generation compute), see imageModelResidencyMB.
          { key: 'image', type: 'image', modelId: id, dirtyMemory: true, sizeMB: imageModelResidencyMB(model).residentMB, isBusy: isImageGenerationInFlight },
          () => this.doUnloadImageModelLocked(true), // eviction keeps the selection
        );
      },
      onError: () => {
        this.loadedImageModelId = null;
        this.loadedImageModelThreads = null;
      },
      onFinally: () => {
        this.loadingState.image = false;
        this.imageLoadPromise = null;
        this.notifyListeners();
      },
    });
    await this.imageLoadPromise;
  }
  async unloadImageModel(keepSelection = false): Promise<void> {
    await modelResidencyManager.runExclusive('unload:image', () =>
      this.doUnloadImageModelLocked(keepSelection),
    );
  }
  /**
   * Non-locking unload core. Safe to call from inside a held lock (eviction).
   * `keepSelection` true for residency eviction (free RAM, keep the model
   * selected so it reloads on demand); false for a user-initiated unload.
   */
  private async doUnloadImageModelLocked(keepSelection = false): Promise<void> {
    if (this.imageLoadPromise !== null) {
      await this.imageLoadPromise;
    }
    const store = useAppStore.getState();
    // "Loaded" (isModelLoaded) means alive AND ready on local-dream (LocalDreamModule.kt
    // isModelLoaded); a server process that is alive but not ready yet (starting, or flagged
    // not-ready after a failed request) still holds its RAM. Its model path stays set from start
    // until stopServer, so a set path also means a process to stop.
    const isNativeLoaded = await onnxImageGeneratorService.isModelLoaded();
    const nativePath = isNativeLoaded ? null : await onnxImageGeneratorService.getLoadedModelPath().catch(() => null);
    const nativeHoldsMemory = isNativeLoaded || nativePath != null;
    if (
      !store.activeImageModelId &&
      !this.loadedImageModelId &&
      !nativeHoldsMemory
    ) {
      return;
    }
    this.loadingState.image = true;
    this.notifyListeners();
    try {
      if (nativeHoldsMemory) {
        // Like the text unload (llm.ts doUnloadModel): the unload is done when the memory is back,
        // so the next load never allocates on top of the outgoing image model. The baseline is read
        // BEFORE the unload: local-dream's unloadModel resolves only after its server process has
        // exited (LocalDreamModule.kt stopServer), so a reading taken after it never sees the drop.
        const readMemory = () => Promise.resolve().then(() => hardwareService.getProcessMemory()).catch(() => null);
        const baseline = await readMemory();
        const residentMB = modelResidencyManager.getResidents().find(r => r.key === 'image')?.sizeMB;
        await onnxImageGeneratorService.unloadModel();
        // A small model frees less than the default 200MB: wait for half its resident size instead.
        await awaitMemoryReclaim(readMemory, {
          baseline,
          minDropMB: residentMB ? Math.max(1, Math.min(200, Math.round(residentMB / 2))) : undefined,
        });
      }
      this.loadedImageModelId = null;
      this.loadedImageModelThreads = null;
      this.imageWeightsMaterialized = false;
      if (!keepSelection) {
        store.setActiveImageModelId(null);
      }
      modelResidencyManager.release('image');
    } finally {
      this.loadingState.image = false;
      this.notifyListeners();
    }
  }
  async unloadAllModels(keepSelection = false): Promise<{ textUnloaded: boolean; imageUnloaded: boolean }> {
    const store = useAppStore.getState();
    const results = { textUnloaded: false, imageUnloaded: false };
    const hasTextModel =
      !!store.activeModelId ||
      !!this.loadedTextModelId ||
      loadedTextEngines().length > 0;
    const hasImageModel =
      !!store.activeImageModelId || !!this.loadedImageModelId;
    if (hasTextModel) {
      try {
        await this.unloadTextModel(keepSelection);
        results.textUnloaded = true;
      } catch {
        /* partial */
      }
    }
    if (hasImageModel) {
      try {
        await this.unloadImageModel(keepSelection);
        results.imageUnloaded = true;
      } catch {
        /* partial */
      }
    }
    return results;
  }
  /**
   * MB the loaded image model still has to allocate when it generates. Before its first generation:
   * its whole estimate (sd.cpp loads weights on first use). After one: for sd.cpp the estimate minus
   * the weights (+ their load buffers) already in RAM; other backends keep the whole estimate (their
   * weights/working-set split is not known here, so the gate errs toward freeing the text model).
   */
  private imageGenerationPendingMB(model: ONNXImageModel): number {
    const fullMB = Math.round(estimateImageModelRamBytes(model) / (1024 * 1024));
    if (!this.imageWeightsMaterialized || model.backend !== 'sdcpp') return fullMB;
    const est = estimateSdCppModelMemory(model);
    // Resident now: weights + LoRA + overhead (est.residentMB). Still to come: the larger compute
    // buffer, charged once with the incoming peak factor.
    return Math.ceil(Math.max(est.unetComputeMB, est.vaeComputeMB) * INCOMING_PEAK_FACTOR);
  }

  /** A local image generation finished on the loaded model: its weights are now in RAM. */
  noteImageGenerationCompleted(): void {
    if (this.loadedImageModelId != null) this.imageWeightsMaterialized = true;
  }

  /**
   * M4: a local image generation is about to run on the loaded image model. When the residency budget
   * (fresh RAM) cannot hold the resident text model next to the image generation, unload the text
   * model and KEEP its selection (same as eviction), so the LLM does not compete for RAM. Returns
   * true when the text model was unloaded here.
   */
  async suspendTextForImageGeneration(): Promise<boolean> {
    return modelResidencyManager.runExclusive('m4:suspend-text', async () => {
      const imageId = this.loadedImageModelId;
      if (!imageId || !this.loadedTextModelId) return false;
      const model = useAppStore.getState().downloadedImageModels.find(m => m.id === imageId);
      if (!model) return false;
      const pendingMB = this.imageGenerationPendingMB(model);
      if (await modelResidencyManager.canKeepOthersDuringGeneration('image', pendingMB)) return false;
      logger.log(`[MODEL-SM] M4 → text ${this.loadedTextModelId} yields RAM to image generation (pendingMB=${pendingMB})`);
      await this.doUnloadTextModelLocked(true);
      return true;
    });
  }

  /**
   * M4: the image generation is over. Reload the text model that was loaded before it (`modelId`)
   * when it is no longer loaded (the image load evicted it, or suspendTextForImageGeneration did).
   * Respects the loading mode: 'conservative' is one model at a time, so the text model stays
   * unloaded and reloads on the next message, as before. Skipped when the user changed or cleared
   * the text selection meanwhile. A failed reload is logged, never thrown into the image result.
   */
  async restoreTextAfterImageGeneration(modelId: string | null): Promise<void> {
    if (!modelId || this.loadedTextModelId != null) return;
    if (modelResidencyManager.getLoadPolicy() === 'conservative') return;
    if (useAppStore.getState().activeModelId !== modelId) return;
    try {
      logger.log(`[MODEL-SM] M4 → reloading text ${modelId} after image generation`);
      await this.loadTextModel(modelId);
    } catch (e) {
      logger.warn('[MODEL-SM] M4 → text reload after image generation failed:', e);
    }
  }

  /**
   * Free ALL model memory: unload local text+image AND disconnect any remote model.
   * THE single owning side-effect for "Eject All" — every screen dispatches this
   * instead of re-implementing the unload sequence (which is how one screen wired it
   * and another stubbed it). Returns how many models were unloaded. Logged for the
   * [MODEL-SM] trace.
   */
  async ejectAll(): Promise<{ count: number }> {
    const remote = useRemoteServerStore.getState();
    const hasRemote = !!(remote.activeRemoteTextModelId || remote.activeRemoteImageModelId);
    logger.log(`[MODEL-SM] ejectAll → start hasRemote=${hasRemote}`);
    // Eject = EVICT from RAM, KEEP the selection. The user's chosen models stay
    // selected (the rows still show them) and reload on demand — ejecting frees
    // memory, it does not un-choose the model. (Clearing selection here is what
    // turned the rows into "Unknown"/"—" after an eject.)
    const results = await this.unloadAllModels(true);
    let count = (results.textUnloaded ? 1 : 0) + (results.imageUnloaded ? 1 : 0);
    // Eject means ALL, not just text+image: free every remaining resident (sidecars — whisper/tts/embedding),
    // which unloadAllModels does not touch. Their real unload runs; they lazy-reload on next use. Previously
    // these leaked and kept charging the memory budget after the user ejected everything (DEV-B1).
    for (const r of modelResidencyManager.getResidents()) {
      const ejected = await modelResidencyManager.evictByKey(r.key);
      logger.log(`[MODEL-SM] ejectAll → sidecar ${r.type} (${r.key}) evicted=${ejected}`);
      if (ejected) count += 1;
    }
    if (hasRemote) {
      remoteServerManager.clearActiveRemoteModel();
      count += 1;
    }
    logger.log(`[MODEL-SM] ejectAll → done count=${count}`);
    return { count };
  }
  async getResourceUsage(): Promise<ResourceUsage> {
    return _getResourceUsage();
  }
  async clearTextModelCache(): Promise<void> {
    if (llmService.isModelLoaded()) {
      await llmService.clearKVCache(false);
    }
  }
  async syncWithNativeState(): Promise<void> {
    await _syncWithNativeState({
      loadedTextModelId: this.loadedTextModelId,
      loadedImageModelId: this.loadedImageModelId,
      setLoadedTextModelId: id => {
        this.setLoadedText(id);
      },
      setLoadedImageModelId: id => {
        this.loadedImageModelId = id;
      },
      setLoadedImageModelThreads: n => {
        this.loadedImageModelThreads = n;
      },
    });
  }
  subscribe(listener: ModelChangeListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  private notifyListeners(): void {
    const info = this.getActiveModels();
    this.listeners.forEach(listener => listener(info));
  }
}
export const activeModelService = new ActiveModelService();
