/**
 * Load-time llama.cpp controls beside the existing batch / KV-cache ones (shared by both settings
 * surfaces): micro-batch (n_ubatch), the V cache type split from K, mmap mode, and a memory readout
 * (model file, estimated KV cache for the current context + K/V types, live free RAM).
 */
import React, { useEffect, useState } from 'react';
import { Text, View } from 'react-native';
import { loadLlamaModelInfo } from 'llama.rn';
import { SliderSetting } from '../SliderSetting';
import { useThemedStyles } from '../../theme';
import { useAppStore } from '../../stores';
import type { CacheType } from '../../types';
import { useTextGenerationAdvanced, CACHE_TYPE_OPTIONS } from '../../hooks/useTextGenerationAdvanced';
import { hardwareService } from '../../services/hardware';
import { buildModelParams, modelMaxContextFromMetadata } from '../../services/llmHelpers';
import { estimateLlamaMemory } from '../../services/llamaMemoryEstimate';
import { formatBytes } from '../../utils/formatBytes';
import { createTextGenAdvancedStyles } from './textGenAdvancedStyles';
import { SegmentedRow, type PillOption } from './segmentedRow';

export const UbatchSlider: React.FC = () => {
  const styles = useThemedStyles(createTextGenAdvancedStyles);
  const { settings, updateSettings } = useAppStore();
  const nBatch = settings.nBatch ?? 512;
  const stored = settings.nUbatch && settings.nUbatch > 0 ? Math.min(settings.nUbatch, nBatch) : 0;
  return (
    <View style={styles.container}>
      <SliderSetting
        testID="ubatch-size-stepper"
        label="Micro-batch Size"
        description="Tokens computed per step inside a batch (n_ubatch). Smaller = smaller compute buffer, slower prompt processing. Requires model reload."
        value={stored}
        min={0} max={nBatch} step={32}
        formatValue={(v) => (v > 0 ? String(v) : `Same as batch (${nBatch})`)}
        onChange={(v) => updateSettings({ nUbatch: v > 0 ? v : 0 })}
      />
    </View>
  );
};

type VCacheId = CacheType | 'same';

export const VCacheTypeToggle: React.FC = () => {
  const { cacheDisabled, currentCacheTypeV, isFlashAttnOn, handleCacheTypeVChange } = useTextGenerationAdvanced();
  const options: PillOption<VCacheId>[] = [
    { id: 'same', label: 'Same as K' },
    ...CACHE_TYPE_OPTIONS.map((ct: CacheType) => ({ id: ct, label: ct })),
  ];
  return (
    <SegmentedRow<VCacheId>
      label="V Cache Type"
      description={`V half of the KV cache. A quantized V cache needs flash attention${isFlashAttnOn ? '' : ' (it will be turned on)'}. Some models (DeepSeek MLA) need K = V; the load then retries with V = K. Requires model reload.`}
      options={options}
      current={currentCacheTypeV}
      onSelect={handleCacheTypeVChange}
      testIdFor={(id) => `cache-type-v-${id}-button`}
      isDisabled={(id) => cacheDisabled && id !== 'same' && id !== 'f16'}
    />
  );
};

type MmapMode = 'auto' | 'on' | 'off';
const MMAP_OPTIONS: PillOption<MmapMode>[] = [
  { id: 'auto', label: 'Auto' },
  { id: 'on', label: 'On' },
  { id: 'off', label: 'Off' },
];
const MMAP_DESCRIPTIONS: Record<MmapMode, string> = {
  auto: 'Maps the model file (pages the OS can drop and re-read) unless a small q4_0 / iq4_nl model fits easily in free RAM. Requires model reload.',
  on: 'Always map the model file: its pages stay reclaimable, so Android is less likely to kill the app. Requires model reload.',
  off: 'Read the whole model into RAM: the OS cannot drop it, so a large model raises the risk of the app being killed. Requires model reload.',
};

export const MmapModeSelector: React.FC = () => {
  const { settings, updateSettings } = useAppStore();
  const current: MmapMode = settings.mmapMode ?? 'auto';
  return (
    <SegmentedRow<MmapMode>
      label="Memory Mapping (mmap)"
      description={MMAP_DESCRIPTIONS[current]}
      options={MMAP_OPTIONS}
      current={current}
      onSelect={(id) => updateSettings({ mmapMode: id })}
      testIdFor={(id) => `mmap-${id}-button`}
    />
  );
};

const FREE_RAM_POLL_MS = 3000;

/**
 * Model file size, the KV cache the current context + K/V types need (llamaMemoryEstimate, the same
 * formula the pre-load fit check uses), and live free RAM (hardwareService.getAppMemoryUsage, the
 * number the load gate reads). The KV figure is an estimate from the GGUF header.
 */
export const MemoryReadout: React.FC = () => {
  const styles = useThemedStyles(createTextGenAdvancedStyles);
  const settings = useAppStore(s => s.settings);
  const model = useAppStore(s => s.downloadedModels.find(m => m.id === s.activeModelId));
  const [freeBytes, setFreeBytes] = useState<number | null>(null);
  const [metadata, setMetadata] = useState<{ path: string; value: Record<string, unknown> | null } | null>(null);
  const llamaPath = model?.engine === 'llama' ? model.filePath : null;

  useEffect(() => {
    let cancelled = false;
    const read = () => {
      hardwareService.getAppMemoryUsage()
        .then(mem => { if (!cancelled) setFreeBytes(mem.available); })
        .catch(() => { if (!cancelled) setFreeBytes(null); });
    };
    read();
    const timer = setInterval(read, FREE_RAM_POLL_MS);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (!llamaPath) return;
    let cancelled = false;
    loadLlamaModelInfo(llamaPath)
      .then(info => { if (!cancelled) setMetadata({ path: llamaPath, value: info as Record<string, unknown> }); })
      .catch(() => { if (!cancelled) setMetadata({ path: llamaPath, value: null }); });
    return () => { cancelled = true; };
  }, [llamaPath]);

  const fileBytes = model ? model.fileSize + (model.engine === 'llama' ? model.mmProjFileSize ?? 0 : 0) : 0;
  let kvLine = 'KV cache: no local llama model selected';
  if (llamaPath) {
    const meta = metadata?.path === llamaPath ? metadata.value : undefined;
    if (meta === undefined) {
      kvLine = 'KV cache: reading model header...';
    } else {
      const params = buildModelParams(llamaPath, { ...settings, modelMaxContext: meta ? modelMaxContextFromMetadata(meta) : null });
      const est = meta
        ? estimateLlamaMemory(meta, { ctxLen: params.ctxLen, cacheType: params.cacheType, cacheTypeV: params.cacheTypeV, nBatch: params.nBatch, weightsBytes: fileBytes })
        : null;
      kvLine = est
        ? `KV cache: ~${formatBytes(est.kvBytes)} at ${params.ctxLen} tokens (K ${params.cacheType}, V ${params.cacheTypeV})`
        : 'KV cache: unknown (model header lacks the fields)';
    }
  }
  return (
    <View style={styles.container} testID="memory-readout">
      <View style={styles.info}>
        <Text style={styles.label}>Memory</Text>
        <Text style={styles.desc} testID="memory-readout-model">
          {model ? `Model file: ${formatBytes(fileBytes)}` : 'Model file: no model selected'}
        </Text>
        <Text style={styles.desc} testID="memory-readout-kv">{kvLine}</Text>
        <Text style={styles.desc} testID="memory-readout-free">
          {freeBytes != null ? `Free RAM now: ${formatBytes(freeBytes)}` : 'Free RAM now: unknown'}
        </Text>
      </View>
    </View>
  );
};
