/**
 * Fast mode: final image in 4, 2 or 1 steps for the active image model.
 *  - stable-diffusion.cpp (GPU): one tap downloads the family's few-step LoRA, registers it in the
 *    LoRA list and applies its steps / guidance / sampler / schedule; Off restores the previous ones.
 *    Plus the "Fast decode (TAESD)" switch (tiny VAE, applies at the next model load).
 *  - LocalDream (NPU): no LoRA at runtime; a few-step checkpoint (by name) gets its preset.
 * Logic lives in services/fastModeService + services/fastModePresets.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { View, Text, Switch, TouchableOpacity } from 'react-native';
import { CustomAlert, hideAlert, initialAlertState, type AlertState } from './CustomAlert';
import { useTheme, useThemedStyles } from '../theme';
import type { ThemeColors } from '../theme';
import { useAppStore } from '../stores';
import { useDownloadStore, isActiveStatus, type DownloadEntry } from '../stores/downloadStore';
import { makeImageModelKey } from '../utils/modelKey';
import { sdFamilyLabel } from '../services/sdCppModelDetection';
import {
  FAST_LORA_PRESETS,
  FAST_STEP_CHOICES,
  TAESD_FILES,
  activeFastSteps,
  fastLoraFamilyOf,
  npuFastPreset,
  taesdFamilyOf,
  type FastSteps,
} from '../services/fastModePresets';
import {
  cancelFastDownload,
  fastLoraDownloadId,
  findPresetLora,
  recoverFastModeDownloads,
  selectNpuFastMode,
  selectSdCppFastMode,
  setTaesdEnabled,
  taesdDownloadId,
  turnOffFastMode,
} from '../services/fastModeService';
import logger from '../utils/logger';

export const NPU_FAST_MODE_HINT = 'Fast mode on NPU needs a few-step model (DMD2 / Hyper) from the catalog.';

const createStyles = (colors: ThemeColors) => ({
  group: { marginTop: 12 },
  label: { color: colors.text, fontSize: 14, fontWeight: '600' as const },
  desc: { color: colors.textMuted, fontSize: 12, marginTop: 2 },
  chips: { flexDirection: 'row' as const, flexWrap: 'wrap' as const, gap: 6, marginTop: 8 },
  chip: { paddingHorizontal: 10, paddingVertical: 6, borderRadius: 14, backgroundColor: colors.surfaceLight },
  chipActive: { backgroundColor: colors.primary },
  chipText: { color: colors.text, fontSize: 12 },
  chipTextActive: { color: colors.surface, fontWeight: '600' as const },
  row: { flexDirection: 'row' as const, alignItems: 'center' as const, justifyContent: 'space-between' as const, marginTop: 8 },
  rowInfo: { flex: 1, marginRight: 10 },
  progress: { color: colors.text, fontSize: 12, flex: 1, fontVariant: ['tabular-nums' as const] },
  cancel: { color: colors.error, fontSize: 12, marginLeft: 10 },
});
type Styles = ReturnType<typeof createStyles>;

const formatMB = (bytes: number) => {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)}GB`;
  return bytes >= 1e8 ? `${Math.round(bytes / 1e6)}MB` : `${(bytes / 1e6).toFixed(1)}MB`;
};

const stepLabel = (s: FastSteps) => `${s} step${s === 1 ? '' : 's'}`;

const isLive = (e: DownloadEntry | undefined): e is DownloadEntry => !!e && (isActiveStatus(e.status) || e.status === 'paused');

const Chip: React.FC<{ label: string; active: boolean; onPress: () => void; testID: string; styles: Styles }> = ({
  label, active, onPress, testID, styles,
}) => (
  <TouchableOpacity style={[styles.chip, active && styles.chipActive]} onPress={onPress} testID={testID}>
    <Text style={[styles.chipText, active && styles.chipTextActive]}>{label}</Text>
  </TouchableOpacity>
);

const DownloadRow: React.FC<{ entry: DownloadEntry; name: string; onCancel: () => void; testID: string; styles: Styles }> = ({
  entry, name, onCancel, testID, styles,
}) => {
  const total = entry.totalBytes || entry.combinedTotalBytes;
  const pct = total > 0 ? Math.floor((entry.bytesDownloaded / total) * 100) : 0;
  return (
    <View style={styles.row}>
      <Text style={styles.progress} numberOfLines={1}>{`Downloading ${name} ${pct}% of ${formatMB(total)}`}</Text>
      <TouchableOpacity onPress={onCancel} testID={testID}>
        <Text style={styles.cancel}>Cancel</Text>
      </TouchableOpacity>
    </View>
  );
};

export const FastModeSection: React.FC = () => {
  const { colors } = useTheme();
  const styles = useThemedStyles(createStyles);
  const {
    settings, downloadedImageModels, activeImageModelId, addDownloadedImageModel, setActiveImageModelId,
  } = useAppStore();
  const downloads = useDownloadStore(s => s.downloads);
  const [alertState, setAlertState] = useState<AlertState>(initialAlertState);
  const model = downloadedImageModels.find(m => m.id === activeImageModelId);

  const deps = useMemo(
    () => ({ addDownloadedImageModel, activeImageModelId, setActiveImageModelId, setAlertState }),
    [addDownloadedImageModel, activeImageModelId, setActiveImageModelId],
  );

  useEffect(() => {
    recoverFastModeDownloads(deps).catch(e => logger.error('[FastMode] recover failed:', e?.message));
  }, [deps]);

  const run = useCallback((p: Promise<unknown>) => {
    p.catch(e => logger.error('[FastMode] action failed:', e?.message));
  }, []);

  if (!model) return null;
  const current = activeFastSteps(settings.imageFastMode, model);
  const isSdCpp = model.backend === 'sdcpp';
  const family = isSdCpp ? fastLoraFamilyOf(model.sdFamily) : null;
  const taesdFamily = isSdCpp ? taesdFamilyOf(model.sdFamily) : null;
  const npuPreset = isSdCpp ? null : npuFastPreset(model.name);
  const trackColor = { false: colors.surfaceLight, true: `${colors.primary}80` };

  const offChip = (
    <Chip label="Off" active={current === null} onPress={turnOffFastMode} testID="fast-mode-off" styles={styles} />
  );

  let body: React.ReactNode;
  if (isSdCpp && family) {
    const presets = FAST_LORA_PRESETS[family];
    const liveRows = FAST_STEP_CHOICES
      .map(s => ({ preset: presets[s], entry: downloads[makeImageModelKey(fastLoraDownloadId(presets[s]))] }))
      .filter(r => isLive(r.entry));
    body = (
      <>
        <Text style={styles.desc}>
          {`Adds a few-step LoRA for ${sdFamilyLabel(family)} and sets steps, guidance, sampler and schedule. Off restores yours.`}
        </Text>
        <View style={styles.chips}>
          {offChip}
          {FAST_STEP_CHOICES.map(s => {
            const onPhone = !!findPresetLora(presets[s], settings.imageLoras);
            const label = onPhone ? stepLabel(s) : `${stepLabel(s)} (${formatMB(presets[s].sizeBytes)})`;
            return (
              <Chip
                key={s}
                label={label}
                active={current === s}
                onPress={() => run(selectSdCppFastMode(family, s, deps))}
                testID={`fast-mode-${s}`}
                styles={styles}
              />
            );
          })}
        </View>
        {liveRows.map(r => (
          <DownloadRow
            key={r.preset.fileName}
            entry={r.entry as DownloadEntry}
            name={r.preset.fileName.replace(/\.safetensors$/i, '')}
            onCancel={() => run(cancelFastDownload(fastLoraDownloadId(r.preset)))}
            testID={`fast-mode-cancel-${r.preset.steps}`}
            styles={styles}
          />
        ))}
      </>
    );
  } else if (isSdCpp) {
    body = (
      <Text style={styles.desc} testID="fast-mode-unsupported">
        {`No few-step LoRA for ${sdFamilyLabel(model.sdFamily)} checkpoints. Fast mode works with SD 1.x and SDXL.`}
      </Text>
    );
  } else if (npuPreset) {
    body = (
      <View style={styles.chips}>
        {offChip}
        {npuPreset.steps.map(s => (
          <Chip
            key={s}
            label={stepLabel(s)}
            active={current === s}
            onPress={() => selectNpuFastMode(npuPreset, s)}
            testID={`fast-mode-${s}`}
            styles={styles}
          />
        ))}
      </View>
    );
  } else {
    body = <Text style={styles.desc} testID="fast-mode-npu-hint">{NPU_FAST_MODE_HINT}</Text>;
  }

  let taesd: React.ReactNode = null;
  if (taesdFamily) {
    const taesdEntry = downloads[makeImageModelKey(taesdDownloadId(taesdFamily))];
    const downloading = isLive(taesdEntry);
    const on = (settings.imageSdTaesd ?? false) || downloading;
    taesd = (
      <>
        <View style={styles.row}>
          <View style={styles.rowInfo}>
            <Text style={styles.label}>Fast decode (TAESD)</Text>
            <Text style={styles.desc}>
              {`Tiny decoder (${formatMB(TAESD_FILES[taesdFamily].sizeBytes)}) instead of the full VAE. Applies at the next model load.`}
            </Text>
          </View>
          <Switch
            value={on}
            onValueChange={v => run(setTaesdEnabled(v, taesdFamily, deps))}
            trackColor={trackColor}
            thumbColor={on ? colors.primary : colors.textMuted}
            testID="fast-mode-taesd"
          />
        </View>
        {downloading && (
          <DownloadRow
            entry={taesdEntry}
            name="TAESD"
            onCancel={() => run(cancelFastDownload(taesdDownloadId(taesdFamily)))}
            testID="fast-mode-cancel-taesd"
            styles={styles}
          />
        )}
      </>
    );
  }

  return (
    <View style={styles.group} testID="fast-mode-section">
      <Text style={styles.label}>Fast mode</Text>
      {body}
      {taesd}
      <CustomAlert
        visible={alertState.visible}
        title={alertState.title}
        message={alertState.message}
        buttons={alertState.buttons}
        onClose={() => setAlertState(hideAlert())}
      />
    </View>
  );
};
