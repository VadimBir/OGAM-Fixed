import React from 'react';
import { View, Text, Switch, TouchableOpacity } from 'react-native';
import RNFS from 'react-native-fs';
import { SliderSetting } from './SliderSetting';
import { useTheme, useThemedStyles } from '../theme';
import type { ThemeColors } from '../theme';
import { useAppStore } from '../stores';
import { ImageLora, ONNXImageModel } from '../types';
import { SD_WEIGHT_TYPES, type SdWeightType } from '../services/sdCppMemory';
import { estimateSdCppModelMemory } from '../services/imageModelMemory';
import { SDCPP_SAMPLERS, SDCPP_SCHEDULERS } from '../services/sdCppGenerator';
import {
  LOCALDREAM_SAMPLERS,
  SDCPP_MAX_SIZE,
  SDCPP_MIN_SIZE,
  SDCPP_SIZE_MULTIPLE,
  snapSdCppDimension,
} from '../services/imageEngineParams';

const createStyles = (colors: ThemeColors) => ({
  group: { marginTop: 12 },
  label: { color: colors.text, fontSize: 14, fontWeight: '600' as const },
  desc: { color: colors.textMuted, fontSize: 12, marginTop: 2 },
  chips: { flexDirection: 'row' as const, flexWrap: 'wrap' as const, gap: 6, marginTop: 8 },
  chip: { paddingHorizontal: 10, paddingVertical: 6, borderRadius: 14, backgroundColor: colors.surfaceLight },
  chipActive: { backgroundColor: colors.primary },
  chipText: { color: colors.text, fontSize: 12 },
  chipTextActive: { color: colors.surface, fontWeight: '600' as const },
  loraRow: { flexDirection: 'row' as const, alignItems: 'center' as const, justifyContent: 'space-between' as const, marginTop: 8 },
  loraName: { color: colors.text, fontSize: 13, flex: 1 },
  remove: { color: colors.error, fontSize: 12, marginRight: 10 },
  estimate: { color: colors.text, fontSize: 12, marginTop: 8, fontVariant: ['tabular-nums' as const] },
});
type Styles = ReturnType<typeof createStyles>;

const ChipRow: React.FC<{
  options: readonly string[];
  value: string;
  onChange: (v: string) => void;
  defaultLabel: string;
  styles: Styles;
  testID: string;
}> = ({ options, value, onChange, defaultLabel, styles, testID }) => (
  <View style={styles.chips}>
    {['', ...options].map(opt => {
      const active = value === opt;
      return (
        <TouchableOpacity
          key={opt || 'default'}
          testID={`${testID}-${opt || 'default'}`}
          style={[styles.chip, active && styles.chipActive]}
          onPress={() => onChange(opt)}
        >
          <Text style={[styles.chipText, active && styles.chipTextActive]}>{opt || defaultLabel}</Text>
        </TouchableOpacity>
      );
    })}
  </View>
);

const LoraList: React.FC<{ styles: Styles }> = ({ styles }) => {
  const { colors } = useTheme();
  const { settings, updateSettings } = useAppStore();
  const loras = settings.imageLoras ?? [];
  const patch = (id: string, change: Partial<ImageLora>) =>
    updateSettings({ imageLoras: loras.map(l => (l.id === id ? { ...l, ...change } : l)) });
  const remove = (lora: ImageLora) => {
    updateSettings({ imageLoras: loras.filter(l => l.id !== lora.id) });
    RNFS.unlink(lora.path).catch(() => {});
  };
  return (
    <View style={styles.group}>
      <Text style={styles.label}>LoRA</Text>
      <Text style={styles.desc}>
        {loras.length === 0
          ? 'Import an SD LoRA .safetensors from Models → Import; it is detected automatically.'
          : 'Enabled LoRAs apply to every generation with this checkpoint.'}
      </Text>
      {loras.map(lora => (
        <View key={lora.id}>
          <View style={styles.loraRow}>
            <Text style={styles.loraName} numberOfLines={1}>{lora.name}</Text>
            <TouchableOpacity onPress={() => remove(lora)} testID={`lora-remove-${lora.id}`}>
              <Text style={styles.remove}>Remove</Text>
            </TouchableOpacity>
            <Switch
              value={lora.enabled}
              onValueChange={v => patch(lora.id, { enabled: v })}
              trackColor={{ false: colors.surfaceLight, true: colors.primary }}
              thumbColor={colors.surface}
            />
          </View>
          {lora.enabled && (
            <SliderSetting
              compact
              testID={`lora-weight-${lora.id}`}
              label="Weight"
              value={lora.weight}
              min={-1} max={2} step={0.05} decimals={2}
              onChange={v => patch(lora.id, { weight: v })}
            />
          )}
        </View>
      ))}
    </View>
  );
};

const SwitchRow: React.FC<{
  label: string; desc: string; value: boolean; onChange: (v: boolean) => void; styles: Styles; testID: string;
}> = ({ label, desc, value, onChange, styles, testID }) => {
  const { colors } = useTheme();
  return (
    <View style={styles.group}>
      <View style={styles.loraRow}>
        <Text style={styles.label}>{label}</Text>
        <Switch
          testID={testID}
          value={value}
          onValueChange={onChange}
          trackColor={{ false: colors.surfaceLight, true: colors.primary }}
          thumbColor={colors.surface}
        />
      </View>
      <Text style={styles.desc}>{desc}</Text>
    </View>
  );
};

/** sd.cpp memory knobs: load-time weight quantization + flash attention, per-image VAE tiling,
 *  with the resulting RAM estimate (the same number the load gate uses). */
const SdCppMemoryControls: React.FC<{ model: ONNXImageModel; styles: Styles }> = ({ model, styles }) => {
  const { settings, updateSettings } = useAppStore();
  const weightType = settings.imageSdWeightType ?? 'auto';
  const est = estimateSdCppModelMemory(model);
  return (
    <>
      <View style={styles.group}>
        <Text style={styles.label}>Weight quantization</Text>
        <Text style={styles.desc}>
          Converted in memory at load (file untouched; reloads the model). UNet attention/linear and
          text-encoder weights convert; 3x3 convs and the VAE keep the file's precision.
        </Text>
        <View style={styles.chips}>
          {SD_WEIGHT_TYPES.map(wt => {
            const active = weightType === wt;
            return (
              <TouchableOpacity
                key={wt}
                testID={`sdcpp-wtype-${wt}`}
                style={[styles.chip, active && styles.chipActive]}
                onPress={() => updateSettings({ imageSdWeightType: wt as SdWeightType })}
              >
                <Text style={[styles.chipText, active && styles.chipTextActive]}>{wt === 'auto' ? 'file (auto)' : wt}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>
      <SwitchRow
        testID="sdcpp-flash-attn"
        styles={styles}
        label="Flash attention"
        desc="Streams UNet attention instead of materialising the score matrix (reloads the model)."
        value={settings.imageSdFlashAttn ?? true}
        onChange={v => updateSettings({ imageSdFlashAttn: v })}
      />
      <SwitchRow
        testID="sdcpp-vae-tiling"
        styles={styles}
        label="VAE tiling"
        desc="Decodes in 256 px tiles: the VAE buffer stays ~0.5 GB at any size instead of ~2 GB at 512 px."
        value={settings.imageSdVaeTiling ?? true}
        onChange={v => updateSettings({ imageSdVaeTiling: v })}
      />
      <Text style={styles.estimate} testID="sdcpp-ram-estimate">
        {`Est. RAM ~${(est.totalMB / 1024).toFixed(2)} GB = weights ${est.weightsMB} MB + max(UNet ${est.unetComputeMB}, VAE ${est.vaeComputeMB}) MB + overhead`}
      </Text>
    </>
  );
};

/**
 * Engine-aware image knobs. The active image model's format decides what is shown:
 * stable-diffusion.cpp checkpoints get sampler + scheduler + free W×H + LoRA; LocalDream
 * (QNN/MNN) models get the two samplers their core supports (DPM default, Euler A).
 */
export const ImageEngineSettings: React.FC<{ compact?: boolean }> = ({ compact }) => {
  const styles = useThemedStyles(createStyles);
  const { settings, updateSettings, downloadedImageModels, activeImageModelId } = useAppStore();
  const model = downloadedImageModels.find(m => m.id === activeImageModelId);
  if (!model || model.backend === 'coreml') return null;
  const sampler = settings.imageSampler ?? '';

  if (model.backend !== 'sdcpp') {
    return (
      <View style={styles.group}>
        <Text style={styles.label}>Sampler</Text>
        <Text style={styles.desc}>The NPU/GPU engine supports DPM (default) and Euler A.</Text>
        <ChipRow
          testID="image-sampler"
          styles={styles}
          options={LOCALDREAM_SAMPLERS.filter(s => s !== 'dpm')}
          value={(LOCALDREAM_SAMPLERS as readonly string[]).includes(sampler) && sampler !== 'dpm' ? sampler : ''}
          defaultLabel="dpm (default)"
          onChange={v => updateSettings({ imageSampler: v })}
        />
      </View>
    );
  }

  const width = snapSdCppDimension(settings.imageWidth);
  const height = snapSdCppDimension(settings.imageHeight, width);
  return (
    <>
      <View style={styles.group}>
        <Text style={styles.label}>Sampler</Text>
        <Text style={styles.desc}>stable-diffusion.cpp sampler. Default = the checkpoint's own.</Text>
        <ChipRow
          testID="image-sampler"
          styles={styles}
          options={SDCPP_SAMPLERS}
          value={(SDCPP_SAMPLERS as readonly string[]).includes(sampler) ? sampler : ''}
          defaultLabel="model default"
          onChange={v => updateSettings({ imageSampler: v })}
        />
      </View>
      <View style={styles.group}>
        <Text style={styles.label}>Scheduler</Text>
        <Text style={styles.desc}>Noise schedule (karras is the common pairing with dpm++2m).</Text>
        <ChipRow
          testID="image-scheduler"
          styles={styles}
          options={SDCPP_SCHEDULERS}
          value={settings.imageScheduler ?? ''}
          defaultLabel="model default"
          onChange={v => updateSettings({ imageScheduler: v })}
        />
      </View>
      <SliderSetting
        compact={compact}
        testID="sdcpp-width"
        label="Width"
        description="Checkpoint resolution, 64 px grid (SD1.x: 512, SDXL: 1024)."
        value={width}
        min={SDCPP_MIN_SIZE} max={SDCPP_MAX_SIZE} step={SDCPP_SIZE_MULTIPLE}
        formatValue={v => `${v}px`}
        onChange={v => updateSettings({ imageWidth: v })}
      />
      <SliderSetting
        compact={compact}
        testID="sdcpp-height"
        label="Height"
        value={height}
        min={SDCPP_MIN_SIZE} max={SDCPP_MAX_SIZE} step={SDCPP_SIZE_MULTIPLE}
        formatValue={v => `${v}px`}
        onChange={v => updateSettings({ imageHeight: v })}
      />
      <SdCppMemoryControls model={model} styles={styles} />
      <LoraList styles={styles} />
    </>
  );
};
