import React from 'react';
import { View, Text, Switch, TouchableOpacity } from 'react-native';
import RNFS from 'react-native-fs';
import { pick, types, isErrorWithCode, errorCodes } from '@react-native-documents/picker';
import { SliderSetting } from './SliderSetting';
import { CustomAlert, showAlert, hideAlert, initialAlertState, type AlertState } from './CustomAlert';
import { importSdCheckpointOrLora } from '../screens/ModelsScreen/importSdCheckpoint';
import { getErrorMessage } from '../screens/ModelsScreen/importHelpers';
import { isSdCppCandidateFileName } from '../services/sdCppModelDetection';
import { textSafetensorsRefusal } from '../services/textLora';
import { isPickerStuck } from '../utils/pickerErrorUtils';
import { useTheme, useThemedStyles } from '../theme';
import type { ThemeColors } from '../theme';
import { useAppStore } from '../stores';
import { ImageLora, ONNXImageModel } from '../types';
import { SD_WEIGHT_TYPES, type SdWeightType } from '../services/sdCppMemory';
import { estimateSdCppModelMemory } from '../services/imageModelMemory';
import { SDCPP_SAMPLERS, SDCPP_SCHEDULERS, sdCppGeneratorService } from '../services/sdCppGenerator';
import { loraCompatibility, sdFamilyLabel, type LoraBase } from '../services/sdCppModelDetection';
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
  warning: { color: colors.error, fontSize: 12, marginTop: 2 },
  button: {
    marginTop: 10, paddingVertical: 8, paddingHorizontal: 12, borderRadius: 8,
    borderWidth: 1, borderColor: colors.border, alignItems: 'center' as const,
  },
  buttonText: { color: colors.primary, fontSize: 14 },
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

/** Base model of each LoRA file, read from its safetensors header (cached in sdCppGeneratorService). */
function useLoraBases(paths: string[]): Record<string, LoraBase> {
  const [bases, setBases] = React.useState<Record<string, LoraBase>>({});
  const key = paths.join('\n');
  React.useEffect(() => {
    let alive = true;
    (async () => {
      const next: Record<string, LoraBase> = {};
      for (const p of key ? key.split('\n') : []) {
        next[p] = (await sdCppGeneratorService.fileFamily(p)).loraBase ?? 'unknown';
      }
      if (alive) setBases(next);
    })().catch(() => {});
    return () => { alive = false; };
  }, [key]);
  return bases;
}

/**
 * How sd.cpp applies LoRAs for the chosen weight type (memory cost), per the pinned sources:
 * AUTO merges into the weights unless they are quantized (diffusion_engine.cpp
 * configure_weight_loading); the JNI resolves AUTO to at_runtime for a q8_0/q5_0/q4_0 load-time
 * conversion (sdcpp_jni.cpp load_ctx). at_runtime keeps every LoRA tensor loaded (LoraModel in
 * runtime_lora_models); immediately frees them after the merge (model_manager.cpp apply_loras_to_params).
 */
function loraMemoryNote(weightType: SdWeightType): string {
  return weightType === 'auto'
    ? 'Applied by merging into the checkpoint weights at the next generation; the LoRA file is not kept in RAM. Changing LoRAs re-reads the affected weights.'
    : `With ${weightType} weights, LoRAs are applied at run time: each enabled LoRA stays in RAM (about its file size, not in the RAM estimate above) and every step does extra work.`;
}

/**
 * Picks one .safetensors/.ckpt and runs the SAME import as Models → Import (importSdCheckpointOrLora:
 * the header decides LoRA vs checkpoint; a checkpoint is registered as an sd.cpp model, a text-weights
 * file is refused). Busy/progress on the button, result in an alert.
 */
const ImportLoraButton: React.FC<{ styles: Styles }> = ({ styles }) => {
  const { addDownloadedImageModel, activeImageModelId, setActiveImageModelId } = useAppStore();
  const [busy, setBusy] = React.useState(false);
  const [progress, setProgress] = React.useState<{ fraction: number; fileName: string } | null>(null);
  const [alertState, setAlertState] = React.useState<AlertState>(initialAlertState);
  const pickingRef = React.useRef(false);

  const handleImport = async () => {
    if (pickingRef.current) return;
    pickingRef.current = true;
    setBusy(true);
    try {
      const [file] = await pick({ type: [types.allFiles], allowMultiSelection: false });
      if (!file) return;
      const name = (file.name?.trim() || decodeURIComponent(file.uri.split('/').pop() ?? '') || 'unknown').split('/').pop() || 'unknown';
      if (!isSdCppCandidateFileName(name)) {
        setAlertState(showAlert('Invalid File', 'Pick a Stable Diffusion LoRA (.safetensors). A .safetensors/.ckpt checkpoint is imported as a model.'));
        return;
      }
      setProgress({ fraction: 0, fileName: name });
      const textRefusal = name.toLowerCase().endsWith('.safetensors') ? await textSafetensorsRefusal(file.uri) : null;
      if (textRefusal) {
        setAlertState(showAlert('Convert to GGUF first', textRefusal));
        return;
      }
      await importSdCheckpointOrLora(
        { uri: file.uri, name, size: file.size },
        { addDownloadedImageModel, activeImageModelId, setActiveImageModelId, setImportProgress: setProgress, setAlertState },
      );
    } catch (error: unknown) {
      if (isErrorWithCode(error) && error.code === errorCodes.OPERATION_CANCELED) return;
      if (isPickerStuck(error)) {
        setAlertState(showAlert('File Picker Unavailable', "The file picker isn't responding. Please close and reopen the app, then try again."));
        return;
      }
      setAlertState(showAlert('Import Failed', getErrorMessage(error)));
    } finally {
      pickingRef.current = false;
      setBusy(false);
      setProgress(null);
    }
  };

  return (
    <>
      <TouchableOpacity style={styles.button} disabled={busy} onPress={handleImport} testID="image-lora-import">
        <Text style={styles.buttonText}>
          {busy ? (progress ? `Importing ${progress.fileName} ${Math.round(progress.fraction * 100)}%` : 'Working...') : 'Import LoRA (.safetensors)'}
        </Text>
      </TouchableOpacity>
      <CustomAlert
        visible={alertState.visible}
        title={alertState.title}
        message={alertState.message}
        buttons={alertState.buttons}
        onClose={() => setAlertState(hideAlert())}
      />
    </>
  );
};

const LoraList: React.FC<{ model: ONNXImageModel; styles: Styles }> = ({ model, styles }) => {
  const { colors } = useTheme();
  const { settings, updateSettings } = useAppStore();
  const loras = settings.imageLoras ?? [];
  const bases = useLoraBases(loras.map(l => l.path));
  const patch = (id: string, change: Partial<ImageLora>) =>
    updateSettings({ imageLoras: loras.map(l => (l.id === id ? { ...l, ...change } : l)) });
  const remove = (lora: ImageLora) => {
    updateSettings({ imageLoras: loras.filter(l => l.id !== lora.id) });
    sdCppGeneratorService.forgetFile(lora.path);
    RNFS.unlink(lora.path).catch(() => {});
  };
  return (
    <View style={styles.group}>
      <Text style={styles.label}>LoRA</Text>
      <Text style={styles.desc}>
        {loras.length === 0
          ? 'No LoRAs yet. Tap "Import LoRA (.safetensors)" below to add an SD LoRA; its base model is detected automatically.'
          : `Enabled LoRAs apply to every generation with this ${sdFamilyLabel(model.sdFamily)} checkpoint. ${loraMemoryNote(settings.imageSdWeightType ?? 'auto')}`}
      </Text>
      {loras.map(lora => {
        const base = bases[lora.path];
        const fit = base ? loraCompatibility(base, model.sdFamily) : 'unknown';
        return (
        <View key={lora.id}>
          <View style={styles.loraRow}>
            <Text style={styles.loraName} numberOfLines={1}>
              {base ? `${lora.name} · ${sdFamilyLabel(base)}` : lora.name}
            </Text>
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
          {fit === 'mismatch' && (
            <Text style={styles.warning} testID={`lora-mismatch-${lora.id}`}>
              {`Made for ${sdFamilyLabel(base)}, not this ${sdFamilyLabel(model.sdFamily)} checkpoint.${lora.enabled ? ' Generation is refused while it is enabled.' : ''}`}
            </Text>
          )}
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
        );
      })}
      <ImportLoraButton styles={styles} />
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
          Converted in memory at load (file untouched; reloads the model). Only linear weights take the
          chosen type. All convolutions (UNet and VAE) are f16, biases and norms f32, embeddings keep the file type.
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
        desc="Applies to the text encoder and the VAE only. Diffusion (UNet) attention always uses flash attention (reloads the model)."
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
 * (QNN/MNN) models get the two samplers their core supports (DPM default, Euler A) and a note that
 * LoRAs need the sd.cpp engine.
 */
export const ImageEngineSettings: React.FC<{ compact?: boolean }> = ({ compact }) => {
  const styles = useThemedStyles(createStyles);
  const { settings, updateSettings, downloadedImageModels, activeImageModelId } = useAppStore();
  const model = downloadedImageModels.find(m => m.id === activeImageModelId);
  if (!model || model.backend === 'coreml') return null;
  const sampler = settings.imageSampler ?? '';

  if (model.backend !== 'sdcpp') {
    // The LocalDream (QNN/MNN) request has no LoRA field (LocalDreamModule.buildGenerationBody) and
    // resolveEngineImageRequest sends none to it, so imported LoRAs never reach this engine.
    const loraCount = (settings.imageLoras ?? []).length;
    return (
      <>
        <View style={styles.group}>
          <Text style={styles.label}>Sampler</Text>
          <Text style={styles.desc}>The NPU/GPU engine supports DPM++ 2M (dpm, default), Euler A, Euler and DPM++ 2M SDE, each also with Karras sigmas, plus LCM.</Text>
          <ChipRow
            testID="image-sampler"
            styles={styles}
            options={LOCALDREAM_SAMPLERS.filter(s => s !== 'dpm')}
            value={(LOCALDREAM_SAMPLERS as readonly string[]).includes(sampler) && sampler !== 'dpm' ? sampler : ''}
            defaultLabel="dpm (default)"
            onChange={v => updateSettings({ imageSampler: v })}
          />
        </View>
        <View style={styles.group}>
          <Text style={styles.label}>LoRA</Text>
          <Text style={styles.desc} testID="lora-engine-note">
            {loraCount > 0
              ? `LoRA needs the sd.cpp (GPU) engine. Your ${loraCount} LoRA${loraCount === 1 ? ' is' : 's are'} not applied with this model; select a .safetensors checkpoint to use ${loraCount === 1 ? 'it' : 'them'}. LoRAs do not apply to NPU models: merge the LoRA into the checkpoint on a PC before NPU conversion.`
              : 'LoRA needs the sd.cpp (GPU) engine (a .safetensors checkpoint). LoRAs do not apply to NPU models: merge the LoRA into the checkpoint on a PC before NPU conversion.'}
          </Text>
          {/* Import works on every engine: the LoRA is kept and applies once a GPU checkpoint is selected. */}
          <ImportLoraButton styles={styles} />
        </View>
      </>
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
      <LoraList model={model} styles={styles} />
    </>
  );
};
