/**
 * Image defaults shared by both image settings surfaces: a negative prompt and a fixed seed used when
 * a request names none (imageParameterPolicy applyImageDefaults, applied once in
 * imageGenerationService for every engine), and CLIP skip. CLIP skip reaches stable-diffusion.cpp
 * (SdCppNative.nativeGenerate clipSkip) and the user's own image server (clip_skip); the LocalDream
 * NPU/GPU engine has no such input, so the control is hidden for those models.
 */
import React, { useEffect, useState } from 'react';
import { Text, TextInput, View } from 'react-native';
import { SliderSetting } from './SliderSetting';
import { useTheme, useThemedStyles } from '../theme';
import { SPACING } from '../constants';
import { useAppStore } from '../stores';
import { createTextGenAdvancedStyles } from './settings/textGenAdvancedStyles';

const SEED_MAX = 2147483647;
const CLIP_SKIP_MAX = 12;

export const ImageDefaultsSettings: React.FC<{ compact?: boolean }> = ({ compact = false }) => {
  const styles = useThemedStyles(createTextGenAdvancedStyles);
  const { colors } = useTheme();
  const { settings, updateSettings, downloadedImageModels, activeImageModelId } = useAppStore();
  const backend = downloadedImageModels.find(m => m.id === activeImageModelId)?.backend;
  // No local image model selected = the image server path; sd.cpp models take it locally.
  const clipSkipApplies = !backend || backend === 'sdcpp';
  const stored = settings.imageNegativePrompt ?? '';
  const [negative, setNegative] = useState(stored);
  useEffect(() => { setNegative(stored); }, [stored]);
  const commit = () => { if (negative !== stored) updateSettings({ imageNegativePrompt: negative }); };

  return (
    <>
      <View style={styles.container}>
        <View style={styles.info}>
          <Text style={styles.label}>Default Negative Prompt</Text>
          <Text style={styles.desc}>What images should avoid. Used when a request gives none.</Text>
        </View>
        <TextInput
          testID="image-negative-prompt"
          multiline
          value={negative}
          onChangeText={setNegative}
          onEndEditing={commit}
          onBlur={commit}
          placeholder="e.g. blurry, low quality, watermark"
          placeholderTextColor={colors.textMuted}
          style={{
            minHeight: 56,
            borderWidth: 1,
            borderColor: colors.border,
            borderRadius: 8,
            padding: SPACING.sm,
            color: colors.text,
            textAlignVertical: 'top',
          }}
        />
      </View>
      <SliderSetting
        testID="image-seed"
        compact={compact}
        label="Image Seed"
        description="Same seed + same settings = same image. -1 = random. Tap the value to type one."
        value={typeof settings.imageSeed === 'number' && settings.imageSeed >= 0 ? settings.imageSeed : -1}
        min={-1} max={SEED_MAX} step={1} decimals={0}
        formatValue={(v) => (v < 0 ? 'Random' : String(v))}
        onChange={(v) => updateSettings({ imageSeed: v < 0 ? -1 : Math.round(v) })}
      />
      {clipSkipApplies && (
        <SliderSetting
          testID="image-clip-skip"
          compact={compact}
          label="CLIP Skip"
          description="Skip the last CLIP layers of the text encoder (2 is common for anime checkpoints). 0 = model default. stable-diffusion.cpp models and your image server only."
          value={settings.imageClipSkip ?? 0}
          min={0} max={CLIP_SKIP_MAX} step={1} decimals={0}
          formatValue={(v) => (v <= 0 ? 'Model default' : String(v))}
          onChange={(v) => updateSettings({ imageClipSkip: Math.max(0, Math.round(v)) })}
        />
      )}
    </>
  );
};
