/**
 * Advanced sampling controls (top-k, min-p, typical-p, penalties, DRY, XTC, Mirostat, seed, extra
 * stop strings) shared by the Model Settings screen and the in-chat Generation Settings modal. The
 * values and their ranges come from useTextGenerationSettings().sampling, the one settings model.
 * They reach local llama.rn (llmHelpers buildCompletionParams) and remote servers
 * (samplingParams remoteSamplingOptions -> each provider sends what its server accepts).
 */
import React, { useEffect, useState } from 'react';
import { Text, TextInput, View } from 'react-native';
import { SliderSetting } from '../SliderSetting';
import { useTheme, useThemedStyles } from '../../theme';
import { useAppStore } from '../../stores';
import { useTextGenerationSettings } from '../../hooks/useTextGenerationSettings';
import { formatStopStringsText, parseStopStringsText, resolveSampling } from '../../services/samplingParams';
import { createTextGenAdvancedStyles } from './textGenAdvancedStyles';
import { SegmentedRow, type PillOption } from './segmentedRow';

type MirostatId = '0' | '1' | '2';
const MIROSTAT_OPTIONS: PillOption<MirostatId>[] = [
  { id: '0', label: 'Off' },
  { id: '1', label: 'v1' },
  { id: '2', label: 'v2' },
];

export const MirostatSelector: React.FC = () => {
  const { settings, updateSettings } = useAppStore();
  const mode = String(resolveSampling(settings).mirostat) as MirostatId;
  return (
    <SegmentedRow<MirostatId>
      label="Mirostat"
      description="Keeps output perplexity near Tau. When on, it replaces top-k / top-p / min-p sampling."
      options={MIROSTAT_OPTIONS}
      current={mode}
      onSelect={(id) => updateSettings({ mirostat: Number(id) })}
      testIdFor={(id) => `mirostat-${id}-button`}
    />
  );
};

export const StopStringsInput: React.FC = () => {
  const styles = useThemedStyles(createTextGenAdvancedStyles);
  const { colors } = useTheme();
  const { settings, updateSettings } = useAppStore();
  const stored = formatStopStringsText(settings.customStopStrings);
  const [text, setText] = useState(stored);
  useEffect(() => { setText(stored); }, [stored]);
  return (
    <View style={styles.container}>
      <View style={styles.info}>
        <Text style={styles.label}>Stop Strings</Text>
        <Text style={styles.desc}>
          One per line, added to the model's own stops. Type \n for a line break. Third-party servers take the first 4.
        </Text>
      </View>
      <TextInput
        testID="custom-stop-strings"
        multiline
        autoCapitalize="none"
        autoCorrect={false}
        value={text}
        onChangeText={setText}
        onEndEditing={() => updateSettings({ customStopStrings: parseStopStringsText(text) })}
        onBlur={() => updateSettings({ customStopStrings: parseStopStringsText(text) })}
        placeholder="e.g. ### User:"
        placeholderTextColor={colors.textMuted}
        style={styles.multilineInput}
      />
    </View>
  );
};

/** Every advanced sampling control, in the order llama.cpp applies the samplers. */
export const AdvancedSamplingSection: React.FC<{ compact?: boolean }> = ({ compact = false }) => {
  const { sampling } = useTextGenerationSettings();
  const sliders = [
    sampling.topK, sampling.minP, sampling.typicalP,
    sampling.penaltyLastN, sampling.presencePenalty, sampling.frequencyPenalty,
    sampling.dryMultiplier, sampling.dryBase, sampling.dryAllowedLength, sampling.dryPenaltyLastN,
    sampling.xtcProbability, sampling.xtcThreshold,
  ];
  return (
    <>
      {sliders.map(setting => (
        <SliderSetting {...setting} key={setting.key} testID={`setting-${setting.key}`} compact={compact} />
      ))}
      <MirostatSelector />
      <SliderSetting testID="setting-mirostatTau" compact={compact} {...sampling.mirostatTau} />
      <SliderSetting testID="setting-mirostatEta" compact={compact} {...sampling.mirostatEta} />
      <SliderSetting testID="setting-seed" compact={compact} {...sampling.seed} />
      <StopStringsInput />
    </>
  );
};
