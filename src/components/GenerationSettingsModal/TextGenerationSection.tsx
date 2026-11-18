import React, { useState } from 'react';
import { View } from 'react-native';
import { AdvancedToggle } from '../AdvancedToggle';
import { SliderSetting } from '../SliderSetting';
import { useThemedStyles } from '../../theme';
import { createStyles } from './styles';
import {
  type NumericSettingModel,
  useTextGenerationSettings,
} from '../../hooks/useTextGenerationSettings';
import {
  AdvancedSamplingSection,
  BackendSelector,
  BatchSizeSlider,
  CpuThreadsSlider,
  FlashAttentionToggle,
  KvCacheTypeToggle,
  LiteRTBackendSelector,
  MemoryReadout,
  MmapModeSelector,
  ModelLoadingModeSelector,
  ShowGenerationDetailsToggle,
  SpeculativeDecodingToggle,
  ThinkingBudgetSelector,
  UbatchSlider,
  VCacheTypeToggle,
} from '../settings/textGenAdvancedSections';
import { TextLoraSection } from '../settings/TextLoraSection';

const ChatSettingSlider: React.FC<{ setting: NumericSettingModel }> = ({
  setting,
}) => <SliderSetting testID={`setting-${setting.key}`} compact {...setting} />;

export const TextGenerationSection: React.FC = () => {
  const styles = useThemedStyles(createStyles);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const { isLiteRT, llama, liteRT, toolCalls } = useTextGenerationSettings();
  const basicSettings = isLiteRT
    ? [liteRT.temperature, liteRT.maxTokens]
    : [llama.temperature, llama.contextLength, llama.maxTokens];
  const advancedSettings = isLiteRT
    ? [liteRT.topP, liteRT.topK, toolCalls]
    : [llama.topP, llama.repeatPenalty, toolCalls];

  return (
    <View style={styles.sectionCard}>
      {basicSettings.map(setting => (
        <ChatSettingSlider key={setting.key} setting={setting} />
      ))}
      {!isLiteRT && <ThinkingBudgetSelector compact />}
      {!isLiteRT && <TextLoraSection compact />}
      <ShowGenerationDetailsToggle />
      <AdvancedToggle
        isExpanded={showAdvanced}
        onPress={() => setShowAdvanced(current => !current)}
        testID="modal-text-advanced-toggle"
      />
      {showAdvanced ? (
        <>
          {advancedSettings.map(setting => (
            <ChatSettingSlider key={setting.key} setting={setting} />
          ))}
          {isLiteRT ? (
            <>
              <LiteRTBackendSelector />
              <MemoryReadout />
              <ModelLoadingModeSelector />
            </>
          ) : (
            <>
              <AdvancedSamplingSection compact />
              <CpuThreadsSlider />
              <BatchSizeSlider />
              <UbatchSlider />
              <BackendSelector />
              <FlashAttentionToggle />
              <SpeculativeDecodingToggle />
              <KvCacheTypeToggle />
              <VCacheTypeToggle />
              <MmapModeSelector />
              <MemoryReadout />
              <ModelLoadingModeSelector />
            </>
          )}
        </>
      ) : null}
    </View>
  );
};
