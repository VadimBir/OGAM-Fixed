import React, { useState } from 'react';
import { Text } from 'react-native';
import { AdvancedToggle, Card } from '../../components';
import { SliderSetting } from '../../components/SliderSetting';
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
} from '../../components/settings/textGenAdvancedSections';
import { TextLoraSection } from '../../components/settings/TextLoraSection';
import { useTextGenerationSettings } from '../../hooks/useTextGenerationSettings';
import { useThemedStyles } from '../../theme';
import { createStyles } from './styles';

export const TextGenerationSection: React.FC = () => {
  const styles = useThemedStyles(createStyles);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const { isLiteRT, llama, liteRT, toolCalls } = useTextGenerationSettings();

  return (
    <Card style={styles.section}>
      <Text style={styles.settingHelp}>
        {isLiteRT
          ? 'Configure LiteRT model behavior.'
          : 'Configure LLM behavior for text responses.'}
      </Text>

      {isLiteRT ? (
        <>
          <SliderSetting testID="litert-temperature" {...liteRT.temperature} />
          <SliderSetting testID="litert-max-tokens" {...liteRT.maxTokens} />
        </>
      ) : (
        <>
          <SliderSetting testID="llama-temperature" {...llama.temperature} />
          <SliderSetting
            testID="llama-context-length"
            {...llama.contextLength}
          />
          <SliderSetting testID="llama-max-tokens" {...llama.maxTokens} />
          <ThinkingBudgetSelector />
          <TextLoraSection />
        </>
      )}

      <ShowGenerationDetailsToggle />
      <AdvancedToggle
        isExpanded={showAdvanced}
        onPress={() => setShowAdvanced(current => !current)}
        testID="text-advanced-toggle"
      />
      {showAdvanced ? (
        isLiteRT ? (
          <>
            <SliderSetting testID="litert-top-p" {...liteRT.topP} />
            <SliderSetting testID="litert-top-k" {...liteRT.topK} />
            <SliderSetting testID="max-tool-calls" {...toolCalls} />
            <LiteRTBackendSelector />
            <MemoryReadout />
            <ModelLoadingModeSelector />
          </>
        ) : (
          <>
            <SliderSetting testID="llama-top-p" {...llama.topP} />
            <SliderSetting testID="repeat-penalty" {...llama.repeatPenalty} />
            <AdvancedSamplingSection />
            <SliderSetting testID="max-tool-calls" {...toolCalls} />
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
        )
      ) : null}
    </Card>
  );
};
