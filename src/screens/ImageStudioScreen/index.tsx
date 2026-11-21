/**
 * Image Studio tab: image-only generation, separate from chat. Never loads or calls the text model
 * itself (see useImageStudio for the service contract).
 */
import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, Image, ScrollView } from 'react-native';
import Icon from 'react-native-vector-icons/Feather';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { ScreenHeader } from '../../components/ScreenHeader';
import { Button } from '../../components/Button';
import { SliderSetting } from '../../components/SliderSetting';
import { ModelSelectorModal } from '../../components/ModelSelectorModal';
import { ImageQualityBasicSliders } from '../../components/GenerationSettingsModal/ImageQualitySliders';
import { useTheme, useThemedStyles } from '../../theme';
import { useAppStore } from '../../stores';
import type { ActiveImageEngine } from '../../services/imageGenerationTypes';
import type { RootStackParamList } from '../../navigation/types';
import type { GeneratedImage } from '../../types';
import { createStyles } from './styles';
import { ReferenceSection } from './ReferenceSection';
import { galleryReference, useImageStudio } from './useImageStudio';

const SEED_MAX = 2147483647;

function engineLabel(engine: ActiveImageEngine): string | null {
  if (!engine) return null;
  if (engine.kind === 'local') return engine.model.name;
  return `${engine.server.mediaModels?.image ?? 'Image model'} (${engine.server.name})`;
}

const noop = () => {};

const SeedSlider: React.FC = () => {
  const seed = useAppStore(s => s.settings.imageSeed);
  const updateSettings = useAppStore(s => s.updateSettings);
  return (
    <SliderSetting
      testID="studio-seed"
      compact
      label="Seed"
      description="Same seed + same settings = same image. -1 = random. Tap the value to type one."
      value={typeof seed === 'number' && seed >= 0 ? seed : -1}
      min={-1}
      max={SEED_MAX}
      step={1}
      decimals={0}
      formatValue={v => (v < 0 ? 'Random' : String(v))}
      onChange={v => updateSettings({ imageSeed: v < 0 ? -1 : Math.round(v) })}
    />
  );
};

const ResultStrip: React.FC<{
  images: GeneratedImage[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}> = ({ images, selectedId, onSelect }) => {
  const styles = useThemedStyles(createStyles);
  if (images.length === 0) return null;
  return (
    <View style={styles.card}>
      <Text style={styles.sectionLabel}>Studio results</Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.strip}
        testID="studio-strip"
      >
        {images.map(img => (
          <TouchableOpacity
            key={img.id}
            testID={`studio-strip-${img.id}`}
            onPress={() => onSelect(img.id)}
            accessibilityLabel={img.prompt}
          >
            <Image
              source={{ uri: `file://${img.imagePath}` }}
              style={[styles.thumb, selectedId === img.id && styles.thumbSelected]}
            />
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
};

export const ImageStudioScreen: React.FC = () => {
  const { colors } = useTheme();
  const styles = useThemedStyles(createStyles);
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const [showModelPicker, setShowModelPicker] = useState(false);
  const studio = useImageStudio();
  const { genState, studioInFlight, busyElsewhere, selectedImage } = studio;
  const modelName = engineLabel(studio.engine);
  const progress = genState.progress;
  const progressFraction =
    progress && progress.totalSteps > 0 ? Math.min(1, progress.step / progress.totalSteps) : 0;
  const canGenerate = !!studio.prompt.trim() && !!studio.engine && !genState.isGenerating;

  return (
    <SafeAreaView testID="image-studio-screen" style={styles.container} edges={['top']}>
      <ScreenHeader title="Image" variant="tab" />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <TouchableOpacity
          testID="studio-model-picker"
          style={[styles.card, styles.modelRow]}
          onPress={() => setShowModelPicker(true)}
          accessibilityRole="button"
          accessibilityLabel="Pick image model"
        >
          <Icon name="image" size={18} color={colors.textSecondary} />
          <Text
            testID="studio-model-name"
            style={[styles.modelName, !modelName && styles.modelNameMissing]}
            numberOfLines={1}
          >
            {modelName ?? 'No image model - tap to pick one'}
          </Text>
          <Icon name="chevron-down" size={18} color={colors.textSecondary} />
        </TouchableOpacity>

        <View style={styles.card}>
          <Text style={styles.sectionLabel}>Prompt</Text>
          <TextInput
            testID="studio-prompt"
            multiline
            value={studio.prompt}
            onChangeText={studio.setPrompt}
            placeholder="Describe the image"
            placeholderTextColor={colors.textMuted}
            style={[styles.input, styles.inputTall]}
          />
          <Text style={styles.sectionLabel}>Negative prompt</Text>
          <TextInput
            testID="studio-negative-prompt"
            multiline
            value={studio.negativePrompt}
            onChangeText={studio.setNegativePrompt}
            placeholder="What to avoid (empty = your default)"
            placeholderTextColor={colors.textMuted}
            style={styles.input}
          />
        </View>

        <ReferenceSection
          blocked={studio.referenceBlocked}
          reference={studio.reference}
          onChangeReference={studio.setReference}
          onPickFromDevice={studio.pickFromDevice}
          error={studio.referenceError}
          keepPercent={studio.keepPercent}
          onChangeKeep={studio.setKeepPercent}
          chain={studio.chain}
          onChangeChain={studio.setChain}
        />

        <View style={styles.card}>
          <ImageQualityBasicSliders />
          <SeedSlider />
        </View>

        {studioInFlight ? (
          <View style={styles.card} testID="studio-progress">
            <View style={styles.progressRow}>
              <Text style={styles.progressText} numberOfLines={2}>
                {genState.status ?? 'Generating...'}
                {progress ? ` (${progress.step}/${progress.totalSteps})` : ''}
              </Text>
              <Button
                title="Cancel"
                variant="outline"
                size="small"
                onPress={studio.cancel}
                testID="studio-cancel"
              />
            </View>
            <View style={styles.progressTrack}>
              <View style={[styles.progressFill, { width: `${Math.round(progressFraction * 100)}%` }]} />
            </View>
            {genState.previewPath && (
              <Image testID="studio-preview" source={{ uri: genState.previewPath }} style={styles.resultImage} />
            )}
          </View>
        ) : (
          <Button
            title="Generate"
            onPress={() => { studio.generate().catch(noop); }}
            disabled={!canGenerate}
            testID="studio-generate"
          />
        )}
        {busyElsewhere && (
          <Text style={styles.note}>An image is being generated in a chat. Try again when it ends.</Text>
        )}

        {selectedImage && !studioInFlight && (
          <View style={styles.card} testID="studio-result">
            <Image
              testID="studio-result-image"
              source={{ uri: `file://${selectedImage.imagePath}` }}
              style={styles.resultImage}
            />
            <Text style={styles.resultMeta} numberOfLines={2}>
              {`${selectedImage.width}x${selectedImage.height} · ${selectedImage.steps} steps · seed ${selectedImage.seed}`}
            </Text>
            {!studio.referenceBlocked && (
              <View style={styles.resultActions}>
                <Button
                  title="Use as reference"
                  variant="secondary"
                  size="small"
                  onPress={() => studio.setReference(galleryReference(selectedImage))}
                  testID="studio-use-as-reference"
                />
              </View>
            )}
          </View>
        )}

        <ResultStrip
          images={studio.studioImages}
          selectedId={selectedImage?.id ?? null}
          onSelect={studio.setSelectedId}
        />
      </ScrollView>

      <ModelSelectorModal
        visible={showModelPicker}
        initialTab="image"
        onClose={() => setShowModelPicker(false)}
        onSelectModel={() => setShowModelPicker(false)}
        onUnloadModel={noop}
        isLoading={false}
        onSelectionComplete={() => setShowModelPicker(false)}
        onBrowseModels={tab => {
          setShowModelPicker(false);
          navigation.navigate('Main', { screen: 'ModelsTab', params: { initialTab: tab } });
        }}
        onAddServer={() => navigation.navigate('RemoteServers')}
      />
    </SafeAreaView>
  );
};
