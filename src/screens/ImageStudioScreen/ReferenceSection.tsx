import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Image, ScrollView } from 'react-native';
import Icon from 'react-native-vector-icons/Feather';
import { SliderSetting } from '../../components/SliderSetting';
import { useTheme, useThemedStyles } from '../../theme';
import { useAppStore } from '../../stores';
import type { GeneratedImage } from '../../types';
import { createStyles } from './styles';
import {
  REFERENCE_BLOCKED_NOTE,
  galleryReference,
  type StudioReference,
} from './useImageStudio';

type ReferenceMode = 'none' | 'device' | 'gallery';

interface ReferenceSectionProps {
  blocked: boolean;
  reference: StudioReference | null;
  onChangeReference: (ref: StudioReference | null) => void;
  onPickFromDevice: () => void;
  error: string | null;
  keepPercent: number;
  onChangeKeep: (keep: number) => void;
  chain: boolean;
  onChangeChain: (on: boolean) => void;
}

const MODES: Array<{ mode: ReferenceMode; label: string }> = [
  { mode: 'none', label: 'None' },
  { mode: 'device', label: 'Device' },
  { mode: 'gallery', label: 'Gallery' },
];

const GalleryChooser: React.FC<{
  images: GeneratedImage[];
  selectedPath: string | null;
  onChoose: (img: GeneratedImage) => void;
}> = ({ images, selectedPath, onChoose }) => {
  const styles = useThemedStyles(createStyles);
  if (images.length === 0) {
    return <Text style={styles.note}>No images in the gallery yet.</Text>;
  }
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.strip}>
      {images.map(img => (
        <TouchableOpacity
          key={img.id}
          testID={`studio-ref-gallery-${img.id}`}
          accessibilityLabel={`Use ${img.prompt} as reference`}
          onPress={() => onChoose(img)}
        >
          <Image
            source={{ uri: `file://${img.imagePath}` }}
            style={[styles.thumb, selectedPath === img.imagePath && styles.thumbSelected]}
          />
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
};

export const ReferenceSection: React.FC<ReferenceSectionProps> = ({
  blocked,
  reference,
  onChangeReference,
  onPickFromDevice,
  error,
  keepPercent,
  onChangeKeep,
  chain,
  onChangeChain,
}) => {
  const { colors } = useTheme();
  const styles = useThemedStyles(createStyles);
  const generatedImages = useAppStore(s => s.generatedImages);
  const [showGallery, setShowGallery] = useState(false);
  const mode: ReferenceMode = reference ? reference.source : 'none';

  const onMode = (next: ReferenceMode) => {
    if (next === 'none') {
      setShowGallery(false);
      onChangeReference(null);
    } else if (next === 'device') {
      setShowGallery(false);
      onPickFromDevice();
    } else {
      setShowGallery(v => !v);
    }
  };

  return (
    <View style={styles.card} testID="studio-reference">
      <Text style={styles.sectionLabel}>Reference</Text>
      {blocked && (
        <Text style={styles.note} testID="studio-reference-blocked">
          {REFERENCE_BLOCKED_NOTE}
        </Text>
      )}
      <View
        style={blocked ? styles.dimmed : undefined}
        pointerEvents={blocked ? 'none' : 'auto'}
        accessibilityState={{ disabled: blocked }}
      >
        <View style={styles.segmentRow}>
          {MODES.map(({ mode: m, label }) => {
            const active = m === mode || (m === 'gallery' && showGallery && mode !== 'device');
            return (
              <TouchableOpacity
                key={m}
                testID={`studio-ref-${m}`}
                disabled={blocked}
                style={[styles.segment, active && styles.segmentActive]}
                onPress={() => onMode(m)}
                accessibilityRole="button"
                accessibilityState={{ disabled: blocked, selected: active }}
              >
                <Text style={[styles.segmentText, active && styles.segmentTextActive]}>{label}</Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {showGallery && !blocked && (
          <GalleryChooser
            images={generatedImages}
            selectedPath={reference?.source === 'gallery' ? reference.imagePath : null}
            onChoose={img => {
              onChangeReference(galleryReference(img));
              setShowGallery(false);
            }}
          />
        )}

        {reference && !blocked && (
          <View style={styles.referenceRow}>
            <Image testID="studio-ref-thumb" source={{ uri: reference.uri }} style={styles.thumb} />
            <Text style={styles.note}>
              {reference.source === 'device' ? 'From device' : 'From gallery'}
            </Text>
          </View>
        )}
        {error && <Text style={styles.errorText}>{error}</Text>}

        <SliderSetting
          testID="studio-keep"
          compact
          label="Keep"
          description="How much of the reference to keep. Lower = more change."
          value={keepPercent}
          min={0}
          max={100}
          step={1}
          decimals={0}
          formatValue={v => `${Math.round(v)}%`}
          onChange={onChangeKeep}
        />

        <TouchableOpacity
          testID="studio-chain-checkbox"
          style={styles.checkboxRow}
          disabled={blocked}
          onPress={() => onChangeChain(!chain)}
          accessibilityRole="checkbox"
          accessibilityState={{ checked: chain, disabled: blocked }}
        >
          <Icon
            name={chain ? 'check-square' : 'square'}
            size={20}
            color={chain ? colors.primary : colors.textMuted}
          />
          <Text style={styles.checkboxLabel}>Use each result as the next reference</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};
