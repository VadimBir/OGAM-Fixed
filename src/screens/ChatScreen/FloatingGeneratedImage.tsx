import React, { useState } from 'react';
import { Image, StyleSheet, TouchableOpacity, View, useWindowDimensions } from 'react-native';
import Icon from 'react-native-vector-icons/Feather';
import { useTheme } from '../../theme';
import type { ImageGenerationState } from '../../services/imageGenerationTypes';

// Dismissed results stay dismissed for the app session, also after the chat screen remounts.
const dismissedResultIds = new Set<string>();

type Props = {
  imageGenState: ImageGenerationState;
  conversationId: string | null;
  /** Distance from the bottom of the chat area: sits just above the composer. */
  bottom: number;
  /** Opens the chat's existing full-screen viewer (Save / Close). */
  onOpenFullscreen: (uri: string) => void;
};

/**
 * Floating card for the image just generated in this chat (local or remote, tool or manual).
 * Not a modal: the composer under it stays usable, so the user keeps typing while it is open.
 * Tap the thumbnail to enlarge it in place; the enlarged card opens the full-screen viewer.
 */
export const FloatingGeneratedImage: React.FC<Props> = ({
  imageGenState,
  conversationId,
  bottom,
  onOpenFullscreen,
}) => {
  const { colors } = useTheme();
  const { width: windowWidth } = useWindowDimensions();
  const [expanded, setExpanded] = useState(false);
  const [, setDismissTick] = useState(0);
  const result = imageGenState.result;
  if (
    imageGenState.phase !== 'done' ||
    !result?.imagePath ||
    !conversationId ||
    imageGenState.conversationId !== conversationId ||
    dismissedResultIds.has(result.id)
  ) {
    return null;
  }

  const uri = result.imagePath.startsWith('file://') ? result.imagePath : `file://${result.imagePath}`;
  const aspect = result.width > 0 && result.height > 0 ? result.width / result.height : 1;
  const size = expanded ? Math.round(windowWidth * 0.6) : 72;
  const dismiss = () => {
    dismissedResultIds.add(result.id);
    setExpanded(false);
    setDismissTick(n => n + 1);
  };

  return (
    <View
      testID="floating-generated-image"
      pointerEvents="box-none"
      style={[
        styles.card,
        { bottom, backgroundColor: colors.surface, borderColor: colors.border },
      ]}
    >
      <TouchableOpacity
        testID="floating-generated-image-thumb"
        accessibilityRole="imagebutton"
        accessibilityLabel={expanded ? 'Open image full screen' : 'Enlarge generated image'}
        activeOpacity={0.85}
        onPress={() => (expanded ? onOpenFullscreen(uri) : setExpanded(true))}
      >
        <Image
          source={{ uri }}
          style={[styles.image, { width: size, height: Math.round(size / aspect) }]}
          resizeMode="cover"
        />
      </TouchableOpacity>
      <View style={styles.actions}>
        {expanded && (
          <TouchableOpacity
            testID="floating-generated-image-collapse"
            accessibilityLabel="Shrink generated image"
            hitSlop={8}
            onPress={() => setExpanded(false)}
          >
            <Icon name="minimize-2" size={14} color={colors.textSecondary} />
          </TouchableOpacity>
        )}
        <TouchableOpacity
          testID="floating-generated-image-close"
          accessibilityLabel="Dismiss generated image"
          hitSlop={8}
          onPress={dismiss}
        >
          <Icon name="x" size={14} color={colors.textSecondary} />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    position: 'absolute',
    left: 16,
    zIndex: 10,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 4,
    padding: 4,
    borderRadius: 8,
    borderWidth: 1,
  },
  image: { borderRadius: 6 },
  actions: { gap: 8, paddingTop: 2 },
});

/** Test seam: forget dismissals between tests. */
export function resetFloatingGeneratedImageDismissals(): void {
  dismissedResultIds.clear();
}
