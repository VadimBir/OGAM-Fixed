import React, { useState } from 'react';
import { View, Text, Image, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import Icon from 'react-native-vector-icons/Feather';
import { useTheme, useThemedStyles } from '../../theme';
import type { ThemeColors, ThemeShadows } from '../../theme';
import { TYPOGRAPHY, SPACING } from '../../constants';
import { useProjectStore } from '../../stores';
import {
  CustomAlert,
  showAlert,
  hideAlert,
  AlertState,
  initialAlertState,
} from '../../components/CustomAlert';
import { importSillyTavernCardFromFile } from '../../services/importCharacterCardFile';
import { CharacterCardParseError } from '../../services/characterCardImport';
import { RootStackParamList } from '../../navigation/types';
import logger from '../../utils/logger';

type Nav = NativeStackNavigationProp<RootStackParamList>;

/**
 * Character-card manager (SillyTavern-style). Characters are `Project`s — the SAME entity the
 * per-chat character dropdown selects, so there is one source of truth. This section adds the two
 * things that were missing: importing a SillyTavern PNG, and editing a character's avatar + card
 * fields (via ProjectEdit). {char} in a character's prompt is that character's name.
 */
export const CharacterSection: React.FC = () => {
  const { colors } = useTheme();
  const styles = useThemedStyles(createStyles);
  const navigation = useNavigation<Nav>();
  const projects = useProjectStore(s => s.projects);
  const [alertState, setAlertState] = useState<AlertState>(initialAlertState);
  const [importing, setImporting] = useState(false);

  const handleImport = async () => {
    if (importing) return;
    setImporting(true);
    try {
      const created = await importSillyTavernCardFromFile();
      if (created) {
        setAlertState(
          showAlert('Character imported', `“${created.name}” was added from the card.`),
        );
      }
    } catch (e) {
      const msg =
        e instanceof CharacterCardParseError
          ? e.message
          : 'Could not import that file. Choose a SillyTavern character-card PNG.';
      logger.warn('[CharacterSection] import failed', e);
      setAlertState(showAlert('Import failed', msg));
    } finally {
      setImporting(false);
    }
  };

  return (
    <View style={styles.section}>
      <Text style={styles.desc}>
        Characters are who the AI plays ({'{char}'}). Import a SillyTavern PNG, or create one by
        hand. The active character is chosen per chat from the character selector.
      </Text>

      <View style={styles.actionRow}>
        <TouchableOpacity
          style={styles.actionBtn}
          onPress={() => navigation.navigate('ProjectEdit', {})}
          testID="character-new"
        >
          <Icon name="plus" size={16} color={colors.primary} />
          <Text style={styles.actionText}>New character</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.actionBtn}
          onPress={handleImport}
          disabled={importing}
          testID="character-import"
        >
          {importing ? (
            <ActivityIndicator size="small" color={colors.primary} />
          ) : (
            <Icon name="download" size={16} color={colors.primary} />
          )}
          <Text style={styles.actionText}>Import PNG card</Text>
        </TouchableOpacity>
      </View>

      {projects.map(p => (
        <TouchableOpacity
          key={p.id}
          style={styles.row}
          onPress={() => navigation.navigate('ProjectEdit', { projectId: p.id })}
          testID={`character-row-${p.id}`}
          activeOpacity={0.7}
        >
          <View style={[styles.avatar, { backgroundColor: p.icon ?? colors.surface }]}>
            {p.avatarUri ? (
              <Image source={{ uri: p.avatarUri }} style={styles.avatarImg} />
            ) : (
              <Icon name="user" size={18} color={colors.textMuted} />
            )}
          </View>
          <View style={styles.rowInfo}>
            <Text style={styles.rowName} numberOfLines={1}>
              {p.name}
            </Text>
            <Text style={styles.rowDesc} numberOfLines={1}>
              {p.description || 'No description'}
            </Text>
          </View>
          <Icon name="chevron-right" size={18} color={colors.textMuted} />
        </TouchableOpacity>
      ))}

      <CustomAlert {...alertState} onClose={() => setAlertState(hideAlert())} />
    </View>
  );
};

const createStyles = (colors: ThemeColors, _shadows: ThemeShadows) => ({
  section: { paddingHorizontal: SPACING.lg, paddingBottom: SPACING.md },
  desc: {
    ...TYPOGRAPHY.bodySmall,
    color: colors.textSecondary,
    marginBottom: SPACING.md,
  },
  actionRow: { flexDirection: 'row' as const, gap: SPACING.lg, marginBottom: SPACING.md },
  actionBtn: { flexDirection: 'row' as const, alignItems: 'center' as const, gap: 6 },
  actionText: { color: colors.primary, fontWeight: '600' as const, fontSize: 14 },
  row: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    paddingVertical: SPACING.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  avatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    marginRight: SPACING.md,
    overflow: 'hidden' as const,
  },
  avatarImg: { width: 40, height: 40, borderRadius: 20 },
  rowInfo: { flex: 1, marginRight: SPACING.sm },
  rowName: { ...TYPOGRAPHY.body, color: colors.text },
  rowDesc: { ...TYPOGRAPHY.bodySmall, color: colors.textMuted },
});
