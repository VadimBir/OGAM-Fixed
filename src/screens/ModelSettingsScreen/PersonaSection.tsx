import React, { useState } from 'react';
import { View, Text, Image, TextInput, TouchableOpacity, ActivityIndicator } from 'react-native';
import Icon from 'react-native-vector-icons/Feather';
import { useTheme, useThemedStyles } from '../../theme';
import type { ThemeColors, ThemeShadows } from '../../theme';
import { TYPOGRAPHY, SPACING } from '../../constants';
import { useAppStore } from '../../stores';
import {
  CustomAlert,
  showAlert,
  hideAlert,
  AlertState,
  initialAlertState,
} from '../../components/CustomAlert';
import { pickAvatarImage } from '../../utils/pickAvatarImage';
import { pickAndParseSillyTavernCard } from '../../services/importCharacterCardFile';
import { CharacterCardParseError } from '../../services/characterCardImport';
import logger from '../../utils/logger';

/**
 * User-PERSONA editor. The persona is who the USER is ({user}) — distinct from the character (the
 * AI, {char}). Stored on AppSettings (personaName / personaPrompt / personaAvatarUri) and folded
 * into the system prompt at send time by personaComposition. Same editing mechanics as a character:
 * name, description/prompt, avatar, and SillyTavern-PNG import.
 */
export const PersonaSection: React.FC = () => {
  const { colors } = useTheme();
  const styles = useThemedStyles(createStyles);
  const personaName = useAppStore(s => s.settings.personaName);
  const personaPrompt = useAppStore(s => s.settings.personaPrompt);
  const personaAvatarUri = useAppStore(s => s.settings.personaAvatarUri);
  const updateSettings = useAppStore(s => s.updateSettings);
  const [alertState, setAlertState] = useState<AlertState>(initialAlertState);
  const [importing, setImporting] = useState(false);

  const handlePickAvatar = async () => {
    try {
      const uri = await pickAvatarImage('persona_avatars');
      if (uri) updateSettings({ personaAvatarUri: uri });
    } catch (e) {
      logger.warn('[PersonaSection] avatar pick failed', e);
      setAlertState(showAlert('Image error', 'Could not load that image. Try a PNG or JPG.'));
    }
  };

  const handleImport = async () => {
    if (importing) return;
    setImporting(true);
    try {
      const result = await pickAndParseSillyTavernCard();
      if (result) {
        const { parsed, avatarUri } = result;
        const body = [parsed.description, parsed.personality]
          .filter(Boolean)
          .join('\n\n');
        updateSettings({
          personaName: parsed.name,
          personaPrompt: body || personaPrompt,
          personaAvatarUri: avatarUri ?? personaAvatarUri,
        });
        setAlertState(showAlert('Persona imported', `Loaded “${parsed.name}” as your persona.`));
      }
    } catch (e) {
      const msg =
        e instanceof CharacterCardParseError
          ? e.message
          : 'Could not import that file. Choose a SillyTavern character-card PNG.';
      logger.warn('[PersonaSection] import failed', e);
      setAlertState(showAlert('Import failed', msg));
    } finally {
      setImporting(false);
    }
  };

  return (
    <View style={styles.section}>
      <Text style={styles.desc}>
        Your persona is who YOU are in the chat ({'{user}'}). The AI stays in character while
        treating this as the person it is talking to.
      </Text>

      <View style={styles.avatarRow}>
        <TouchableOpacity
          style={styles.avatarCircle}
          onPress={handlePickAvatar}
          activeOpacity={0.8}
          testID="persona-avatar"
        >
          {personaAvatarUri ? (
            <Image source={{ uri: personaAvatarUri }} style={styles.avatarImg} />
          ) : (
            <Icon name="user" size={26} color={colors.textMuted} />
          )}
          <View style={styles.avatarBadge}>
            <Icon name="camera" size={11} color="#fff" />
          </View>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.importBtn}
          onPress={handleImport}
          disabled={importing}
          testID="persona-import"
        >
          {importing ? (
            <ActivityIndicator size="small" color={colors.primary} />
          ) : (
            <Icon name="download" size={15} color={colors.primary} />
          )}
          <Text style={styles.importText}>Import PNG card</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.label}>Your name ({'{user}'})</Text>
      <TextInput
        style={styles.input}
        value={personaName ?? ''}
        onChangeText={t => updateSettings({ personaName: t })}
        placeholder="e.g. Alex"
        placeholderTextColor={colors.textMuted}
        testID="persona-name"
      />

      <Text style={styles.label}>About you</Text>
      <Text style={styles.hint}>
        Folded into the system prompt as who the user is. Supports {'{char}'}/{'{user}'}.
      </Text>
      <TextInput
        style={[styles.input, styles.textArea]}
        value={personaPrompt ?? ''}
        onChangeText={t => updateSettings({ personaPrompt: t })}
        placeholder="{user} is a 30-year-old software engineer who prefers concise answers..."
        placeholderTextColor={colors.textMuted}
        multiline
        textAlignVertical="top"
        testID="persona-prompt"
      />

      <CustomAlert {...alertState} onClose={() => setAlertState(hideAlert())} />
    </View>
  );
};

const createStyles = (colors: ThemeColors, _shadows: ThemeShadows) => ({
  section: { paddingHorizontal: SPACING.lg, paddingBottom: SPACING.md },
  desc: { ...TYPOGRAPHY.bodySmall, color: colors.textSecondary, marginBottom: SPACING.md },
  avatarRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: SPACING.lg,
    marginBottom: SPACING.sm,
  },
  avatarCircle: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.surface,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    borderWidth: 1,
    borderColor: colors.border,
  },
  avatarImg: { width: 56, height: 56, borderRadius: 28 },
  avatarBadge: {
    position: 'absolute' as const,
    right: -2,
    bottom: -2,
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: colors.primary,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    borderWidth: 2,
    borderColor: colors.background,
  },
  importBtn: { flexDirection: 'row' as const, alignItems: 'center' as const, gap: 6 },
  importText: { color: colors.primary, fontWeight: '600' as const, fontSize: 14 },
  label: {
    ...TYPOGRAPHY.label,
    color: colors.text,
    marginTop: SPACING.md,
    marginBottom: SPACING.sm,
    textTransform: 'uppercase' as const,
  },
  hint: { ...TYPOGRAPHY.bodySmall, color: colors.textSecondary, marginBottom: SPACING.sm },
  input: {
    ...TYPOGRAPHY.body,
    backgroundColor: colors.surface,
    borderRadius: 8,
    padding: SPACING.md,
    color: colors.text,
  },
  textArea: { minHeight: 120, maxHeight: 220, textAlignVertical: 'top' as const },
});
