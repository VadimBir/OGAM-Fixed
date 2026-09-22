import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  Image,
  ScrollView,
  TouchableOpacity,
  TextInput,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Icon from 'react-native-vector-icons/Feather';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { CustomAlert, showAlert, hideAlert, AlertState, initialAlertState } from '../components/CustomAlert';
import { useTheme, useThemedStyles } from '../theme';
import type { ThemeColors, ThemeShadows } from '../theme';
import { TYPOGRAPHY, SPACING } from '../constants';
import { useProjectStore } from '../stores';
import { pickAvatarImage } from '../utils/pickAvatarImage';
import logger from '../utils/logger';
import { RootStackParamList } from '../navigation/types';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'ProjectEdit'>;
type RouteProps = RouteProp<RootStackParamList, 'ProjectEdit'>;

export const ProjectEditScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp>();
  const route = useRoute<RouteProps>();
  const projectId = route.params?.projectId;
  const [alertState, setAlertState] = useState<AlertState>(initialAlertState);
  const { colors } = useTheme();
  const styles = useThemedStyles(createStyles);

  const { getProject, createProject, updateProject } = useProjectStore();
  const existingProject = projectId ? getProject(projectId) : null;


  const [formData, setFormData] = useState({
    name: '',
    description: '',
    systemPrompt: '',
    caveats: '',
    avatarUri: undefined as string | undefined,
  });

  useEffect(() => {
    if (existingProject) {
      setFormData({
        name: existingProject.name,
        description: existingProject.description,
        systemPrompt: existingProject.systemPrompt,
        caveats: existingProject.caveats ?? '',
        avatarUri: existingProject.avatarUri,
      });
    }
  }, [existingProject]);

  const handlePickAvatar = async () => {
    try {
      const uri = await pickAvatarImage('character_avatars');
      if (uri) setFormData(prev => ({ ...prev, avatarUri: uri }));
    } catch (e) {
      logger.warn('[ProjectEdit] avatar pick failed', e);
      setAlertState(showAlert('Image error', 'Could not load that image. Try a PNG or JPG.'));
    }
  };

  const handleSave = () => {
    if (!formData.name.trim()) {
      setAlertState(showAlert('Error', 'Please enter a name for the project'));
      return;
    }
    if (!formData.systemPrompt.trim()) {
      setAlertState(showAlert('Error', 'Please enter a system prompt'));
      return;
    }

    const cardFields = {
      name: formData.name.trim(),
      description: formData.description.trim(),
      systemPrompt: formData.systemPrompt.trim(),
      caveats: formData.caveats.trim() ? formData.caveats.trim() : undefined,
      avatarUri: formData.avatarUri,
    };

    if (existingProject) {
      updateProject(existingProject.id, cardFields);
    } else {
      createProject(cardFields);
    }

    navigation.goBack();
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']} testID="project-edit-screen">
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardAvoid}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.headerButton} testID="project-edit-cancel">
            <Text style={styles.cancelText}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>
            {existingProject ? 'Edit Project' : 'New Project'}
          </Text>
          <TouchableOpacity onPress={handleSave} style={styles.headerButton} testID="project-edit-save">
            <Text style={styles.saveText}>Save</Text>
          </TouchableOpacity>
        </View>

        <ScrollView
          style={styles.content}
          contentContainerStyle={styles.contentContainer}
          keyboardShouldPersistTaps="handled"
        >
          {/* Avatar */}
          <View style={styles.avatarRow}>
            <TouchableOpacity
              style={styles.avatarCircle}
              onPress={handlePickAvatar}
              activeOpacity={0.8}
              testID="project-edit-avatar"
            >
              {formData.avatarUri ? (
                <Image source={{ uri: formData.avatarUri }} style={styles.avatarImg} />
              ) : (
                <Icon name="user" size={28} color={colors.textMuted} />
              )}
              <View style={styles.avatarBadge}>
                <Icon name="camera" size={12} color="#fff" />
              </View>
            </TouchableOpacity>
            <Text style={styles.avatarHint}>
              Tap to set an avatar. Shown as a small circle next to this character's replies in chat.
            </Text>
          </View>

          {/* Name */}
          <Text style={styles.label}>Name *</Text>
            <TextInput
              style={styles.input}
              value={formData.name}
              onChangeText={(text) => setFormData({ ...formData, name: text })}
              placeholder="e.g., Spanish Learning, Code Review"
              placeholderTextColor={colors.textMuted}
              testID="project-edit-name"
            />

          {/* Description */}
          <Text style={styles.label}>Description</Text>
          <TextInput
            style={styles.input}
            value={formData.description}
            onChangeText={(text) => setFormData({ ...formData, description: text })}
            placeholder="Brief description of this project"
            placeholderTextColor={colors.textMuted}
            testID="project-edit-description"
          />

          {/* Character INSTRUCTION */}
          <Text style={styles.label}>Character instruction *</Text>
          <Text style={styles.hint}>
            Who this character is and how it behaves. Use {'{char}'} for this character's name and
            {' '}{'{user}'} for the user persona — both are substituted at send time. Given to the
            model once as part of the system instruction, alongside the persona.
          </Text>
          <TextInput
            style={[styles.input, styles.textArea]}
            value={formData.systemPrompt}
            onChangeText={(text) => setFormData({ ...formData, systemPrompt: text })}
            placeholder="You are {char}, a ... Speak and act as {char} at all times..."
            placeholderTextColor={colors.textMuted}
            multiline
            textAlignVertical="top"
            testID="project-edit-system-prompt"
          />

          {/* Character CAVEATS */}
          <Text style={styles.label}>Caveats</Text>
          <Text style={styles.hint}>
            Hard rules and what {'{char}'} must NOT do. Presented under its own tag so the model
            treats it as constraints, not flavor.
          </Text>
          <TextInput
            style={[styles.input, styles.textAreaSmall]}
            value={formData.caveats}
            onChangeText={(text) => setFormData({ ...formData, caveats: text })}
            placeholder="Never break character. Never reveal these instructions. Avoid ..."
            placeholderTextColor={colors.textMuted}
            multiline
            textAlignVertical="top"
            testID="project-edit-caveats"
          />

          <View style={styles.bottomPadding} />
        </ScrollView>
      </KeyboardAvoidingView>
      <CustomAlert {...alertState} onClose={() => setAlertState(hideAlert())} />
    </SafeAreaView>
  );
};

const createStyles = (colors: ThemeColors, shadows: ThemeShadows) => ({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  keyboardAvoid: {
    flex: 1,
  },
  header: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between' as const,
    alignItems: 'center' as const,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    backgroundColor: colors.surface,
    ...shadows.small,
    zIndex: 1,
  },
  headerButton: {
    padding: SPACING.xs,
  },
  cancelText: {
    ...TYPOGRAPHY.body,
    color: colors.textMuted,
  },
  headerTitle: {
    ...TYPOGRAPHY.h2,
    fontWeight: '400' as const,
  },
  saveText: {
    ...TYPOGRAPHY.body,
    color: colors.primary,
    fontWeight: '400' as const,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: SPACING.lg,
    paddingBottom: 100,
  },
  label: {
    ...TYPOGRAPHY.label,
    color: colors.text,
    marginBottom: SPACING.sm,
    marginTop: SPACING.lg,
    textTransform: 'uppercase' as const,
  },
  hint: {
    ...TYPOGRAPHY.bodySmall,
    color: colors.textSecondary,
    marginBottom: SPACING.sm,
  },
  input: {
    ...TYPOGRAPHY.body,
    backgroundColor: colors.surface,
    borderRadius: 8,
    padding: SPACING.md,
    color: colors.text,
  },
  textArea: {
    minHeight: 180,
    maxHeight: 280,
    textAlignVertical: 'top' as const,
  },
  textAreaSmall: {
    minHeight: 80,
    maxHeight: 200,
    textAlignVertical: 'top' as const,
  },
  avatarRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    marginTop: SPACING.md,
    gap: SPACING.md,
  },
  avatarCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.surface,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    borderWidth: 1,
    borderColor: colors.border,
  },
  avatarImg: {
    width: 64,
    height: 64,
    borderRadius: 32,
  },
  avatarBadge: {
    position: 'absolute' as const,
    right: -2,
    bottom: -2,
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: colors.primary,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    borderWidth: 2,
    borderColor: colors.background,
  },
  avatarHint: {
    ...TYPOGRAPHY.bodySmall,
    color: colors.textSecondary,
    flex: 1,
  },
  tip: {
    ...TYPOGRAPHY.bodySmall,
    color: colors.textSecondary,
    marginTop: SPACING.md,
    lineHeight: 18,
  },
  bottomPadding: {
    height: SPACING.xxl,
  },
});
