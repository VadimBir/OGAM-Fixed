import React, { useState } from 'react';
import { View, Text, Switch, TouchableOpacity, ScrollView, TextInput, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/Feather';
import IconMC from 'react-native-vector-icons/MaterialCommunityIcons';
import { useTheme, useThemedStyles } from '../../theme';
import { FONTS, TYPOGRAPHY, SPACING } from '../../constants';
import { AVAILABLE_TOOLS, getToolsAsOpenAISchema } from '../../services/tools';
import {
  MAX_SKILL_REPEAT,
  applyRepeat,
  clampRepeat,
  repeatContent,
  toolInstructionBlock,
  type RepeatMode,
} from '../../services/tools/registry';
import { useAppStore, useSkillStore } from '../../stores';
import { useOpenProTools } from '../../hooks/useOpenProTools';
import { CalendarUndoPanel } from './CalendarUndoPanel';
import type { ThemeColors, ThemeShadows } from '../../theme';

const TOOL_WARNING_COLOR = '#F59E0B';

/**
 * Tool / skill CARDS editor.
 *
 * Built-in tools are listed as cards: each can be enabled/disabled AND its model-facing
 * description edited — that description is literally the instruction the model reads on how/when
 * to use the tool (see services/tools/registry). Edits persist in `skillStore` and flow into the
 * generation pipeline immediately. Below, the user can create prompt-only "custom skills"
 * (guidance injected into the system prompt, never callable as a tool).
 */
export const ToolsScreen: React.FC = () => {
  const navigation = useNavigation<any>();
  const { colors } = useTheme();
  const styles = useThemedStyles(createStyles);
  const openProTools = useOpenProTools();

  const enabledTools = useAppStore(st => st.settings.enabledTools) || [];
  const updateSettings = useAppStore(st => st.updateSettings);
  const showProTools = useAppStore(st => st.settings.showProTools ?? true);
  const toolCountHintDismissed = useAppStore(st => st.toolCountHintDismissed);
  const setToolCountHintDismissed = useAppStore(st => st.setToolCountHintDismissed);

  // Skill-card store (editable tool descriptions + custom skills).
  const overrides = useSkillStore(s => s.overrides);
  const customSkills = useSkillStore(s => s.customSkills);
  const getToolDescription = useSkillStore(s => s.getToolDescription);
  const setToolOverride = useSkillStore(s => s.setToolOverride);
  const resetToolOverride = useSkillStore(s => s.resetToolOverride);
  const createCustomSkill = useSkillStore(s => s.createCustomSkill);
  const updateCustomSkill = useSkillStore(s => s.updateCustomSkill);
  const deleteCustomSkill = useSkillStore(s => s.deleteCustomSkill);

  // Which built-in tool's description is being edited, and the working draft.
  const [editingToolId, setEditingToolId] = useState<string | null>(null);
  const [draftDesc, setDraftDesc] = useState('');
  const [draftRepeat, setDraftRepeat] = useState(1);
  const [draftMode, setDraftMode] = useState<RepeatMode>('content');
  // Custom-skill editor: 'new' for a fresh card, an id for editing, null when closed.
  const [customEditId, setCustomEditId] = useState<string | 'new' | null>(null);
  const [customName, setCustomName] = useState('');
  const [customBody, setCustomBody] = useState('');
  const [customRepeat, setCustomRepeat] = useState(1);
  const [customMode, setCustomMode] = useState<RepeatMode>('content');

  const handleToggleTool = (toolId: string) => {
    const cur = useAppStore.getState().settings.enabledTools || [];
    updateSettings({
      enabledTools: cur.includes(toolId) ? cur.filter(id => id !== toolId) : [...cur, toolId],
    });
  };

  const openEditor = (toolId: string) => {
    setEditingToolId(toolId);
    setDraftDesc(getToolDescription(toolId));
    setDraftRepeat(clampRepeat(overrides[toolId]?.repeat));
    setDraftMode(overrides[toolId]?.repeatMode ?? 'content');
  };
  const saveEditor = () => {
    if (editingToolId) {
      setToolOverride(editingToolId, { description: draftDesc, repeat: draftRepeat, repeatMode: draftMode });
    }
    setEditingToolId(null);
  };

  // The literal JSON tool-call schema the model receives, with the live draft description applied,
  // so the user sees exactly how the model is told to call this tool.
  const schemaPreview = (toolId: string, desc: string): string => {
    const schema = getToolsAsOpenAISchema([toolId])[0];
    if (!schema) return '{}';
    const inSchema = draftMode === 'block' ? desc : repeatContent(desc, draftRepeat);
    const withDraft = { ...schema, function: { ...schema.function, description: inSchema } };
    const json = JSON.stringify(withDraft, null, 2);
    if (draftMode !== 'block') return json;
    const block = toolInstructionBlock(schema.function.name, desc);
    const blocks = Array.from({ length: clampRepeat(draftRepeat) }, () => block).join('\n');
    return `${json}\n\n+ system prompt (every engine):\n<tool_instructions>\n${blocks}\n</tool_instructions>`;
  };

  const openCustomEditor = (id: string | 'new') => {
    if (id === 'new') {
      setCustomName('');
      setCustomBody('');
      setCustomRepeat(1);
      setCustomMode('content');
    } else {
      const s = customSkills.find(c => c.id === id);
      setCustomName(s?.name ?? '');
      setCustomBody(s?.description ?? '');
      setCustomRepeat(clampRepeat(s?.repeat));
      setCustomMode(s?.repeatMode ?? 'content');
    }
    setCustomEditId(id);
  };
  const saveCustomEditor = () => {
    const name = customName.trim();
    const body = customBody.trim();
    if (!name || !body) {
      Alert.alert('Missing fields', 'A skill needs both a name and instructions.');
      return;
    }
    if (customEditId === 'new') {
      createCustomSkill({ name, description: body, enabled: true, repeat: customRepeat, repeatMode: customMode });
    } else if (customEditId) {
      updateCustomSkill(customEditId, { name, description: body, repeat: customRepeat, repeatMode: customMode });
    }
    setCustomEditId(null);
  };

  const showHint = enabledTools.length > 3 && !toolCountHintDismissed;

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backBtn}
          hitSlop={{ top: 8, right: 8, bottom: 8, left: 8 }}
          testID="tools-back"
        >
          <Icon name="arrow-left" size={20} color={colors.text} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Tools &amp; Skills</Text>
      </View>

      <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
        {/* Pro Tools sits on top of the listing unless the user hid it (toggle at the bottom). */}
        {showProTools && <TouchableOpacity
          style={styles.proToolsButton}
          onPress={openProTools}
          activeOpacity={0.75}
          testID="tools-pro-tools"
        >
          <View style={styles.proToolsIcon}>
            <IconMC name="crown" size={20} color={colors.primary} />
          </View>
          <View style={styles.toolInfo}>
            <Text style={styles.toolName}>Pro Tools</Text>
            <Text style={styles.toolDescription}>Email, calendar and MCP servers</Text>
          </View>
          <Icon name="chevron-right" size={18} color={colors.textMuted} />
        </TouchableOpacity>}

        {showHint && (
          <View style={[styles.hintBanner, { backgroundColor: colors.surface }]}>
            <Icon name="alert-circle" size={16} color={TOOL_WARNING_COLOR} style={styles.hintIcon} />
            <View style={styles.hintBody}>
              <Text style={[styles.hintText, { color: colors.text }]}>
                Too many tools can confuse the model and increase latency on the first response. Stick to 2-3 tools for best results.
              </Text>
              <TouchableOpacity onPress={setToolCountHintDismissed} style={styles.hintDismiss}>
                <Text style={styles.hintDismissText}>Got it</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {AVAILABLE_TOOLS.map(tool => {
          const isEnabled = enabledTools.includes(tool.id);
          const isEditing = editingToolId === tool.id;
          const isCustomized = !!overrides[tool.id];
          const toolRepeat = clampRepeat(overrides[tool.id]?.repeat);
          const desc = getToolDescription(tool.id);
          return (
            <View key={tool.id} style={styles.toolCard} testID={`tool-picker-row-${tool.id}`}>
              <View style={styles.toolRow}>
                <View style={styles.toolIcon}>
                  <Icon name={tool.icon} size={20} color={isEnabled ? colors.primary : colors.textMuted} />
                </View>
                <View style={styles.toolInfo}>
                  <View style={styles.toolNameRow}>
                    <Text style={styles.toolName} testID={`tool-picker-name-${tool.id}`}>{tool.displayName}</Text>
                    {tool.requiresNetwork && (
                      <Icon name="wifi" size={12} color={colors.textMuted} style={styles.networkIcon} />
                    )}
                    {isCustomized && (
                      <Text style={styles.editedBadge}>edited</Text>
                    )}
                    {toolRepeat > 1 && (
                      <Text style={styles.editedBadge}>×{toolRepeat}</Text>
                    )}
                  </View>
                  <Text style={styles.toolDescription}>{desc}</Text>
                </View>
                <Switch
                  value={isEnabled}
                  onValueChange={() => handleToggleTool(tool.id)}
                  trackColor={{ false: colors.border, true: `${colors.primary}80` }}
                  thumbColor={isEnabled ? colors.primary : colors.textMuted}
                  accessibilityLabel={`${tool.displayName}, ${isEnabled ? 'ON' : 'OFF'}`}
                  testID={`tool-picker-toggle-${tool.id}`}
                />
              </View>

              {!isEditing ? (
                <View style={styles.rowActions}>
                  <TouchableOpacity
                    onPress={() => openEditor(tool.id)}
                    style={styles.linkBtn}
                    testID={`tool-edit-${tool.id}`}
                  >
                    <Icon name="edit-2" size={13} color={colors.primary} />
                    <Text style={styles.linkBtnText}>Edit instructions</Text>
                  </TouchableOpacity>
                  {isCustomized && (
                    <TouchableOpacity
                      onPress={() => resetToolOverride(tool.id)}
                      style={styles.linkBtn}
                      testID={`tool-reset-${tool.id}`}
                    >
                      <Icon name="rotate-ccw" size={13} color={colors.textMuted} />
                      <Text style={[styles.linkBtnText, { color: colors.textMuted }]}>Reset</Text>
                    </TouchableOpacity>
                  )}
                </View>
              ) : (
                <View style={styles.editorBox}>
                  <Text style={styles.editorLabel}>
                    How the model is told to use this tool (edit exactly as you want it). This text
                    becomes the tool's `description` in the JSON the model receives below:
                  </Text>
                  <TextInput
                    style={styles.editorInput}
                    value={draftDesc}
                    onChangeText={setDraftDesc}
                    multiline
                    placeholder="Instruction the model reads for this tool…"
                    placeholderTextColor={colors.textMuted}
                    testID={`tool-edit-input-${tool.id}`}
                  />
                  <RepeatControl count={draftRepeat} setCount={setDraftRepeat} mode={draftMode} setMode={setDraftMode} testID={`tool-${tool.id}`} isTool styles={styles} />
                  <Text style={[styles.editorLabel, styles.sectionGap]}>
                    Exact tool-call schema sent to the model (name + parameters), with your edited
                    instructions applied:
                  </Text>
                  <Text style={styles.jsonPreview} selectable testID={`tool-json-${tool.id}`}>
                    {schemaPreview(tool.id, draftDesc)}
                  </Text>
                  <View style={styles.editorActions}>
                    <TouchableOpacity onPress={() => setEditingToolId(null)} style={styles.btnGhost}>
                      <Text style={styles.btnGhostText}>Cancel</Text>
                    </TouchableOpacity>
                    <TouchableOpacity onPress={saveEditor} style={styles.btnPrimary} testID={`tool-edit-save-${tool.id}`}>
                      <Text style={styles.btnPrimaryText}>Save</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              )}
            </View>
          );
        })}

        {/* ---- Custom skills (prompt-only guidance) ---- */}
        <View style={styles.sectionHeaderRow}>
          <Text style={styles.sectionHeader}>Custom skills</Text>
          <TouchableOpacity onPress={() => openCustomEditor('new')} style={styles.addBtn} testID="custom-skill-add">
            <Icon name="plus" size={16} color={colors.primary} />
            <Text style={styles.addBtnText}>Add</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.sectionSub}>
          Prompt-only instructions added to the model's system prompt. Not callable tools.
        </Text>

        {customSkills.map(skill => {
          const isEditing = customEditId === skill.id;
          if (isEditing) return renderCustomEditor(skill.id);
          return (
            <View key={skill.id} style={styles.toolCard} testID={`custom-skill-${skill.id}`}>
              <View style={styles.toolRow}>
                <View style={styles.toolIcon}>
                  <Icon name="feather" size={20} color={skill.enabled ? colors.primary : colors.textMuted} />
                </View>
                <View style={styles.toolInfo}>
                  <View style={styles.toolNameRow}>
                    <Text style={styles.toolName}>{skill.name}</Text>
                    {clampRepeat(skill.repeat) > 1 && (
                      <Text style={styles.editedBadge}>×{clampRepeat(skill.repeat)}</Text>
                    )}
                  </View>
                  <Text style={styles.toolDescription} numberOfLines={3}>{skill.description}</Text>
                </View>
                <Switch
                  value={skill.enabled}
                  onValueChange={v => updateCustomSkill(skill.id, { enabled: v })}
                  trackColor={{ false: colors.border, true: `${colors.primary}80` }}
                  thumbColor={skill.enabled ? colors.primary : colors.textMuted}
                />
              </View>
              <View style={styles.rowActions}>
                <TouchableOpacity onPress={() => openCustomEditor(skill.id)} style={styles.linkBtn}>
                  <Icon name="edit-2" size={13} color={colors.primary} />
                  <Text style={styles.linkBtnText}>Edit</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => deleteCustomSkill(skill.id)} style={styles.linkBtn}>
                  <Icon name="trash-2" size={13} color={TOOL_WARNING_COLOR} />
                  <Text style={[styles.linkBtnText, { color: TOOL_WARNING_COLOR }]}>Delete</Text>
                </TouchableOpacity>
              </View>
            </View>
          );
        })}
        {customEditId === 'new' && renderCustomEditor('new')}

        <CalendarUndoPanel />

        <View style={[styles.toolRow, styles.sectionGap]}>
          <View style={styles.toolInfo}>
            <Text style={styles.toolName}>Show Pro Tools</Text>
            <Text style={styles.toolDescription}>Pro Tools entry here and in chat quick settings.</Text>
          </View>
          <Switch
            testID="tools-show-pro-toggle"
            value={showProTools}
            onValueChange={v => updateSettings({ showProTools: v })}
            trackColor={{ false: colors.border, true: `${colors.primary}80` }}
            thumbColor={showProTools ? colors.primary : colors.textMuted}
          />
        </View>

        <Text style={styles.hint}>
          Enabling more tools can confuse the model and increases latency on first response.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );

  function renderCustomEditor(id: string | 'new') {
    return (
      <View key={`editor-${id}`} style={[styles.toolCard, styles.editorBox]} testID={`custom-skill-editor-${id}`}>
        <Text style={styles.editorLabel}>Skill name</Text>
        <TextInput
          style={styles.nameInput}
          value={customName}
          onChangeText={setCustomName}
          placeholder="e.g. Roleplay narrator"
          placeholderTextColor={colors.textMuted}
          testID="custom-skill-name"
        />
        <Text style={styles.editorLabel}>Instructions (added to the system prompt verbatim)</Text>
        <TextInput
          style={styles.editorInput}
          value={customBody}
          onChangeText={setCustomBody}
          multiline
          placeholder="Always narrate in third person, present tense…"
          placeholderTextColor={colors.textMuted}
          testID="custom-skill-body"
        />
        <RepeatControl count={customRepeat} setCount={setCustomRepeat} mode={customMode} setMode={setCustomMode} testID="custom-skill" styles={styles} />
        <Text style={[styles.editorLabel, styles.sectionGap]}>Injected into the system prompt:</Text>
        <Text style={styles.jsonPreview} selectable testID="custom-skill-preview">
          {applyRepeat(
            customBody.trim(),
            d => `<skill name="${customName.trim().replaceAll('"', "'")}">\n${d}\n</skill>`,
            { repeat: customRepeat, repeatMode: customMode },
          )}
        </Text>
        <View style={styles.editorActions}>
          <TouchableOpacity onPress={() => setCustomEditId(null)} style={styles.btnGhost}>
            <Text style={styles.btnGhostText}>Cancel</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={saveCustomEditor} style={styles.btnPrimary} testID="custom-skill-save">
            <Text style={styles.btnPrimaryText}>Save</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }
};

type ToolsStyles = ReturnType<typeof createStyles>;

const REPEAT_MODES: { id: RepeatMode; label: string }[] = [
  { id: 'content', label: 'Repeat text in block' },
  { id: 'block', label: 'Repeat whole block' },
];

/** Emphasis multiplier: count (1..MAX) + mode (repeat the text inside one block / repeat the block). */
const RepeatControl: React.FC<{
  count: number;
  setCount: (n: number) => void;
  mode: RepeatMode;
  setMode: (m: RepeatMode) => void;
  testID: string;
  isTool?: boolean;
  styles: ToolsStyles;
}> = ({ count, setCount, mode, setMode, testID, isTool, styles }) => {
  const { colors } = useTheme();
  return (
    <View style={styles.repeatBox}>
      <View style={styles.repeatRow}>
        <Text style={styles.editorLabel}>Emphasis multiplier</Text>
        <View style={styles.repeatStepper}>
          <TouchableOpacity onPress={() => setCount(clampRepeat(count - 1))} testID={`${testID}-repeat-dec`} style={styles.stepBtn}>
            <Icon name="minus" size={14} color={colors.primary} />
          </TouchableOpacity>
          <Text style={styles.repeatCount} testID={`${testID}-repeat-count`}>×{count}</Text>
          <TouchableOpacity onPress={() => setCount(clampRepeat(count + 1))} testID={`${testID}-repeat-inc`} style={styles.stepBtn}>
            <Icon name="plus" size={14} color={colors.primary} />
          </TouchableOpacity>
        </View>
      </View>
      <View style={styles.repeatRow}>
        {REPEAT_MODES.map(m => (
          <TouchableOpacity
            key={m.id}
            onPress={() => setMode(m.id)}
            style={[styles.modeChip, mode === m.id && styles.modeChipActive]}
            testID={`${testID}-repeat-mode-${m.id}`}
          >
            <Text style={[styles.modeChipText, mode === m.id && styles.modeChipTextActive]}>{m.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
      <Text style={styles.editorLabel}>
        {`Sends this instruction ${count}× (max ${MAX_SKILL_REPEAT}). `}
        {isTool
          ? 'Text: repeats the description inside the tool definition. Whole block: definition once + N separate <tool> blocks in the system prompt (every engine).'
          : 'Each copy costs context tokens.'}
      </Text>
    </View>
  );
};

const createStyles = (colors: ThemeColors, _shadows: ThemeShadows) => ({
  root: { flex: 1, backgroundColor: colors.background },
  header: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    paddingHorizontal: SPACING.lg,
    paddingVertical: SPACING.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  backBtn: { marginRight: SPACING.md },
  headerTitle: { ...TYPOGRAPHY.h2, fontSize: 18, color: colors.text, flex: 1 },
  container: { flex: 1 },
  contentContainer: { paddingHorizontal: SPACING.lg, paddingBottom: SPACING.xl },
  proToolsButton: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  proToolsIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: `${colors.primary}20`,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    marginRight: 12,
  },
  toolCard: {
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    paddingBottom: 10,
  },
  toolRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    paddingVertical: 14,
  },
  toolIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.surface,
    alignItems: 'center' as const,
    justifyContent: 'center' as const,
    marginRight: 12,
  },
  toolInfo: { flex: 1, marginRight: 12 },
  toolNameRow: { flexDirection: 'row' as const, alignItems: 'center' as const },
  toolName: {
    fontSize: 15,
    fontFamily: FONTS.mono,
    fontWeight: '400' as const,
    color: colors.text,
  },
  networkIcon: { marginLeft: 6 },
  editedBadge: {
    marginLeft: 8,
    fontSize: 10,
    fontFamily: FONTS.mono,
    color: colors.primary,
    borderWidth: 1,
    borderColor: colors.primary,
    borderRadius: 4,
    paddingHorizontal: 4,
    paddingVertical: 1,
  },
  toolDescription: {
    fontSize: 12,
    fontFamily: FONTS.mono,
    color: colors.textMuted,
    marginTop: 2,
  },
  rowActions: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    marginLeft: 52,
    gap: SPACING.lg,
    marginBottom: 4,
  },
  sectionGap: { marginTop: 12 },
  repeatBox: { marginTop: 12, gap: 6 },
  repeatRow: { flexDirection: 'row' as const, alignItems: 'center' as const, justifyContent: 'space-between' as const, gap: 6 },
  repeatStepper: { flexDirection: 'row' as const, alignItems: 'center' as const, gap: 10 },
  stepBtn: { padding: 6, borderRadius: 6, borderWidth: 1, borderColor: colors.primary },
  repeatCount: { fontSize: 14, fontFamily: FONTS.mono, color: colors.text, minWidth: 28, textAlign: 'center' as const },
  modeChip: { flex: 1, paddingVertical: 6, borderRadius: 14, backgroundColor: colors.background, alignItems: 'center' as const },
  modeChipActive: { backgroundColor: colors.primary },
  modeChipText: { fontSize: 12, color: colors.text },
  modeChipTextActive: { color: colors.background, fontWeight: '600' as const },
  linkBtn: { flexDirection: 'row' as const, alignItems: 'center' as const, gap: 4 },
  linkBtnText: { fontSize: 12, color: colors.primary, fontWeight: '500' as const },
  editorBox: {
    marginLeft: 0,
    marginTop: 4,
    backgroundColor: colors.surface,
    borderRadius: 10,
    padding: SPACING.md,
  },
  editorLabel: { fontSize: 12, color: colors.textMuted, marginBottom: 6, marginTop: 4 },
  jsonPreview: {
    fontFamily: FONTS.mono,
    fontSize: 11,
    color: colors.text,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
    padding: 10,
    marginTop: 4,
  },
  editorInput: {
    minHeight: 90,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
    padding: 10,
    color: colors.text,
    fontSize: 13,
    fontFamily: FONTS.mono,
    textAlignVertical: 'top' as const,
  },
  nameInput: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
    padding: 10,
    color: colors.text,
    fontSize: 14,
    marginBottom: 4,
  },
  editorActions: {
    flexDirection: 'row' as const,
    justifyContent: 'flex-end' as const,
    gap: SPACING.md,
    marginTop: 10,
  },
  btnGhost: { paddingHorizontal: 14, paddingVertical: 8 },
  btnGhostText: { color: colors.textMuted, fontWeight: '500' as const },
  btnPrimary: {
    backgroundColor: colors.primary,
    borderRadius: 8,
    paddingHorizontal: 18,
    paddingVertical: 8,
  },
  btnPrimaryText: { color: '#fff', fontWeight: '600' as const },
  sectionHeaderRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    justifyContent: 'space-between' as const,
    marginTop: SPACING.xl,
  },
  sectionHeader: { ...TYPOGRAPHY.h2, fontSize: 16, color: colors.text },
  addBtn: { flexDirection: 'row' as const, alignItems: 'center' as const, gap: 4 },
  addBtnText: { color: colors.primary, fontWeight: '600' as const },
  sectionSub: { fontSize: 12, color: colors.textMuted, marginTop: 4, marginBottom: SPACING.sm },
  hint: {
    ...TYPOGRAPHY.meta,
    color: colors.textMuted,
    marginTop: SPACING.lg,
    textAlign: 'center' as const,
  },
  hintBanner: {
    flexDirection: 'row' as const,
    alignItems: 'flex-start' as const,
    borderWidth: 1,
    borderColor: TOOL_WARNING_COLOR,
    borderRadius: 10,
    padding: SPACING.md,
    marginTop: SPACING.sm,
  },
  hintIcon: { marginRight: SPACING.sm, marginTop: 1 },
  hintBody: { flex: 1 },
  hintText: { ...TYPOGRAPHY.bodySmall, lineHeight: 18 },
  hintDismiss: { marginTop: SPACING.sm },
  hintDismissText: {
    ...TYPOGRAPHY.bodySmall,
    fontWeight: '400' as const,
    color: TOOL_WARNING_COLOR,
  },
});
