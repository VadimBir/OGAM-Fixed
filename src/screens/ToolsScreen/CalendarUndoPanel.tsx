import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Alert } from 'react-native';
import Icon from 'react-native-vector-icons/Feather';
import { useTheme, useThemedStyles } from '../../theme';
import type { ThemeColors } from '../../theme';
import { useCalendarLogStore } from '../../stores/calendarLogStore';
import { removeAiCalendarEvents } from '../../services/tools/calendarTool';

const createStyles = (colors: ThemeColors) => ({
  box: { marginTop: 16, padding: 12, borderRadius: 10, backgroundColor: colors.surface },
  title: { color: colors.text, fontSize: 14, fontWeight: '600' as const },
  desc: { color: colors.textMuted, fontSize: 12, marginTop: 2 },
  actions: { flexDirection: 'row' as const, gap: 10, marginTop: 10 },
  btn: { flexDirection: 'row' as const, alignItems: 'center' as const, gap: 6, paddingVertical: 6, paddingHorizontal: 10, borderRadius: 8, backgroundColor: colors.surfaceLight },
  btnText: { color: colors.text, fontSize: 12 },
  danger: { color: colors.error },
  row: { flexDirection: 'row' as const, alignItems: 'center' as const, justifyContent: 'space-between' as const, marginTop: 8 },
  rowText: { color: colors.text, fontSize: 12, flex: 1, marginRight: 8 },
});

const MAX_ROWS = 20;

/**
 * Undo surface for add_calendar_event: every AI-created event is logged, so the user can remove one,
 * the last one, or ALL of them in one tap (a runaway model can't bury the calendar).
 */
export const CalendarUndoPanel: React.FC = () => {
  const styles = useThemedStyles(createStyles);
  const { colors } = useTheme();
  const entries = useCalendarLogStore(s => s.entries);
  const [busy, setBusy] = useState(false);
  if (entries.length === 0) return null;

  const run = async (ids: string[]) => {
    setBusy(true);
    try {
      const r = await removeAiCalendarEvents(ids);
      if (r.failed > 0) Alert.alert('Some events not removed', `${r.failed} event(s) could not be deleted.`);
    } catch (e: any) {
      Alert.alert('Calendar', e?.message ?? 'Could not remove events.');
    } finally {
      setBusy(false);
    }
  };
  const confirmAll = () =>
    Alert.alert('Delete all AI events?', `Removes all ${entries.length} calendar events the AI added.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete all', style: 'destructive', onPress: () => run(entries.map(e => e.eventId)) },
    ]);
  const recent = [...entries].reverse().slice(0, MAX_ROWS);

  return (
    <View style={styles.box} testID="calendar-undo-panel">
      <Text style={styles.title}>{`Calendar events added by AI (${entries.length})`}</Text>
      <Text style={styles.desc}>Undo anything the model put in your calendar.</Text>
      <View style={styles.actions}>
        <TouchableOpacity disabled={busy} style={styles.btn} onPress={() => run([entries[entries.length - 1].eventId])} testID="calendar-undo-last">
          <Icon name="rotate-ccw" size={13} color={colors.primary} />
          <Text style={styles.btnText}>Undo last</Text>
        </TouchableOpacity>
        <TouchableOpacity disabled={busy} style={styles.btn} onPress={confirmAll} testID="calendar-undo-all">
          <Icon name="trash-2" size={13} color={colors.error} />
          <Text style={[styles.btnText, styles.danger]}>Delete all</Text>
        </TouchableOpacity>
      </View>
      {recent.map(e => (
        <View key={e.eventId} style={styles.row}>
          <Text style={styles.rowText} numberOfLines={1}>{`${new Date(e.startDate).toLocaleString()} · ${e.title}`}</Text>
          <TouchableOpacity disabled={busy} onPress={() => run([e.eventId])} testID={`calendar-undo-${e.eventId}`}>
            <Icon name="x" size={16} color={colors.textMuted} />
          </TouchableOpacity>
        </View>
      ))}
    </View>
  );
};
