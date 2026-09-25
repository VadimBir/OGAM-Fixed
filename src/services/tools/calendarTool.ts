/**
 * Free, built-in calendar write tool (add_calendar_event) on react-native-calendar-events.
 * Every created event is written to useCalendarLogStore so the user can undo one, the last one,
 * or all AI-created events (Tools → "Calendar events added by AI").
 */
import { Platform } from 'react-native';
import RNCalendarEvents from 'react-native-calendar-events';
import type { ToolCall } from './types';
import { useCalendarLogStore, type AiCalendarEntry } from '../../stores/calendarLogStore';

const HOUR_MS = 60 * 60 * 1000;

/** "2026-09-26T14:30" (no offset) is LOCAL time per ECMAScript; a date-only value is all-day. */
export function parseLocalDateTime(value: unknown): Date | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  const v = value.trim().replace(' ', 'T');
  const d = /^\d{4}-\d{2}-\d{2}$/.test(v) ? new Date(`${v}T00:00:00`) : new Date(v);
  return Number.isNaN(d.getTime()) ? null : d;
}

async function ensureCalendarPermission(): Promise<void> {
  let status = await RNCalendarEvents.checkPermissions(false);
  if (status !== 'authorized') status = await RNCalendarEvents.requestPermissions(false);
  if (status !== 'authorized') {
    throw new Error('Calendar permission was not granted. Ask the user to allow calendar access in Android settings.');
  }
}

export async function handleAddCalendarEvent(call: ToolCall): Promise<string> {
  const a = call.arguments ?? {};
  const title = typeof a.title === 'string' ? a.title.trim() : '';
  if (!title) throw new Error('Missing required parameter: title');
  const start = parseLocalDateTime(a.start);
  if (!start) throw new Error('Missing or invalid start: use local ISO date-time like 2026-09-26T14:30');
  const allDay = a.all_day === true || a.all_day === 'true' || /^\d{4}-\d{2}-\d{2}$/.test(String(a.start).trim());
  const parsedEnd = parseLocalDateTime(a.end);
  const end = parsedEnd && parsedEnd > start ? parsedEnd : new Date(start.getTime() + (allDay ? 24 * HOUR_MS : HOUR_MS));
  const reminder = Number(a.reminder_minutes);
  const hasReminder = Number.isFinite(reminder) && reminder >= 0 && reminder <= 40320;

  await ensureCalendarPermission();
  const eventId = await RNCalendarEvents.saveEvent(title, {
    startDate: start.toISOString(),
    endDate: end.toISOString(),
    allDay,
    ...(typeof a.location === 'string' && a.location.trim() ? { location: a.location.trim() } : {}),
    notes: `${typeof a.notes === 'string' ? `${a.notes.trim()}\n\n` : ''}Added by Off Grid AI`,
    // Android stores Reminders.MINUTES (before start) as given; iOS takes a negative relative offset.
    ...(hasReminder ? { alarms: [{ date: Platform.OS === 'ios' ? -reminder : reminder }] } : {}),
  });
  const entry: AiCalendarEntry = {
    eventId: String(eventId),
    title,
    startDate: start.toISOString(),
    endDate: end.toISOString(),
    createdAt: new Date().toISOString(),
    conversationId: call.context?.conversationId,
  };
  useCalendarLogStore.getState().add(entry);
  const when = allDay ? start.toDateString() : start.toLocaleString();
  return `Added "${title}" to the calendar on ${when}${hasReminder ? ` with a reminder ${reminder} min before` : ''}. The user can undo it in Tools → Calendar events added by AI.`;
}

/** Remove AI-created events (native + log). Events the user already deleted are dropped from the log too. */
export async function removeAiCalendarEvents(eventIds: string[]): Promise<{ removed: number; failed: number }> {
  if (eventIds.length === 0) return { removed: 0, failed: 0 };
  await ensureCalendarPermission();
  let removed = 0;
  let failed = 0;
  const done: string[] = [];
  for (const id of eventIds) {
    try {
      await RNCalendarEvents.removeEvent(id);
      removed++;
      done.push(id);
    } catch {
      // Already gone (deleted in the calendar app) → nothing left to undo; forget it.
      const still = await RNCalendarEvents.findEventById(id).catch(() => null);
      if (still) failed++;
      else done.push(id);
    }
  }
  useCalendarLogStore.getState().remove(done);
  return { removed, failed };
}
