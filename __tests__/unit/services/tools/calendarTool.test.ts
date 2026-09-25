const mockSave = jest.fn();
const mockRemove = jest.fn();
const mockFind = jest.fn();
jest.mock('react-native-calendar-events', () => ({
  __esModule: true,
  default: {
    checkPermissions: jest.fn(async () => 'authorized'),
    requestPermissions: jest.fn(async () => 'authorized'),
    saveEvent: (...a: unknown[]) => mockSave(...a),
    removeEvent: (...a: unknown[]) => mockRemove(...a),
    findEventById: (...a: unknown[]) => mockFind(...a),
  },
}));
jest.mock('../../../../src/utils/hydrationGatedStorage', () => ({
  createHydrationGatedStorage: () => ({ storage: undefined, markHydrated: () => undefined }),
}));

import { Platform } from 'react-native';
import { parseLocalDateTime, handleAddCalendarEvent, removeAiCalendarEvents } from '../../../../src/services/tools/calendarTool';
import { useCalendarLogStore } from '../../../../src/stores/calendarLogStore';

describe('add_calendar_event', () => {
  beforeEach(() => {
    mockSave.mockReset(); mockRemove.mockReset(); mockFind.mockReset();
    useCalendarLogStore.setState({ entries: [] });
  });

  it('parses local ISO date-times and date-only; rejects junk', () => {
    const d = parseLocalDateTime('2026-09-26T14:30')!;
    expect([d.getFullYear(), d.getMonth(), d.getDate(), d.getHours(), d.getMinutes()]).toEqual([2026, 8, 26, 14, 30]);
    expect(parseLocalDateTime('2026-09-26')!.getHours()).toBe(0);
    expect(parseLocalDateTime('tomorrow')).toBeNull();
    expect(parseLocalDateTime(undefined)).toBeNull();
  });

  it('saves the event (default 1h, reminder) and logs it for undo', async () => {
    mockSave.mockResolvedValue('42');
    const out = await handleAddCalendarEvent({ name: 'add_calendar_event', arguments: { title: 'Dentist', start: '2026-09-26T14:30', reminder_minutes: 15 } });
    const [title, details] = mockSave.mock.calls[0];
    expect(title).toBe('Dentist');
    expect(new Date(details.endDate).getTime() - new Date(details.startDate).getTime()).toBe(3600000);
    // Android: Reminders.MINUTES before start (positive); iOS: negative relative offset.
    expect(details.alarms).toEqual([{ date: Platform.OS === 'ios' ? -15 : 15 }]);
    expect(useCalendarLogStore.getState().entries.map(e => e.eventId)).toEqual(['42']);
    expect(out).toContain('undo');
  });

  it('rejects a missing title or start without writing anything', async () => {
    await expect(handleAddCalendarEvent({ name: 'add_calendar_event', arguments: { start: '2026-09-26T14:30' } })).rejects.toThrow('title');
    await expect(handleAddCalendarEvent({ name: 'add_calendar_event', arguments: { title: 'x', start: 'soon' } })).rejects.toThrow('start');
    expect(mockSave).not.toHaveBeenCalled();
  });

  it('undo removes events and forgets ones the user already deleted', async () => {
    useCalendarLogStore.setState({ entries: ['1', '2', '3'].map(id => ({ eventId: id, title: id, startDate: '', endDate: '', createdAt: '' })) });
    mockRemove.mockImplementation(async (id: string) => { if (id === '2') throw new Error('gone'); return true; });
    mockFind.mockResolvedValue(null);
    const r = await removeAiCalendarEvents(['1', '2', '3']);
    expect(r).toEqual({ removed: 2, failed: 0 });
    expect(useCalendarLogStore.getState().entries).toEqual([]);
  });
});
