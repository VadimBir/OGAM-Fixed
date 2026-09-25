/**
 * Undo log for calendar events the LLM created (add_calendar_event). Every event the model writes
 * is recorded here with its native event id, so the user can remove one, the last one, or ALL of
 * them from Tools → Calendar undo — a runaway model can never leave hundreds of events to delete by
 * hand. Persisted; entries survive restarts until undone.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { createHydrationGatedStorage } from '../utils/hydrationGatedStorage';

export interface AiCalendarEntry {
  eventId: string;
  title: string;
  startDate: string;
  endDate: string;
  createdAt: string;
  conversationId?: string;
}

interface CalendarLogState {
  entries: AiCalendarEntry[];
  add: (entry: AiCalendarEntry) => void;
  remove: (eventIds: string[]) => void;
}

type Persisted = Pick<CalendarLogState, 'entries'>;
const storage = createHydrationGatedStorage<Persisted>();

export const useCalendarLogStore = create<CalendarLogState>()(
  persist(
    set => ({
      entries: [],
      add: entry => set(s => ({ entries: [...s.entries, entry] })),
      remove: ids => set(s => ({ entries: s.entries.filter(e => !ids.includes(e.eventId)) })),
    }),
    {
      name: 'local-llm-ai-calendar-log',
      storage: storage.storage,
      partialize: s => ({ entries: s.entries }),
      onRehydrateStorage: () => () => storage.markHydrated(),
    },
  ),
);
