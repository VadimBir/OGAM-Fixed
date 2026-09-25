import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { generateId } from '../utils/generateId';
import { createHydrationGatedStorage } from '../utils/hydrationGatedStorage';
import {
  AVAILABLE_TOOLS,
  clampRepeat,
  setSkillOverrides,
  type SkillOverride,
  type CustomSkill,
} from '../services/tools/registry';

/**
 * Skill cards store.
 *
 * A "skill card" is the user-editable surface over the assistant's tools/skills:
 * - Built-in tool cards wrap a `AVAILABLE_TOOLS` entry. The user may edit the model-facing
 *   `description` (the text the model reads on HOW/WHEN to use the tool) and the `displayName`.
 *   Params + handler stay in code. Overrides are keyed by tool id.
 * - Custom skill cards are user-authored, prompt-only guidance (no code handler). They inject
 *   their text into the system-prompt hint but are never offered as callable tools.
 *
 * The effective overrides + custom skills are pushed into `services/tools/registry` via
 * `setSkillOverrides`, so the generation pipeline resolves the edited text without any call-site
 * changes. `sync()` is invoked on hydrate and after every mutation.
 */
interface SkillState {
  /** Per-built-in-tool overrides, keyed by tool id. Absent id ⇒ use the built-in default. */
  overrides: Record<string, SkillOverride>;
  /** User-authored prompt-only skills. */
  customSkills: CustomSkill[];

  /** Effective model-facing description for a built-in tool id (override wins). */
  getToolDescription: (toolId: string) => string;
  /** Effective display name for a built-in tool id (override wins). */
  getToolDisplayName: (toolId: string) => string;
  /** Set/replace the override for a built-in tool. Empty strings clear that field. */
  setToolOverride: (toolId: string, patch: SkillOverride) => void;
  /** Reset a built-in tool card back to its shipped defaults. */
  resetToolOverride: (toolId: string) => void;

  createCustomSkill: (skill: Omit<CustomSkill, 'id'>) => CustomSkill;
  updateCustomSkill: (id: string, patch: Partial<Omit<CustomSkill, 'id'>>) => void;
  deleteCustomSkill: (id: string) => void;

  /** Push the current overrides + custom skills into the tool registry. */
  sync: () => void;
}

type PersistedSkillState = Pick<SkillState, 'overrides' | 'customSkills'>;

const skillStorage = createHydrationGatedStorage<PersistedSkillState>();

export const useSkillStore = create<SkillState>()(
  persist(
    (set, get) => ({
      overrides: {},
      customSkills: [],

      getToolDescription: toolId => {
        const tool = AVAILABLE_TOOLS.find(t => t.id === toolId);
        const override = get().overrides[toolId]?.description;
        return override && override.trim() ? override : tool?.description ?? '';
      },

      getToolDisplayName: toolId => {
        const tool = AVAILABLE_TOOLS.find(t => t.id === toolId);
        const override = get().overrides[toolId]?.displayName;
        return override && override.trim() ? override : tool?.displayName ?? toolId;
      },

      setToolOverride: (toolId, patch) => {
        set(state => {
          const next: SkillOverride = { ...state.overrides[toolId] };
          // An empty string means "clear this field back to the built-in default".
          if (patch.description !== undefined) {
            if (patch.description.trim()) next.description = patch.description;
            else delete next.description;
          }
          if (patch.displayName !== undefined) {
            if (patch.displayName.trim()) next.displayName = patch.displayName;
            else delete next.displayName;
          }
          // Multiplier: 1× 'content' is the default, so it is stored as absent.
          if (patch.repeat !== undefined) {
            const n = clampRepeat(patch.repeat);
            if (n > 1) next.repeat = n;
            else delete next.repeat;
          }
          if (patch.repeatMode !== undefined) {
            if (patch.repeatMode === 'block') next.repeatMode = 'block';
            else delete next.repeatMode;
          }
          const overrides = { ...state.overrides };
          if (Object.keys(next).length === 0) delete overrides[toolId];
          else overrides[toolId] = next;
          return { overrides };
        });
        get().sync();
      },

      resetToolOverride: toolId => {
        set(state => {
          const overrides = { ...state.overrides };
          delete overrides[toolId];
          return { overrides };
        });
        get().sync();
      },

      createCustomSkill: skill => {
        const created: CustomSkill = { ...skill, id: generateId() };
        set(state => ({ customSkills: [...state.customSkills, created] }));
        get().sync();
        return created;
      },

      updateCustomSkill: (id, patch) => {
        set(state => ({
          customSkills: state.customSkills.map(s =>
            s.id === id ? { ...s, ...patch } : s,
          ),
        }));
        get().sync();
      },

      deleteCustomSkill: id => {
        set(state => ({
          customSkills: state.customSkills.filter(s => s.id !== id),
        }));
        get().sync();
      },

      sync: () => {
        setSkillOverrides(get().overrides, get().customSkills);
      },
    }),
    {
      name: 'local-llm-skill-storage',
      storage: skillStorage.storage,
      onRehydrateStorage: () => state => {
        skillStorage.markHydrated();
        // Push persisted edits into the registry as soon as they load, so the very first
        // generation after launch already sees the user's tool-description edits.
        state?.sync();
      },
      partialize: (state): PersistedSkillState => ({
        overrides: state.overrides,
        customSkills: state.customSkills,
      }),
    },
  ),
);
