/**
 * Thinking Budget scale: 0 = no thinking, 1..maxTokens-1 = exact cap (1-token steps),
 * top slider position = Auto; request mapping; one-time migration of the legacy 0 = Auto value.
 */
jest.mock('@offgrid/models', () => ({
  REASONING_BUDGET_AUTO: -1,
  thinkingBudgetPayload: (on: boolean, b: number) => (on && b > 0 ? { thinking_budget_tokens: b } : {}),
}), { virtual: true });
jest.mock('../../../src/services/llmNativeLog', () => ({
  ensureNativeLogCapture: jest.fn(), resetNativeLogCapture: jest.fn(), recentNativeLog: jest.fn(),
}));
jest.mock('../../../src/utils/messageContent', () => ({ templateEmitsReasoning: jest.fn() }));
jest.mock('../../../src/utils/logger', () => ({ __esModule: true, default: { log: jest.fn(), warn: jest.fn(), error: jest.fn() } }));
jest.mock('react-native-fs', () => ({}), { virtual: true });

import { budgetToSliderValue, sliderValueToBudget } from '../../../src/utils/thinkingBudget';
import { buildThinkingCompletionParams } from '../../../src/services/llmHelpers';
import { migratePersistedState } from '../../../src/stores/appStoreMigrations';

describe('Thinking Budget slider scale', () => {
  it('every integer 0..maxTokens-1 round-trips; the top position is Auto', () => {
    for (const v of [0, 1, 2, 3, 257, 4095]) {
      expect(sliderValueToBudget(v, 4096)).toBe(v);
      expect(budgetToSliderValue(v, 4096)).toBe(v);
    }
    expect(sliderValueToBudget(4096, 4096)).toBe(-1);
    expect(budgetToSliderValue(-1, 4096)).toBe(4096);
    expect(budgetToSliderValue(undefined, 4096)).toBe(4096);
    expect(budgetToSliderValue(9000, 4096)).toBe(4096);
  });
});

describe('buildThinkingCompletionParams budget mapping', () => {
  it('0 = no thinking (template off + 0-token cap), N = exact cap, Auto = no cap', () => {
    expect(buildThinkingCompletionParams(true, false, 0)).toEqual({ enable_thinking: false, reasoning_format: 'none', thinking_budget_tokens: 0 });
    expect(buildThinkingCompletionParams(true, false, 3)).toMatchObject({ enable_thinking: true, thinking_budget_tokens: 3 });
    expect(buildThinkingCompletionParams(true, false, -1)).not.toHaveProperty('thinking_budget_tokens');
    expect(buildThinkingCompletionParams(false, false, 3)).toEqual({ enable_thinking: false, reasoning_format: 'none' });
  });
});

describe('reasoningBudget migration (legacy 0 = Auto)', () => {
  const run = (settings: Record<string, unknown>) =>
    (migratePersistedState({ settings }, { settings: {} }, { defaultSettings: { reasoningBudget: -1, reasoningBudgetZeroIsOff: true } as any }) as any).settings;
  it('maps a legacy 0 to Auto once and keeps a real cap', () => {
    expect(run({ reasoningBudget: 0, cacheType: 'q8_0', inferenceBackend: 'cpu' })).toMatchObject({ reasoningBudget: -1, reasoningBudgetZeroIsOff: true });
    expect(run({ reasoningBudget: 2048, cacheType: 'q8_0', inferenceBackend: 'cpu' })).toMatchObject({ reasoningBudget: 2048, reasoningBudgetZeroIsOff: true });
  });
  it('after migration a chosen 0 (no thinking) is kept', () => {
    expect(run({ reasoningBudget: 0, reasoningBudgetZeroIsOff: true, cacheType: 'q8_0', inferenceBackend: 'cpu' }).reasoningBudget).toBe(0);
  });
});
