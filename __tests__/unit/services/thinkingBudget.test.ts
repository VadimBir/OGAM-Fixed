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
import { buildThinkingCompletionParams, autoThinkingBudgetTokens } from '../../../src/services/llmHelpers';
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
  const hardOff = { enable_thinking: false, reasoning_format: 'none', thinking_budget_tokens: 0, thinking_start_tag: '<think>', thinking_end_tag: '</think>' };
  it('toggle OFF is a hard off: template switch + 0-token budget sampler with think tags', () => {
    expect(buildThinkingCompletionParams(false, false, 3)).toEqual(hardOff);
    expect(buildThinkingCompletionParams(false, false, undefined)).toEqual(hardOff);
    expect(buildThinkingCompletionParams(false, true)).toMatchObject({ thinking_budget_tokens: 0, thinking_start_tag: '<|channel>thought', thinking_end_tag: '<channel|>' });
  });
  it('budget 0 = same hard off; N = exact cap; Auto = no cap', () => {
    expect(buildThinkingCompletionParams(true, false, 0)).toEqual(hardOff);
    expect(buildThinkingCompletionParams(true, false, 3)).toMatchObject({ enable_thinking: true, thinking_budget_tokens: 3 });
    expect(buildThinkingCompletionParams(true, false, -1)).not.toHaveProperty('thinking_budget_tokens');
  });
  // Tool path: llama.rn builds the reasoning-budget sampler only for thinking_budget_tokens >= 0
  // (JSIParams.cpp:497); without it the lazy tool grammar can fire inside <think>. Auto + n_predict
  // must therefore send a finite budget that leaves room for the call.
  it('Auto with n_predict sends a finite budget leaving >= 128 tokens for the answer', () => {
    expect(autoThinkingBudgetTokens(2048)).toBe(1536);
    expect(autoThinkingBudgetTokens(1024)).toBe(768);
    expect(autoThinkingBudgetTokens(600)).toBe(472); // max(512,450)=512 > 600-128 -> 472
    expect(autoThinkingBudgetTokens(256)).toBe(256); // tiny cap: sampler exists, never closes early
    expect(autoThinkingBudgetTokens(100)).toBe(100);
    for (const n of [257, 300, 640, 1024, 2048, 8192]) {
      const b = autoThinkingBudgetTokens(n);
      expect(b).toBeGreaterThan(0);
      expect(b).toBeLessThanOrEqual(n - 128);
    }
    expect(buildThinkingCompletionParams(true, false, -1, 2048)).toEqual({ enable_thinking: true, reasoning_format: 'deepseek', thinking_budget_tokens: 1536 });
    expect(buildThinkingCompletionParams(true, true, undefined, 1024)).toMatchObject({ reasoning_format: 'auto', thinking_budget_tokens: 768 });
    // An explicit user cap still wins; off still wins.
    expect(buildThinkingCompletionParams(true, false, 300, 2048)).toMatchObject({ thinking_budget_tokens: 300 });
    expect(buildThinkingCompletionParams(false, false, -1, 2048)).toEqual(hardOff);
  });
});

describe('maxTokens default 1024 -> 2048 migration', () => {
  const run = (settings: Record<string, unknown>) =>
    (migratePersistedState({ settings }, { settings: {} }, { defaultSettings: { maxTokens: 2048, maxTokensDefault2048: true } as any }) as any).settings;
  it('raises only the exact old default, once', () => {
    expect(run({ maxTokens: 1024, cacheType: 'q8_0', inferenceBackend: 'cpu' })).toMatchObject({ maxTokens: 2048, maxTokensDefault2048: true });
    expect(run({ maxTokens: 512, cacheType: 'q8_0', inferenceBackend: 'cpu' })).toMatchObject({ maxTokens: 512, maxTokensDefault2048: true });
    // After the one-time migration a user-chosen 1024 is kept.
    expect(run({ maxTokens: 1024, maxTokensDefault2048: true, cacheType: 'q8_0', inferenceBackend: 'cpu' }).maxTokens).toBe(1024);
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
