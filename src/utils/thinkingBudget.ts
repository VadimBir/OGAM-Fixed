import { REASONING_BUDGET_AUTO } from '@offgrid/models';

/**
 * Thinking Budget slider scale (1-token steps): 0 = no thinking, 1..maxTokens-1 = exact cap,
 * maxTokens (the top position) = Auto (no cap; Max Tokens bounds the reply anyway).
 * The request side maps 0 to thinking off (buildThinkingCompletionParams, remote helpers).
 */
export function budgetToSliderValue(budget: number | undefined, maxTokens: number): number {
  if (budget == null || budget < 0 || budget >= maxTokens) return maxTokens;
  return Math.round(budget);
}

export function sliderValueToBudget(value: number, maxTokens: number): number {
  const v = Math.round(value);
  return v >= maxTokens ? REASONING_BUDGET_AUTO : Math.max(0, v);
}
