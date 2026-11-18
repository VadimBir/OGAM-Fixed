export declare const REASONING_BUDGET_AUTO: number;
export declare const REASONING_BUDGET_OPTIONS: number[];
export declare function reasoningBudgetLabel(value: number | null | undefined): string;
export declare function thinkingBudgetPayload(enableThinking: boolean, budget: number): { thinking_budget_tokens?: number };
export declare function reasoningBudgetPayload(enableThinking: boolean, budget: number): { thinking: { type: 'enabled' | 'disabled'; budget_tokens?: number } };
export declare function openRouterReasoningPayload(enableThinking: boolean, budget: number): { reasoning: { max_tokens?: number } };

export declare const MIN_MAX_TOOL_CALLS: number;
export declare const MAX_MAX_TOOL_CALLS: number;
export declare const DEFAULT_MAX_TOOL_CALLS: number;
export declare function normalizeMaxToolCalls(value: unknown): number;
export declare function toolLimitFinalAnswerInstruction(maximum: number): string;
export declare const MAX_TOOL_RESULT_CHARS: number;
export declare function boundToolResult(name: string, content: string, maxChars?: number): string;
export declare function toolPromptChars(messages: readonly unknown[], tools?: readonly unknown[]): number;
export declare function toolResultCharBudget(input: { contextLength?: number; promptChars: number; replyReserveTokens?: number }): number;
export declare function finalResponseFromToolResults(prefix: string, results: readonly string[]): string;

export interface ImageParameterEntry {
  steps?: number;
  cfgScale?: number;
  size?: number;
}
export type ImageParameterStore = Record<string, ImageParameterEntry | undefined>;
export declare const DEFAULT_IMAGE_STEPS: number;
export declare const DEFAULT_IMAGE_CFG_SCALE: number;
export declare const DEFAULT_IMAGE_SIZE: number;
export declare function resolveImageParameters(
  model: { id: string; name?: string },
  store: ImageParameterStore,
): { steps: number; cfgScale: number; size: number };
export declare function effectiveImageParameter(requestValue: number | undefined, fallback: number): number;
