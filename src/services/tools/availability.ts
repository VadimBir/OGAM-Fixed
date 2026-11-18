/**
 * Which enabled built-in tools can actually execute right now. A tool whose engine is absent is not
 * offered to the model: a 3B model shown generate_image with no image model calls it, gets an error,
 * and the user sees a failed turn instead of a plain answer.
 *
 * One predicate per engine-backed tool. generate_image asks imageGenerationService.activeImageEngine,
 * the same answer generateImage itself routes on (remote image server or selected local model), so a
 * new engine (e.g. a future remote image server) satisfies this by being added there, not here.
 */
const TOOL_AVAILABILITY: Record<string, () => boolean> = {
  generate_image: () =>
    require('../imageGenerationService').imageGenerationService.activeImageEngine() !== null, // NOSONAR (cycle-safe lazy import)
};

export function isToolAvailable(toolId: string): boolean {
  const check = TOOL_AVAILABILITY[toolId];
  return check ? check() : true;
}

/** The subset of `toolIds` that can execute now (order kept). */
export function availableToolIds(toolIds: readonly string[]): string[] {
  return toolIds.filter(isToolAvailable);
}
