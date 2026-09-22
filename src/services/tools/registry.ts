import { ToolDefinition } from './types';

export const AVAILABLE_TOOLS: ToolDefinition[] = [
  {
    id: 'web_search',
    name: 'web_search',
    displayName: 'Web Search',
    description: 'Search the live web and return real-time result titles, snippets, and URLs. Use this for any question about current events, prices, weather, news, or anything that requires up-to-date information. When the snippet is insufficient, call read_url on the most relevant result URL to get the full page content.',
    icon: 'globe',
    requiresNetwork: true,
    parameters: {
      query: {
        type: 'string',
        description: 'Search query',
        required: true,
      },
    },
  },
  {
    id: 'calculator',
    name: 'calculator',
    displayName: 'Calculator',
    description: 'Evaluate math expressions',
    icon: 'hash',
    parameters: {
      expression: {
        type: 'string',
        description: 'Math expression',
        required: true,
      },
    },
  },
  {
    id: 'get_current_datetime',
    name: 'get_current_datetime',
    displayName: 'Date & Time',
    description: 'Get current date and time',
    icon: 'clock',
    parameters: {
      timezone: {
        type: 'string',
        description: 'IANA timezone, e.g. America/New_York',
      },
    },
  },
  {
    id: 'get_device_info',
    name: 'get_device_info',
    displayName: 'Device Info',
    description: 'Get device hardware info',
    icon: 'smartphone',
    parameters: {
      info_type: {
        type: 'string',
        description: 'Info type',
        enum: ['battery', 'storage', 'memory', 'all'],
      },
    },
  },
  {
    id: 'search_knowledge_base',
    name: 'search_knowledge_base',
    displayName: 'Knowledge Base',
    description: 'Search uploaded project documents',
    icon: 'book-open',
    parameters: {
      query: {
        type: 'string',
        description: 'Search query',
        required: true,
      },
    },
  },
  {
    id: 'read_url',
    name: 'read_url',
    displayName: 'URL Reader',
    description: 'Fetch the full live content of any URL. Use this after web_search to read the complete text of a result page, or directly when the user shares a link.',
    icon: 'link',
    requiresNetwork: true,
    parameters: {
      url: {
        type: 'string',
        description: 'Full URL to fetch',
        required: true,
      },
    },
  },
  {
    id: 'generate_image',
    name: 'generate_image',
    displayName: 'Generate Image',
    description: 'Generate an image on-device from a text description and show it directly in the chat. Call this whenever the user asks to create, draw, paint, render, generate, or make an image, picture, photo, wallpaper, avatar, logo, or artwork. Write a vivid, detailed English `prompt` describing the desired subject, style, and composition. The image is produced by the selected local image model and appears as an attachment in the conversation.',
    icon: 'image',
    parameters: {
      prompt: {
        type: 'string',
        description: 'Detailed English description of the image to generate (subject, style, composition, lighting).',
        required: true,
      },
      negative_prompt: {
        type: 'string',
        description: 'Optional: things to avoid in the image (e.g. "blurry, extra fingers, text").',
      },
      steps: {
        type: 'integer',
        description: 'Optional denoising steps. Higher = more detail but slower; typical 8-30. Omit to use the user\'s configured default.',
      },
    },
  },
];

/**
 * Skill-card override layer.
 *
 * `AVAILABLE_TOOLS` holds the built-in tool DEFINITIONS whose params + handlers are wired in
 * native/JS code and must not change. What the user is allowed to edit via the Skills/Tools
 * cards UI is the model-facing TEXT — the `description` (the instruction the model reads on how
 * and when to use the tool) and the `displayName`. Those edits live in `skillStore` and are
 * pushed here through `setSkillOverrides`, so `registry` stays the single place that resolves the
 * effective tool text without threading a store through every generation call site.
 *
 * Custom skill cards are user-created, prompt-only guidance (no code handler): they never appear
 * in the OpenAI tool schema (the model must not call a tool that cannot execute) — they only
 * contribute their text to the system-prompt hint.
 */
export interface SkillOverride {
  /** Replaces the model-facing tool description (the "how to use this tool" text). */
  description?: string;
  /** Replaces the human-facing display name. */
  displayName?: string;
}

export interface CustomSkill {
  id: string;
  /** Short label for the guidance block. */
  name: string;
  /** Free-form instruction injected verbatim into the system-prompt hint. */
  description: string;
  enabled: boolean;
}

let skillOverrides: Record<string, SkillOverride> = {};
let customSkills: CustomSkill[] = [];

/** Called by `skillStore` on init and whenever the user edits a skill card. */
export function setSkillOverrides(
  overrides: Record<string, SkillOverride>,
  custom: CustomSkill[] = [],
): void {
  skillOverrides = overrides ?? {};
  customSkills = custom ?? [];
}

/** The description the model should see for a built-in tool (user override wins over the default). */
export function effectiveToolDescription(tool: ToolDefinition): string {
  const o = skillOverrides[tool.id];
  return (o?.description && o.description.trim()) ? o.description : tool.description;
}

export function getToolsAsOpenAISchema(enabledToolIds: readonly string[]) {
  return AVAILABLE_TOOLS
    .filter(tool => enabledToolIds.includes(tool.id))
    .map(tool => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: effectiveToolDescription(tool),
        parameters: {
          type: 'object',
          properties: Object.fromEntries(
            Object.entries(tool.parameters).map(([key, param]) => [
              key,
              {
                type: param.type,
                description: param.description,
                ...(param.enum ? { enum: param.enum } : {}),
              },
            ]),
          ),
          required: Object.entries(tool.parameters)
            .filter(([_, param]) => param.required)
            .map(([key]) => key),
        },
      },
    }));
}

/**
 * Text listing of the ENABLED TOOLS for engines that lack native tool-calling (llama without Jinja).
 * Native-tool engines (LiteRT / remote) receive the tools structurally, so this text is suppressed
 * for them by the caller to avoid a double-inject.
 *
 * NOTE: custom-skill text is intentionally NOT here — see `buildCustomSkillPromptHint`. Custom skills
 * are prompt-only guidance (never callable tools), so they must reach EVERY engine regardless of
 * native tool support; bundling them here dropped them silently on LiteRT/remote (the model "never
 * saw" an enabled custom skill).
 */
export function buildToolSystemPromptHint(enabledToolIds: string[]): string {
  const enabledTools = AVAILABLE_TOOLS.filter(t => enabledToolIds.includes(t.id));
  if (enabledTools.length === 0) return '';
  const toolList = enabledTools
    .map(t => `- ${t.name}: ${effectiveToolDescription(t)}`)
    .join('\n');
  return `\n\nTools available:\n${toolList}\nUse these tools proactively and precisely — call the right tool at the right moment rather than guessing or saying you cannot help.`;
}

/**
 * Text of the ENABLED custom skills (prompt-only guidance the user authored in Tools & Skills).
 * Injected on EVERY engine so a native-tool model still receives the guidance. Empty when none.
 */
export function buildCustomSkillPromptHint(): string {
  const activeCustom = customSkills.filter(s => s.enabled && s.description.trim());
  if (activeCustom.length === 0) return '';
  const skillList = activeCustom.map(s => `- ${s.name}: ${s.description}`).join('\n');
  return `\n\nAdditional skills:\n${skillList}`;
}
