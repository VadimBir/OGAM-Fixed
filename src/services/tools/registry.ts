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
  {
    id: 'add_calendar_event',
    name: 'add_calendar_event',
    displayName: 'Add to Calendar',
    description: 'Add an event or reminder to the user\'s phone calendar. Call this when the user asks to schedule, remind, book, or put something in the calendar. Resolve relative times ("tomorrow at 5", "in 2 hours") against the current local date and time and pass LOCAL times in ISO format. Make exactly one call per event. Every event created is logged so the user can undo it.',
    icon: 'calendar',
    parameters: {
      title: { type: 'string', description: 'Event name, e.g. "Dentist".', required: true },
      start: { type: 'string', description: 'Local start, ISO 8601: "2026-09-26T14:30" (or "2026-09-26" for an all-day event).', required: true },
      end: { type: 'string', description: 'Optional local end, ISO 8601. Default: start + 1 hour.' },
      reminder_minutes: { type: 'integer', description: 'Optional: notify this many minutes before the start (0 = at start).' },
      all_day: { type: 'boolean', description: 'Optional: true for an all-day event.' },
      location: { type: 'string', description: 'Optional location.' },
      notes: { type: 'string', description: 'Optional notes / description.' },
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
/**
 * Emphasis multiplier for a card's model-facing text (small models drop instructions they see once).
 *  - 'content': ONE block/entry whose instruction text is written N times inside it.
 *  - 'block':   the whole block/entry is emitted N times back to back.
 * Native tool schemas cannot carry duplicate functions, so a tool's JSON `description` always uses
 * 'content'; 'block' applies to the tool's text-hint entry (engines without native tool calling).
 */
export type RepeatMode = 'block' | 'content';
export const MAX_SKILL_REPEAT = 5;

export interface RepeatSettings {
  /** 1 (default) = emitted once; clamped to 1..MAX_SKILL_REPEAT. */
  repeat?: number;
  /** Default 'content'. */
  repeatMode?: RepeatMode;
}

export interface SkillOverride extends RepeatSettings {
  /** Replaces the model-facing tool description (the "how to use this tool" text). */
  description?: string;
  /** Replaces the human-facing display name. */
  displayName?: string;
}

export interface CustomSkill extends RepeatSettings {
  id: string;
  /** Short label for the guidance block. */
  name: string;
  /** Free-form instruction injected verbatim into the system-prompt hint. */
  description: string;
  enabled: boolean;
}

export function clampRepeat(n?: number): number {
  const v = Math.round(Number(n));
  return Number.isFinite(v) ? Math.min(MAX_SKILL_REPEAT, Math.max(1, v)) : 1;
}

/** `text` written `n` times, newline-separated (the 'content' multiplier). */
export function repeatContent(text: string, n?: number): string {
  return Array.from({ length: clampRepeat(n) }, () => text).join('\n');
}

/** Emit a block per the multiplier: 'content' repeats the body inside one block, 'block' repeats the block. */
export function applyRepeat(
  body: string,
  wrap: (body: string) => string,
  r: RepeatSettings | undefined,
): string {
  const n = clampRepeat(r?.repeat);
  if ((r?.repeatMode ?? 'content') === 'block') {
    return Array.from({ length: n }, () => wrap(body)).join('\n');
  }
  return wrap(repeatContent(body, n));
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

const isBlockMode = (r: RepeatSettings | undefined): boolean => r?.repeatMode === 'block';

/** Tool description as written INSIDE the tool definition (schema / text list): 'content' mode
 *  repeats it there; 'block' mode keeps it single and repeats standalone system blocks instead. */
export function toolDefinitionDescription(tool: ToolDefinition): string {
  const o = skillOverrides[tool.id];
  const desc = effectiveToolDescription(tool);
  return isBlockMode(o) ? desc : repeatContent(desc, o?.repeat);
}

/** One standalone system block for a built-in tool in 'block' mode. */
export function toolInstructionBlock(name: string, desc: string): string {
  return `<tool name="${name.replaceAll('"', "'")}">\n${desc}\n</tool>`;
}

/**
 * 'block'-mode tools: N standalone `<tool>` blocks in the SYSTEM prompt, on EVERY engine (native
 * tool calling included). Native engines only receive the JSON schema, and a schema cannot hold a
 * function twice, so this is the only place a repeated whole block can exist. Empty when no enabled
 * tool uses block mode.
 */
export function buildToolBlockPromptHint(enabledToolIds: readonly string[]): string {
  const blocks = AVAILABLE_TOOLS
    .filter(t => enabledToolIds.includes(t.id) && isBlockMode(skillOverrides[t.id]))
    .map(t => {
      const n = clampRepeat(skillOverrides[t.id]?.repeat);
      const block = toolInstructionBlock(t.name, effectiveToolDescription(t));
      return Array.from({ length: n }, () => block).join('\n');
    });
  if (blocks.length === 0) return '';
  return (
    '\n\n<tool_instructions>\nStanding SYSTEM instructions for how to use these tools. Follow them ' +
    `every time the tool is relevant.\n${blocks.join('\n')}\n</tool_instructions>`
  );
}

export function getToolsAsOpenAISchema(enabledToolIds: readonly string[]) {
  return AVAILABLE_TOOLS
    .filter(tool => enabledToolIds.includes(tool.id))
    .map(tool => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        // 'content' repeats the text here; 'block' keeps it single (duplicate functions are invalid)
        // and repeats standalone system blocks via buildToolBlockPromptHint.
        description: toolDefinitionDescription(tool),
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
/** `name (type, required) — description` lines, so a text-hint model knows the argument names. */
function toolParameterLines(tool: ToolDefinition): string {
  return Object.entries(tool.parameters)
    .map(([key, p]) => `    ${key} (${p.type}${p.required ? ', required' : ''}${p.enum ? `, one of: ${p.enum.join('|')}` : ''}) — ${p.description}`)
    .join('\n');
}

/** A literal call the parser accepts (parseToolCallsFromText: <tool_call>{"name","arguments"}). */
function toolCallExample(tool: ToolDefinition): string {
  const args = Object.fromEntries(
    Object.entries(tool.parameters)
      .filter(([, p]) => p.required)
      .map(([key, p]) => [key, p.type === 'integer' || p.type === 'number' ? 1 : '...']),
  );
  return `<tool_call>${JSON.stringify({ name: tool.name, arguments: args })}</tool_call>`;
}

export function buildToolSystemPromptHint(enabledToolIds: string[]): string {
  const enabledTools = AVAILABLE_TOOLS.filter(t => enabledToolIds.includes(t.id));
  if (enabledTools.length === 0) return '';
  // Without native tool calling the model sees ONLY this text, so it must carry the exact call
  // syntax, every argument name and a literal example — a bare "- name: description" list left
  // small models believing they could not act (e.g. "I cannot generate images").
  const toolList = enabledTools
    .map(t => {
      const params = toolParameterLines(t);
      return `- ${t.name}: ${toolDefinitionDescription(t)}\n${params ? `  arguments:\n${params}\n` : ''}  example: ${toolCallExample(t)}`;
    })
    .join('\n');
  return (
    '\n\nTools available. You CAN use them: to call one, output exactly one line\n' +
    '<tool_call>{"name": "TOOL_NAME", "arguments": {"ARG": "VALUE"}}</tool_call>\n' +
    'and nothing else in that turn; the result comes back in the next message.\n' +
    `${toolList}\nUse these tools proactively and precisely — call the right tool at the right moment rather than guessing or saying you cannot help.`
  );
}

/**
 * Text of the ENABLED custom skills (prompt-only guidance the user authored in Tools & Skills).
 * Injected on EVERY engine so a native-tool model still receives the guidance. Empty when none.
 *
 * Framed as binding SYSTEM directives in the same tag style as `<character>` / `<user_persona>`
 * (personaComposition), so the model reads them as standing instructions rather than as reference
 * context or chat content. Each skill's multiplier applies to its own `<skill>` block.
 */
export function buildCustomSkillPromptHint(): string {
  const activeCustom = customSkills.filter(s => s.enabled && s.description.trim());
  if (activeCustom.length === 0) return '';
  const blocks = activeCustom
    .map(s => {
      const name = s.name.trim().replaceAll('"', "'");
      return applyRepeat(s.description.trim(), d => `<skill name="${name}">\n${d}\n</skill>`, s);
    })
    .join('\n');
  return (
    '\n\n<skills>\nThe user configured the skills below as standing SYSTEM instructions. They are not ' +
    'conversation content. Apply every skill in every reply, together with the instructions above.\n' +
    `${blocks}\n</skills>`
  );
}
