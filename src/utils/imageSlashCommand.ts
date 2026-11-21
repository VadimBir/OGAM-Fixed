/**
 * `/img <prompt>` (alias `/image`) typed in the composer: draw directly, without the LLM.
 *
 * Optional inline flags override the app settings for that one image:
 *   --w N  --h N  --steps N  --cfg X  --seed N  --neg "text"
 * Flags may appear anywhere after the command and are stripped from the prompt. A numeric flag
 * whose value is not a number is dropped (the value stays prompt text). Unknown `--words` stay in
 * the prompt. The values are returned under the generate_image argument names, so the caller bounds
 * them with the same rules as a tool call (imageTool.resolveImageToolRequest: clamp, snap, seed < 0
 * = random). Pure: no store, no service.
 */

export interface ImageSlashArgs {
  width?: number;
  height?: number;
  steps?: number;
  guidance?: number;
  seed?: number;
  negative_prompt?: string;
}

export interface ImageSlashCommand {
  /** The prompt with the command and every recognised flag removed. May be '' (command alone). */
  prompt: string;
  /** Only the flags the user passed. */
  args: ImageSlashArgs;
}

const COMMAND_RE = /^\/(?:img|image)(?=\s|$)/i;
const NUMERIC_FLAGS = new Map<string, keyof Omit<ImageSlashArgs, 'negative_prompt'>>([
  ['--w', 'width'],
  ['--h', 'height'],
  ['--steps', 'steps'],
  ['--cfg', 'guidance'],
  ['--seed', 'seed'],
]);
const NEG_FLAG = '--neg';
/** A double- or single-quoted run, or a run of non-space characters. */
const TOKEN_RE = /"[^"]*"|'[^']*'|\S+/g;
const NUMBER_RE = /^-?\d+(?:\.\d+)?$/;

const isFlag = (token: string | undefined): boolean =>
  token !== undefined && (NUMERIC_FLAGS.has(token.toLowerCase()) || token.toLowerCase() === NEG_FLAG);

function unquote(token: string): string {
  const q = token[0];
  return token.length >= 2 && (q === '"' || q === "'") && token.at(-1) === q ? token.slice(1, -1) : token;
}

/** Parse the composer text. Null when it is not an image command. */
export function parseImageSlashCommand(text: string): ImageSlashCommand | null {
  const trimmed = text.trim();
  const command = COMMAND_RE.exec(trimmed);
  if (!command) return null;
  const tokens = trimmed.slice(command[0].length).match(TOKEN_RE) ?? [];
  const args: ImageSlashArgs = {};
  const promptParts: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const flag = tokens[i].toLowerCase();
    const next = tokens[i + 1];
    const numericKey = NUMERIC_FLAGS.get(flag);
    if (numericKey) {
      if (next !== undefined && NUMBER_RE.test(next)) {
        args[numericKey] = Number(next);
        i++;
      }
      continue;
    }
    if (flag === NEG_FLAG) {
      // `--neg --w 256`: a flag is not the negative text.
      if (next === undefined || isFlag(next)) continue;
      const neg = unquote(next).trim();
      if (neg) args.negative_prompt = neg;
      i++;
      continue;
    }
    promptParts.push(tokens[i]);
  }
  return { prompt: promptParts.join(' ').trim(), args };
}
