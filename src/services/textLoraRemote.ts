/**
 * Text LoRA on the user's own llama-server: the adapter list (GET /lora-adapters) and the
 * per-request `lora: [{ id, scale }]` field.
 *
 * The pinned llama.cpp vendored in llama.rn has no tools/server source; it only names the endpoint
 * ("apply later via POST /lora-adapters", vendor common/arg.cpp:3791). The GET list shape and the
 * per-request field follow llama.cpp's tools/server/README.md upstream, NOT verified against a
 * vendored copy. The parser therefore accepts only entries with a numeric id and ignores the rest.
 *
 * The field is sent only to a server that takes llama.cpp extension fields (acceptsServerExtensions,
 * remoteTransportPolicy.ts:66-77): a public OpenAI-compatible host answers unknown fields with 400.
 */
import { useAppStore } from '../stores/appStore';
import { fetchWithTimeout } from './httpClient';
import {
  REMOTE_FETCH_REDIRECT_POLICY,
  acceptsServerExtensions,
  remoteApiBase,
  remoteApiUrl,
  remoteAuthorizationHeaders,
} from './remoteTransportPolicy';
import { clampTextLoraScale } from './textLoraParams';
import type { RemoteTextLoraScale } from './textLoraTypes';

export interface ServerLoraAdapter {
  id: number;
  path: string;
  /** The server's current global scale. */
  scale: number;
}

/** Settings key for a server: the normalized API root, so `http://pc:8080` and `.../v1/` match. */
export const remoteLoraServerKey = (endpoint: string): string => remoteApiBase(endpoint).toLowerCase();

export function parseServerLoraAdapters(body: unknown): ServerLoraAdapter[] {
  const data = typeof body === 'string' ? safeJson(body) : body;
  if (!Array.isArray(data)) return [];
  return data
    .filter((a): a is Record<string, unknown> => !!a && typeof a === 'object' && Number.isInteger((a as { id?: unknown }).id))
    .map(a => ({
      id: a.id as number,
      path: typeof a.path === 'string' ? a.path : `adapter ${a.id as number}`,
      scale: typeof a.scale === 'number' && Number.isFinite(a.scale) ? a.scale : 0,
    }));
}

function safeJson(text: string): unknown {
  try { return JSON.parse(text); } catch { return null; }
}

/** GET /lora-adapters on the user's server. Refuses a host that does not take extension fields. */
export async function fetchServerLoraAdapters(endpoint: string, apiKey?: string | null): Promise<ServerLoraAdapter[]> {
  if (!acceptsServerExtensions(endpoint)) {
    throw new Error('LoRA adapters are listed only for your own llama-server (LAN, localhost or Tailscale).');
  }
  const url = remoteApiUrl(endpoint, '/lora-adapters');
  const body = await fetchWithTimeout<unknown>(url, {
    method: 'GET',
    headers: { Accept: 'application/json', ...remoteAuthorizationHeaders(url, apiKey) },
    redirect: REMOTE_FETCH_REDIRECT_POLICY,
    timeout: 8000,
    retries: 0,
  });
  return parseServerLoraAdapters(body);
}

/**
 * The per-request `lora` field for this endpoint: the user's saved scales, only for a server that
 * takes extension fields, and only when the user set any (otherwise the server's own scales apply).
 */
export function remoteLoraPayload(
  endpoint: string,
  saved: Record<string, RemoteTextLoraScale[]> | undefined = useAppStore.getState().settings.remoteTextLoras,
): { lora?: Array<{ id: number; scale: number }> } {
  if (!acceptsServerExtensions(endpoint)) return {};
  const entries = saved?.[remoteLoraServerKey(endpoint)];
  if (!entries || entries.length === 0) return {};
  return { lora: entries.map(e => ({ id: e.id, scale: clampTextLoraScale(e.scale) })) };
}
