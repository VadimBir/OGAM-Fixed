import { isTailscaleIPv4 } from '../utils/network';

export const PUBLIC_HTTP_REMOTE_ERROR =
  'Remote HTTP servers must use a private LAN or Tailscale address.';

/** Private IPv6: unique-local fc00::/7 (includes fd00::/8) and link-local fe80::/10. */
function isPrivateIPv6(hostname: string): boolean {
  const address = hostname.replace(/^\[|\]$/g, '');
  if (!address.includes(':')) return false;
  if (address === '::1') return true;
  const firstGroup = address.split(':')[0];
  if (!/^[0-9a-f]{1,4}$/i.test(firstGroup)) return false;
  const first = Number.parseInt(firstGroup, 16);
  return (first >= 0xfc00 && first <= 0xfdff) || (first >= 0xfe80 && first <= 0xfebf);
}

function isPrivateIPv4(hostname: string): boolean {
  const octets = hostname.split('.');
  if (
    octets.length !== 4 ||
    octets.some(octet => !/^\d{1,3}$/.test(octet) || Number(octet) > 255)
  )
    return false;

  const first = Number(octets[0]);
  const second = Number(octets[1]);
  return (
    first === 127 ||
    first === 10 ||
    (first === 172 && second >= 16 && second <= 31) ||
    (first === 192 && second === 168) ||
    (first === 169 && second === 254) ||
    isTailscaleIPv4(hostname)
  );
}

/**
 * A host whose cleartext HTTP may carry the user's API key: loopback, RFC1918, link-local,
 * CGNAT / Tailscale 100.64/10, `.local`, private IPv6. Never a public or DNS-resolved name.
 */
function isCredentialSafeHttpHost(hostname: string): boolean {
  return (
    hostname === 'localhost' ||
    hostname.endsWith('.local') ||
    isPrivateIPv4(hostname) ||
    isPrivateIPv6(hostname)
  );
}

function isPrivateHost(hostname: string): boolean {
  return (
    isCredentialSafeHttpHost(hostname) ||
    // Single-label names use the device's local DNS search domain. Public DNS
    // cannot resolve them without a local resolver.
    (!hostname.includes('.') && !hostname.includes(':'))
  );
}

/**
 * Does this server take the llama.cpp / Off Grid Desktop / LM Studio extension fields (top_k,
 * repeat_penalty, cache_prompt, image steps / cfg_scale / seed / negative_prompt / loras ...)?
 * Yes for a server the user runs: a Desktop contract, or a LAN / loopback / Tailscale address
 * (Tailscale MagicDNS names end in `.ts.net`). A public host (api.openai.com, a cloud gateway)
 * gets standard OpenAI fields only: a strict server answers an unknown field with HTTP 400.
 */
export function acceptsServerExtensions(
  endpoint: string,
  modelManagement?: string,
): boolean {
  if (modelManagement === 'offgrid-desktop-v1') return true;
  let hostname: string;
  try {
    hostname = new URL(endpoint).hostname.toLowerCase();
  } catch {
    return false;
  }
  return isPrivateHost(hostname) || hostname.endsWith('.ts.net');
}

/** Validate the transport before any remote request starts. */
export function validateRemoteEndpoint(endpoint: string): URL {
  const url = new URL(endpoint);
  if (url.protocol === 'https:') return url;
  if (url.protocol === 'http:' && isPrivateHost(url.hostname)) return url;
  throw new Error(PUBLIC_HTTP_REMOTE_ERROR);
}

/**
 * Bearer goes over TLS, or over cleartext HTTP only to a host the user's own network owns
 * (a keyed llama-server on the user's PC). A single-label name over HTTP gets no key.
 */
export function remoteAuthorizationHeaders(
  endpoint: string,
  apiKey?: string | null,
): Record<string, string> {
  const url = validateRemoteEndpoint(endpoint);
  const keySafe =
    url.protocol === 'https:' || isCredentialSafeHttpHost(url.hostname);
  return apiKey && keySafe ? { Authorization: `Bearer ${apiKey}` } : {};
}

/** Automatic discovery must never carry a saved credential into cleartext. */
export function canReconcileCredentialedEndpoint(
  endpoint: string,
  hasStoredCredential: boolean,
): boolean {
  const url = validateRemoteEndpoint(endpoint);
  return !hasStoredCredential || url.protocol === 'https:';
}

/**
 * The server root a saved endpoint names. A user may save `http://pc:8080` or `http://pc:8080/v1`;
 * both mean the same server, so trailing slashes and one trailing `/v1` are dropped here and every
 * request path carries its own `/v1` (or none, for `/props`, `/api/chat`, `/api/show`).
 */
export function remoteApiBase(endpoint: string): string {
  let base = endpoint.trim();
  while (base.endsWith('/')) base = base.slice(0, -1);
  return base.toLowerCase().endsWith('/v1') ? base.slice(0, -3) : base;
}

/** One URL builder for every remote request: never `/v1/v1`, never a missing `/v1`. */
export function remoteApiUrl(endpoint: string, path: string): string {
  return `${remoteApiBase(endpoint)}${path.startsWith('/') ? path : `/${path}`}`;
}

/** React Native fetch must reject redirects instead of following a TLS downgrade. */
export const REMOTE_FETCH_REDIRECT_POLICY = 'error' as const;

/** Reject a credentialed response if the native client followed a TLS downgrade. */
export function isCredentialTransportDowngrade(
  requestUrl: string,
  responseUrl: string | undefined,
  hasAuthorization: boolean,
): boolean {
  if (!hasAuthorization || !responseUrl) return false;
  const response = new URL(responseUrl);
  if (response.protocol === 'https:') return false;
  // HTTPS to HTTP is always a downgrade; an HTTP redirect may not carry the key off the private network.
  return (
    new URL(requestUrl).protocol === 'https:' ||
    !isCredentialSafeHttpHost(response.hostname)
  );
}
