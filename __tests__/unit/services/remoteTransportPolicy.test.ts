import {
  PUBLIC_HTTP_REMOTE_ERROR,
  acceptsServerExtensions,
  canReconcileCredentialedEndpoint,
  isCredentialTransportDowngrade,
  remoteAuthorizationHeaders,
  validateRemoteEndpoint,
} from '../../../src/services/remoteTransportPolicy';
import { fetchModelsFromServer } from '../../../src/stores/remoteServerHelpers';

describe('remote transport policy', () => {
  const originalFetch = global.fetch;
  afterEach(() => {
    global.fetch = originalFetch;
  });
  it('sends the key over HTTP only to loopback, private, link-local, CGNAT, .local, private IPv6', () => {
    const bearer = { Authorization: 'Bearer secret' };
    for (const endpoint of [
      'http://localhost:8080',
      'http://127.0.0.1:8080',
      'http://127.5.0.1:8080',
      'http://[::1]:8080',
      'http://10.0.0.5:8080',
      'http://172.16.0.5:8080',
      'http://172.31.255.5:8080',
      'http://192.168.1.30:7878',
      'http://169.254.10.20:8080',
      'http://100.64.0.1:8080',
      'http://100.127.255.254:8080',
      'http://desktop.local:7878',
      'http://[fd00::1]:8080',
      'http://[fc12:3456::1]:8080',
      'http://[fe80::1]:8080',
      'http://[febf::1]:8080',
    ]) {
      expect(remoteAuthorizationHeaders(endpoint, 'secret')).toEqual(bearer);
    }
    // Allowed as an HTTP endpoint but not in the key list: a single-label DNS name.
    expect(remoteAuthorizationHeaders('http://desktop:7878', 'secret')).toEqual({});
    expect(remoteAuthorizationHeaders('http://192.168.1.30:7878', null)).toEqual({});
  });

  it('never sends the key to a public HTTP host', () => {
    for (const endpoint of [
      'http://8.8.8.8:8080',
      'http://172.32.0.1:8080',
      'http://100.128.0.1:8080',
      'http://[2001:db8::1]:8080',
      'http://[fec0::1]:8080',
      'http://192.168.attacker.example:8080',
      'http://example.test:8080',
    ]) {
      expect(() => remoteAuthorizationHeaders(endpoint, 'secret')).toThrow(
        PUBLIC_HTTP_REMOTE_ERROR,
      );
    }
  });

  it('allows bearer credentials only on HTTPS', () => {
    expect(
      remoteAuthorizationHeaders('https://desktop.example.test', 'secret'),
    ).toEqual({
      Authorization: 'Bearer secret',
    });
  });

  it('sends extension fields only to a server the user runs', () => {
    for (const endpoint of [
      'http://192.168.1.30:8080',
      'http://10.0.0.5:7878/v1',
      'http://localhost:8080',
      'http://desktop.local:7878',
      'http://desktop:7878',
      'http://100.101.102.103:8080',
      'https://studio.tail1234.ts.net',
      'http://[fd00::1]:8080',
      'http://[fe80::1]:8080',
    ]) {
      expect(acceptsServerExtensions(endpoint)).toBe(true);
    }
    for (const endpoint of [
      'https://api.openai.com',
      'https://openrouter.ai/api',
      'https://192.168.attacker.example',
      'https://[2001:db8::1]',
      'not a url',
    ]) {
      expect(acceptsServerExtensions(endpoint)).toBe(false);
    }
    expect(acceptsServerExtensions('https://desktop.example.test', 'offgrid-desktop-v1')).toBe(true);
  });

  it('rejects public cleartext endpoints before a request starts', () => {
    expect(() => validateRemoteEndpoint('http://example.test:7878')).toThrow(
      PUBLIC_HTTP_REMOTE_ERROR,
    );
    expect(() =>
      validateRemoteEndpoint('http://192.168.attacker.example:7878'),
    ).toThrow(PUBLIC_HTTP_REMOTE_ERROR);
    expect(() =>
      validateRemoteEndpoint('http://10.attacker.example:7878'),
    ).toThrow(PUBLIC_HTTP_REMOTE_ERROR);
  });

  it('does not reconcile a credentialed server onto a discovered HTTP endpoint', () => {
    expect(
      canReconcileCredentialedEndpoint('http://192.168.1.31:7878', true),
    ).toBe(false);
    expect(
      canReconcileCredentialedEndpoint('http://192.168.1.31:7878', false),
    ).toBe(true);
    expect(
      canReconcileCredentialedEndpoint('https://desktop.local:7878', true),
    ).toBe(true);
  });

  it('detects an HTTPS-to-HTTP response downgrade only for credentialed requests', () => {
    expect(
      isCredentialTransportDowngrade(
        'https://desktop.example.test/v1/chat',
        'http://192.168.1.30:7878/v1/chat',
        true,
      ),
    ).toBe(true);
    expect(
      isCredentialTransportDowngrade(
        'https://desktop.example.test/v1/chat',
        'http://192.168.1.30:7878/v1/chat',
        false,
      ),
    ).toBe(false);
  });

  it('rejects a keyed HTTP redirect off the private network, allows one inside it', () => {
    expect(
      isCredentialTransportDowngrade(
        'http://192.168.1.30:8080/v1/chat',
        'http://203.0.113.9/v1/chat',
        true,
      ),
    ).toBe(true);
    expect(
      isCredentialTransportDowngrade(
        'http://192.168.1.30:8080/v1/chat',
        'http://192.168.1.31:8080/v1/chat',
        true,
      ),
    ).toBe(false);
    expect(
      isCredentialTransportDowngrade(
        'http://192.168.1.30:8080/v1/chat',
        'https://desktop.example.test/v1/chat',
        true,
      ),
    ).toBe(false);
  });

  it('sends the key on model discovery to a private-LAN HTTP server', async () => {
    const calls: Array<RequestInit | undefined> = [];
    global.fetch = jest.fn(async (_url, init) => {
      calls.push(init);
      return {
        ok: true,
        json: async () => ({ object: 'list', data: [] }),
      } as Response;
    }) as typeof fetch;

    await fetchModelsFromServer({
      id: 'desktop',
      name: 'Desktop',
      endpoint: 'http://192.168.1.30:7878',
      providerType: 'openai-compatible',
      apiKey: 'secret',
      createdAt: '2026-08-30',
    });

    expect(calls[0]?.headers).toMatchObject({ Authorization: 'Bearer secret' });
    expect(calls[0]?.redirect).toBe('error');
  });
});
