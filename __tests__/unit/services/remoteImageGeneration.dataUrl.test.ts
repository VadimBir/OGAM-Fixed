import { runRemoteImageGeneration } from '../../../src/services/remoteImageGeneration';
import { remoteMediaRuntime } from '../../../src/services/remoteMediaRuntime';
import { PUBLIC_HTTP_REMOTE_ERROR } from '../../../src/services/remoteTransportPolicy';
import type { RemoteServer } from '../../../src/types';

const server: RemoteServer = {
  id: 'pc',
  name: 'PC',
  endpoint: 'http://192.168.1.30:8080',
  providerType: 'openai-compatible',
  createdAt: '2026-11-18T00:00:00.000Z',
  mediaModels: { image: 'sd-turbo' },
};

function deps() {
  return {
    updateState: jest.fn(),
    fail: jest.fn((_message: string, _cause?: unknown) => null),
    isCancelled: () => false,
    setRequest: jest.fn(),
  };
}

describe('runRemoteImageGeneration data: URL', () => {
  afterEach(() => jest.restoreAllMocks());

  it('names an unsupported data: image format instead of the private-LAN transport error', async () => {
    jest.spyOn(remoteMediaRuntime, 'generateImage').mockResolvedValue({
      url: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=',
      used: {},
    } as never);
    const d = deps();

    await expect(
      runRemoteImageGeneration({ prompt: 'a cat' }, server, d),
    ).resolves.toBeNull();

    const message = d.fail.mock.calls[0][0];
    expect(message).toContain('image/gif');
    expect(message).toContain('PNG, JPEG, or WebP');
    expect(message).not.toBe(PUBLIC_HTTP_REMOTE_ERROR);
  });
});
