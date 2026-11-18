/**
 * Remote text LoRA: the per-request `lora: [{ id, scale }]` field goes only to the user's own
 * llama-server (acceptsServerExtensions) and only when the user set scales; GET /lora-adapters is
 * parsed defensively.
 */
import * as httpClient from '../../../src/services/httpClient';
import { OpenAICompatibleProvider } from '../../../src/services/providers/openAICompatibleProvider';
import {
  fetchServerLoraAdapters,
  parseServerLoraAdapters,
  remoteLoraPayload,
  remoteLoraServerKey,
} from '../../../src/services/textLoraRemote';
import { clearRemoteTextLoras, setRemoteTextLoraScale } from '../../../src/services/textLoraActions';
import { useAppStore } from '../../../src/stores/appStore';

jest.mock('../../../src/services/httpClient', () => ({
  createStreamingRequest: jest.fn(),
  createNDJSONStreamingRequest: jest.fn(),
  imageToBase64DataUrl: jest.fn(),
  fetchWithTimeout: jest.fn(),
  parseOpenAIMessage: jest.fn((event: { data: string }) => {
    try { return JSON.parse(event.data); } catch { return null; }
  }),
}));
jest.mock('../../../src/utils/logger', () => ({ __esModule: true, default: { log: jest.fn(), warn: jest.fn(), error: jest.fn() } }));

const LAN = 'http://192.168.1.50:8080';
const PUBLIC = 'https://api.openai.com';

const setRemote = (remoteTextLoras: Record<string, Array<{ id: number; scale: number }>>) =>
  useAppStore.setState({ settings: { ...useAppStore.getState().settings, remoteTextLoras } });

beforeEach(() => {
  jest.clearAllMocks();
  setRemote({});
});

describe('remoteLoraPayload', () => {
  it('sends the saved scales to the user\'s own server', () => {
    expect(remoteLoraPayload(`${LAN}/v1/`, { [remoteLoraServerKey(LAN)]: [{ id: 0, scale: 0.7 }, { id: 1, scale: 3 }] }))
      .toEqual({ lora: [{ id: 0, scale: 0.7 }, { id: 1, scale: 2 }] });
  });

  it('never sends the field to a public host', () => {
    expect(remoteLoraPayload(PUBLIC, { [remoteLoraServerKey(PUBLIC)]: [{ id: 0, scale: 1 }] })).toEqual({});
  });

  it('sends nothing when no scale was set (the server keeps its own)', () => {
    expect(remoteLoraPayload(LAN, {})).toEqual({});
  });
});

describe('request body', () => {
  const bodyFor = async (endpoint: string) => {
    const mock = httpClient.createStreamingRequest as jest.Mock;
    mock.mockImplementation((_url, _req, onEvent) => {
      onEvent({ data: '{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}' });
      return Promise.resolve();
    });
    const provider = new OpenAICompatibleProvider('s', { endpoint, modelId: 'm' });
    await provider.generate(
      [{ id: '1', role: 'user', content: 'Hi', timestamp: 0 }],
      {},
      { onToken: jest.fn(), onComplete: jest.fn(), onError: jest.fn() },
    );
    return mock.mock.calls[0][1].body as Record<string, unknown>;
  };

  it('carries lora for a LAN llama-server with saved scales', async () => {
    setRemoteTextLoraScale(LAN, 1, 0.5, [{ id: 0, scale: 1 }, { id: 1, scale: 1 }]);
    const body = await bodyFor(LAN);
    expect(body.lora).toEqual([{ id: 0, scale: 1 }, { id: 1, scale: 0.5 }]);
  });

  it('has no lora field for a public host even with saved scales', async () => {
    setRemote({ [remoteLoraServerKey(PUBLIC)]: [{ id: 0, scale: 1 }] });
    const body = await bodyFor(PUBLIC);
    expect('lora' in body).toBe(false);
  });

  it('has no lora field after the overrides are cleared', async () => {
    setRemoteTextLoraScale(LAN, 0, 0.5, [{ id: 0, scale: 1 }]);
    clearRemoteTextLoras(LAN);
    const body = await bodyFor(LAN);
    expect('lora' in body).toBe(false);
  });
});

describe('GET /lora-adapters', () => {
  it('parses the list and ignores entries without a numeric id', () => {
    expect(parseServerLoraAdapters([{ id: 0, path: '/m/a.gguf', scale: 1 }, { id: 'x' }, null, { id: 2 }]))
      .toEqual([{ id: 0, path: '/m/a.gguf', scale: 1 }, { id: 2, path: 'adapter 2', scale: 0 }]);
    expect(parseServerLoraAdapters('[{"id":1,"path":"b.gguf","scale":0.5}]')).toEqual([{ id: 1, path: 'b.gguf', scale: 0.5 }]);
    expect(parseServerLoraAdapters({ error: 'x' })).toEqual([]);
  });

  it('calls the server root, not /v1', async () => {
    (httpClient.fetchWithTimeout as jest.Mock).mockResolvedValue([{ id: 0, path: 'a.gguf', scale: 1 }]);
    await expect(fetchServerLoraAdapters(`${LAN}/v1`)).resolves.toEqual([{ id: 0, path: 'a.gguf', scale: 1 }]);
    expect(httpClient.fetchWithTimeout).toHaveBeenCalledWith(`${LAN}/lora-adapters`, expect.objectContaining({ method: 'GET' }));
  });

  it('refuses a public host', async () => {
    await expect(fetchServerLoraAdapters(PUBLIC)).rejects.toThrow('own llama-server');
    expect(httpClient.fetchWithTimeout).not.toHaveBeenCalled();
  });
});
