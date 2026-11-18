/**
 * ModelKeepAliveStopRequested (notification Stop / swipe-away) must unload all models and must not
 * restart the native service while unloading.
 */
const mockStart = jest.fn(async (_l: string) => true);
const mockStop = jest.fn(async () => undefined);
const mockRemove = jest.fn();
let stopHandler: ((e?: { reason?: string }) => void) | undefined;
const mockAddListener = jest.fn((name: string, h: (e?: { reason?: string }) => void) => {
  if (name === 'ModelKeepAliveStopRequested') stopHandler = h;
  return { remove: mockRemove };
});

// A Proxy, not `{...actual}`: spreading evaluates every lazy react-native getter, and the DevMenu
// TurboModule getter throws under jest ("'DevMenu' could not be found"), failing the whole suite.
jest.mock('react-native', () => {
  const actual = jest.requireActual('react-native');
  const overrides: Record<string, unknown> = {
    Platform: { ...actual.Platform, OS: 'android' },
    NativeEventEmitter: jest.fn().mockImplementation(() => ({ addListener: mockAddListener })),
    NativeModules: {
      ...actual.NativeModules,
      // Delegating arrows: the import below is hoisted above the mock* consts, so this factory runs first.
      ModelKeepAliveModule: {
        start: (label: string) => mockStart(label),
        stop: () => mockStop(),
        addListener: jest.fn(),
        removeListeners: jest.fn(),
      },
    },
  };
  return new Proxy(actual, {
    get: (target, key) => (typeof key === 'string' && key in overrides ? overrides[key] : target[key]),
  });
});

const mockUnloadAll = jest.fn(async (_keep?: boolean) => ({ textUnloaded: true, imageUnloaded: true }));
let modelListener: (() => void) | undefined;
let mockTextLoaded = true;
jest.mock('../../../src/services/activeModelService', () => ({
  activeModelService: {
    getActiveModels: () => ({
      text: { isLoaded: mockTextLoaded, isLoading: false, model: { name: 'T' } },
      image: { isLoaded: false, isLoading: false, model: null },
    }),
    subscribe: (l: () => void) => { modelListener = l; return jest.fn(); },
    unloadAllModels: (k?: boolean) => mockUnloadAll(k),
  },
}));

import { startModelKeepAliveSync } from '../../../src/services/modelKeepAlive';

const flush = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };

describe('keep-alive stop request', () => {
  beforeEach(() => {
    mockTextLoaded = true;
    mockStart.mockClear();
    mockStop.mockClear();
  });

  it('unloads all models (selection kept) and does not restart the service meanwhile', async () => {
    const unsub = startModelKeepAliveSync();
    expect(mockStart).toHaveBeenCalledTimes(1);
    mockStart.mockClear();
    let finish!: () => void;
    mockUnloadAll.mockImplementationOnce(() => new Promise(r => { finish = () => r({ textUnloaded: true, imageUnloaded: false }); }));
    stopHandler?.({ reason: 'task_removed' });
    expect(mockUnloadAll).toHaveBeenCalledWith(true);
    modelListener?.(); // model state notifies mid-unload
    expect(mockStart).not.toHaveBeenCalled();
    finish();
    await Promise.resolve();
    unsub();
    expect(mockRemove).toHaveBeenCalled();
  });

  it('re-syncs after a FAILED unload: the still-resident model turns keep-alive back on', async () => {
    const unsub = startModelKeepAliveSync();
    mockStart.mockClear();
    mockUnloadAll.mockImplementationOnce(async () => { throw new Error('unload failed'); });
    stopHandler?.({ reason: 'notification' });
    await flush();
    expect(mockStart).toHaveBeenCalledWith('T');
    unsub();
  });

  it('does not restart the service after a successful unload', async () => {
    const unsub = startModelKeepAliveSync();
    mockStart.mockClear();
    mockUnloadAll.mockImplementationOnce(async () => {
      mockTextLoaded = false;
      return { textUnloaded: true, imageUnloaded: true };
    });
    stopHandler?.({ reason: 'notification' });
    await flush();
    expect(mockStart).not.toHaveBeenCalled();
    unsub();
  });
});
