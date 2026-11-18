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

jest.mock('react-native', () => {
  const actual = jest.requireActual('react-native');
  return {
    ...actual,
    Platform: { ...actual.Platform, OS: 'android' },
    NativeEventEmitter: jest.fn().mockImplementation(() => ({ addListener: mockAddListener })),
    NativeModules: {
      ...actual.NativeModules,
      ModelKeepAliveModule: { start: mockStart, stop: mockStop, addListener: jest.fn(), removeListeners: jest.fn() },
    },
  };
});

const mockUnloadAll = jest.fn(async (_keep?: boolean) => ({ textUnloaded: true, imageUnloaded: true }));
let modelListener: (() => void) | undefined;
jest.mock('../../../src/services/activeModelService', () => ({
  activeModelService: {
    getActiveModels: () => ({
      text: { isLoaded: true, isLoading: false, model: { name: 'T' } },
      image: { isLoaded: false, isLoading: false, model: null },
    }),
    subscribe: (l: () => void) => { modelListener = l; return jest.fn(); },
    unloadAllModels: (k?: boolean) => mockUnloadAll(k),
  },
}));

import { startModelKeepAliveSync } from '../../../src/services/modelKeepAlive';

describe('keep-alive stop request', () => {
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
});
