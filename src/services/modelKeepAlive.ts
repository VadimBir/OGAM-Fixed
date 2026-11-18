/**
 * Projects "a model is resident" onto the Android keep-alive foreground service
 * (ModelKeepAliveService): while a text or image model is loaded (or loading) the process runs at
 * foreground-service priority, so the low-memory killer reclaims cached/background apps first
 * instead of dropping the loaded model. Opt-out via settings.keepModelsAlive (default on).
 * No-op off Android.
 */
import { NativeModules, NativeEventEmitter, Platform } from 'react-native';
import { useAppStore } from '../stores';
import { activeModelService } from './activeModelService';
import type { ActiveModelInfo } from './activeModelService/types';
import logger from '../utils/logger';

const native = NativeModules.ModelKeepAliveModule as
  | { start(label: string): Promise<boolean>; stop(): Promise<void> }
  | undefined;

export function keepAliveLabel(info: ActiveModelInfo): string | null {
  const names = [
    info.text.isLoaded || info.text.isLoading ? info.text.model?.name : null,
    info.image.isLoaded || info.image.isLoading ? info.image.model?.name : null,
  ].filter((n): n is string => !!n);
  return names.length ? names.join(' + ') : null;
}

/** Native event (ModelKeepAliveModule.EVENT_STOP_REQUESTED) {reason: "notification" | "task_removed"}. */
export const KEEPALIVE_STOP_EVENT = 'ModelKeepAliveStopRequested';

let activeUnsubscribe: (() => void) | null = null;

export function startModelKeepAliveSync(): () => void {
  if (activeUnsubscribe) return activeUnsubscribe;
  if (Platform.OS !== 'android' || !native) return () => {};
  let running: string | null = null;
  // True while the models are being unloaded after a native stop request: apply() must not
  // restart the service the user just stopped (models still read "loaded" until unload ends).
  let stopping = false;
  const apply = () => {
    if (stopping) return;
    const enabled = useAppStore.getState().settings?.keepModelsAlive ?? true;
    const label = enabled ? keepAliveLabel(activeModelService.getActiveModels()) : null;
    if (label === running) return;
    running = label;
    if (label) {
      native.start(label)
        .then(ok => { if (!ok) { running = null; logger.warn('[KEEPALIVE] start denied (app not in foreground?)'); } })
        .catch(e => { running = null; logger.warn('[KEEPALIVE] start failed', e); });
    } else {
      native.stop().catch(() => {});
    }
    logger.log(`[KEEPALIVE] ${label ? `on: ${label}` : 'off'}`);
  };
  // "I cannot stop it": notification Stop / swipe-away stop the native service; the models must
  // be released too or the process keeps the RAM. activeModelService does not import this
  // module, so there is no import cycle and a direct call is safe.
  const stopSub = new NativeEventEmitter(NativeModules.ModelKeepAliveModule).addListener(
    KEEPALIVE_STOP_EVENT,
    (event?: { reason?: string }) => {
      logger.log(`[KEEPALIVE] stop requested (${event?.reason ?? 'unknown'}): unloading all models`);
      running = null;
      stopping = true;
      activeModelService.unloadAllModels(true)
        .catch(e => logger.warn('[KEEPALIVE] unload after stop request failed', e))
        .finally(() => { stopping = false; });
    },
  );
  const unsubModels = activeModelService.subscribe(apply);
  const unsubStore = useAppStore.subscribe(apply);
  apply();
  activeUnsubscribe = () => {
    stopSub.remove();
    unsubModels();
    unsubStore();
    native.stop().catch(() => {});
    activeUnsubscribe = null;
  };
  return activeUnsubscribe;
}
