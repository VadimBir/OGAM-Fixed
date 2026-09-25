/**
 * Projects "a model is resident" onto the Android keep-alive foreground service
 * (ModelKeepAliveService): while a text or image model is loaded (or loading) the process runs at
 * foreground-service priority, so the low-memory killer reclaims cached/background apps first
 * instead of dropping the loaded model. Opt-out via settings.keepModelsAlive (default on).
 * No-op off Android.
 */
import { NativeModules, Platform } from 'react-native';
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

let activeUnsubscribe: (() => void) | null = null;

export function startModelKeepAliveSync(): () => void {
  if (activeUnsubscribe) return activeUnsubscribe;
  if (Platform.OS !== 'android' || !native) return () => {};
  let running: string | null = null;
  const apply = () => {
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
  const unsubModels = activeModelService.subscribe(apply);
  const unsubStore = useAppStore.subscribe(apply);
  apply();
  activeUnsubscribe = () => {
    unsubModels();
    unsubStore();
    native.stop().catch(() => {});
    activeUnsubscribe = null;
  };
  return activeUnsubscribe;
}
