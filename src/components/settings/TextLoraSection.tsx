/**
 * Text LoRA settings, shared by the Model Settings screen and the in-chat Generation Settings modal.
 *
 * Local model: the imported GGUF adapters (enable, scale 0-2, remove, import). Changes apply live to
 * the loaded model (llama.rn applyLoraAdapters); the last llama.cpp error and a CPU-fallback warning
 * are shown. Remote text model: the adapters the user's llama-server lists (GET /lora-adapters) with
 * a per-request scale each.
 */
import React, { useCallback, useEffect, useState, useSyncExternalStore } from 'react';
import { Alert, Switch, Text, TouchableOpacity, View } from 'react-native';
import { pick, types, isErrorWithCode, errorCodes } from '@react-native-documents/picker';
import { SliderSetting } from '../SliderSetting';
import { useTheme, useThemedStyles } from '../../theme';
import type { ThemeColors } from '../../theme';
import { SPACING, TYPOGRAPHY } from '../../constants';
import { useAppStore, useRemoteServerStore } from '../../stores';
import { llmService } from '../../services/llm';
import { remoteServerManager } from '../../services/remoteServerManager';
import { acceptsServerExtensions } from '../../services/remoteTransportPolicy';
import {
  TEXT_LORA_MISMATCH_NOTE,
  getTextLoraStatus,
  subscribeTextLoraStatus,
} from '../../services/textLora';
import {
  clearRemoteTextLoras,
  importPickedTextLora,
  patchTextLora,
  removeTextLora,
  setRemoteTextLoraScale,
} from '../../services/textLoraActions';
import {
  fetchServerLoraAdapters,
  remoteLoraServerKey,
  type ServerLoraAdapter,
} from '../../services/textLoraRemote';
import { TEXT_LORA_SCALE_MAX, TEXT_LORA_SCALE_MIN } from '../../services/textLoraTypes';

const createStyles = (colors: ThemeColors) => ({
  container: { marginBottom: SPACING.lg },
  label: { ...TYPOGRAPHY.body, color: colors.text, marginBottom: SPACING.xs },
  desc: { ...TYPOGRAPHY.bodySmall, color: colors.textSecondary, lineHeight: 18 },
  warn: { ...TYPOGRAPHY.bodySmall, color: colors.warning, lineHeight: 18, marginTop: SPACING.xs },
  error: { ...TYPOGRAPHY.bodySmall, color: colors.error, lineHeight: 18, marginTop: SPACING.xs },
  row: { flexDirection: 'row' as const, alignItems: 'center' as const, justifyContent: 'space-between' as const, marginTop: SPACING.sm },
  name: { ...TYPOGRAPHY.bodySmall, color: colors.text, flex: 1 },
  remove: { ...TYPOGRAPHY.bodySmall, color: colors.error, marginRight: SPACING.md },
  button: {
    marginTop: SPACING.sm, paddingVertical: SPACING.sm, paddingHorizontal: SPACING.md, borderRadius: 8,
    borderWidth: 1, borderColor: colors.border, alignItems: 'center' as const,
  },
  buttonText: { ...TYPOGRAPHY.body, color: colors.primary },
});
type Styles = ReturnType<typeof createStyles>;

const fileName = (path: string): string => path.split('/').pop() || path;

const LocalTextLoras: React.FC<{ styles: Styles; compact: boolean }> = ({ styles, compact }) => {
  const { colors } = useTheme();
  const loras = useAppStore(s => s.settings.textLoras) ?? [];
  const status = useSyncExternalStore(subscribeTextLoraStatus, getTextLoraStatus);
  const [busy, setBusy] = useState(false);

  const run = useCallback(async (work: () => Promise<unknown>) => {
    setBusy(true);
    try { await work(); } finally { setBusy(false); }
  }, []);

  const handleImport = () => run(async () => {
    try {
      const [file] = await pick({ type: [types.allFiles], allowMultiSelection: false });
      if (!file) return;
      const name = (file.name?.trim() || decodeURIComponent(file.uri.split('/').pop() ?? '')).split('/').pop() || 'adapter.gguf';
      if (!name.toLowerCase().endsWith('.gguf')) {
        Alert.alert('Not a GGUF adapter', 'Pick a .gguf LoRA adapter. A .safetensors LoRA must be converted on a PC with llama.cpp\'s convert_lora_to_gguf.py first.');
        return;
      }
      const outcome = await importPickedTextLora({ uri: file.uri, name });
      if (!outcome.handled) {
        Alert.alert('Not a LoRA adapter', 'This GGUF is a model (general.type is not "adapter"). Import models from the Models screen.');
        return;
      }
      Alert.alert(outcome.title ?? 'Text LoRA', outcome.message ?? '');
    } catch (error) {
      if (isErrorWithCode(error) && error.code === errorCodes.OPERATION_CANCELED) return;
      Alert.alert('Import failed', error instanceof Error ? error.message : String(error));
    }
  });

  const modelLoaded = llmService.isModelLoaded();
  const onCpu = modelLoaded && !llmService.getGpuInfo().gpu;
  const anyEnabled = loras.some(l => l.enabled && l.scale > 0);

  return (
    <View style={styles.container} testID="text-lora-section">
      <Text style={styles.label}>Text LoRA</Text>
      <Text style={styles.desc}>
        {loras.length === 0
          ? 'Import a GGUF LoRA adapter (.gguf). Enabled adapters apply to the loaded text model without a reload.'
          : 'Enabled adapters apply to the loaded text model without a reload. Their file size counts against free RAM.'}
      </Text>
      <Text style={styles.desc}>{TEXT_LORA_MISMATCH_NOTE}</Text>
      {loras.map(lora => (
        <View key={lora.id}>
          <View style={styles.row}>
            <Text style={styles.name} numberOfLines={1}>{lora.name}</Text>
            <TouchableOpacity disabled={busy} onPress={() => run(() => removeTextLora(lora.id))} testID={`text-lora-remove-${lora.id}`}>
              <Text style={styles.remove}>Remove</Text>
            </TouchableOpacity>
            <Switch
              testID={`text-lora-enable-${lora.id}`}
              disabled={busy}
              value={lora.enabled}
              onValueChange={v => { run(() => patchTextLora(lora.id, { enabled: v })); }}
              trackColor={{ false: colors.surfaceLight, true: colors.primary }}
              thumbColor={colors.surface}
            />
          </View>
          {lora.enabled && (
            <SliderSetting
              compact={compact}
              testID={`text-lora-scale-${lora.id}`}
              label="Scale"
              value={lora.scale}
              min={TEXT_LORA_SCALE_MIN} max={TEXT_LORA_SCALE_MAX} step={0.05} decimals={2}
              onChange={v => { run(() => patchTextLora(lora.id, { scale: v })); }}
            />
          )}
        </View>
      ))}
      {status.lastError ? <Text style={styles.error} testID="text-lora-error">{status.lastError}</Text> : null}
      {anyEnabled && status.cpuFallback ? (
        <Text style={styles.warn} testID="text-lora-cpu-warning">
          llama.cpp placed some LoRA tensors on the CPU (their base weights sit in a CPU repack buffer). That part of the LoRA math runs on the CPU, not the GPU/NPU.
        </Text>
      ) : null}
      {anyEnabled && onCpu ? (
        <Text style={styles.warn} testID="text-lora-cpu-model-warning">
          The loaded model runs on the CPU, so the LoRA math does too.
        </Text>
      ) : null}
      <TouchableOpacity style={styles.button} disabled={busy} onPress={handleImport} testID="text-lora-import">
        <Text style={styles.buttonText}>{busy ? 'Working...' : 'Import LoRA (.gguf)'}</Text>
      </TouchableOpacity>
    </View>
  );
};

const RemoteTextLoras: React.FC<{ styles: Styles; compact: boolean }> = ({ styles, compact }) => {
  const server = useRemoteServerStore(s => s.servers.find(x => x.id === s.activeServerId) ?? null);
  const saved = useAppStore(s => s.settings.remoteTextLoras);
  const [adapters, setAdapters] = useState<ServerLoraAdapter[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const endpoint = server?.endpoint ?? '';
  const supported = !!endpoint && acceptsServerExtensions(endpoint);

  const load = useCallback(async () => {
    if (!server || !supported) return;
    setError(null);
    try {
      const apiKey = await remoteServerManager.getApiKey(server.id);
      setAdapters(await fetchServerLoraAdapters(server.endpoint, apiKey));
    } catch (e) {
      setAdapters(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [server, supported]);

  useEffect(() => { load(); }, [load]);

  if (!server) return null;
  const overrides = saved?.[remoteLoraServerKey(endpoint)] ?? [];
  return (
    <View style={styles.container} testID="remote-text-lora-section">
      <Text style={styles.label}>Server LoRA</Text>
      {!supported ? (
        <Text style={styles.desc}>LoRA scales are sent only to your own llama-server (LAN, localhost or Tailscale).</Text>
      ) : (
        <>
          <Text style={styles.desc}>
            Adapters loaded on {server.name}. A scale set here is sent with each request; with none set the server's own scales apply.
          </Text>
          {error ? <Text style={styles.error} testID="remote-text-lora-error">{error}</Text> : null}
          {adapters && adapters.length === 0 ? (
            <Text style={styles.desc}>The server lists no LoRA adapters (start llama-server with --lora).</Text>
          ) : null}
          {(adapters ?? []).map(a => {
            const override = overrides.find(o => o.id === a.id);
            return (
              <SliderSetting
                key={a.id}
                compact={compact}
                testID={`remote-text-lora-scale-${a.id}`}
                label={fileName(a.path)}
                value={override?.scale ?? a.scale}
                min={TEXT_LORA_SCALE_MIN} max={TEXT_LORA_SCALE_MAX} step={0.05} decimals={2}
                onChange={v => setRemoteTextLoraScale(endpoint, a.id, v, adapters ?? [])}
              />
            );
          })}
          <TouchableOpacity style={styles.button} onPress={load} testID="remote-text-lora-refresh">
            <Text style={styles.buttonText}>Refresh list</Text>
          </TouchableOpacity>
          {overrides.length > 0 ? (
            <TouchableOpacity style={styles.button} onPress={() => clearRemoteTextLoras(endpoint)} testID="remote-text-lora-clear">
              <Text style={styles.buttonText}>Use server scales</Text>
            </TouchableOpacity>
          ) : null}
        </>
      )}
    </View>
  );
};

export const TextLoraSection: React.FC<{ compact?: boolean }> = ({ compact = false }) => {
  const styles = useThemedStyles(createStyles);
  const isRemote = useRemoteServerStore(s => !!s.activeServerId && !!s.activeRemoteTextModelId);
  return isRemote
    ? <RemoteTextLoras styles={styles} compact={compact} />
    : <LocalTextLoras styles={styles} compact={compact} />;
};
