import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, Image, Switch, StyleSheet } from 'react-native';
import Icon from 'react-native-vector-icons/Feather';
import { LoadingDots } from '../../components/LoadingDots';
import { useTheme, useThemedStyles } from '../../theme';
import type { ThemeColors } from '../../theme';
import { TYPOGRAPHY, SPACING } from '../../constants';
import { useDownloadStore, isActiveStatus, isFailedStatus } from '../../stores/downloadStore';
import { makeImageModelKey } from '../../utils/modelKey';
import { formatBytes } from '../../utils/formatBytes';
import type { AlertState } from '../../components/CustomAlert';
import type { CivitaiFamily, CivitaiModelType, CivitaiResult } from '../../services/civitaiBrowser';
import { civitaiImageModelId } from '../../services/civitaiDownload';
import { createStyles } from './styles';
import { useCivitaiSearch, type CivitaiSearchViewModel } from './useCivitaiSearch';

const TYPE_OPTIONS: { key: CivitaiModelType; label: string }[] = [
  { key: 'Checkpoint', label: 'Checkpoints' },
  { key: 'LORA', label: 'LoRAs' },
];
const FAMILY_OPTIONS: { key: CivitaiFamily; label: string }[] = [
  { key: 'sd1', label: 'SD 1.5' },
  { key: 'sd2', label: 'SD 2.x' },
  { key: 'sdxl', label: 'SDXL' },
];

/** The PC server API has no model-download route (offGridDesktopModels.ts calls only catalog /
 *  installed / active / activate), so the server target is shown disabled with this reason. */
export const SERVER_DOWNLOAD_UNAVAILABLE =
  'Download on server: not available yet. Your PC server has no model-download API.';

const createLocalStyles = (colors: ThemeColors) => ({
  note: { ...TYPOGRAPHY.meta, color: colors.textSecondary, marginBottom: SPACING.xs },
  row: { flexDirection: 'row' as const, alignItems: 'center' as const, gap: SPACING.sm, marginBottom: SPACING.sm },
  chipRow: { flexDirection: 'row' as const, flexWrap: 'wrap' as const, gap: SPACING.xs, marginBottom: SPACING.sm },
  label: { ...TYPOGRAPHY.bodySmall, color: colors.text, flex: 1 },
  smallButton: { paddingHorizontal: 10, paddingVertical: 6, borderRadius: 8, backgroundColor: colors.surface },
  smallButtonText: { ...TYPOGRAPHY.meta, color: colors.primary },
  card: {
    flexDirection: 'row' as const, gap: SPACING.sm, padding: SPACING.sm, marginBottom: SPACING.sm,
    borderRadius: 12, borderWidth: 1, borderColor: colors.border, backgroundColor: colors.surface,
  },
  thumb: { width: 72, height: 96, borderRadius: 8, backgroundColor: colors.border },
  thumbEmpty: { alignItems: 'center' as const, justifyContent: 'center' as const },
  cardBody: { flex: 1 },
  title: { ...TYPOGRAPHY.body, color: colors.text },
  meta: { ...TYPOGRAPHY.meta, color: colors.textMuted, marginTop: 2 },
  actions: { flexDirection: 'row' as const, flexWrap: 'wrap' as const, gap: SPACING.xs, marginTop: SPACING.xs },
  action: { paddingHorizontal: 10, paddingVertical: 6, borderRadius: 8, backgroundColor: colors.primary },
  actionText: { ...TYPOGRAPHY.meta, color: colors.background },
  actionMuted: { backgroundColor: colors.border },
  actionMutedText: { color: colors.textMuted },
  input: {
    ...TYPOGRAPHY.bodySmall, flex: 1, color: colors.text, paddingHorizontal: 10, paddingVertical: 6,
    borderRadius: 8, borderWidth: 1, borderColor: colors.border,
  },
});

type LocalStyles = ReturnType<typeof createLocalStyles>;

function Chips<K extends string>({ options, value, onChange, testPrefix }: {
  options: { key: K; label: string }[]; value: K; onChange: (k: K) => void; testPrefix: string;
}) {
  const styles = useThemedStyles(createStyles);
  return (
    <>
      {options.map(o => (
        <TouchableOpacity
          key={o.key}
          testID={`${testPrefix}-${o.key}`}
          style={[styles.filterChip, value === o.key && styles.filterChipActive]}
          onPress={() => onChange(o.key)}
        >
          <Text style={[styles.filterChipText, value === o.key && styles.filterChipTextActive]}>{o.label}</Text>
        </TouchableOpacity>
      ))}
    </>
  );
}

const PhoneAction: React.FC<{ result: CivitaiResult; index: number; vm: CivitaiSearchViewModel; local: LocalStyles }> = ({
  result, index, vm, local,
}) => {
  const entry = useDownloadStore(s => s.downloads[makeImageModelKey(civitaiImageModelId(result))]);
  if (vm.isOnPhone(result)) {
    return (
      <View style={[local.action, local.actionMuted]} testID={`civitai-onphone-${index}`}>
        <Text style={[local.actionText, local.actionMutedText]}>On phone</Text>
      </View>
    );
  }
  if (entry && (isActiveStatus(entry.status) || entry.status === 'paused')) {
    const pct = Math.round((entry.progress ?? 0) * 100);
    const label = entry.status === 'processing' ? 'Installing...' : entry.status === 'paused' ? 'Paused' : `Downloading ${pct}%`;
    return (
      <>
        <View style={[local.action, local.actionMuted]} testID={`civitai-progress-${index}`}>
          <Text style={[local.actionText, local.actionMutedText]}>{label}</Text>
        </View>
        {entry.status !== 'processing' && (
          <TouchableOpacity style={local.action} onPress={() => vm.cancelDownload(result)} testID={`civitai-cancel-${index}`}>
            <Text style={local.actionText}>Cancel</Text>
          </TouchableOpacity>
        )}
      </>
    );
  }
  const failed = !!entry && isFailedStatus(entry.status);
  return (
    <TouchableOpacity style={local.action} onPress={() => vm.downloadToPhone(result)} testID={`civitai-download-phone-${index}`}>
      <Text style={local.actionText}>{failed ? 'Retry download to phone' : 'Download to phone'}</Text>
    </TouchableOpacity>
  );
};

const ResultCard: React.FC<{ result: CivitaiResult; index: number; vm: CivitaiSearchViewModel; local: LocalStyles }> = ({
  result, index, vm, local,
}) => {
  const { colors } = useTheme();
  const kind = result.type === 'LORA' ? 'LoRA' : 'Checkpoint';
  const fp = result.file.fp ? ` · ${result.file.fp}` : '';
  return (
    <View style={local.card} testID={`civitai-result-${index}`}>
      {result.previewUrl
        ? <Image source={{ uri: result.previewUrl }} style={local.thumb} resizeMode="cover" testID={`civitai-thumb-${index}`} />
        : <View style={[local.thumb, local.thumbEmpty]}><Icon name="image" size={20} color={colors.textMuted} /></View>}
      <View style={local.cardBody}>
        <Text style={local.title} numberOfLines={2}>{result.name}{result.versionName ? ` · ${result.versionName}` : ''}</Text>
        <Text style={local.meta}>{result.baseModel} {'·'} {formatBytes(result.file.sizeBytes)} {'·'} {kind}{fp}</Text>
        {result.creator ? <Text style={local.meta}>by {result.creator}</Text> : null}
        <View style={local.actions}>
          <PhoneAction result={result} index={index} vm={vm} local={local} />
          <View
            style={[local.action, local.actionMuted]}
            accessibilityState={{ disabled: true }}
            accessibilityHint={SERVER_DOWNLOAD_UNAVAILABLE}
            testID={`civitai-download-server-${index}`}
          >
            <Text style={[local.actionText, local.actionMutedText]}>Download on server</Text>
          </View>
        </View>
      </View>
    </View>
  );
};

const ApiKeyRow: React.FC<{ vm: CivitaiSearchViewModel; local: LocalStyles }> = ({ vm, local }) => {
  const { colors } = useTheme();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  if (editing) {
    return (
      <View style={local.row}>
        <TextInput
          style={local.input}
          value={draft}
          onChangeText={setDraft}
          placeholder="Civitai API key"
          placeholderTextColor={colors.textMuted}
          secureTextEntry
          autoCapitalize="none"
          autoCorrect={false}
          testID="civitai-apikey-input"
        />
        <TouchableOpacity
          style={local.smallButton}
          testID="civitai-apikey-save"
          onPress={async () => { await vm.saveApiKey(draft); setDraft(''); setEditing(false); }}
        >
          <Text style={local.smallButtonText}>Save</Text>
        </TouchableOpacity>
      </View>
    );
  }
  return (
    <View style={local.row}>
      <Text style={local.label}>Civitai API key (optional): {vm.hasApiKey ? 'saved in Keychain' : 'not set'}</Text>
      <TouchableOpacity style={local.smallButton} onPress={() => setEditing(true)} testID="civitai-apikey-edit">
        <Text style={local.smallButtonText}>{vm.hasApiKey ? 'Change' : 'Set'}</Text>
      </TouchableOpacity>
      {vm.hasApiKey && (
        <TouchableOpacity style={local.smallButton} onPress={vm.clearApiKey} testID="civitai-apikey-remove">
          <Text style={local.smallButtonText}>Remove</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

export const CivitaiSection: React.FC<{ setAlertState: (s: AlertState) => void }> = ({ setAlertState }) => {
  const vm = useCivitaiSearch(setAlertState);
  const { colors } = useTheme();
  const styles = useThemedStyles(createStyles);
  const local = React.useMemo(() => StyleSheet.create(createLocalStyles(colors)), [colors]);

  return (
    <ScrollView keyboardShouldPersistTaps="handled" testID="civitai-section">
      <View style={styles.imageModelsList}>
        <Text style={local.note}>
          Civitai models run on stable-diffusion.cpp on the GPU (Vulkan) in this build. Devices without a Vulkan GPU cannot run them.
        </Text>
        <Text style={local.note}>{SERVER_DOWNLOAD_UNAVAILABLE}</Text>

        <View style={[styles.searchContainer, styles.searchContainerNoPadding]}>
          <TextInput
            style={styles.searchInput}
            placeholder="Search Civitai..."
            placeholderTextColor={colors.textMuted}
            value={vm.query}
            onChangeText={vm.setQuery}
            onSubmitEditing={vm.search}
            returnKeyType="search"
            testID="civitai-search-input"
          />
          <TouchableOpacity style={styles.filterToggle} onPress={vm.search} testID="civitai-search-button">
            <Icon name="search" size={14} color={colors.primary} />
          </TouchableOpacity>
        </View>

        <View style={local.chipRow}>
          <Chips options={TYPE_OPTIONS} value={vm.type} onChange={vm.setType} testPrefix="civitai-type" />
        </View>
        <View style={local.chipRow}>
          <Chips options={FAMILY_OPTIONS} value={vm.family} onChange={vm.setFamily} testPrefix="civitai-family" />
        </View>

        <View style={local.row}>
          <Text style={local.label}>Show NSFW models</Text>
          <Switch value={vm.showNsfw} onValueChange={vm.setShowNsfw} testID="civitai-nsfw-switch" />
        </View>
        <ApiKeyRow vm={vm} local={local} />

        {vm.loading && (
          <View style={styles.hfLoadingContainer}>
            <LoadingDots color={colors.primary} />
            <Text style={styles.loadingText}>Searching Civitai...</Text>
          </View>
        )}

        {vm.error && !vm.loading && (
          <View style={styles.hfErrorContainer}>
            <Text style={styles.hfErrorText}>{vm.error}</Text>
            <TouchableOpacity style={styles.retryButton} onPress={vm.search} testID="civitai-retry">
              <Text style={styles.retryButtonText}>Retry</Text>
            </TouchableOpacity>
          </View>
        )}

        {!vm.loading && vm.results.map((r, i) => <ResultCard key={r.key} result={r} index={i} vm={vm} local={local} />)}

        {!vm.loading && !vm.error && vm.hasSearched && vm.results.length === 0 && (
          <Text style={styles.allDownloadedText} testID="civitai-empty">No runnable models found. Try another search or base model.</Text>
        )}
        {!vm.loading && vm.hiddenCount > 0 && (
          <Text style={local.note} testID="civitai-hidden-count">
            {vm.hiddenCount} hidden: no SafeTensor file, another base model, NSFW, or too large for this phone's RAM.
          </Text>
        )}
        {!vm.loading && vm.nextCursor && (
          <TouchableOpacity style={styles.retryButton} onPress={vm.loadMore} disabled={vm.loadingMore} testID="civitai-load-more">
            <Text style={styles.retryButtonText}>{vm.loadingMore ? 'Loading...' : 'Load more'}</Text>
          </TouchableOpacity>
        )}
      </View>
    </ScrollView>
  );
};
