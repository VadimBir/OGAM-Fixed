import { renderHook, act } from '@testing-library/react-native';
import { resetStores } from '../../utils/testHelpers';
import { useAppStore } from '../../../src/stores/appStore';
import { useTextGenerationAdvanced } from '../../../src/hooks/useTextGenerationAdvanced';

describe('useTextGenerationAdvanced', () => {
  beforeEach(() => {
    resetStores();
  });

  // No backend coerces the KV cache to f16 any more (llmHelpers.backendForcesF16Cache returns false):
  // the user's cacheType is honored on HTP too, so the picker stays enabled and shows their choice.
  it('keeps the chosen KV cache type when HTP backend is selected', () => {
    act(() => {
      useAppStore.getState().updateSettings({ inferenceBackend: 'htp', cacheType: 'q4_0' });
    });

    const { result } = renderHook(() => useTextGenerationAdvanced());

    expect(result.current.gpuForcesF16).toBe(false);
    expect(result.current.cacheDisabled).toBe(false);
    expect(result.current.displayCacheType).toBe('q4_0');
  });

  it('shows Auto (N) for cpu threads when nThreads uses the auto sentinel', async () => {
    act(() => {
      useAppStore.getState().updateSettings({ nThreads: 0 });
    });

    const { result } = renderHook(() => useTextGenerationAdvanced());

    await act(async () => {});

    expect(result.current.cpuThreadsDisplayValue).toMatch(/^Auto \(\d+\)$/);
    expect(result.current.cpuThreadsSliderValue).toBe(1);
  });
});
