/**
 * Fast mode section: per active model - sd.cpp SD 1.x/SDXL shows Off/4/2/1 (download size until the
 * LoRA is on the phone) and the TAESD switch; another sd.cpp family says why there is no preset; an
 * NPU few-step model (by name) gets its choices; any other NPU model gets the one-line hint.
 */
import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { useAppStore } from '../../../src/stores/appStore';
import { resetStores } from '../../utils/testHelpers';

jest.mock('../../../src/services', () => ({
  backgroundDownloadService: {
    isAvailable: () => true,
    startDownload: jest.fn(),
    moveCompletedDownload: jest.fn(),
    cancelDownload: jest.fn(),
    onComplete: () => () => {},
    onError: () => () => {},
    startProgressPolling: jest.fn(),
  },
}));
jest.mock('../../../src/services/sdCppGenerator', () => ({
  sdCppGeneratorService: { isAvailable: () => true },
}));
jest.mock('../../../src/services/civitaiDownload', () => ({ finalizeSdCppFileDownload: jest.fn() }));

import { FastModeSection, NPU_FAST_MODE_HINT } from '../../../src/components/FastModeSection';

function setModel(backend: string, name: string, sdFamily?: string) {
  useAppStore.setState({
    downloadedImageModels: [{ id: 'm', name, description: '', modelPath: '/m', downloadedAt: '', size: 2e9, backend, sdFamily } as any],
    activeImageModelId: 'm',
  });
  useAppStore.getState().updateSettings({ imageSteps: 20, imageGuidanceScale: 7.5, imageSampler: '', imageLoras: [], imageFastMode: null });
}

describe('FastModeSection', () => {
  beforeEach(() => resetStores());

  it('no active image model: renders nothing', () => {
    useAppStore.setState({ downloadedImageModels: [], activeImageModelId: null });
    expect(render(<FastModeSection />).queryByTestId('fast-mode-section')).toBeNull();
  });

  it('sd.cpp SD 1.x: Off/4/2/1 with download sizes + TAESD switch', () => {
    setModel('sdcpp', 'DreamShaper', 'sd1');
    const view = render(<FastModeSection />);
    expect(view.getByText('4 steps (269MB)')).toBeTruthy();
    expect(view.getByText('1 step (269MB)')).toBeTruthy();
    expect(view.getByTestId('fast-mode-off')).toBeTruthy();
    expect(view.getByTestId('fast-mode-taesd').props.value).toBe(false);
  });

  it('sd.cpp SDXL with the 2-step LoRA on the phone: no size on that chip', () => {
    setModel('sdcpp', 'Juggernaut', 'sdxl');
    useAppStore.getState().updateSettings({ imageLoras: [
      { id: 'l', name: 'sdxl_lightning_2step_lora', path: '/d/image_loras/sdxl_lightning_2step_lora.safetensors', weight: 1, enabled: false },
    ] });
    const view = render(<FastModeSection />);
    expect(view.getByText('2 steps')).toBeTruthy();
    expect(view.getByText('4 steps (394MB)')).toBeTruthy();
    expect(view.getByText('1 step (787MB)')).toBeTruthy();
  });

  it('sd.cpp SD 2.x: no LoRA preset, TAESD still offered', () => {
    setModel('sdcpp', 'SD21', 'sd2');
    const view = render(<FastModeSection />);
    expect(view.getByTestId('fast-mode-unsupported')).toBeTruthy();
    expect(view.queryByTestId('fast-mode-4')).toBeNull();
    expect(view.getByTestId('fast-mode-taesd')).toBeTruthy();
  });

  it('NPU model without a few-step name: the one-line hint, no TAESD', () => {
    setModel('qnn', 'AnythingV5');
    const view = render(<FastModeSection />);
    expect(view.getByTestId('fast-mode-npu-hint').props.children).toBe(NPU_FAST_MODE_HINT);
    expect(NPU_FAST_MODE_HINT).toBe('Fast mode on NPU needs a few-step model (DMD2 / Hyper) from the catalog.');
    expect(view.queryByTestId('fast-mode-taesd')).toBeNull();
  });

  it('NPU few-step model: its choices only; tapping applies the preset, Off restores', () => {
    setModel('qnn', 'HyperSpireV5');
    const view = render(<FastModeSection />);
    expect(view.queryByTestId('fast-mode-1')).toBeNull();
    fireEvent.press(view.getByTestId('fast-mode-2'));
    expect(useAppStore.getState().settings).toMatchObject({ imageSteps: 2, imageGuidanceScale: 1, imageSampler: 'euler' });
    fireEvent.press(view.getByTestId('fast-mode-off'));
    expect(useAppStore.getState().settings).toMatchObject({ imageSteps: 20, imageGuidanceScale: 7.5, imageSampler: '', imageFastMode: null });
  });
});
