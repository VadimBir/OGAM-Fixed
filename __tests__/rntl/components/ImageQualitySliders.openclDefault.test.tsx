/**
 * GPU Acceleration (OpenCL) switch: an unset imageUseOpenCL reads OFF (store default is false,
 * appStore.ts), so the Clear GPU Cache button is hidden until the user turns it on.
 */
import React from 'react';
import { Platform } from 'react-native';
import { render } from '@testing-library/react-native';
import { ImageQualityAdvancedSliders } from '../../../src/components/GenerationSettingsModal/ImageQualitySliders';
import { useAppStore } from '../../../src/stores/appStore';
import { resetStores } from '../../utils/testHelpers';

describe('ImageQualityAdvancedSliders GPU Acceleration default', () => {
  let originalOS: string;
  beforeEach(() => {
    resetStores();
    originalOS = Platform.OS;
    Object.defineProperty(Platform, 'OS', { get: () => 'android', configurable: true });
  });
  afterEach(() => {
    Object.defineProperty(Platform, 'OS', { get: () => originalOS, configurable: true });
  });

  it('unset imageUseOpenCL -> switch OFF, no Clear GPU Cache button', () => {
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, imageUseOpenCL: undefined as any } });
    const view = render(<ImageQualityAdvancedSliders />);
    expect(view.getByTestId('image-gpu-acceleration').props.value).toBe(false);
    expect(view.queryByText('Clear GPU Cache')).toBeNull();
  });

  it('imageUseOpenCL true -> switch ON and the cache button shows', () => {
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, imageUseOpenCL: true } });
    const view = render(<ImageQualityAdvancedSliders />);
    expect(view.getByTestId('image-gpu-acceleration').props.value).toBe(true);
    expect(view.queryByText('Clear GPU Cache')).not.toBeNull();
  });
});
