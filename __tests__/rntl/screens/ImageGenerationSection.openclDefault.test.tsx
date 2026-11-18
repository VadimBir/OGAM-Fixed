/**
 * Model Settings > Image > Advanced > "OpenCL GPU Acceleration": an unset imageUseOpenCL must
 * read OFF, matching the store default (appStore DEFAULT_SETTINGS.imageUseOpenCL = false), the
 * engine (imageGenerationService `?? false`) and the chat-modal switch (ImageQualitySliders).
 * The section used `?? true`, so the screen showed ON while generation ran without OpenCL.
 */
import React from 'react';
import { Platform, Switch } from 'react-native';
import { render, fireEvent } from '@testing-library/react-native';
import { ImageGenerationSection } from '../../../src/screens/ModelSettingsScreen/ImageGenerationSection';
import { useAppStore, DEFAULT_SETTINGS } from '../../../src/stores/appStore';
import { resetStores } from '../../utils/testHelpers';

function openclSwitch(view: ReturnType<typeof render>) {
  fireEvent.press(view.getByTestId('image-advanced-toggle'));
  const label = view.getByText('OpenCL GPU Acceleration');
  // The Switch is the sibling of the label's container in the same toggle row.
  const switches = view.UNSAFE_getAllByType(Switch);
  expect(label).toBeTruthy();
  return switches[switches.length - 1];
}

describe('ImageGenerationSection OpenCL default', () => {
  let originalOS: string;
  beforeEach(() => {
    resetStores();
    originalOS = Platform.OS;
    Object.defineProperty(Platform, 'OS', { get: () => 'android', configurable: true });
  });
  afterEach(() => {
    Object.defineProperty(Platform, 'OS', { get: () => originalOS, configurable: true });
  });

  it('store default is OFF', () => {
    expect(DEFAULT_SETTINGS.imageUseOpenCL).toBe(false);
  });

  it('unset imageUseOpenCL -> switch OFF, no Clear GPU Cache row', () => {
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, imageUseOpenCL: undefined as any } });
    const view = render(<ImageGenerationSection />);
    expect(openclSwitch(view).props.value).toBe(false);
    expect(view.queryByText('Clear GPU Cache')).toBeNull();
  });

  it('imageUseOpenCL true -> switch ON and the Clear GPU Cache row shows', () => {
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, imageUseOpenCL: true } });
    const view = render(<ImageGenerationSection />);
    expect(openclSwitch(view).props.value).toBe(true);
    expect(view.queryByText('Clear GPU Cache')).not.toBeNull();
  });
});
