import React from 'react';
import { Platform } from 'react-native';
import { render, fireEvent } from '@testing-library/react-native';
import { ImageModelsTab } from '../../../src/screens/ModelsScreen/ImageModelsTab';

jest.mock('../../../src/screens/ModelsScreen/CivitaiSection', () => {
  const { Text } = require('react-native');
  return { CivitaiSection: () => <Text testID="civitai-section">civitai</Text> };
});

const noop = jest.fn();
const baseProps: any = {
  imageSearchQuery: '', setImageSearchQuery: noop,
  hfModelsLoading: false, hfModelsError: null,
  filteredHFModels: [], availableHFModels: [{ id: 'x', name: 'X', displayName: 'X', backend: 'mnn', fileName: 'x.zip', downloadUrl: 'u', size: 1, repo: 'r' }],
  backendFilter: 'all', setBackendFilter: noop, styleFilter: 'all', setStyleFilter: noop,
  sdVersionFilter: 'all', setSdVersionFilter: noop, imageFilterExpanded: null, setImageFilterExpanded: noop,
  imageFiltersVisible: false, setImageFiltersVisible: noop, hasActiveImageFilters: false,
  showRecommendedOnly: false, setShowRecommendedOnly: noop, showRecHint: false, setShowRecHint: noop,
  imageRec: null, ramGB: 12, imageRecommendation: 'ok',
  handleDownloadImageModel: noop, handleCancelImageDownload: noop, loadHFModels: noop,
  clearImageFilters: noop, setUserChangedBackendFilter: noop, isRecommendedModel: () => false,
};

const originalOS = Platform.OS;
afterEach(() => { (Platform as any).OS = originalOS; });

describe('ImageModelsTab source switch', () => {
  it('offers Civitai on Android and switches to it', () => {
    (Platform as any).OS = 'android';
    const { getByTestId, queryByTestId } = render(<ImageModelsTab {...baseProps} setAlertState={noop} />);
    expect(queryByTestId('civitai-section')).toBeNull();
    fireEvent.press(getByTestId('image-source-civitai'));
    expect(getByTestId('civitai-section')).toBeTruthy();
    fireEvent.press(getByTestId('image-source-catalog'));
    expect(queryByTestId('civitai-section')).toBeNull();
  });

  it('has no Civitai switch on iOS (no stable-diffusion.cpp there)', () => {
    (Platform as any).OS = 'ios';
    const { queryByTestId } = render(<ImageModelsTab {...baseProps} setAlertState={noop} />);
    expect(queryByTestId('image-source-civitai')).toBeNull();
  });

  it('says a searched model is already downloaded instead of "no match"', () => {
    const { getByText } = render(<ImageModelsTab {...baseProps} imageSearchQuery="x" downloadedSearchMatches={1} />);
    expect(getByText('The matching model is already downloaded')).toBeTruthy();
  });
});
