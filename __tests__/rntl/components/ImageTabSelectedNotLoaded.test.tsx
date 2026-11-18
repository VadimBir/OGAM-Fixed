/**
 * H1-B3: after the user unloads the image model its selection stays, and the sheet must SHOW that
 * it is not in RAM: "Selected, not loaded", no Unload button, and the row stays tappable to load it.
 * A loaded model keeps "Currently Loaded" + Unload.
 */
import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { ImageTab } from '../../../src/components/ModelSelectorModal/ImageTab';
import { createONNXImageModel } from '../../utils/factories';

const model = createONNXImageModel({ id: 'sd', name: 'Dream SD' });

function renderTab(isImageModelLoaded: boolean) {
  const onSelectImageModel = jest.fn();
  const onUnloadImageModel = jest.fn();
  const utils = render(
    <ImageTab
      downloadedImageModels={[model]}
      remoteVisionModels={[]}
      activeImageModelId="sd"
      activeRemoteImageModelId={null}
      activeRemoteImageServerId={null}
      isAnyLoading={false}
      isLoadingImage={false}
      isImageModelLoaded={isImageModelLoaded}
      onSelectImageModel={onSelectImageModel}
      onSelectRemoteVisionModel={jest.fn()}
      onUnloadImageModel={onUnloadImageModel}
    />,
  );
  return { ...utils, onSelectImageModel, onUnloadImageModel };
}

describe('ImageTab — selected vs loaded', () => {
  it('loaded: "Currently Loaded" with an Unload button', () => {
    const { getByText, queryByText, queryByTestId, onUnloadImageModel } = renderTab(true);
    expect(getByText('Currently Loaded')).toBeTruthy();
    expect(queryByText('Selected, not loaded')).toBeNull();
    expect(queryByTestId('image-model-selected-not-loaded')).toBeNull();
    fireEvent.press(getByText('Unload'));
    expect(onUnloadImageModel).toHaveBeenCalledTimes(1);
  });

  it('unloaded but selected: "Selected, not loaded", no Unload, row tappable to load', () => {
    const { getByText, queryByText, getByTestId, onSelectImageModel } = renderTab(false);
    expect(getByText('Selected, not loaded')).toBeTruthy();
    expect(getByTestId('image-model-selected-not-loaded')).toBeTruthy();
    expect(queryByText('Currently Loaded')).toBeNull();
    expect(queryByText('Unload')).toBeNull();
    // the selection is still shown (generate_image stays available)
    expect(getByTestId('currently-loaded-image-model-name')).toBeTruthy();
    fireEvent.press(getByTestId('image-model-row-sd'));
    expect(onSelectImageModel).toHaveBeenCalledWith(model);
  });

  it('loaded: the selected row is disabled (nothing to do)', () => {
    const { getByTestId, onSelectImageModel } = renderTab(true);
    fireEvent.press(getByTestId('image-model-row-sd'));
    expect(onSelectImageModel).not.toHaveBeenCalled();
  });
});
