/**
 * ImageStudioScreen (the Image tab) tests:
 * - Generate calls the image service with the prompt (settings read fresh by the service), never the LLM
 * - reference + Keep 60% sends initImage and denoiseStrength 0.4
 * - chain mode feeds the last result back as the reference
 * - sdcpp disables the reference and never sends initImage
 * - cancel works
 */
import React from 'react';
import { render, fireEvent, act, waitFor } from '@testing-library/react-native';
import RNFS from 'react-native-fs';
import { launchImageLibrary } from 'react-native-image-picker';
import { useAppStore, DEFAULT_SETTINGS } from '../../../src/stores/appStore';
import { imageGenerationService } from '../../../src/services/imageGenerationService';
import type { ImageGenerationState } from '../../../src/services/imageGenerationService';
import { activeModelService } from '../../../src/services/activeModelService';
import { generationService } from '../../../src/services/generationService';
import { resolveMobileImageParameters } from '../../../src/services/imageParameterPolicy';
import { resetStores } from '../../utils/testHelpers';
import { createGeneratedImage, createONNXImageModel } from '../../utils/factories';
import type { ONNXImageModel } from '../../../src/types';

jest.mock('@react-navigation/native', () => {
  const actual = jest.requireActual('@react-navigation/native');
  return {
    ...actual,
    useNavigation: () => ({ navigate: jest.fn(), goBack: jest.fn(), addListener: jest.fn(() => jest.fn()) }),
  };
});

jest.mock('../../../src/components/ModelSelectorModal', () => ({
  ModelSelectorModal: ({ visible, initialTab }: any) => {
    if (!visible) return null;
    const { Text } = require('react-native');
    return <Text testID="model-selector-open">{initialTab}</Text>;
  },
}));

jest.mock('../../../src/components/Button', () => ({
  Button: ({ title, onPress, disabled, testID }: any) => {
    const { TouchableOpacity, Text } = require('react-native');
    return (
      <TouchableOpacity onPress={onPress} disabled={disabled} testID={testID} accessibilityState={{ disabled }}>
        <Text>{title}</Text>
      </TouchableOpacity>
    );
  },
}));

import { ImageStudioScreen } from '../../../src/screens/ImageStudioScreen';

const idleState: ImageGenerationState = {
  phase: 'idle',
  isGenerating: false,
  progress: null,
  status: null,
  previewPath: null,
  prompt: null,
  conversationId: null,
  messageId: null,
  error: null,
  result: null,
};

let emitState: (s: ImageGenerationState) => void = () => {};
let currentState: ImageGenerationState = idleState;

function setupModel(backend: string = 'qnn'): ONNXImageModel {
  const model = { ...createONNXImageModel({ name: 'NPU Model' }), backend } as ONNXImageModel;
  useAppStore.setState({
    downloadedImageModels: [model],
    activeImageModelId: model.id,
  });
  return model;
}

function typePrompt(getByTestId: any, text: string) {
  fireEvent.changeText(getByTestId('studio-prompt'), text);
}

describe('ImageStudioScreen', () => {
  let generateSpy: jest.SpyInstance;
  let cancelSpy: jest.SpyInstance;

  beforeEach(() => {
    resetStores();
    useAppStore.setState({ settings: { ...DEFAULT_SETTINGS }, generatedImages: [] });
    currentState = idleState;
    jest.spyOn(imageGenerationService, 'getState').mockImplementation(() => currentState);
    jest.spyOn(imageGenerationService, 'subscribe').mockImplementation(listener => {
      emitState = s => {
        currentState = s;
        listener(s);
      };
      listener(currentState);
      return () => true;
    });
    generateSpy = jest.spyOn(imageGenerationService, 'generateImage');
    cancelSpy = jest.spyOn(imageGenerationService, 'cancelGeneration').mockResolvedValue(undefined);
    (RNFS.readFile as jest.Mock).mockClear();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('Generate calls the image service with the prompt and settings, never the LLM', async () => {
    const model = setupModel();
    useAppStore.getState().updateSettings({ imageSteps: 12, imageGuidanceScale: 6 });
    const loadText = jest.spyOn(activeModelService, 'loadTextModel');
    const textGen = jest.spyOn(generationService, 'generateResponse');
    const image = createGeneratedImage({ id: 'img-1' });
    generateSpy.mockResolvedValue(image);

    const { getByTestId } = render(<ImageStudioScreen />);
    expect(getByTestId('studio-model-name').props.children).toBe('NPU Model');
    typePrompt(getByTestId, 'a red fox');
    fireEvent.changeText(getByTestId('studio-negative-prompt'), 'blurry');
    await act(async () => {
      fireEvent.press(getByTestId('studio-generate'));
    });

    expect(generateSpy).toHaveBeenCalledTimes(1);
    const params = generateSpy.mock.calls[0][0];
    expect(params).toEqual({ prompt: 'a red fox', negativePrompt: 'blurry', previewInterval: 2 });
    // No conversationId: no chat message. No initImage: plain txt2img.
    expect(params.conversationId).toBeUndefined();
    expect(params.initImage).toBeUndefined();
    // The settings the service resolves for exactly these params are the store's (read fresh).
    const resolved = resolveMobileImageParameters(model, useAppStore.getState().settings, params);
    expect(resolved.steps).toBe(12);
    expect(resolved.guidanceScale).toBe(6);
    expect(loadText).not.toHaveBeenCalled();
    expect(textGen).not.toHaveBeenCalled();
    // The result is recorded for the Studio strip.
    expect(useAppStore.getState().settings.imageStudioImageIds).toEqual(['img-1']);
  });

  it('reference from device + Keep 60% sends initImage and denoiseStrength 0.4', async () => {
    setupModel('qnn');
    (launchImageLibrary as jest.Mock).mockResolvedValueOnce({
      assets: [{ uri: 'file:///mock/ref.jpg', base64: 'UkVGX0JBU0U2NA==', type: 'image/jpeg' }],
    });
    generateSpy.mockResolvedValue(createGeneratedImage({ id: 'img-2' }));

    const { getByTestId } = render(<ImageStudioScreen />);
    // Default keep = 60% (imageDenoiseStrength default 0.4).
    expect(DEFAULT_SETTINGS.imageDenoiseStrength).toBe(0.4);
    expect(getByTestId('studio-keep-value').props.children).toBe('60%');

    await act(async () => {
      fireEvent.press(getByTestId('studio-ref-device'));
    });
    expect(launchImageLibrary).toHaveBeenCalledWith(expect.objectContaining({ includeBase64: true }));
    expect(getByTestId('studio-ref-thumb')).toBeTruthy();

    typePrompt(getByTestId, 'same scene at night');
    await act(async () => {
      fireEvent.press(getByTestId('studio-generate'));
    });
    const params = generateSpy.mock.calls[0][0];
    expect(params.initImage).toBe('UkVGX0JBU0U2NA==');
    expect(params.initImage.startsWith('data:')).toBe(false);
    expect(params.denoiseStrength).toBe(0.4);
  });

  it('Keep slider writes denoiseStrength = 1 - keep to the settings store', () => {
    setupModel('qnn');
    const { getByTestId } = render(<ImageStudioScreen />);
    fireEvent(getByTestId('studio-keep-slider'), 'onSlidingComplete', 75);
    expect(useAppStore.getState().settings.imageDenoiseStrength).toBe(0.25);
  });

  it('chain mode feeds the last result back as the next reference', async () => {
    setupModel('qnn');
    const first = createGeneratedImage({ id: 'chain-1', imagePath: '/mock/generated/chain-1.png' });
    const second = createGeneratedImage({ id: 'chain-2', imagePath: '/mock/generated/chain-2.png' });
    generateSpy.mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    (RNFS.readFile as jest.Mock).mockResolvedValue('Q0hBSU4x');

    const { getByTestId } = render(<ImageStudioScreen />);
    fireEvent.press(getByTestId('studio-chain-checkbox'));
    expect(useAppStore.getState().settings.imageStudioChain).toBe(true);

    typePrompt(getByTestId, 'a castle');
    await act(async () => {
      fireEvent.press(getByTestId('studio-generate'));
    });
    expect(generateSpy.mock.calls[0][0].initImage).toBeUndefined();

    await act(async () => {
      fireEvent.press(getByTestId('studio-generate'));
    });
    expect(RNFS.readFile).toHaveBeenCalledWith('/mock/generated/chain-1.png', 'base64');
    const second_params = generateSpy.mock.calls[1][0];
    expect(second_params.initImage).toBe('Q0hBSU4x');
    expect(second_params.denoiseStrength).toBe(0.4);
    expect(useAppStore.getState().settings.imageStudioImageIds).toEqual(['chain-2', 'chain-1']);
  });

  it('sdcpp disables the reference and never sends initImage', async () => {
    setupModel('qnn');
    const galleryImg = createGeneratedImage({ id: 'g-1', imagePath: '/mock/generated/g-1.png' });
    useAppStore.setState({ generatedImages: [galleryImg] });
    (RNFS.readFile as jest.Mock).mockResolvedValue('R0FMTEVSWQ==');
    generateSpy.mockResolvedValue(createGeneratedImage({ id: 'sd-1' }));

    const { getByTestId, queryByTestId, getByText } = render(<ImageStudioScreen />);
    // Pick a gallery reference while the NPU model is active.
    fireEvent.press(getByTestId('studio-ref-gallery'));
    fireEvent.press(getByTestId('studio-ref-gallery-g-1'));
    expect(getByTestId('studio-ref-thumb')).toBeTruthy();

    // Switch to a stable-diffusion.cpp model.
    act(() => {
      setupModel('sdcpp');
    });
    expect(getByText('Reference images need an NPU model for now')).toBeTruthy();
    expect(getByTestId('studio-ref-device').props.accessibilityState).toEqual(
      expect.objectContaining({ disabled: true }),
    );
    expect(queryByTestId('studio-ref-thumb')).toBeNull();

    typePrompt(getByTestId, 'a lighthouse');
    await act(async () => {
      fireEvent.press(getByTestId('studio-generate'));
    });
    const params = generateSpy.mock.calls[0][0];
    expect(params.initImage).toBeUndefined();
    expect(params.denoiseStrength).toBeUndefined();
    expect(RNFS.readFile).not.toHaveBeenCalled();
  });

  it('cancel works while a Studio generation is in flight', async () => {
    setupModel('qnn');
    const { getByTestId, queryByTestId } = render(<ImageStudioScreen />);
    act(() => {
      emitState({
        ...idleState,
        phase: 'generating',
        isGenerating: true,
        status: 'Generating image (3/20)...',
        progress: { step: 3, totalSteps: 20 },
      });
    });
    expect(getByTestId('studio-progress')).toBeTruthy();
    expect(queryByTestId('studio-generate')).toBeNull();
    await act(async () => {
      fireEvent.press(getByTestId('studio-cancel'));
    });
    expect(cancelSpy).toHaveBeenCalledTimes(1);
    act(() => {
      emitState(idleState);
    });
    await waitFor(() => expect(getByTestId('studio-generate')).toBeTruthy());
  });

  it('model row opens the existing image model picker on the image tab', () => {
    setupModel('qnn');
    const { getByTestId } = render(<ImageStudioScreen />);
    fireEvent.press(getByTestId('studio-model-picker'));
    expect(getByTestId('model-selector-open').props.children).toBe('image');
  });
});
