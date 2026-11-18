import React from 'react';
import { TextInput, View } from 'react-native';
import { fireEvent, render, screen } from '@testing-library/react-native';
import {
  FloatingGeneratedImage,
  resetFloatingGeneratedImageDismissals,
} from '../../../../src/screens/ChatScreen/FloatingGeneratedImage';
import type { ImageGenerationState } from '../../../../src/services/imageGenerationTypes';

function doneState(overrides: Partial<ImageGenerationState> = {}): ImageGenerationState {
  return {
    phase: 'done',
    isGenerating: false,
    progress: null,
    status: null,
    previewPath: null,
    prompt: 'a fox',
    conversationId: 'conv-1',
    messageId: 'm-1',
    error: null,
    result: {
      id: 'img-1',
      prompt: 'a fox',
      imagePath: '/data/generated_images/img-1.png',
      width: 512,
      height: 768,
      steps: 4,
      seed: 1,
      modelId: 'sd-turbo',
      createdAt: '2026-11-18T00:00:00.000Z',
    },
    ...overrides,
  };
}

describe('FloatingGeneratedImage', () => {
  beforeEach(() => resetFloatingGeneratedImageDismissals());

  it('floats the finished image of this chat and opens the full-screen viewer on the enlarged card', () => {
    const onOpen = jest.fn();
    render(
      <FloatingGeneratedImage
        imageGenState={doneState()}
        conversationId="conv-1"
        bottom={92}
        onOpenFullscreen={onOpen}
      />,
    );
    expect(screen.getByTestId('floating-generated-image')).toBeTruthy();
    // First tap enlarges in place (no modal), second opens the existing full-screen viewer.
    fireEvent.press(screen.getByTestId('floating-generated-image-thumb'));
    expect(onOpen).not.toHaveBeenCalled();
    expect(screen.getByTestId('floating-generated-image-collapse')).toBeTruthy();
    fireEvent.press(screen.getByTestId('floating-generated-image-thumb'));
    expect(onOpen).toHaveBeenCalledWith('file:///data/generated_images/img-1.png');
  });

  it('keeps the composer usable while the card is open', () => {
    const onChangeText = jest.fn();
    render(
      <View>
        <FloatingGeneratedImage
          imageGenState={doneState()}
          conversationId="conv-1"
          bottom={92}
          onOpenFullscreen={jest.fn()}
        />
        <TextInput testID="composer" onChangeText={onChangeText} />
      </View>,
    );
    fireEvent.press(screen.getByTestId('floating-generated-image-thumb'));
    fireEvent.changeText(screen.getByTestId('composer'), 'next message');
    expect(onChangeText).toHaveBeenCalledWith('next message');
    expect(screen.getByTestId('floating-generated-image')).toBeTruthy();
  });

  it('stays dismissed for that image, and shows the next image', () => {
    const { rerender } = render(
      <FloatingGeneratedImage
        imageGenState={doneState()}
        conversationId="conv-1"
        bottom={92}
        onOpenFullscreen={jest.fn()}
      />,
    );
    fireEvent.press(screen.getByTestId('floating-generated-image-close'));
    expect(screen.queryByTestId('floating-generated-image')).toBeNull();

    const next = doneState();
    next.result = { ...next.result!, id: 'img-2', imagePath: '/data/generated_images/img-2.png' };
    rerender(
      <FloatingGeneratedImage
        imageGenState={next}
        conversationId="conv-1"
        bottom={92}
        onOpenFullscreen={jest.fn()}
      />,
    );
    expect(screen.getByTestId('floating-generated-image')).toBeTruthy();
  });

  it('shows nothing for another chat, while generating, or without a result', () => {
    for (const [state, conversationId] of [
      [doneState(), 'conv-2'],
      [doneState({ phase: 'generating', isGenerating: true, result: null }), 'conv-1'],
      [doneState({ result: null }), 'conv-1'],
      [doneState(), null],
    ] as const) {
      const { unmount } = render(
        <FloatingGeneratedImage
          imageGenState={state}
          conversationId={conversationId}
          bottom={92}
          onOpenFullscreen={jest.fn()}
        />,
      );
      expect(screen.queryByTestId('floating-generated-image')).toBeNull();
      unmount();
    }
  });
});
