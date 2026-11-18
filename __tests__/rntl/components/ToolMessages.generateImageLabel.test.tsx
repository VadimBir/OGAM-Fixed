/**
 * ToolResultMessage label for generate_image: success says "Generated image"; a failed tool result
 * (model-facing text from toolResultModelContent) says it failed; a user cancel says cancelled.
 */
import React from 'react';
import { render } from '@testing-library/react-native';
import { ToolResultMessage } from '../../../src/components/ChatMessage/components/ToolMessages';
import type { Message } from '../../../src/types';

const colors = { primary: '#00f', textMuted: '#888' };

function renderResult(content: string) {
  const message = {
    id: 'm1',
    role: 'tool',
    content,
    timestamp: 0,
    toolName: 'generate_image',
    toolCallId: 'c1',
  } as Message;
  return render(<ToolResultMessage message={message} styles={{}} colors={colors} />);
}

describe('ToolResultMessage generate_image label', () => {
  it('success -> Generated image', () => {
    const view = renderResult('{"status":"ok","shown_in_chat":true}');
    expect(view.getByTestId('tool-result-label-generate_image').props.children.join('')).toBe('Generated image');
  });

  it('failed result -> Image generation failed', () => {
    const view = renderResult(
      'Tool "generate_image" failed (internal): generate_image engine_error: The image engine failed.. It returned no data',
    );
    expect(view.getByTestId('tool-result-label-generate_image').props.children.join('')).toBe('Image generation failed');
  });

  it('user cancel -> Image generation cancelled', () => {
    const view = renderResult(
      'Tool "generate_image" failed (internal): generate_image cancelled: The user cancelled the image. Do not retry.. It returned no data',
    );
    expect(view.getByTestId('tool-result-label-generate_image').props.children.join('')).toBe('Image generation cancelled');
  });
});
