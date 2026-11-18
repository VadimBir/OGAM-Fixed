/**
 * Chat avatars: the user's PERSONA avatar beside the user's messages, the active CHARACTER's avatar
 * beside assistant replies; each falls back to its glyph when no avatar is set.
 */
import React from 'react';
import { Image } from 'react-native';
import { render } from '@testing-library/react-native';
import { ChatMessage } from '../../../src/components/ChatMessage';
import { useAppStore, useChatStore, useProjectStore } from '../../../src/stores';
import { createUserMessage, createAssistantMessage, createProject } from '../../utils/factories';
import { resetStores } from '../../utils/testHelpers';

function imageUris(view: ReturnType<typeof render>): string[] {
  return view.UNSAFE_queryAllByType(Image).map(node => node.props.source?.uri);
}

describe('ChatMessage avatars', () => {
  beforeEach(() => {
    resetStores();
  });

  it('shows the persona avatar beside a user message', () => {
    useAppStore.getState().updateSettings({ personaAvatarUri: 'file:///persona_avatars/me.png' });
    const view = render(<ChatMessage message={createUserMessage('hi')} />);
    expect(view.getByTestId('user-avatar')).toBeTruthy();
    expect(imageUris(view)).toContain('file:///persona_avatars/me.png');
  });

  it('falls back to the glyph when no persona avatar is set', () => {
    const view = render(<ChatMessage message={createUserMessage('hi')} />);
    expect(view.getByTestId('user-avatar')).toBeTruthy();
    expect(imageUris(view)).toHaveLength(0);
  });

  it('shows the character avatar beside an assistant reply, not the persona avatar', () => {
    useAppStore.getState().updateSettings({ personaAvatarUri: 'file:///persona_avatars/me.png' });
    useProjectStore.setState({
      projects: [createProject({ id: 'char-1', name: 'Ava', avatarUri: 'file:///avatars/ava.png' })],
    });
    const convId = useChatStore.getState().createConversation('txt', undefined, 'char-1');
    useChatStore.getState().setActiveConversation(convId);
    const view = render(<ChatMessage message={createAssistantMessage('hello')} />);
    expect(view.queryByTestId('user-avatar')).toBeNull();
    const uris = imageUris(view);
    expect(uris).toContain('file:///avatars/ava.png');
    expect(uris).not.toContain('file:///persona_avatars/me.png');
  });
});
