/**
 * EmptyChat project hint — shows the character's avatar image when it has one, else its initial.
 */
import React from 'react';
import { render } from '@testing-library/react-native';

jest.mock('../../../src/components', () => ({
  ModelSelectorModal: () => null,
}));
jest.mock('../../../src/services', () => ({
  llmService: { getLoadedModelPath: () => null },
}));

import { EmptyChat } from '../../../src/screens/ChatScreen/ChatScreenComponents';
import { createStyles } from '../../../src/screens/ChatScreen/styles';
import { getTheme } from '../../../src/theme';

const theme = getTheme('light');
const styles = createStyles(theme.colors, theme.shadows);

function renderEmpty(activeProject: any) {
  return render(
    <EmptyChat
      styles={styles}
      colors={theme.colors}
      activeModel={{ name: 'M' }}
      activeProject={activeProject}
      setShowProjectSelector={jest.fn()}
    />,
  );
}

describe('EmptyChat project hint', () => {
  it('shows the avatar image when the character has an avatar', () => {
    const { getByTestId, queryByText } = renderEmpty({ id: 'p1', name: 'Nova', avatarUri: 'file:///a/nova.png' });
    expect(getByTestId('project-hint-avatar').props.source).toEqual({ uri: 'file:///a/nova.png' });
    expect(queryByText('N')).toBeNull();
  });

  it('shows the initial when the character has no avatar', () => {
    const { getByText, queryByTestId } = renderEmpty({ id: 'p1', name: 'Nova' });
    expect(getByText('N')).toBeTruthy();
    expect(queryByTestId('project-hint-avatar')).toBeNull();
  });

  it('shows D for the default (no character)', () => {
    const { getByText } = renderEmpty(null);
    expect(getByText('D')).toBeTruthy();
  });
});
