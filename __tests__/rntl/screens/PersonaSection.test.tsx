/**
 * Importing a persona card replaces the persona, so the previous persona's caveats must not survive.
 */
import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';

jest.mock('../../../src/services/importCharacterCardFile', () => ({
  pickAndParseSillyTavernCard: jest.fn(),
}));
jest.mock('../../../src/utils/pickAvatarImage', () => ({
  pickAvatarImage: jest.fn(),
}));

import { PersonaSection } from '../../../src/screens/ModelSettingsScreen/PersonaSection';
import { pickAndParseSillyTavernCard } from '../../../src/services/importCharacterCardFile';
import { useAppStore } from '../../../src/stores';
import { resetStores } from '../../utils/testHelpers';

describe('PersonaSection import', () => {
  beforeEach(() => {
    resetStores();
    useAppStore.getState().updateSettings({
      personaName: 'Old',
      personaPrompt: 'old prompt',
      personaCaveats: 'OLD CAVEAT',
    });
  });

  it('clears the previous persona caveats on import', async () => {
    (pickAndParseSillyTavernCard as jest.Mock).mockResolvedValue({
      parsed: { name: 'Nova', description: 'a pilot', personality: '' },
      avatarUri: 'file:///persona_avatars/nova.png',
    });
    const { getByTestId } = render(<PersonaSection />);
    fireEvent.press(getByTestId('persona-import'));
    await waitFor(() =>
      expect(useAppStore.getState().settings.personaName).toBe('Nova'),
    );
    const settings = useAppStore.getState().settings;
    expect(settings.personaCaveats).toBe('');
    expect(settings.personaPrompt).toBe('a pilot');
    expect(settings.personaAvatarUri).toBe('file:///persona_avatars/nova.png');
  });

  it('resets the persona prompt to empty when the card has no description or personality', async () => {
    (pickAndParseSillyTavernCard as jest.Mock).mockResolvedValue({
      parsed: { name: 'Blank', description: '', personality: '' },
      avatarUri: undefined,
    });
    const { getByTestId } = render(<PersonaSection />);
    fireEvent.press(getByTestId('persona-import'));
    await waitFor(() =>
      expect(useAppStore.getState().settings.personaName).toBe('Blank'),
    );
    expect(useAppStore.getState().settings.personaPrompt).toBe('');
  });
});
