/**
 * Text LoRA settings: local rows (enable, scale, remove) call the live-apply actions and show the
 * mismatch risk and llama.cpp's error; with a remote text model active, the server's adapter list is
 * shown instead.
 */
import React from 'react';
import { act, fireEvent, render, waitFor } from '@testing-library/react-native';
import { TextLoraSection } from '../../../src/components/settings/TextLoraSection';
import { useAppStore } from '../../../src/stores/appStore';
import { useRemoteServerStore } from '../../../src/stores/remoteServerStore';
import { setTextLoraStatus } from '../../../src/services/textLora';
import { resetStores } from '../../utils/testHelpers';

const mockPatch = jest.fn((..._args: unknown[]) => Promise.resolve(null));
const mockRemove = jest.fn((..._args: unknown[]) => Promise.resolve());
const mockSetRemote = jest.fn();
jest.mock('../../../src/services/textLoraActions', () => ({
  patchTextLora: (...a: unknown[]) => mockPatch(...a),
  removeTextLora: (...a: unknown[]) => mockRemove(...a),
  importPickedTextLora: jest.fn(),
  setRemoteTextLoraScale: (...a: unknown[]) => mockSetRemote(...a),
  clearRemoteTextLoras: jest.fn(),
}));
const mockFetchAdapters = jest.fn();
jest.mock('../../../src/services/textLoraRemote', () => ({
  ...jest.requireActual('../../../src/services/textLoraRemote'),
  fetchServerLoraAdapters: (...a: unknown[]) => mockFetchAdapters(...a),
}));
jest.mock('../../../src/services/remoteServerManager', () => ({
  remoteServerManager: { getApiKey: jest.fn(() => Promise.resolve(null)) },
}));

const textLoras = [
  { id: 'a', name: 'qwen-chat-lora', path: '/d/text_loras/a.gguf', scale: 1, enabled: true },
  { id: 'b', name: 'off-lora', path: '/d/text_loras/b.gguf', scale: 1, enabled: false },
];

describe('TextLoraSection', () => {
  beforeEach(() => {
    resetStores();
    jest.clearAllMocks();
    setTextLoraStatus({ lastError: null, cpuFallback: false });
    useRemoteServerStore.setState({ activeServerId: null, activeRemoteTextModelId: null });
    useAppStore.setState({ settings: { ...useAppStore.getState().settings, textLoras } });
  });

  it('lists the adapters with the mismatch note; only enabled ones show a scale', () => {
    const view = render(<TextLoraSection />);
    expect(view.getByText('qwen-chat-lora')).toBeTruthy();
    expect(view.getByText('off-lora')).toBeTruthy();
    expect(view.getByText(/same architecture and same layer sizes/)).toBeTruthy();
    expect(view.getByTestId('text-lora-scale-a-slider')).toBeTruthy();
    expect(view.queryByTestId('text-lora-scale-b-slider')).toBeNull();
  });

  it('enable switch and remove call the live-apply actions', async () => {
    const view = render(<TextLoraSection />);
    await act(async () => { fireEvent(view.getByTestId('text-lora-enable-b'), 'valueChange', true); });
    expect(mockPatch).toHaveBeenCalledWith('b', { enabled: true });
    await act(async () => { fireEvent.press(view.getByTestId('text-lora-remove-a')); });
    expect(mockRemove).toHaveBeenCalledWith('a');
  });

  it('a typed scale is committed through the action', async () => {
    const view = render(<TextLoraSection />);
    fireEvent.press(view.getByTestId('text-lora-scale-a-value-button'));
    const input = view.getByTestId('text-lora-scale-a-input');
    fireEvent.changeText(input, '1.6');
    await act(async () => { fireEvent(input, 'submitEditing'); });
    expect(mockPatch).toHaveBeenCalledWith('a', { scale: 1.6 });
  });

  it('shows llama.cpp\'s error and the CPU-fallback warning', () => {
    setTextLoraStatus({ lastError: 'model arch and LoRA arch mismatch', cpuFallback: true });
    const view = render(<TextLoraSection />);
    expect(view.getByTestId('text-lora-error').props.children).toBe('model arch and LoRA arch mismatch');
    expect(view.getByTestId('text-lora-cpu-warning')).toBeTruthy();
  });

  it('remote text model: shows the server adapters and saves a per-request scale', async () => {
    useRemoteServerStore.setState({
      servers: [{ id: 's1', name: 'PC', endpoint: 'http://192.168.1.50:8080', providerType: 'openai-compatible', createdAt: '' } as any],
      activeServerId: 's1',
      activeRemoteTextModelId: 'm',
    });
    mockFetchAdapters.mockResolvedValue([{ id: 0, path: '/models/style.gguf', scale: 1 }]);
    const view = render(<TextLoraSection />);
    await waitFor(() => expect(view.getByText('style.gguf')).toBeTruthy());
    expect(view.queryByTestId('text-lora-section')).toBeNull();
    fireEvent.press(view.getByTestId('remote-text-lora-scale-0-value-button'));
    const input = view.getByTestId('remote-text-lora-scale-0-input');
    fireEvent.changeText(input, '0.4');
    fireEvent(input, 'submitEditing');
    expect(mockSetRemote).toHaveBeenCalledWith('http://192.168.1.50:8080', 0, 0.4, [{ id: 0, path: '/models/style.gguf', scale: 1 }]);
  });

  it('remote public host: explains LoRA scales go only to the user\'s own server', () => {
    useRemoteServerStore.setState({
      servers: [{ id: 's1', name: 'Cloud', endpoint: 'https://api.openai.com', providerType: 'openai-compatible', createdAt: '' } as any],
      activeServerId: 's1',
      activeRemoteTextModelId: 'gpt',
    });
    const view = render(<TextLoraSection />);
    expect(view.getByText(/only to your own llama-server/)).toBeTruthy();
    expect(mockFetchAdapters).not.toHaveBeenCalled();
  });
});
