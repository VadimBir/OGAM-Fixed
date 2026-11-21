/**
 * Image Generation settings, LoRA section: an sd.cpp checkpoint lists each LoRA with its detected
 * base and flags a mismatch; a LocalDream (QNN/MNN) model says LoRA needs the sd.cpp (GPU) engine
 * (LocalDreamModule.buildGenerationBody sends no LoRA field), never a fake LoRA control.
 */
import React from 'react';
import { render, waitFor, fireEvent, act } from '@testing-library/react-native';
import { ImageEngineSettings } from '../../../src/components/ImageEngineSettings';
import { useAppStore } from '../../../src/stores/appStore';
import { resetStores } from '../../utils/testHelpers';

const mockFileFamily = jest.fn(async (path: string) =>
  path.includes('xl') ? { family: 'sdxl', loraBase: 'sdxl' } : { family: 'sd1', loraBase: 'sd1' });

jest.mock('../../../src/services/sdCppGenerator', () => ({
  SDCPP_SAMPLERS: ['euler_a'],
  SDCPP_SCHEDULERS: ['karras'],
  sdCppGeneratorService: { fileFamily: (p: string) => mockFileFamily(p), forgetFile: jest.fn() },
}));

const mockPick = jest.fn();
jest.mock('@react-native-documents/picker', () => ({
  pick: (...a: unknown[]) => mockPick(...a),
  types: { allFiles: '*/*' },
  isErrorWithCode: (e: any) => !!e && typeof e === 'object' && 'code' in e,
  errorCodes: { OPERATION_CANCELED: 'OPERATION_CANCELED' },
}));
const mockImport = jest.fn();
jest.mock('../../../src/screens/ModelsScreen/importSdCheckpoint', () => ({
  importSdCheckpointOrLora: (...a: unknown[]) => mockImport(...a),
}));
const mockTextRefusal = jest.fn(async (_uri: string): Promise<string | null> => null);
jest.mock('../../../src/services/textLora', () => ({
  textSafetensorsRefusal: (u: string) => mockTextRefusal(u),
}));

const loras = [
  { id: 'a', name: 'detail', path: '/l/detail.safetensors', weight: 1, enabled: true },
  { id: 'b', name: 'xl_style', path: '/l/xl_style.safetensors', weight: 1, enabled: true },
];

function setModel(backend: string, sdFamily?: string) {
  useAppStore.setState({
    downloadedImageModels: [{ id: 'm', name: 'M', description: '', modelPath: '/m', downloadedAt: '', size: 2e9, backend, sdFamily } as any],
    activeImageModelId: 'm',
    settings: { ...useAppStore.getState().settings, imageLoras: loras },
  });
}

describe('ImageEngineSettings LoRA', () => {
  beforeEach(() => resetStores());

  it('sd.cpp SD 1.x checkpoint: shows bases, flags the SDXL LoRA only', async () => {
    setModel('sdcpp', 'sd1');
    const view = render(<ImageEngineSettings />);
    await waitFor(() => expect(view.getByText('xl_style · SDXL')).toBeTruthy());
    expect(view.getByText('detail · SD 1.x')).toBeTruthy();
    expect(view.getByTestId('lora-mismatch-b').props.children).toMatch(/Made for SDXL, not this SD 1\.x checkpoint\. Generation is refused/);
    expect(view.queryByTestId('lora-mismatch-a')).toBeNull();
  });

  it('LocalDream model: "LoRA needs the sd.cpp (GPU) engine", no LoRA rows', () => {
    setModel('qnn');
    const view = render(<ImageEngineSettings />);
    expect(view.getByTestId('lora-engine-note').props.children).toMatch(/^LoRA needs the sd\.cpp \(GPU\) engine\. Your 2 LoRAs are not applied/);
    expect(view.queryByText(/xl_style/)).toBeNull();
    expect(mockFileFamily).not.toHaveBeenCalled();
    expect(view.getByTestId('lora-engine-note').props.children).toMatch(/merge the LoRA into the checkpoint on a PC before NPU conversion/);
    expect(view.getByTestId('image-lora-import')).toBeTruthy(); // import works on every engine
  });

  describe('Import LoRA button', () => {
    beforeEach(() => {
      mockPick.mockReset();
      mockImport.mockReset();
      mockTextRefusal.mockClear();
    });

    it('empty state points at the button', () => {
      setModel('sdcpp', 'sd1');
      useAppStore.setState({ settings: { ...useAppStore.getState().settings, imageLoras: [] } });
      const view = render(<ImageEngineSettings />);
      expect(view.getByText(/Tap "Import LoRA \(\.safetensors\)" below/)).toBeTruthy();
      expect(view.getByTestId('image-lora-import')).toBeTruthy();
    });

    it('picks a file and runs importSdCheckpointOrLora with the store deps; shows busy then idle', async () => {
      setModel('sdcpp', 'sd1');
      mockPick.mockResolvedValue([{ uri: 'content://x/style.safetensors', name: 'style.safetensors', size: 1234 }]);
      let release!: () => void;
      mockImport.mockImplementation((_f: unknown, deps: any) => new Promise<void>(res => {
        deps.setImportProgress({ fraction: 0.45, fileName: 'style.safetensors' });
        release = res;
      }));
      const view = render(<ImageEngineSettings />);
      fireEvent.press(view.getByTestId('image-lora-import'));
      await waitFor(() => expect(view.getByText('Importing style.safetensors 45%')).toBeTruthy());
      expect(mockImport.mock.calls[0][0]).toEqual({ uri: 'content://x/style.safetensors', name: 'style.safetensors', size: 1234 });
      expect(Object.keys(mockImport.mock.calls[0][1]).sort()).toEqual(
        ['activeImageModelId', 'addDownloadedImageModel', 'setActiveImageModelId', 'setAlertState', 'setImportProgress']);
      await act(async () => { release(); });
      await waitFor(() => expect(view.getByText('Import LoRA (.safetensors)')).toBeTruthy());
    });

    it('shows the alert the shared import produces', async () => {
      setModel('sdcpp', 'sd1');
      mockPick.mockResolvedValue([{ uri: 'content://x/a.safetensors', name: 'a.safetensors', size: 1 }]);
      mockImport.mockImplementation(async (_f: unknown, deps: any) => {
        deps.setAlertState({ visible: true, title: 'LoRA Imported', message: 'a.safetensors added' });
      });
      const view = render(<ImageEngineSettings />);
      fireEvent.press(view.getByTestId('image-lora-import'));
      await waitFor(() => expect(view.getByText('LoRA Imported')).toBeTruthy());
    });

    it('rejects a non-safetensors file without importing', async () => {
      setModel('sdcpp', 'sd1');
      mockPick.mockResolvedValue([{ uri: 'content://x/a.gguf', name: 'a.gguf', size: 1 }]);
      const view = render(<ImageEngineSettings />);
      fireEvent.press(view.getByTestId('image-lora-import'));
      await waitFor(() => expect(view.getByText('Invalid File')).toBeTruthy());
      expect(mockImport).not.toHaveBeenCalled();
    });

    it('text-weights safetensors are refused before import', async () => {
      setModel('sdcpp', 'sd1');
      mockPick.mockResolvedValue([{ uri: 'content://x/t.safetensors', name: 't.safetensors', size: 1 }]);
      mockTextRefusal.mockResolvedValueOnce('convert me');
      const view = render(<ImageEngineSettings />);
      fireEvent.press(view.getByTestId('image-lora-import'));
      await waitFor(() => expect(view.getByText('Convert to GGUF first')).toBeTruthy());
      expect(mockImport).not.toHaveBeenCalled();
    });

    it('picker cancel is silent', async () => {
      setModel('sdcpp', 'sd1');
      mockPick.mockRejectedValue({ code: 'OPERATION_CANCELED' });
      const view = render(<ImageEngineSettings />);
      fireEvent.press(view.getByTestId('image-lora-import'));
      await waitFor(() => expect(mockPick).toHaveBeenCalled());
      await waitFor(() => expect(view.getByText('Import LoRA (.safetensors)')).toBeTruthy());
      expect(view.queryByText('Import Failed')).toBeNull();
      expect(mockImport).not.toHaveBeenCalled();
    });

    it('import error surfaces as Import Failed', async () => {
      setModel('sdcpp', 'sd1');
      mockPick.mockResolvedValue([{ uri: 'content://x/a.safetensors', name: 'a.safetensors', size: 1 }]);
      mockImport.mockRejectedValue(new Error('disk exploded'));
      const view = render(<ImageEngineSettings />);
      fireEvent.press(view.getByTestId('image-lora-import'));
      await waitFor(() => expect(view.getByText('Import Failed')).toBeTruthy());
    });
  });
});
