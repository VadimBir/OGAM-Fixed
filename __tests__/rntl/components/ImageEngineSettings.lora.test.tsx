/**
 * Image Generation settings, LoRA section: an sd.cpp checkpoint lists each LoRA with its detected
 * base and flags a mismatch; a LocalDream (QNN/MNN) model says LoRA needs the sd.cpp (GPU) engine
 * (LocalDreamModule.buildGenerationBody sends no LoRA field), never a fake LoRA control.
 */
import React from 'react';
import { render, waitFor } from '@testing-library/react-native';
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
  });
});
