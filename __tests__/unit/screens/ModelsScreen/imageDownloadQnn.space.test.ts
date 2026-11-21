/**
 * A zip image model needs room for the zip AND its extraction at once (the zip is deleted only
 * after the unzip). The native worker checks the zip size alone, so the download entry point
 * refuses up front: a 3.8 GB SDXL zip needs ~7.6 GB (+ headroom) free.
 */
const mockGetFSInfo = jest.fn();
jest.mock('react-native-fs', () => ({ getFSInfo: () => mockGetFSInfo() }));

import { refuseZipWithoutExtractionSpace, zipExtractionBytes } from '../../../../src/services/imageDownloadQnn';
import { ImageModelDescriptor } from '../../../../src/screens/ModelsScreen/types';
import { makeImageDownloadDeps } from '../../../utils/factories';

const GB = 1024 * 1024 * 1024;

function zipModel(overrides: Partial<ImageModelDescriptor> = {}): ImageModelDescriptor {
  return {
    id: 'illustrious_v16_sdxl_npu_8gen3',
    name: 'Illustrious v16 (SDXL NPU)',
    description: 'SDXL NPU model (1024 px) from xororz/sdxl-qnn',
    downloadUrl: 'https://huggingface.co/xororz/sdxl-qnn/resolve/main/illustrious_v16_qnn2.28_8gen3.zip',
    size: 3.8 * GB,
    style: 'anime',
    backend: 'qnn',
    variant: '8gen3',
    ...overrides,
  };
}

describe('zip extraction space', () => {
  beforeEach(() => mockGetFSInfo.mockReset());

  it('a zip needs twice its size: 3.8 GB -> 7.6 GB', () => {
    expect(zipExtractionBytes(3.8 * GB) / GB).toBeCloseTo(7.6);
    expect(zipExtractionBytes(0)).toBe(0);
  });

  it('refuses with an alert when free space is below zip + extraction + headroom', async () => {
    mockGetFSInfo.mockResolvedValue({ freeSpace: 7 * GB, totalSpace: 256 * GB });
    const deps = makeImageDownloadDeps();
    await expect(refuseZipWithoutExtractionSpace(zipModel(), deps)).resolves.toBe(true);
    expect(deps.setAlertState).toHaveBeenCalledTimes(1);
    const alert = (deps.setAlertState as jest.Mock).mock.calls[0][0];
    expect(alert.title).toBe('Not enough storage');
    expect(alert.message).toContain('7.6 GB');
  });

  it('lets the download through when it fits', async () => {
    mockGetFSInfo.mockResolvedValue({ freeSpace: 8 * GB, totalSpace: 256 * GB });
    const deps = makeImageDownloadDeps();
    await expect(refuseZipWithoutExtractionSpace(zipModel(), deps)).resolves.toBe(false);
    expect(deps.setAlertState).not.toHaveBeenCalled();
  });

  it('does not block when free space is unknown', async () => {
    mockGetFSInfo.mockRejectedValue(new Error('no statfs'));
    await expect(refuseZipWithoutExtractionSpace(zipModel(), makeImageDownloadDeps())).resolves.toBe(false);
  });

  it('skips multi-file and Core ML downloads (no extraction)', async () => {
    mockGetFSInfo.mockResolvedValue({ freeSpace: 0, totalSpace: 256 * GB });
    const deps = makeImageDownloadDeps();
    await expect(refuseZipWithoutExtractionSpace(
      zipModel({ huggingFaceRepo: 'r', huggingFaceFiles: [{ path: 'a', size: 1 }] }), deps,
    )).resolves.toBe(false);
    await expect(refuseZipWithoutExtractionSpace(
      zipModel({ coremlFiles: [{ path: 'a', relativePath: 'a', size: 1, downloadUrl: 'u' }] }), deps,
    )).resolves.toBe(false);
    expect(mockGetFSInfo).not.toHaveBeenCalled();
  });
});
