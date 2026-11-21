/**
 * SDXL NPU RAM estimate (local-dream v2.8.1 `--type sdxl --lowram`): unet.bin + 0.09 GB spill-fill +
 * 0.2 GB embeddings + the core process baseline. The real unet.bin size is used once
 * validateImageModelDir has listed it; before that, zip size x 0.72. The load is charged
 * INCOMING_PEAK_FACTOR once; the resident charge carries no factor.
 */
const listings: Record<string, Array<{ name: string; size: number; isFile: boolean }>> = {};
const mockExists = jest.fn(async (p: string) => {
  const cut = p.lastIndexOf('/');
  return (listings[p.slice(0, cut)] ?? []).some(e => e.name === p.slice(cut + 1));
});
const mockReadDir = jest.fn(async (p: string) => {
  if (!(p in listings)) throw new Error(`ENOENT ${p}`);
  return listings[p].map(e => ({
    name: e.name, size: e.size, path: `${p}/${e.name}`, isFile: () => e.isFile, isDirectory: () => !e.isFile,
  }));
});
jest.mock('react-native-fs', () => ({ exists: (p: string) => mockExists(p), readDir: (p: string) => mockReadDir(p) }));

import {
  estimateImageModelRamBytes,
  estimateSdxlNpuPeakBytes,
  imageModelResidencyMB,
  isSdxlNpuModel,
  LOCALDREAM_PROCESS_BASELINE_BYTES,
  SDXL_EMBEDDINGS_BYTES,
  SDXL_SPILL_FILL_BYTES,
  SDXL_UNET_OF_ZIP,
} from '../../../src/services/imageModelMemory';
import { INCOMING_PEAK_FACTOR } from '../../../src/services/memoryBudget';
import { hardwareService } from '../../../src/services/hardware';
import { validateImageModelDir } from '../../../src/utils/imageModelIntegrity';
import { createONNXImageModel } from '../../utils/factories';

const GiB = 1024 * 1024 * 1024;
const MiB = 1024 * 1024;
const OVERHEAD = SDXL_SPILL_FILL_BYTES + SDXL_EMBEDDINGS_BYTES + LOCALDREAM_PROCESS_BASELINE_BYTES;

function sdxlListing(unetBytes: number) {
  return [
    'SDXL', 'tokenizer.json', 'clip.mnn', 'clip_2.mnn', 'clip_2.mnn.weight', 'pos_emb.bin', 'token_emb.bin',
    'pos_emb_2.bin', 'token_emb_2.bin', 'vae_decoder.bin',
  ].map(name => ({ name, size: name === 'SDXL' ? 0 : 10, isFile: true }))
    .concat([{ name: 'unet.bin', size: unetBytes, isFile: true }]);
}

describe('SDXL NPU RAM estimate', () => {
  it('uses the Phase A constants: 0.09 GB spill-fill, 0.2 GB embeddings, unet = 0.72 x zip', () => {
    expect(SDXL_SPILL_FILL_BYTES).toBeCloseTo(0.09 * GiB);
    expect(SDXL_EMBEDDINGS_BYTES).toBeCloseTo(0.2 * GiB);
    expect(SDXL_UNET_OF_ZIP).toBe(0.72);
  });

  it('before the files were listed: a catalog SDXL id is SDXL and the zip size x 0.72 stands in for unet.bin', () => {
    const model = createONNXImageModel({
      id: 'illustrious_v16_sdxl_npu_8gen3', backend: 'qnn', size: 3.8 * GiB, modelPath: '/models/never-listed',
    });
    expect(isSdxlNpuModel(model)).toBe(true);
    const expected = 3.8 * GiB * 0.72 + OVERHEAD;
    expect(estimateSdxlNpuPeakBytes(model)).toBeCloseTo(expected);
    // ~2.74 GB UNet + 0.29 GB = ~3.0 GB before the process baseline (the Phase A figure).
    expect((expected - LOCALDREAM_PROCESS_BASELINE_BYTES) / GiB).toBeCloseTo(3.03, 1);
    expect(estimateImageModelRamBytes(model)).toBeCloseTo(expected);
  });

  it('after validateImageModelDir listed the folder: the real unet.bin size is used and the kind comes from disk', async () => {
    const root = '/models/custom-sdxl';
    listings[root] = [{ name: 'output', size: 0, isFile: false }];
    listings[`${root}/output`] = [{ name: 'qnn_models_sdxl_8gen3', size: 0, isFile: false }];
    listings[`${root}/output/qnn_models_sdxl_8gen3`] = sdxlListing(2.5 * GiB);

    const result = await validateImageModelDir(root, 'qnn');
    expect(result).toMatchObject({ complete: true, kind: 'sdxl', unetBytes: 2.5 * GiB });

    // An id with no `_sdxl_` (e.g. a renamed folder) is still SDXL once its files were seen.
    const model = createONNXImageModel({ id: 'my-model', backend: 'qnn', size: 3.8 * GiB, modelPath: root });
    expect(isSdxlNpuModel(model)).toBe(true);
    expect(estimateSdxlNpuPeakBytes(model)).toBeCloseTo(2.5 * GiB + OVERHEAD);
  });

  it('charges the load INCOMING_PEAK_FACTOR once and the resident model without a factor', () => {
    const model = createONNXImageModel({
      id: 'x_sdxl_npu_8gen3', backend: 'qnn', size: 3.5 * GiB, modelPath: '/models/unlisted-2',
    });
    const bytes = estimateSdxlNpuPeakBytes(model);
    expect(imageModelResidencyMB(model)).toEqual({
      peakMB: Math.ceil((bytes * INCOMING_PEAK_FACTOR) / MiB),
      residentMB: Math.ceil(bytes / MiB),
    });
  });

  it('SD1.5 NPU and GPU models keep the file-size estimate', async () => {
    const root = '/models/sd15-npu';
    listings[root] = ['tokenizer.json', 'clip_v2.mnn', 'pos_emb.bin', 'token_emb.bin', 'unet.bin', 'vae_decoder.bin']
      .map(name => ({ name, size: 10, isFile: true }));
    await validateImageModelDir(root, 'qnn');
    const npu = createONNXImageModel({ id: 'anythingv5_npu_8gen2', backend: 'qnn', size: GiB, modelPath: root });
    const gpu = createONNXImageModel({ id: 'x_sdxl_npu_8gen3', backend: 'mnn', size: GiB });
    expect(isSdxlNpuModel(npu)).toBe(false);
    expect(isSdxlNpuModel(gpu)).toBe(false);
    expect(estimateImageModelRamBytes(npu)).toBe(hardwareService.estimateImageModelRam(npu));
    const mb = Math.round(hardwareService.estimateImageModelRam(npu) / MiB);
    expect(imageModelResidencyMB(npu)).toEqual({ peakMB: mb, residentMB: mb });
  });
});
