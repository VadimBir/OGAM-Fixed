/**
 * RED-FLOW (integration) — Q14: the advisory "safe to load?" check and the authoritative load gate used
 * to size the SAME image model with DIFFERENT multipliers (1.5/1.8 vs 1.8/2.5), so a user was told
 * "Safe to load" and then hit a hard "Insufficient memory" refusal. The advisory check had no callers
 * and was removed; the gate (makeRoomFor) and the register now both read ONE estimator,
 * imageModelResidencyMB. Integration boundary: only the RAM/platform leaf is faked.
 */
import { installNativeBoundary, GB } from '../../harness/nativeBoundary';
import { createONNXImageModel } from '../../utils/factories';

describe('Q14 — one image-RAM estimate for the load gate and the register (red-flow)', () => {
  it('a LocalDream model is charged its file-size estimate at load and once resident (no split known)', async () => {
    installNativeBoundary({ ram: { platform: 'android', totalBytes: 12 * GB, availBytes: 8 * GB } });

    const { imageModelResidencyMB } = require('../../../src/services/imageModelMemory');
    const { hardwareService } = require('../../../src/services/hardware');

    await hardwareService.refreshMemoryInfo();

    const model = createONNXImageModel({ id: 'sd', name: 'SD', size: 2 * GB, backend: 'mnn' });
    const gateMB = Math.round(hardwareService.estimateImageModelRam(model) / (1024 * 1024));

    expect(imageModelResidencyMB(model)).toEqual({ peakMB: gateMB, residentMB: gateMB });
  });
});
