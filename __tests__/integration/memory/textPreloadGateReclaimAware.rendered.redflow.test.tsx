/**
 * RED-FLOW (UI, rendered) - the TEXT model load on Android is gated on LIVE free RAM.
 *
 * HISTORY: this file used to pin the Android "reclaim credit" (memoryBudget.effectiveAvailableMB returned
 * max(live, 70% of total RAM)), which admitted the qwythos cell below (12GB / 5GB free / 5GB on-disk model).
 * That credit is what let a load fill RAM faster than the low-memory killer reclaimed it, so on a 12GB phone
 * at ~75% used the app itself was killed. The owner ordered the gates driven by live available RAM with a
 * stated safety margin, a refusal that states needed vs available MB, and "Load Anyway" kept.
 *
 * THE CELL (numbers pinned):
 *  - 12GB total / 5GB live available, ANDROID, Aggressive (live margin 512MB).
 *  - declared model 3.5GB → residency spec sizeMB 5376 > live room 5120 - 512 = 4608 → residency REFUSES
 *    (overridable) with "it needs ~5376 MB but only ~4608 MB is free for it right now".
 *  - even past residency, the pre-load gate sizes the 5GB file at 6144MB > 5120 - 512 - 200 → refuses.
 * Either refusal carries the "it needs ~" signature and no reply renders. iOS (second case) is unchanged.
 */
import { setupChatScreen } from '../../harness/chatHarness';
import { GB } from '../../harness/nativeBoundary';

jest.mock('@react-navigation/native', () => ({
  useNavigation: () => ({ navigate: () => {}, goBack: () => {}, setOptions: () => {}, addListener: () => () => {} }),
  useRoute: () => require('../../harness/chatHarness').routeHolder,
  useFocusEffect: () => {},
  useIsFocused: () => true,
}));

describe('text model load gate reads LIVE available on Android (rendered, red-flow)', () => {
  it('ANDROID Aggressive: the live gate REFUSES (overridable, needed vs available MB) instead of loading into a kill', async () => {
    // declared 3.5GB → residency spec 5376MB vs live room 4608MB. deferInitialLoad → the first send
    // triggers the real lazy load. android + 12GB total / 5GB live-avail = the qwythos profile.
    const h = await setupChatScreen({
      engine: 'llama',
      platform: 'android',
      modelFileSizeBytes: 3.5 * GB,
      ram: { platform: 'android', totalBytes: 12 * GB, availBytes: 5 * GB },
      deferInitialLoad: true,
    });

    const React = require('react');
    const { ModelLoadingModeSelector } = require('../../../src/components/settings/textGenAdvancedSections');
    const { startLoadPolicySync } = require('../../../src/services/loadPolicySync');

    // BOUNDARY: the ACTUAL on-disk model file is 5GB (the harness seeds a 500MB placeholder; a real 5GB
    // download is the device reality) - so the pre-load gate would refuse too if residency admitted.
    h.boundary.fs!.seedFile('/docs/models/ggml-small.gguf', Math.round(5 * GB));

    h.render();

    // Real app wiring: App.tsx boots this so the settings toggle drives the residency manager.
    const stopSync = startLoadPolicySync();
    // GESTURE: turn on Aggressive via the real segmented control (the device was in Aggressive).
    const toggle = h.rtl.render(React.createElement(ModelLoadingModeSelector, {}));
    h.rtl.fireEvent.press(toggle.getByTestId('model-loading-mode-aggressive-button'));
    await h.rtl.waitFor(() => { expect(require('../../../src/services/modelResidency').modelResidencyManager.getLoadPolicy()).toBe('aggressive'); });
    toggle.unmount();

    // Precondition: no reply and no refusal surface yet.
    expect(h.view!.queryByText(/Paris\./)).toBeNull();
    expect(h.view!.queryByText(/it needs ~/)).toBeNull();

    // GESTURE: the real first-send lazy load → the live gate decides.
    await h.send('what is the capital of France', { text: 'Paris.' });

    // TERMINAL ARTIFACT: the refusal (with its "it needs ~" MB signature) renders; the reply does NOT.
    await h.rtl.waitFor(() => {
      expect(h.view!.queryByText(/it needs ~/)).not.toBeNull();
    }, { timeout: 8000 });
    expect(h.view!.queryByText(/Paris\./)).toBeNull();

    stopSync();
  }, 30000);

  it('iOS UNCHANGED: the pre-load gate stays RAW (jetsam kills us, no reclaim credit) so the same model is refused', async () => {
    // Identical intersection on iOS. effectiveAvailableMB returns the RAW snapshot on iOS, so the pre-load
    // gate refuses the 5GB-on-disk model on 5GB raw-available. iOS behavior is untouched by the Android change.
    const h = await setupChatScreen({
      engine: 'llama',
      platform: 'ios',
      modelFileSizeBytes: 3.5 * GB,
      ram: { platform: 'ios', totalBytes: 12 * GB, availBytes: 5 * GB },
      deferInitialLoad: true,
    });

    const React = require('react');
    const { ModelLoadingModeSelector } = require('../../../src/components/settings/textGenAdvancedSections');
    const { startLoadPolicySync } = require('../../../src/services/loadPolicySync');

    // BOUNDARY: the ACTUAL on-disk file is 5GB → the gate sizes weights at 6144MB > 5120MB raw available.
    h.boundary.fs!.seedFile('/docs/models/ggml-small.gguf', Math.round(5 * GB));

    h.render();

    const stopSync = startLoadPolicySync();
    // GESTURE: Aggressive (matches the Android case).
    const toggle = h.rtl.render(React.createElement(ModelLoadingModeSelector, {}));
    h.rtl.fireEvent.press(toggle.getByTestId('model-loading-mode-aggressive-button'));
    await h.rtl.waitFor(() => { expect(require('../../../src/services/modelResidency').modelResidencyManager.getLoadPolicy()).toBe('aggressive'); });
    toggle.unmount();

    expect(h.view!.queryByText(/it needs ~/)).toBeNull();

    // GESTURE: send → residency admits (iOS physical cap holds the 3.5GB spec) → the RAW pre-load gate refuses.
    await h.send('what is the capital of France', { text: 'Paris.' });

    // TERMINAL ARTIFACT: iOS still refuses on the raw snapshot — the "it needs ~" refusal renders and the reply
    // does NOT. iOS jetsam behavior is unchanged by the fix.
    await h.rtl.waitFor(() => {
      expect(h.view!.queryByText(/it needs ~/)).not.toBeNull();
    }, { timeout: 8000 });
    expect(h.view!.queryByText(/Paris\./)).toBeNull();

    stopSync();
  }, 30000);
});
