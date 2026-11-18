/**
 * Unit tests for useChatMessageHandlers — the resend/retry + edit-and-resend
 * handlers.
 *
 * Regression: ejecting all models then tapping Resend/Retry used to silently
 * no-op (no message, no feedback). Now it alerts "No Model Selected", mirroring
 * the send path (handleSendFn). See handleRetryMessageFn / handleEditMessageFn.
 */

import { handleRetryMessageFn, handleEditMessageFn } from '../../../src/screens/ChatScreen/useChatMessageHandlers';
import * as generationActions from '../../../src/screens/ChatScreen/useChatGenerationActions';
import { createMessage } from '../../utils/factories';

// Light mocks — the no-model path bails before any of these are touched, but
// they still need to exist so the module imports cleanly.
jest.mock('../../../src/services/modelResidency', () => ({
  modelResidencyManager: { getResidents: jest.fn(() => []), reclaimSttForGeneration: jest.fn() },
}));
jest.mock('../../../src/services/tools/useExtensionToolCount', () => ({
  enableAssistantTools: jest.fn(() => true),
}));
jest.mock('../../../src/services/hardware', () => ({
  hardwareService: { getAvailableMemoryGB: jest.fn(() => 4), getTotalMemoryGB: jest.fn(() => 8) },
}));

const makeDeps = (overrides: Partial<any> = {}): any => ({
  setAlertState: jest.fn(),
  ...overrides,
});

describe('handleRetryMessageFn — no active model', () => {
  let regenSpy: jest.SpyInstance;

  beforeEach(() => {
    regenSpy = jest.spyOn(generationActions, 'regenerateResponseFn').mockResolvedValue(undefined);
  });
  afterEach(() => jest.restoreAllMocks());

  it('alerts "No Model Selected" and does not regenerate when no model is loaded', async () => {
    const genDeps = makeDeps();
    const message = createMessage({ role: 'user', content: 'hi' });

    await handleRetryMessageFn(message, genDeps, {
      activeConversationId: 'conv-1',
      hasActiveModel: false,
      deleteMessagesAfter: jest.fn(),
      setDebugInfo: jest.fn(),
    });

    expect(genDeps.setAlertState).toHaveBeenCalledWith(
      expect.objectContaining({ visible: true, title: 'No Model Selected' }),
    );
    expect(regenSpy).not.toHaveBeenCalled();
  });

  it('does NOT alert and bails quietly when a model exists but there is no conversation', async () => {
    const genDeps = makeDeps();
    const message = createMessage({ role: 'user', content: 'hi' });

    await handleRetryMessageFn(message, genDeps, {
      activeConversationId: null,
      hasActiveModel: true,
      deleteMessagesAfter: jest.fn(),
      setDebugInfo: jest.fn(),
    });

    expect(genDeps.setAlertState).not.toHaveBeenCalled();
    expect(regenSpy).not.toHaveBeenCalled();
  });
});

describe('handleEditMessageFn — no active model', () => {
  let regenSpy: jest.SpyInstance;

  beforeEach(() => {
    regenSpy = jest.spyOn(generationActions, 'regenerateResponseFn').mockResolvedValue(undefined);
  });
  afterEach(() => jest.restoreAllMocks());

  it('alerts "No Model Selected" and does not regenerate when no model is loaded', async () => {
    const genDeps = makeDeps();
    const message = createMessage({ role: 'user', content: 'original' });

    await handleEditMessageFn(genDeps, {
      message,
      newContent: 'edited',
      activeConversationId: 'conv-1',
      hasActiveModel: false,
      updateMessageContent: jest.fn(),
      deleteMessagesAfter: jest.fn(),
      setDebugInfo: jest.fn(),
    });

    expect(genDeps.setAlertState).toHaveBeenCalledWith(
      expect.objectContaining({ visible: true, title: 'No Model Selected' }),
    );
    expect(regenSpy).not.toHaveBeenCalled();
  });
});

describe('handleEditMessageFn — Assistant selection carried like retry', () => {
  let regenSpy: jest.SpyInstance;
  const { enableAssistantTools } = jest.requireMock('../../../src/services/tools/useExtensionToolCount');

  beforeEach(() => {
    regenSpy = jest.spyOn(generationActions, 'regenerateResponseFn').mockResolvedValue(undefined);
    (enableAssistantTools as jest.Mock).mockReset();
  });
  afterEach(() => jest.restoreAllMocks());

  const editParams = (overrides: Partial<any> = {}) => ({
    message: createMessage({ role: 'user', content: 'original' }),
    newContent: 'edited',
    activeConversationId: 'conv-1',
    hasActiveModel: true,
    updateMessageContent: jest.fn(),
    deleteMessagesAfter: jest.fn(),
    setDebugInfo: jest.fn(),
    ...overrides,
  });

  it('passes assistantEnabled through to regenerate after re-enabling the Assistant tools', async () => {
    (enableAssistantTools as jest.Mock).mockReturnValue(true);
    await handleEditMessageFn(makeDeps(), editParams({ assistantEnabled: true }));
    expect(enableAssistantTools).toHaveBeenCalledTimes(1);
    expect(regenSpy).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ assistantEnabled: true }),
    );
  });

  it('bails with "Assistant unavailable" before touching the turn when the tools cannot be enabled', async () => {
    (enableAssistantTools as jest.Mock).mockReturnValue(false);
    const genDeps = makeDeps();
    const params = editParams({ assistantEnabled: true });
    await handleEditMessageFn(genDeps, params);
    expect(genDeps.setAlertState).toHaveBeenCalledWith(
      expect.objectContaining({ visible: true, title: 'Assistant unavailable' }),
    );
    expect(params.updateMessageContent).not.toHaveBeenCalled();
    expect(params.deleteMessagesAfter).not.toHaveBeenCalled();
    expect(regenSpy).not.toHaveBeenCalled();
  });

  it('does not consult the Assistant tools when the selection is off', async () => {
    await handleEditMessageFn(makeDeps(), editParams({ assistantEnabled: false }));
    expect(enableAssistantTools).not.toHaveBeenCalled();
    expect(regenSpy).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ assistantEnabled: false }),
    );
  });
});
