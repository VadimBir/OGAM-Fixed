/**
 * Edit / retry / delete must drop a compaction summary that no longer describes the history.
 * The summary stands in for messages[0..cutoff]; touching any of those (or removing the cutoff) makes
 * it stale, and applyCompactionPrefix would otherwise send it beside the changed history.
 */
import { useChatStore } from '../../../src/stores/chatStore';
import { dropStaleCompaction } from '../../../src/stores/chatMessageMutationActions';
import type { Conversation } from '../../../src/types';
import { resetStores } from '../../utils/testHelpers';

function seed(count: number, cutoffIndex: number) {
  const store = useChatStore.getState();
  const convId = store.createConversation('txt');
  for (let i = 0; i < count; i++) {
    store.addMessage(convId, {
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `m${i}`,
    });
  }
  const ids = store.getConversationMessages(convId).map(m => m.id);
  store.updateCompactionState(convId, 'SUMMARY', ids[cutoffIndex]);
  return { convId, ids };
}

function conv(convId: string): Conversation {
  return useChatStore.getState().conversations.find(c => c.id === convId)!;
}

beforeEach(() => {
  resetStores();
});

describe('deleteMessagesAfter (retry / edit)', () => {
  it('clears the summary when the cutoff is removed', () => {
    const { convId, ids } = seed(6, 3);
    useChatStore.getState().deleteMessagesAfter(convId, ids[1]);
    expect(conv(convId).compactionSummary).toBeUndefined();
    expect(conv(convId).compactionCutoffMessageId).toBeUndefined();
  });

  it('keeps the summary when only messages after the cutoff are removed', () => {
    const { convId, ids } = seed(6, 1);
    useChatStore.getState().deleteMessagesAfter(convId, ids[3]);
    expect(conv(convId).compactionSummary).toBe('SUMMARY');
    expect(conv(convId).compactionCutoffMessageId).toBe(ids[1]);
  });

  it('keeps the summary when the kept message IS the cutoff', () => {
    const { convId, ids } = seed(6, 3);
    useChatStore.getState().deleteMessagesAfter(convId, ids[3]);
    expect(conv(convId).compactionSummary).toBe('SUMMARY');
  });
});

describe('deleteMessage', () => {
  it('clears the summary when a summarized message is deleted', () => {
    const { convId, ids } = seed(6, 3);
    useChatStore.getState().deleteMessage(convId, ids[2]);
    expect(conv(convId).compactionSummary).toBeUndefined();
  });

  it('clears the summary when the cutoff itself is deleted', () => {
    const { convId, ids } = seed(6, 3);
    useChatStore.getState().deleteMessage(convId, ids[3]);
    expect(conv(convId).compactionCutoffMessageId).toBeUndefined();
  });

  it('keeps the summary when a message after the cutoff is deleted', () => {
    const { convId, ids } = seed(6, 1);
    useChatStore.getState().deleteMessage(convId, ids[5]);
    expect(conv(convId).compactionSummary).toBe('SUMMARY');
  });
});

describe('updateMessageContent (edit)', () => {
  it('clears the summary when an edited message index <= cutoff index', () => {
    const { convId, ids } = seed(6, 3);
    useChatStore.getState().updateMessageContent(convId, ids[3], 'edited');
    expect(conv(convId).compactionSummary).toBeUndefined();
    expect(conv(convId).messages[3].content).toBe('edited');
  });

  it('keeps the summary when the edited message is after the cutoff', () => {
    const { convId, ids } = seed(6, 1);
    useChatStore.getState().updateMessageContent(convId, ids[4], 'edited');
    expect(conv(convId).compactionSummary).toBe('SUMMARY');
  });
});

describe('dropStaleCompaction', () => {
  it('is a no-op without compaction state', () => {
    const c = { id: 'c', messages: [] } as unknown as Conversation;
    expect(dropStaleCompaction(c, 0)).toBe(c);
  });

  it('drops a summary whose cutoff is already missing', () => {
    const c = {
      id: 'c',
      messages: [{ id: 'a' }],
      compactionSummary: 'S',
      compactionCutoffMessageId: 'gone',
    } as unknown as Conversation;
    expect(dropStaleCompaction(c, -1).compactionSummary).toBeUndefined();
  });
});
