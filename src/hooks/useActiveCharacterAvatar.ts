import { useChatStore, useProjectStore } from '../stores';

/**
 * The avatar URI of the ACTIVE conversation's character (its `Project`), or undefined when the
 * chat has no character or the character has no avatar. Rendered as the small circle next to
 * assistant replies. Kept as a tiny standalone selector hook so the (memoized) message list is not
 * forced to re-render on unrelated store ticks — only this leaf subscribes.
 */
export function useActiveCharacterAvatarUri(): string | undefined {
  const projectId = useChatStore(s => {
    const activeId = s.activeConversationId;
    return activeId
      ? s.conversations.find(c => c.id === activeId)?.projectId ?? undefined
      : undefined;
  });
  return useProjectStore(s =>
    projectId ? s.projects.find(p => p.id === projectId)?.avatarUri : undefined,
  );
}
