/**
 * HAPPY-PATH (UI integration, HEAVY entry point) - the `/img` slash command: on the real ChatScreen,
 * `/img <prompt> --flags` draws directly (no classifier, no LLM turn), the flags reach the native
 * generator for that one image, and the user bubble shows the prompt without the command or flags.
 *
 * Real ChatScreen + real ChatInput + real dispatchGenerationFn + real imageGenerationService; only the
 * native LiteRT + diffusion are faked (harness boundary).
 */
import { setupChatScreen } from '../../harness/chatHarness';
import { createONNXImageModel } from '../../utils/factories';

jest.mock('@react-navigation/native', () => ({
  useNavigation: () => ({ navigate: () => {}, goBack: () => {}, setOptions: () => {}, addListener: () => () => {} }),
  useRoute: () => require('../../harness/chatHarness').routeHolder,
  useFocusEffect: () => {},
  useIsFocused: () => true,
}));

// Same setup as imageIntentRouting.happy: the image model is downloaded + active (documented there).
async function withImageModel(h: Awaited<ReturnType<typeof setupChatScreen>>) {
  const imageModel = createONNXImageModel({ id: 'sd', name: 'SD', modelPath: '/models/sd', backend: 'coreml' as never });
  h.useAppStore.setState({ downloadedImageModels: [imageModel], activeImageModelId: 'sd' });
  h.boundary.diffusion.module.getLoadedModelPath.mockResolvedValue(imageModel.modelPath);
}

describe('happy - /img slash command (heavy entry point)', () => {
  it('draws directly with the inline flags and shows only the prompt in the user bubble', async () => {
    const h = await setupChatScreen({ engine: 'litert', platform: 'ios' });
    await withImageModel(h);
    h.render();

    await h.tapSend('/img a cat wearing a hat --steps 7 --seed 42 --cfg 9 --neg "blurry"');

    await h.rtl.waitFor(() => { expect(h.boundary.diffusion.calls.generateImage.length).toBe(1); });
    const native = h.boundary.diffusion.calls.generateImage[0];
    expect(native).toEqual(expect.objectContaining({ steps: 7, seed: 42, guidanceScale: 9, negativePrompt: 'blurry' }));
    expect(String(native.prompt)).toContain('a cat wearing a hat');
    expect(String(native.prompt)).not.toMatch(/\/img|--steps|--seed/);
    const conversation = h.useChatStore.getState().conversations.find(
      (c: { id: string }) => c.id === h.conversationId,
    );
    const userMessage = conversation?.messages.find((m: { role: string }) => m.role === 'user');
    expect(userMessage?.content).toBe('a cat wearing a hat');
    expect(userMessage?.turnKind).toBe('image');
    // No LLM turn ran for the command (the only engine call is the harness load warm-up "Hi").
    expect(JSON.stringify(h.boundary.litert.calls.sendMessage)).not.toMatch(/cat wearing a hat|\/img/);
  });

  it('without flags the app settings are used', async () => {
    const h = await setupChatScreen({ engine: 'litert', platform: 'ios' });
    await withImageModel(h);
    h.useAppStore.getState().updateSettings({ imageSteps: 9 });
    h.render();

    await h.tapSend('/image a red fox');

    await h.rtl.waitFor(() => { expect(h.boundary.diffusion.calls.generateImage.length).toBe(1); });
    expect(h.boundary.diffusion.calls.generateImage[0]).toEqual(expect.objectContaining({ steps: 9 }));
  });
});
