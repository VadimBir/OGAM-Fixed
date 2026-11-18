/**
 * Text LoRA import: a GGUF adapter is copied into text_loras/ and registered as a text LoRA (never a
 * model); a model GGUF or another adapter type is not; .safetensors text weights are refused with
 * the PC conversion step.
 */
import RNFS from 'react-native-fs';
import { useAppStore } from '../../../src/stores/appStore';
import {
  SAFETENSORS_LLM_REFUSAL,
  SAFETENSORS_TEXT_LORA_REFUSAL,
  TEXT_LORAS_DIR,
  importTextLora,
  textSafetensorsRefusal,
} from '../../../src/services/textLora';
import { importPickedTextLora } from '../../../src/services/textLoraActions';
import { buildGguf, buildSafetensorsHeader, serveBytes } from '../../utils/ggufFixture';

jest.mock('../../../src/utils/logger', () => ({ __esModule: true, default: { log: jest.fn(), warn: jest.fn(), error: jest.fn() } }));

const mockedRNFS = RNFS as jest.Mocked<typeof RNFS>;

const LORA_GGUF = buildGguf({
  kv: [['general.architecture', 'qwen3'], ['general.type', 'adapter'], ['adapter.type', 'lora'], ['adapter.lora.alpha', { f32: 32 }]],
  tensors: [{ name: 'blk.0.attn_q.weight.lora_a', type: 1 }, { name: 'blk.0.attn_q.weight.lora_b', type: 8 }],
});
const MODEL_GGUF = buildGguf({ kv: [['general.architecture', 'qwen3'], ['general.type', 'model']] });

beforeEach(() => {
  jest.clearAllMocks();
  mockedRNFS.exists.mockResolvedValue(false);
  useAppStore.setState({ settings: { ...useAppStore.getState().settings, textLoras: [] } });
});

describe('importTextLora', () => {
  it('copies a GGUF LoRA into text_loras/ and registers it enabled at scale 1', async () => {
    mockedRNFS.read.mockImplementation(serveBytes(LORA_GGUF) as any);
    const result = await importTextLora({ uri: 'content://docs/my-lora.gguf', name: 'my-lora.gguf' });

    expect(mockedRNFS.copyFile).toHaveBeenCalledWith('content://docs/my-lora.gguf', `${TEXT_LORAS_DIR}/my-lora.gguf`);
    expect(result.architecture).toBe('qwen3');
    expect(result.lora).toMatchObject({ name: 'my-lora', path: `${TEXT_LORAS_DIR}/my-lora.gguf`, scale: 1, enabled: true });
    expect(useAppStore.getState().settings.textLoras).toEqual([result.lora]);
    // One tensor is Q8_0: not proven on OpenCL/HTP, so the user is told.
    expect(result.warnings[0]).toContain('Q8_0');
  });

  it('refuses a model GGUF and removes the copy', async () => {
    mockedRNFS.read.mockImplementation(serveBytes(MODEL_GGUF) as any);
    await expect(importTextLora({ uri: 'file:///x/model.gguf', name: 'model.gguf' })).rejects.toThrow('not a LoRA adapter');
    expect(mockedRNFS.unlink).toHaveBeenCalledWith(`${TEXT_LORAS_DIR}/model.gguf`);
    expect(useAppStore.getState().settings.textLoras).toEqual([]);
  });

  it('refuses a name that already exists', async () => {
    mockedRNFS.exists.mockResolvedValue(true);
    await expect(importTextLora({ uri: 'file:///x/a.gguf', name: 'a.gguf' })).rejects.toThrow('already exists');
    expect(mockedRNFS.copyFile).not.toHaveBeenCalled();
  });
});

describe('importPickedTextLora', () => {
  it('a model GGUF is not handled (the model import runs)', async () => {
    mockedRNFS.read.mockImplementation(serveBytes(MODEL_GGUF) as any);
    await expect(importPickedTextLora({ uri: 'file:///x/m.gguf', name: 'm.gguf' })).resolves.toEqual({ handled: false });
    expect(mockedRNFS.copyFile).not.toHaveBeenCalled();
  });

  it('a non-LoRA adapter is refused without a copy', async () => {
    mockedRNFS.read.mockImplementation(serveBytes(buildGguf({ kv: [['general.type', 'adapter'], ['adapter.type', 'cvec']] })) as any);
    const outcome = await importPickedTextLora({ uri: 'file:///x/c.gguf', name: 'c.gguf' });
    expect(outcome).toMatchObject({ handled: true, title: 'Unsupported adapter' });
    expect(mockedRNFS.copyFile).not.toHaveBeenCalled();
  });

  it('a LoRA is imported and the message names the architecture and the mismatch risk', async () => {
    mockedRNFS.read.mockImplementation(serveBytes(LORA_GGUF) as any);
    const outcome = await importPickedTextLora({ uri: 'file:///x/l.gguf', name: 'l.gguf' });
    expect(outcome.handled).toBe(true);
    expect(outcome.message).toContain('Trained for architecture: qwen3');
    expect(outcome.message).toContain('same architecture and same layer sizes');
  });
});

describe('textSafetensorsRefusal', () => {
  it('refuses a PEFT text LoRA, naming convert_lora_to_gguf.py', async () => {
    mockedRNFS.read.mockImplementation(serveBytes(buildSafetensorsHeader([
      'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight',
      'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight',
    ])) as any);
    const message = await textSafetensorsRefusal('content://docs/adapter_model.safetensors');
    expect(message).toBe(SAFETENSORS_TEXT_LORA_REFUSAL);
    expect(message).toContain('convert_lora_to_gguf.py');
  });

  it('refuses safetensors LLM weights, naming convert_hf_to_gguf.py', async () => {
    mockedRNFS.read.mockImplementation(serveBytes(buildSafetensorsHeader([
      'model.embed_tokens.weight', 'model.layers.0.mlp.up_proj.weight', 'lm_head.weight',
    ])) as any);
    const message = await textSafetensorsRefusal('file:///x/model.safetensors');
    expect(message).toBe(SAFETENSORS_LLM_REFUSAL);
    expect(message).toContain('convert_hf_to_gguf.py');
  });

  it('leaves an SD LoRA to the image import', async () => {
    mockedRNFS.read.mockImplementation(serveBytes(buildSafetensorsHeader([
      'lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight',
      'lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight',
    ])) as any);
    await expect(textSafetensorsRefusal('file:///x/style.safetensors')).resolves.toBeNull();
  });
});
