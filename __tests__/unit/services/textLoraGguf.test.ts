/**
 * Text LoRA header detection: GGUF adapter keys (general.type / adapter.type, the keys llama.cpp
 * checks in vendor src/llama-adapter.cpp:202-216), truncated headers, tensor types, safetensors names.
 */
import {
  classifyGgufAdapter,
  nonAcceleratedLoraTypes,
  parseGgufHeader,
  safetensorsKeysFromHeaderText,
} from '../../../src/services/textLoraGguf';
import { buildGguf } from '../../utils/ggufFixture';

describe('textLoraGguf', () => {
  it('detects a GGUF LoRA adapter from general.type / adapter.type', () => {
    const bytes = buildGguf({
      kv: [
        ['general.architecture', 'llama'],
        ['general.type', 'adapter'],
        ['adapter.type', 'lora'],
        ['adapter.lora.alpha', { f32: 16 }],
      ],
      tensors: [
        { name: 'blk.0.attn_q.weight.lora_a', type: 1 },
        { name: 'blk.0.attn_q.weight.lora_b', type: 1 },
      ],
    });
    const header = parseGgufHeader(bytes)!;
    expect(header.truncated).toBe(false);
    expect(header.tensorTypes).toEqual([1, 1]);
    expect(classifyGgufAdapter(header)).toEqual({ kind: 'text-lora', architecture: 'llama', alpha: 16 });
    expect(nonAcceleratedLoraTypes(header.tensorTypes)).toEqual([]);
  });

  it('a model GGUF (general.type model, or none) is not an adapter', () => {
    const model = parseGgufHeader(buildGguf({ kv: [['general.type', 'model'], ['general.architecture', 'qwen3']] }))!;
    expect(classifyGgufAdapter(model)).toEqual({ kind: 'model' });
    const legacy = parseGgufHeader(buildGguf({ kv: [['general.architecture', 'llama']] }))!;
    expect(classifyGgufAdapter(legacy)).toEqual({ kind: 'model' });
  });

  it('an adapter that is not a LoRA is reported with its adapter.type', () => {
    const header = parseGgufHeader(buildGguf({ kv: [['general.type', 'adapter'], ['adapter.type', 'control_vector']] }))!;
    expect(classifyGgufAdapter(header)).toEqual({ kind: 'other-adapter', adapterType: 'control_vector' });
  });

  it('skips array KVs and reads keys after them', () => {
    const header = parseGgufHeader(buildGguf({
      kv: [['tokenizer.ggml.tokens', { strArray: ['a', 'bb', 'ccc'] }], ['general.type', 'adapter'], ['adapter.type', 'lora']],
    }))!;
    expect(header.kv['general.type']).toBe('adapter');
    expect(header.kv['tokenizer.ggml.tokens']).toBeUndefined();
  });

  it('a prefix that ends early is returned as truncated, not thrown', () => {
    const full = buildGguf({ kv: [['general.type', 'adapter'], ['adapter.type', 'lora']], tensors: [{ name: 't.lora_a', type: 8 }] });
    const header = parseGgufHeader(full.subarray(0, full.length - 6))!;
    expect(header.truncated).toBe(true);
    expect(header.kv['adapter.type']).toBe('lora');
  });

  it('non-GGUF bytes are null', () => {
    expect(parseGgufHeader(new Uint8Array([1, 2, 3, 4, 5]))).toBeNull();
  });

  it('flags tensor types not proven on OpenCL/HTP', () => {
    expect(nonAcceleratedLoraTypes([1, 0, 8, 8, 30])).toEqual(['Q8_0', 'BF16']);
  });

  it('reads tensor names from a safetensors JSON header, without __metadata__', () => {
    const text = JSON.stringify({
      __metadata__: { format: 'pt' },
      'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight': { dtype: 'F16', shape: [8, 64], data_offsets: [0, 1] },
      'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight': { dtype: 'F16', shape: [64, 8], data_offsets: [1, 2] },
    });
    expect(safetensorsKeysFromHeaderText(text)).toEqual([
      'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight',
      'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight',
    ]);
  });
});
