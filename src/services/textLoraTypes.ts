/**
 * Text LoRA types. No imports, so the store and the services can both name them without a cycle.
 */

/** A GGUF LoRA adapter for the local text engine (llama.rn), applied to the loaded base model. */
export interface TextLora {
  id: string;
  name: string;
  /** Absolute path of the copy inside the app's text_loras folder. */
  path: string;
  /** llama.cpp adapter scale, 0-2 in the UI (llama.rn `scaled`, JSIParams.cpp:126). */
  scale: number;
  enabled: boolean;
}

/** One per-request override for an adapter the user's llama-server has loaded. */
export interface RemoteTextLoraScale {
  /** The adapter id the server lists at GET /lora-adapters. */
  id: number;
  scale: number;
}

export const TEXT_LORA_SCALE_MIN = 0;
export const TEXT_LORA_SCALE_MAX = 2;
