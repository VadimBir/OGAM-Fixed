# sd.cpp image engine: memory audit (sd.cpp @ c92d73c, 2026-11-18)

All numbers are COMPUTED from source rules and the published SD architectures, never measured on a
device. sd.cpp paths are relative to a checkout of leejet/stable-diffusion.cpp at
`c92d73c408515c94beef32161bb5960764fde7a0` (pinned in `scripts/build-sdcpp.sh`).

## 1. What a load-time `wtype` really converts

| Tensor | Type in RAM | Proof |
| --- | --- | --- |
| Linear weight, row length % 32 == 0, name not excluded | requested wtype | `src/model_loader.cpp:1511-1546`, `src/model/common/ggml_block.hpp:159-163` |
| Conv2d weight (3x3 AND 1x1) | always f16, whatever the file or wtype | `ggml_block.hpp:444-445` |
| Bias, LayerNorm, GroupNorm | always f32 | `ggml_block.hpp:165, 448, 750-754, 800-803` |
| Names containing `embedding`, `time_embed.`, `label_emb.` | file type | `model_loader.cpp:1532-1536` |

SD 1.5 (fp16 file), MiB, script `calc/sd15.py` (LDM config, parameter totals match the published
859.5 M / 123.1 M / 49.5 M + 34.2 M):

| | UNet | CLIP | VAE dec | txt2img total |
| --- | --- | --- | --- | --- |
| params (M) | 859.5 (conv 589.0, linear 268.0) | 123.1 (embed 38.0, linear 84.9) | 49.5 (all conv) | 1032 |
| auto (f16) | 1640 | 235 | 94 | 1970 |
| q8_0 | 1401 | 159 | 94 | 1654 |
| q4_0 | 1273 | 119 | 94 | 1486 |
| naive "all 4-bit" (0.5 B/param) | | | | 492 |

q4_0 saves 484 MiB of 1970 (25 %), not 75 %. Upstream's own table agrees in direction:
`docs/quantization_and_gguf.md:13-16` (SD1.x 512 px: f16 ~2.3G, q4_0 ~2.0G).
Linear share of the UNet: SD1 0.312, SD2 0.346 (`calc/sd2.py`), SDXL 0.867 (`calc/sdxl.py`).

## 2. Load path

- Lazy: weights load on first use (first generation), not in `new_sd_ctx`:
  `sd_ctx_params_init` sets `eager_load = false` (`src/stable-diffusion.cpp:337`,
  `include/stable-diffusion.h:240`).
- mmap: `enable_mmap` maps the whole file; a tensor is used zero-copy only when file type == RAM type
  (`model_loader.cpp:875-936`, `model_manager.cpp:819-829`). fp16 file + auto: weights are
  file-backed pages. Any converted tensor (quant override, fp32 file) is an anonymous heap copy.
- No full-file read. Per loader thread: `read_buffer` grows to the largest converted source tensor
  and lives until the thread ends; `convert_tensor` allocates an f32 copy per quantized tensor
  (`model_loader.cpp:1173-1179, 112-160`). SD1.5 largest Linear (ff.net.0.proj at 1280 ch,
  13.1 M params): 25 MiB f16 + 50 MiB f32 = <= 75 MiB per thread; threads = `n_threads`
  (`model_loader.cpp:188`). Same-type tensors read straight into the destination.
- Residency: CPU-only build -> auto-fit has no GPU -> plan invalid -> params stay on CPU
  (`src/core/backend_fit.cpp:205-216, 441-447`). Weights are released at module completion only
  for `params_backend=disk` (`model_manager.cpp:1537-1547`, `src/core/ggml_runner.cpp:478-497`,
  `docs/performance.md` "Use disk params").
- LoRA: AUTO picks `immediately` unless the FILE has quantized tensors (`src/pipeline/diffusion_engine.cpp:1019-1034`
  reads `get_wtype_stat`, `model_loader.cpp:686-697`, which ignores the load-time override).
  `immediately` + mmap maps the file copy-on-write (`diffusion_engine.cpp:1054-1058`).
- Legacy flags `free_params_immediately`, `keep_clip_on_cpu`, `vae_decode_only` do not exist at this
  commit. The AutoEncoderKL is built with its encoder (`src/pipeline/model_builders.cpp:557-563`).

## 3. Pre-quantized GGUF

`sd-cli -M convert --type q4_0` applies the same `tensor_should_be_converted` (`src/convert.cpp:33-52`)
and writes GGUF unless the output ends in `.safetensors` (`convert.cpp:331`). Loading it: q4_0
Linear tensors match their RAM type -> mmap, no conversion transient. The app does not accept it
today: `.gguf` imports go to the text importer (`src/screens/ModelsScreen/useModelsScreen.ts:125-128,
175-182`; `SDCPP_CHECKPOINT_EXTENSIONS` in `src/services/sdCppModelDetection.ts:29`). The JNI and
Kotlin load path do not check the extension. Tool: `scripts/sdcpp-convert-gguf.sh`.

## 4. JNI options (this pass)

`nativeLoadEx` adds `paramsBackend`, `loraApplyMode`, `vaeConvDirect`; `nativeLoad` is kept.
Default changes: a quantized `wtype` resolves LoRA AUTO to `at_runtime` (upstream intent,
`docs/lora.md:19-25`); `diffusion_flash_attn` is always on (upstream default false,
`src/stable-diffusion.cpp:339`). Flash attention is `ggml_flash_attn_ext` with K/V cast to f16 and
f32 precision (`src/core/ggml_extend.cpp:616-720`); when the backend rejects the op it falls back to
plain attention (`ggml_extend.cpp:747-772`). ggml @ 4bf5f600: CPU supports it (online softmax per row,
`src/ggml-cpu/ops.cpp:8509-8613`), Vulkan needs head dims % 8 (`src/ggml-vulkan/ggml-vulkan.cpp:18270-18276`),
OpenCL lists head dims {40,64,80,96,112,128,...} and skips Adreno E17 compilers
(`src/ggml-opencl/ggml-opencl.cpp:7384-7400`). No separate "flash attention 2" option exists in sd.cpp
or llama.cpp (llama: `LLAMA_FLASH_ATTN_TYPE_AUTO/DISABLED/ENABLED`, `include/llama.h:191-193`).
Needs a rebuilt `libsdcpp_jni.so` (`scripts/build-sdcpp.sh`).

## Open (device measurement only)

- Real compute buffers: sd.cpp logs "compute buffer size" per runner (`docs/performance.md:16-18`).
- Whether Android lmkd counts mmap'd clean pages against the app at kill time.
- CPU vs OpenCL/Vulkan vs local-dream NPU speed on the S26.
