// JNI bridge: ai.offgridmobile.sdcpp.SdCppNative -> stable-diffusion.cpp C API.
// One sd_ctx_t per load (returned to Kotlin as a jlong handle). Generation is serialized by
// g_gen_mutex; sd_cancel_generation is called from another thread while it runs.
// Lock order: g_gen_mutex before g_ctx_mutex. g_ctx_mutex guards the live-handle registry, so
// nativeCancel / nativeModelVersion never touch a context nativeFree already released.
// No C++ exception may cross the JNI boundary (it aborts the process): every entry point catches
// and rethrows as java.lang.RuntimeException.
#include <jni.h>
#include <android/log.h>

#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "stable-diffusion.h"

#define TAG "SdCppJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

JavaVM* g_vm = nullptr;
std::mutex g_gen_mutex;
// Progress listener of the generation currently running (global ref), null otherwise.
jobject g_listener = nullptr;
jmethodID g_on_progress = nullptr;

// Per-context state. cancel_requested is set by nativeCancel, so a cancel issued before the
// generation takes g_gen_mutex is not erased by the SD_CANCEL_RESET at generation start.
struct CtxState {
  std::atomic<bool> cancel_requested{false};
};
std::mutex g_ctx_mutex;
std::unordered_map<sd_ctx_t*, std::unique_ptr<CtxState>> g_live;  // guarded by g_ctx_mutex

// Caller holds g_ctx_mutex. Null when the handle was never returned by load or was already freed.
CtxState* find_live(sd_ctx_t* ctx) {
  auto it = g_live.find(ctx);
  return it == g_live.end() ? nullptr : it->second.get();
}

// Releases the generation's progress listener on every exit path (normal, Java error, C++ throw).
struct ListenerGuard {
  JNIEnv* env;
  explicit ListenerGuard(JNIEnv* e) : env(e) {}
  ListenerGuard(const ListenerGuard&) = delete;
  ListenerGuard& operator=(const ListenerGuard&) = delete;
  ~ListenerGuard() {
    if (g_listener != nullptr) {
      env->DeleteGlobalRef(g_listener);  // legal with a pending Java exception
      g_listener = nullptr;
    }
    g_on_progress = nullptr;
  }
};

std::string to_std(JNIEnv* env, jstring s) {
  if (s == nullptr) return {};
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string out = c ? c : "";
  if (c) env->ReleaseStringUTFChars(s, c);
  return out;
}

void log_cb(enum sd_log_level_t level, const char* text, void*) {
  int prio = ANDROID_LOG_DEBUG;
  if (level == SD_LOG_ERROR) prio = ANDROID_LOG_ERROR;
  else if (level == SD_LOG_WARN) prio = ANDROID_LOG_WARN;
  else if (level == SD_LOG_INFO) prio = ANDROID_LOG_INFO;
  __android_log_print(prio, TAG, "%s", text ? text : "");
}

void progress_cb(int step, int steps, float time, void*) {
  // Fires for tensor loading too; only forwarded while a generation owns a listener.
  if (g_vm == nullptr || g_listener == nullptr || g_on_progress == nullptr) return;
  JNIEnv* env = nullptr;
  bool attached = false;
  if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
    attached = true;
  }
  env->CallVoidMethod(g_listener, g_on_progress, (jint)step, (jint)steps, (jfloat)time);
  if (env->ExceptionCheck()) env->ExceptionClear();
  if (attached) g_vm->DetachCurrentThread();
}

void throw_java(JNIEnv* env, const char* msg) {
  if (env->ExceptionCheck()) return;  // keep the first (already pending) Java exception
  jclass cls = env->FindClass("java/lang/RuntimeException");
  if (cls) env->ThrowNew(cls, msg);
}

// Maps an escaped C++ exception to a Java exception. Call only from inside a catch block.
// Fixed buffer, no heap: the message must still build after std::bad_alloc.
void throw_from_current(JNIEnv* env, const char* where) noexcept {
  char msg[512];
  try {
    throw;
  } catch (const std::bad_alloc&) {
    snprintf(msg, sizeof(msg), "ERR_OUT_OF_MEMORY: native allocation failed in %s", where);
  } catch (const std::exception& e) {
    snprintf(msg, sizeof(msg), "stable-diffusion.cpp error in %s: %s", where, e.what());
  } catch (...) {
    snprintf(msg, sizeof(msg), "stable-diffusion.cpp unknown native error in %s", where);
  }
  LOGE("%s", msg);
  throw_java(env, msg);
}

// Block-quantized (or ternary/fp4) weight types; the float and integer storage types are not.
bool is_quantized_type(enum sd_type_t t) {
  switch (t) {
    case SD_TYPE_F32: case SD_TYPE_F16: case SD_TYPE_BF16: case SD_TYPE_F64:
    case SD_TYPE_F8_E4M3: case SD_TYPE_F8_E5M2:
    case SD_TYPE_I8: case SD_TYPE_I16: case SD_TYPE_I32: case SD_TYPE_I64:
    case SD_TYPE_COUNT:
      return false;
    default:
      return true;
  }
}

struct LoadOptions {
  std::string model, vae, taesd, wtype;
  int n_threads = 0;
  bool flash_attn = false;
  bool mmap = false;
  // "" = every module keeps its weights resident (upstream default). "disk" or "te=disk,vae=disk,
  // diffusion=disk" reloads that module's weights from the file per run and frees them when the
  // module finishes: lower resident RAM, slower generations, identical output.
  std::string params_backend;
  // "" / "auto" / "immediately" / "at_runtime" (str_to_lora_apply_mode).
  std::string lora_apply_mode;
  // ggml_conv_2d_direct in the VAE: no im2col buffer during decode; upstream notes it may be slower.
  bool vae_conv_direct = false;
};

sd_ctx_t* load_ctx(JNIEnv* env, const LoadOptions& o) {
  const std::string empty;

  sd_ctx_params_t p;
  sd_ctx_params_init(&p);
  p.model_path = o.model.c_str();
  p.clip_l_path = empty.c_str();
  p.clip_g_path = empty.c_str();
  p.clip_vision_path = empty.c_str();
  p.t5xxl_path = empty.c_str();
  p.llm_path = empty.c_str();
  p.llm_vision_path = empty.c_str();
  p.tokenizer = empty.c_str();
  p.diffusion_model_path = empty.c_str();
  p.high_noise_diffusion_model_path = empty.c_str();
  p.uncond_diffusion_model_path = empty.c_str();
  p.embeddings_connectors_path = empty.c_str();
  p.vae_path = o.vae.c_str();
  p.audio_vae_path = empty.c_str();
  p.audio_encoder_path = empty.c_str();
  p.taesd_path = o.taesd.c_str();
  p.control_net_path = empty.c_str();
  p.ip_adapter_path = empty.c_str();
  p.motion_module_path = empty.c_str();
  p.photo_maker_path = empty.c_str();
  p.pulid_weights_path = empty.c_str();
  p.tensor_type_rules = empty.c_str();
  p.max_vram = empty.c_str();
  p.backend = empty.c_str();
  p.params_backend = o.params_backend.c_str();
  p.split_mode = empty.c_str();
  p.rpc_servers = empty.c_str();
  p.model_args = nullptr;
  if (o.n_threads > 0) p.n_threads = o.n_threads;
  // "auto"/"" keeps the checkpoint's own tensor types; q8_0/q4_0/... quantizes in memory at load
  // (the file on disk is never rewritten). Only Linear weights whose row length is a multiple of the
  // block size convert; Conv2d weights are always f16 and biases/norms f32 (ggml_block.hpp).
  bool quantized_override = false;
  if (!o.wtype.empty() && o.wtype != "auto") {
    const enum sd_type_t t = str_to_sd_type(o.wtype.c_str());
    if (t != SD_TYPE_COUNT) {
      p.wtype = t;
      quantized_override = is_quantized_type(t);
    }
  }
  // flash_attn covers the text encoder + VAE; the diffusion model always uses it (owner requirement).
  // This library is built CPU-only (CMakeLists.txt enables no GPU backend), so there is no backend
  // fallback here. If a backend rejects the flash op, ggml_ext_attention_ext uses plain attention
  // (ggml_backend_supports_op, core/ggml_extend.cpp at the pinned commit; not re-verified here).
  p.flash_attn = o.flash_attn;
  p.diffusion_flash_attn = true;
  p.enable_mmap = o.mmap;
  p.vae_conv_direct = o.vae_conv_direct;
  // Upstream AUTO picks at_runtime when the weights are quantized (docs/lora.md), but it reads the
  // FILE's tensor types (ModelLoader::get_wtype_stat), not the load-time wtype override. So a
  // safetensors loaded as q4_0 would merge LoRAs into the re-quantized weights and map the file
  // copy-on-write. Resolve AUTO as upstream intends for a quantized override.
  enum lora_apply_mode_t lora_mode = LORA_APPLY_AUTO;
  if (!o.lora_apply_mode.empty()) {
    const enum lora_apply_mode_t m = str_to_lora_apply_mode(o.lora_apply_mode.c_str());
    if (m != LORA_APPLY_MODE_COUNT) lora_mode = m;
  }
  if (lora_mode == LORA_APPLY_AUTO && quantized_override) lora_mode = LORA_APPLY_AT_RUNTIME;
  p.lora_apply_mode = lora_mode;

  LOGI("load model=%s vae=%s taesd=%s threads=%d wtype=%s flash=%d diffusion_flash=%d mmap=%d params_backend=%s lora_apply=%s vae_conv_direct=%d",
       o.model.c_str(), o.vae.c_str(), o.taesd.c_str(), p.n_threads, o.wtype.c_str(), (int)p.flash_attn,
       (int)p.diffusion_flash_attn, (int)p.enable_mmap, o.params_backend.c_str(), sd_lora_apply_mode_name(p.lora_apply_mode),
       (int)p.vae_conv_direct);
  sd_ctx_t* ctx = new_sd_ctx(&p);
  if (ctx == nullptr) {
    throw_java(env, "stable-diffusion.cpp failed to load the checkpoint (see SdCppJNI logcat)");
    return nullptr;
  }
  if (!sd_ctx_supports_image_generation(ctx)) {
    free_sd_ctx(ctx);
    throw_java(env, "Checkpoint loaded but is not an image-generation model");
    return nullptr;
  }
  LOGI("loaded: %s", sd_get_model_version_name(ctx));
  try {
    auto state = std::make_unique<CtxState>();
    std::lock_guard<std::mutex> reg(g_ctx_mutex);
    g_live.emplace(ctx, std::move(state));
  } catch (...) {
    free_sd_ctx(ctx);  // never leak the loaded weights when registration fails
    throw;
  }
  return ctx;
}

// Shared body of nativeLoad / nativeLoadEx: no C++ exception (std::bad_alloc while loading
// weights) reaches the JVM; it becomes a RuntimeException and the handle is 0.
jlong load_guarded(JNIEnv* env, const LoadOptions& o) {
  try {
    return reinterpret_cast<jlong>(load_ctx(env, o));
  } catch (...) {
    throw_from_current(env, "load");
    return 0;
  }
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  g_vm = vm;
  sd_set_log_callback(log_cb, nullptr);
  sd_set_progress_callback(progress_cb, nullptr);
  return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeSystemInfo(JNIEnv* env, jclass) {
  try {
    std::string info = std::string("sd.cpp ") + sd_version() + " (" + sd_commit() + ") | " + sd_get_system_info();
    return env->NewStringUTF(info.c_str());
  } catch (...) {
    throw_from_current(env, "nativeSystemInfo");
    return nullptr;
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeLoad(JNIEnv* env, jclass, jstring jModelPath,
                                                    jstring jVaePath, jstring jTaesdPath,
                                                    jint nThreads, jstring jWtype,
                                                    jboolean flashAttn, jboolean mmap) {
  // Every path field is a (possibly empty) string, exactly as the upstream CLI passes them.
  try {
    LoadOptions o;
    o.model = to_std(env, jModelPath);
    o.vae = to_std(env, jVaePath);
    o.taesd = to_std(env, jTaesdPath);
    o.wtype = to_std(env, jWtype);
    if (env->ExceptionCheck()) return 0;  // GetStringUTFChars out of memory
    o.n_threads = nThreads;
    o.flash_attn = flashAttn == JNI_TRUE;
    o.mmap = mmap == JNI_TRUE;
    return load_guarded(env, o);
  } catch (...) {
    throw_from_current(env, "nativeLoad");
    return 0;
  }
}

// nativeLoad plus the memory options; nativeLoad stays so an older Kotlin side keeps linking.
extern "C" JNIEXPORT jlong JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeLoadEx(JNIEnv* env, jclass, jstring jModelPath,
                                                      jstring jVaePath, jstring jTaesdPath,
                                                      jint nThreads, jstring jWtype,
                                                      jboolean flashAttn, jboolean mmap,
                                                      jstring jParamsBackend, jstring jLoraApplyMode,
                                                      jboolean vaeConvDirect) {
  try {
    LoadOptions o;
    o.model = to_std(env, jModelPath);
    o.vae = to_std(env, jVaePath);
    o.taesd = to_std(env, jTaesdPath);
    o.wtype = to_std(env, jWtype);
    o.n_threads = nThreads;
    o.flash_attn = flashAttn == JNI_TRUE;
    o.mmap = mmap == JNI_TRUE;
    o.params_backend = to_std(env, jParamsBackend);
    o.lora_apply_mode = to_std(env, jLoraApplyMode);
    if (env->ExceptionCheck()) return 0;  // GetStringUTFChars out of memory
    o.vae_conv_direct = vaeConvDirect == JNI_TRUE;
    return load_guarded(env, o);
  } catch (...) {
    throw_from_current(env, "nativeLoadEx");
    return 0;
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeModelVersion(JNIEnv* env, jclass, jlong handle) {
  auto* ctx = reinterpret_cast<sd_ctx_t*>(handle);
  try {
    std::string name;
    if (ctx != nullptr) {
      std::lock_guard<std::mutex> reg(g_ctx_mutex);  // nativeFree cannot release ctx meanwhile
      if (find_live(ctx) != nullptr) {
        const char* n = sd_get_model_version_name(ctx);
        name = n ? n : "";
      } else {
        LOGE("nativeModelVersion: handle is not a live context");
      }
    }
    return env->NewStringUTF(name.c_str());
  } catch (...) {
    throw_from_current(env, "nativeModelVersion");
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeFree(JNIEnv* env, jclass, jlong handle) {
  auto* ctx = reinterpret_cast<sd_ctx_t*>(handle);
  if (ctx == nullptr) return;
  try {
    std::lock_guard<std::mutex> lock(g_gen_mutex);  // never free under a running generation
    {
      std::lock_guard<std::mutex> reg(g_ctx_mutex);
      auto it = g_live.find(ctx);
      if (it == g_live.end()) {
        LOGE("nativeFree: handle is not a live context (double free ignored)");
        return;
      }
      g_live.erase(it);  // from here nativeCancel / nativeModelVersion no longer see ctx
    }
    free_sd_ctx(ctx);
  } catch (...) {
    throw_from_current(env, "nativeFree");
  }
}

// Sets the context's cancel_requested flag (consumed by nativeGenerate, so a cancel that lands
// before the generation starts is honoured) and stops a running generation. Holding g_ctx_mutex
// keeps nativeFree from releasing ctx during the call.
extern "C" JNIEXPORT void JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeCancel(JNIEnv* env, jclass, jlong handle) {
  auto* ctx = reinterpret_cast<sd_ctx_t*>(handle);
  if (ctx == nullptr) return;
  try {
    std::lock_guard<std::mutex> reg(g_ctx_mutex);
    CtxState* st = find_live(ctx);
    if (st == nullptr) {
      LOGI("nativeCancel: handle already freed, nothing to cancel");
      return;
    }
    st->cancel_requested.store(true);
    sd_cancel_generation(ctx, SD_CANCEL_ALL);
  } catch (...) {
    throw_from_current(env, "nativeCancel");
  }
}

// Message prefix "CANCELLED:" marks a user cancel (nativeCancel during or before this generation);
// any other RuntimeException is a real failure.
static const char* const kCancelledMsg = "CANCELLED: stable-diffusion.cpp generation was cancelled";

// Returns [int32 LE width][int32 LE height][int32 LE channels][pixels...] or null on failure.
// The resolved sampler/scheduler names are written back into outNames[0..1] for logging.
extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeGenerate(
    JNIEnv* env, jclass, jlong handle, jstring jPrompt, jstring jNegative, jint width,
    jint height, jint steps, jfloat cfg, jlong seed, jstring jSampler, jstring jScheduler,
    jint clipSkip, jboolean vaeTiling, jobjectArray jLoraPaths, jfloatArray jLoraMults, jobjectArray outNames,
    jobject listener) {
  auto* ctx = reinterpret_cast<sd_ctx_t*>(handle);
  if (ctx == nullptr) {
    throw_java(env, "ERR_NO_MODEL: no stable-diffusion.cpp model loaded");
    return nullptr;
  }
  try {
    std::lock_guard<std::mutex> lock(g_gen_mutex);
    // st stays valid while g_gen_mutex is held: nativeFree needs g_gen_mutex to erase it.
    CtxState* st = nullptr;
    {
      std::lock_guard<std::mutex> reg(g_ctx_mutex);
      st = find_live(ctx);
      if (st == nullptr) {
        throw_java(env, "ERR_NO_MODEL: stable-diffusion.cpp model was unloaded");
        return nullptr;
      }
      if (st->cancel_requested.exchange(false)) {
        LOGI("generate: cancel requested before start, not running");
        throw_java(env, kCancelledMsg);
        return nullptr;
      }
      // Clears the core's flag left by a previous cancel; an early cancel is kept in st above.
      sd_cancel_generation(ctx, SD_CANCEL_RESET);
    }

    const std::string prompt = to_std(env, jPrompt);
    const std::string negative = to_std(env, jNegative);
    const std::string sampler = to_std(env, jSampler);
    const std::string scheduler = to_std(env, jScheduler);
    if (env->ExceptionCheck()) return nullptr;  // GetStringUTFChars out of memory

    std::vector<std::string> lora_paths;
    std::vector<sd_lora_t> loras;
    const jsize lora_count = jLoraPaths ? env->GetArrayLength(jLoraPaths) : 0;
    if (lora_count > 0) {
      lora_paths.reserve(lora_count);
      // Copied, not pinned: no Get/Release pair to leak if anything below throws.
      std::vector<jfloat> mults;
      if (jLoraMults) {
        const jsize mult_count = env->GetArrayLength(jLoraMults);
        mults.resize(static_cast<size_t>(mult_count));
        if (mult_count > 0) env->GetFloatArrayRegion(jLoraMults, 0, mult_count, mults.data());
      }
      for (jsize i = 0; i < lora_count; ++i) {
        auto js = static_cast<jstring>(env->GetObjectArrayElement(jLoraPaths, i));
        lora_paths.push_back(to_std(env, js));
        if (js) env->DeleteLocalRef(js);
      }
      if (env->ExceptionCheck()) return nullptr;
      for (jsize i = 0; i < lora_count; ++i) {
        sd_lora_t l{};
        l.is_high_noise = false;
        l.multiplier = static_cast<size_t>(i) < mults.size() ? mults[i] : 1.0f;
        l.path = lora_paths[i].c_str();
        loras.push_back(l);
      }
    }

    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    gp.prompt = prompt.c_str();
    gp.negative_prompt = negative.c_str();
    gp.width = width;
    gp.height = height;
    gp.seed = seed;
    gp.batch_count = 1;
    gp.clip_skip = clipSkip > 0 ? clipSkip : -1;
    // Defaults from sd_img_gen_params_init: 32-latent (256 px) tiles, 0.5 overlap.
    gp.vae_tiling_params.enabled = vaeTiling == JNI_TRUE;
    gp.loras = loras.empty() ? nullptr : loras.data();
    gp.lora_count = static_cast<uint32_t>(loras.size());
    gp.sample_params.sample_steps = steps;
    gp.sample_params.guidance.txt_cfg = cfg;
    // Unknown/empty name -> the checkpoint's own default (resolved explicitly, never left implicit).
    enum sample_method_t method = sampler.empty() ? SAMPLE_METHOD_COUNT : str_to_sample_method(sampler.c_str());
    if (method == SAMPLE_METHOD_COUNT) method = sd_get_default_sample_method(ctx);
    enum scheduler_t sched = scheduler.empty() ? SCHEDULER_COUNT : str_to_scheduler(scheduler.c_str());
    if (sched == SCHEDULER_COUNT) sched = sd_get_default_scheduler(ctx, method);
    gp.sample_params.sample_method = method;
    gp.sample_params.scheduler = sched;
    if (outNames && env->GetArrayLength(outNames) >= 2) {
      jstring m = env->NewStringUTF(sd_sample_method_name(method));
      jstring s = env->NewStringUTF(sd_scheduler_name(sched));
      env->SetObjectArrayElement(outNames, 0, m);
      env->SetObjectArrayElement(outNames, 1, s);
      if (m) env->DeleteLocalRef(m);
      if (s) env->DeleteLocalRef(s);
      if (env->ExceptionCheck()) return nullptr;
    }

    LOGI("generate %dx%d steps=%d cfg=%.2f seed=%lld sampler=%s scheduler=%s loras=%d clip_skip=%d vae_tiling=%d",
         width, height, steps, cfg, (long long)seed, sd_sample_method_name(method),
         sd_scheduler_name(sched), (int)loras.size(), gp.clip_skip, (int)gp.vae_tiling_params.enabled);

    // Declared after `lock`, so it releases g_listener before g_gen_mutex on every exit path.
    ListenerGuard listener_guard(env);
    if (listener != nullptr) {
      jclass cls = env->GetObjectClass(listener);
      g_on_progress = env->GetMethodID(cls, "onProgress", "(IIF)V");
      env->DeleteLocalRef(cls);
      if (g_on_progress != nullptr) g_listener = env->NewGlobalRef(listener);
      if (env->ExceptionCheck()) {
        LOGE("generate: progress listener unavailable, running without progress");
        env->ExceptionClear();
      }
    }

    sd_image_t* images = nullptr;
    int num_images = 0;
    const bool ok = generate_image(ctx, &gp, &images, &num_images);
    struct ImagesGuard {
      sd_image_t* p;
      int n;
      ~ImagesGuard() {
        if (p) free_sd_images(p, n);
      }
    } images_guard{images, num_images};

    bool cancelled = false;
    {
      std::lock_guard<std::mutex> reg(g_ctx_mutex);
      cancelled = st->cancel_requested.exchange(false);  // consumed: never leaks into the next run
    }
    if (cancelled) {
      LOGI("generate: cancelled (core ok=%d)", (int)ok);
      throw_java(env, kCancelledMsg);
      return nullptr;
    }
    if (!ok || images == nullptr || num_images < 1 || images[0].data == nullptr) {
      throw_java(env, "stable-diffusion.cpp generation failed (see SdCppJNI logcat)");
      return nullptr;
    }

    const sd_image_t& img = images[0];
    const size_t pixels = static_cast<size_t>(img.width) * img.height * img.channel;
    if (pixels > static_cast<size_t>(INT_MAX) - 12) {
      throw_java(env, "stable-diffusion.cpp image too large for a Java byte array");
      return nullptr;
    }
    jbyteArray out = env->NewByteArray(static_cast<jsize>(12 + pixels));
    if (out != nullptr) {
      const int32_t hdr[3] = {(int32_t)img.width, (int32_t)img.height, (int32_t)img.channel};
      env->SetByteArrayRegion(out, 0, 12, reinterpret_cast<const jbyte*>(hdr));
      env->SetByteArrayRegion(out, 12, static_cast<jsize>(pixels), reinterpret_cast<const jbyte*>(img.data));
    }
    return out;  // null => OutOfMemoryError already pending
  } catch (...) {
    throw_from_current(env, "nativeGenerate");
    return nullptr;
  }
}
