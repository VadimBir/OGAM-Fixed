// JNI bridge: ai.offgridmobile.sdcpp.SdCppNative -> stable-diffusion.cpp C API.
// One sd_ctx_t per load (returned to Kotlin as a jlong handle). Generation is serialized by
// g_gen_mutex; sd_cancel_generation is safe to call from another thread while it runs.
#include <jni.h>
#include <android/log.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
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
  jclass cls = env->FindClass("java/lang/RuntimeException");
  if (cls) env->ThrowNew(cls, msg);
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
  std::string info = std::string("sd.cpp ") + sd_version() + " (" + sd_commit() + ") | " + sd_get_system_info();
  return env->NewStringUTF(info.c_str());
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeLoad(JNIEnv* env, jclass, jstring jModelPath,
                                                    jstring jVaePath, jstring jTaesdPath,
                                                    jint nThreads, jstring jWtype,
                                                    jboolean flashAttn, jboolean mmap) {
  // Every path field is a (possibly empty) string, exactly as the upstream CLI passes them.
  const std::string model = to_std(env, jModelPath);
  const std::string vae = to_std(env, jVaePath);
  const std::string taesd = to_std(env, jTaesdPath);
  const std::string wtype = to_std(env, jWtype);
  const std::string empty;

  sd_ctx_params_t p;
  sd_ctx_params_init(&p);
  p.model_path = model.c_str();
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
  p.vae_path = vae.c_str();
  p.audio_vae_path = empty.c_str();
  p.audio_encoder_path = empty.c_str();
  p.taesd_path = taesd.c_str();
  p.control_net_path = empty.c_str();
  p.ip_adapter_path = empty.c_str();
  p.motion_module_path = empty.c_str();
  p.photo_maker_path = empty.c_str();
  p.pulid_weights_path = empty.c_str();
  p.tensor_type_rules = empty.c_str();
  p.max_vram = empty.c_str();
  p.backend = empty.c_str();
  p.params_backend = empty.c_str();
  p.split_mode = empty.c_str();
  p.rpc_servers = empty.c_str();
  p.model_args = nullptr;
  if (nThreads > 0) p.n_threads = nThreads;
  // "auto"/"" keeps the checkpoint's own tensor types; q8_0/q4_0/... quantizes in memory at load
  // (the file on disk is never rewritten).
  if (!wtype.empty() && wtype != "auto") {
    const enum sd_type_t t = str_to_sd_type(wtype.c_str());
    if (t != SD_TYPE_COUNT) p.wtype = t;
  }
  p.flash_attn = flashAttn == JNI_TRUE;
  p.diffusion_flash_attn = flashAttn == JNI_TRUE;
  p.enable_mmap = mmap == JNI_TRUE;

  LOGI("load model=%s vae=%s taesd=%s threads=%d wtype=%s flash=%d mmap=%d", model.c_str(),
       vae.c_str(), taesd.c_str(), p.n_threads, wtype.c_str(), (int)p.flash_attn, (int)p.enable_mmap);
  sd_ctx_t* ctx = new_sd_ctx(&p);
  if (ctx == nullptr) {
    throw_java(env, "stable-diffusion.cpp failed to load the checkpoint (see SdCppJNI logcat)");
    return 0;
  }
  if (!sd_ctx_supports_image_generation(ctx)) {
    free_sd_ctx(ctx);
    throw_java(env, "Checkpoint loaded but is not an image-generation model");
    return 0;
  }
  LOGI("loaded: %s", sd_get_model_version_name(ctx));
  return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeModelVersion(JNIEnv* env, jclass, jlong handle) {
  auto* ctx = reinterpret_cast<sd_ctx_t*>(handle);
  return env->NewStringUTF(ctx ? sd_get_model_version_name(ctx) : "");
}

extern "C" JNIEXPORT void JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeFree(JNIEnv*, jclass, jlong handle) {
  auto* ctx = reinterpret_cast<sd_ctx_t*>(handle);
  if (ctx == nullptr) return;
  std::lock_guard<std::mutex> lock(g_gen_mutex);  // never free under a running generation
  free_sd_ctx(ctx);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeCancel(JNIEnv*, jclass, jlong handle) {
  auto* ctx = reinterpret_cast<sd_ctx_t*>(handle);
  if (ctx) sd_cancel_generation(ctx, SD_CANCEL_ALL);
}

// Returns [int32 LE width][int32 LE height][int32 LE channels][pixels...] or null on failure.
// The resolved sampler/scheduler names are written back into outNames[0..1] for logging.
extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_offgridmobile_sdcpp_SdCppNative_nativeGenerate(
    JNIEnv* env, jclass, jlong handle, jstring jPrompt, jstring jNegative, jint width,
    jint height, jint steps, jfloat cfg, jlong seed, jstring jSampler, jstring jScheduler,
    jint clipSkip, jobjectArray jLoraPaths, jfloatArray jLoraMults, jobjectArray outNames,
    jobject listener) {
  auto* ctx = reinterpret_cast<sd_ctx_t*>(handle);
  if (ctx == nullptr) {
    throw_java(env, "ERR_NO_MODEL: no stable-diffusion.cpp model loaded");
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(g_gen_mutex);
  sd_cancel_generation(ctx, SD_CANCEL_RESET);

  const std::string prompt = to_std(env, jPrompt);
  const std::string negative = to_std(env, jNegative);
  const std::string sampler = to_std(env, jSampler);
  const std::string scheduler = to_std(env, jScheduler);

  std::vector<std::string> lora_paths;
  std::vector<sd_lora_t> loras;
  const jsize lora_count = jLoraPaths ? env->GetArrayLength(jLoraPaths) : 0;
  if (lora_count > 0) {
    lora_paths.reserve(lora_count);
    jfloat* mults = jLoraMults ? env->GetFloatArrayElements(jLoraMults, nullptr) : nullptr;
    const jsize mult_count = jLoraMults ? env->GetArrayLength(jLoraMults) : 0;
    for (jsize i = 0; i < lora_count; ++i) {
      auto js = static_cast<jstring>(env->GetObjectArrayElement(jLoraPaths, i));
      lora_paths.push_back(to_std(env, js));
      env->DeleteLocalRef(js);
    }
    for (jsize i = 0; i < lora_count; ++i) {
      sd_lora_t l{};
      l.is_high_noise = false;
      l.multiplier = (mults && i < mult_count) ? mults[i] : 1.0f;
      l.path = lora_paths[i].c_str();
      loras.push_back(l);
    }
    if (mults) env->ReleaseFloatArrayElements(jLoraMults, mults, JNI_ABORT);
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
    env->DeleteLocalRef(m);
    env->DeleteLocalRef(s);
  }

  LOGI("generate %dx%d steps=%d cfg=%.2f seed=%lld sampler=%s scheduler=%s loras=%d clip_skip=%d",
       width, height, steps, cfg, (long long)seed, sd_sample_method_name(method),
       sd_scheduler_name(sched), (int)loras.size(), gp.clip_skip);

  if (listener != nullptr) {
    jclass cls = env->GetObjectClass(listener);
    g_on_progress = env->GetMethodID(cls, "onProgress", "(IIF)V");
    env->DeleteLocalRef(cls);
    if (g_on_progress != nullptr) g_listener = env->NewGlobalRef(listener);
    if (env->ExceptionCheck()) env->ExceptionClear();
  }

  sd_image_t* images = nullptr;
  int num_images = 0;
  const bool ok = generate_image(ctx, &gp, &images, &num_images);

  if (g_listener != nullptr) {
    env->DeleteGlobalRef(g_listener);
    g_listener = nullptr;
  }
  g_on_progress = nullptr;

  if (!ok || images == nullptr || num_images < 1 || images[0].data == nullptr) {
    if (images) free_sd_images(images, num_images);
    throw_java(env, "stable-diffusion.cpp generation failed or was cancelled");
    return nullptr;
  }

  const sd_image_t& img = images[0];
  const size_t pixels = static_cast<size_t>(img.width) * img.height * img.channel;
  jbyteArray out = env->NewByteArray(static_cast<jsize>(12 + pixels));
  if (out != nullptr) {
    const int32_t hdr[3] = {(int32_t)img.width, (int32_t)img.height, (int32_t)img.channel};
    env->SetByteArrayRegion(out, 0, 12, reinterpret_cast<const jbyte*>(hdr));
    env->SetByteArrayRegion(out, 12, static_cast<jsize>(pixels), reinterpret_cast<const jbyte*>(img.data));
  }
  free_sd_images(images, num_images);
  return out;
}
