package ai.offgridmobile.sdcpp

import android.graphics.Bitmap
import android.util.Log
import ai.offgridmobile.SafePromise
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.Executors

/**
 * Second, isolated image engine: stable-diffusion.cpp (ggml) running raw Stable-Diffusion
 * checkpoints (.safetensors / .ckpt) directly — no conversion. Independent of LocalDreamModule
 * (QNN/MNN): own native lib, own event name, own context. Output PNGs go to the same
 * filesDir/generated_images/<id>.png store so the gallery and delete paths are shared.
 */
class SdCppImageModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val TAG = "SdCppImageModule"
        private const val EVENT_PROGRESS = "SdCppProgress"
        private const val MAX_HEADER_BYTES = 100L * 1024 * 1024

        /** SD latents are 1/8 of the image and the UNet downsamples 3x: snap to 64 px, 64..2048. */
        internal fun snapDimension(px: Int): Int {
            val safe = if (px in 64..2048) px else 512
            return maxOf(64, (safe / 64) * 64)
        }

        /** Parse a .safetensors header: u64 LE length + JSON {name: {dtype, shape, offsets}, __metadata__}. */
        internal fun readSafetensorsHeader(file: File): JSONObject {
            RandomAccessFile(file, "r").use { raf ->
                if (raf.length() < 8) throw IllegalArgumentException("File too small for safetensors")
                val lenBuf = ByteArray(8)
                raf.readFully(lenBuf)
                val headerLen = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).long
                if (headerLen <= 1 || headerLen > MAX_HEADER_BYTES || headerLen > raf.length() - 8) {
                    throw IllegalArgumentException("Invalid safetensors header length $headerLen")
                }
                val header = ByteArray(headerLen.toInt())
                raf.readFully(header)
                return JSONObject(String(header, Charsets.UTF_8))
            }
        }
    }

    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "sdcpp-worker") }

    @Volatile private var handle: Long = 0L
    @Volatile private var loadedModelPath: String? = null
    @Volatile private var generating = false

    override fun getName(): String = "SdCppImageModule"

    private fun reject(promise: Promise, code: String, message: String, t: Throwable? = null) =
        SafePromise(promise, TAG).reject(code, message, t)

    private fun resolve(promise: Promise, value: Any?) = SafePromise(promise, TAG).resolve(value)

    private fun sendEvent(name: String, params: WritableMap) {
        try {
            reactApplicationContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit(name, params)
        } catch (e: Exception) {
            Log.w(TAG, "emit $name failed: ${e.message}")
        }
    }

    override fun getConstants(): MutableMap<String, Any> = hashMapOf(
        "DEFAULT_STEPS" to 20,
        "DEFAULT_GUIDANCE_SCALE" to 7.0,
        "DEFAULT_WIDTH" to 512,
        "DEFAULT_HEIGHT" to 512,
        "SIZE_MULTIPLE" to 64,
        "MIN_SIZE" to 64,
        "MAX_SIZE" to 2048,
    )

    @ReactMethod
    fun isAvailable(promise: Promise) = resolve(promise, SdCppNative.isLoaded)

    @ReactMethod
    fun getSystemInfo(promise: Promise) {
        if (!SdCppNative.isLoaded) return reject(promise, "ERR_NATIVE", SdCppNative.loadErrorMessage())
        resolve(promise, SdCppNative.nativeSystemInfo())
    }

    /** Header-only probe used by JS to route a .safetensors file (SD checkpoint vs LLM vs LoRA). */
    @ReactMethod
    fun readSafetensorsHeader(path: String, promise: Promise) {
        executor.execute {
            try {
                val json = readSafetensorsHeader(File(path))
                val keys = Arguments.createArray()
                val names = json.keys()
                while (names.hasNext()) {
                    val k = names.next()
                    if (k != "__metadata__") keys.pushString(k)
                }
                val meta = Arguments.createMap()
                json.optJSONObject("__metadata__")?.let { m ->
                    val mk = m.keys()
                    while (mk.hasNext()) {
                        val k = mk.next()
                        meta.putString(k, m.optString(k).take(512))
                    }
                }
                resolve(promise, Arguments.createMap().apply {
                    putArray("keys", keys)
                    putMap("metadata", meta)
                })
            } catch (e: Exception) {
                reject(promise, "ERR_SAFETENSORS_HEADER", e.message ?: "Unreadable safetensors header", e)
            }
        }
    }

    @ReactMethod
    fun isModelLoaded(promise: Promise) = resolve(promise, handle != 0L)

    @ReactMethod
    fun getLoadedModelPath(promise: Promise) = resolve(promise, if (handle != 0L) loadedModelPath else null)

    @ReactMethod
    fun loadModel(params: ReadableMap, promise: Promise) {
        if (!SdCppNative.isLoaded) return reject(promise, "ERR_NATIVE", SdCppNative.loadErrorMessage())
        val modelPath = params.getString("modelPath")
        if (modelPath.isNullOrBlank() || !File(modelPath).isFile) {
            return reject(promise, "ERR_MODEL_NOT_FOUND", "Checkpoint not found: $modelPath")
        }
        val threads = if (params.hasKey("threads")) params.getInt("threads") else 0
        val vaePath = params.getString("vaePath") ?: ""
        val taesdPath = params.getString("taesdPath") ?: ""
        val wtype = params.getString("weightType") ?: "auto"
        val flashAttn = params.hasKey("flashAttn") && params.getBoolean("flashAttn")
        val mmap = !params.hasKey("mmap") || params.getBoolean("mmap")
        executor.execute {
            try {
                if (handle != 0L) {
                    SdCppNative.nativeFree(handle)
                    handle = 0L
                    loadedModelPath = null
                }
                val t0 = System.currentTimeMillis()
                val h = SdCppNative.nativeLoad(modelPath, vaePath, taesdPath, threads, wtype, flashAttn, mmap)
                handle = h
                loadedModelPath = modelPath
                Log.i(TAG, "[SDCPP-LOAD] ${SdCppNative.nativeModelVersion(h)} in ${System.currentTimeMillis() - t0} ms: $modelPath")
                resolve(promise, true)
            } catch (e: Throwable) {
                Log.e(TAG, "load failed", e)
                reject(promise, "ERR_LOAD", e.message ?: "stable-diffusion.cpp load failed", e)
            }
        }
    }

    @ReactMethod
    fun unloadModel(promise: Promise) {
        val h = handle
        if (h != 0L) SdCppNative.nativeCancel(h)
        executor.execute {
            if (handle != 0L) {
                SdCppNative.nativeFree(handle)
                handle = 0L
                loadedModelPath = null
            }
            resolve(promise, true)
        }
    }

    @ReactMethod
    fun cancelGeneration(promise: Promise) {
        val h = handle
        if (h != 0L && generating) SdCppNative.nativeCancel(h)
        resolve(promise, true)
    }

    @ReactMethod
    fun generateImage(params: ReadableMap, promise: Promise) {
        if (handle == 0L) return reject(promise, "ERR_NO_MODEL", "No stable-diffusion.cpp model loaded")
        val prompt = params.getString("prompt") ?: ""
        val negative = params.getString("negativePrompt") ?: ""
        val steps = (if (params.hasKey("steps")) params.getInt("steps") else 20).coerceIn(1, 150)
        val cfg = (if (params.hasKey("guidanceScale")) params.getDouble("guidanceScale") else 7.0).toFloat()
        val seed = if (params.hasKey("seed")) params.getDouble("seed").toLong() else -1L
        val width = snapDimension(if (params.hasKey("width")) params.getInt("width") else 512)
        val height = snapDimension(if (params.hasKey("height")) params.getInt("height") else 512)
        val sampler = params.getString("sampler") ?: ""
        val scheduler = params.getString("scheduler") ?: ""
        val clipSkip = if (params.hasKey("clipSkip")) params.getInt("clipSkip") else -1
        val loraPaths = ArrayList<String>()
        val loraMults = ArrayList<Float>()
        params.getArray("loras")?.let { arr ->
            for (i in 0 until arr.size()) {
                val l = arr.getMap(i) ?: continue
                val p = l.getString("path") ?: continue
                if (!File(p).isFile) {
                    return reject(promise, "ERR_LORA_NOT_FOUND", "LoRA not found: $p")
                }
                loraPaths.add(p)
                loraMults.add(if (l.hasKey("weight")) l.getDouble("weight").toFloat() else 1f)
            }
        }
        generating = true
        executor.execute {
            val names = arrayOfNulls<String>(2)
            val t0 = System.currentTimeMillis()
            try {
                val listener = object : SdCppProgressListener {
                    override fun onProgress(step: Int, steps: Int, seconds: Float) {
                        if (steps <= 0) return
                        sendEvent(EVENT_PROGRESS, Arguments.createMap().apply {
                            putInt("step", step)
                            putInt("totalSteps", steps)
                            putDouble("progress", step.toDouble() / steps)
                        })
                    }
                }
                val out = SdCppNative.nativeGenerate(
                    handle, prompt, negative, width, height, steps, cfg, seed, sampler, scheduler,
                    clipSkip, loraPaths.toTypedArray(), loraMults.toFloatArray(), names, listener,
                ) ?: throw IllegalStateException("stable-diffusion.cpp returned no image")
                val hdr = ByteBuffer.wrap(out, 0, 12).order(ByteOrder.LITTLE_ENDIAN)
                val w = hdr.int
                val h = hdr.int
                val c = hdr.int
                val imageId = UUID.randomUUID().toString()
                val dir = File(reactApplicationContext.filesDir, "generated_images").apply { if (!exists()) mkdirs() }
                val outPath = File(dir, "$imageId.png").absolutePath
                savePixelsToPng(out, 12, w, h, c, outPath)
                val ms = System.currentTimeMillis() - t0
                Log.i(TAG, "[SDCPP-GEN] ${w}x$h sampler=${names[0]} scheduler=${names[1]} steps=$steps seed=$seed loras=${loraPaths.size} in $ms ms")
                resolve(promise, Arguments.createMap().apply {
                    putString("id", imageId)
                    putString("imagePath", outPath)
                    putInt("width", w)
                    putInt("height", h)
                    putDouble("seed", seed.toDouble())
                    putString("sampler", names[0])
                    putString("scheduler", names[1])
                    putDouble("generationTimeMs", ms.toDouble())
                })
            } catch (e: Throwable) {
                Log.e(TAG, "generate failed", e)
                reject(promise, "GENERATION_ERROR", e.message ?: "stable-diffusion.cpp generation failed", e)
            } finally {
                generating = false
            }
        }
    }

    private fun savePixelsToPng(buf: ByteArray, offset: Int, w: Int, h: Int, c: Int, path: String) {
        require(c == 3 || c == 4) { "Unsupported channel count $c" }
        require(buf.size - offset >= w * h * c) { "Pixel buffer ${buf.size - offset} < ${w}x${h}x$c" }
        val pixels = IntArray(w * h)
        var src = offset
        for (i in pixels.indices) {
            val r = buf[src].toInt() and 0xFF
            val g = buf[src + 1].toInt() and 0xFF
            val b = buf[src + 2].toInt() and 0xFF
            val a = if (c == 4) buf[src + 3].toInt() and 0xFF else 0xFF
            pixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
            src += c
        }
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
        FileOutputStream(path).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bmp.recycle()
    }

    @ReactMethod
    fun addListener(eventName: String) { /* required by NativeEventEmitter */ }

    @ReactMethod
    fun removeListeners(count: Int) { /* required by NativeEventEmitter */ }

    override fun invalidate() {
        val h = handle
        if (h != 0L) SdCppNative.nativeCancel(h)
        executor.execute {
            if (handle != 0L) {
                SdCppNative.nativeFree(handle)
                handle = 0L
            }
        }
        executor.shutdown()
        super.invalidate()
    }
}
