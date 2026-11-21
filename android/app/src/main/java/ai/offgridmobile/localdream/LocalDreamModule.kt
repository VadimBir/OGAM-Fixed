package ai.offgridmobile.localdream

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.util.Base64
import android.util.Log
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.InetAddress
import java.net.ServerSocket
import java.net.URL
import java.time.Instant
import java.util.UUID
import java.util.concurrent.TimeUnit
import ai.offgridmobile.SafePromise
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.json.JSONObject

/**
 * Generation cancel state by run id. Each generateImage call gets the next id; a cancel (or unload)
 * cancels every id issued so far. A queued generation therefore stays cancelled when a newer
 * generateImage arrives, and an idle cancel cannot poison a later run. Pure JVM.
 */
internal class GenerationCancelGate {
    private var issued = 0L
    private var cancelledThrough = 0L

    @Synchronized fun issue(): Long = ++issued

    @Synchronized fun cancelAll() {
        cancelledThrough = issued
    }

    @Synchronized fun isCancelled(runId: Long): Boolean = runId <= cancelledThrough
}

/**
 * React Native native module that manages local-dream's inference server process.
 *
 * Architecture:
 * - Spawns libstable_diffusion_core.so as a subprocess
 * - The subprocess runs an HTTP server on 127.0.0.1:18081, or a free port when 18081 is taken
 * - This module POSTs /generate to that server and parses its SSE stream (TypeScript never talks HTTP)
 * - This module handles: process lifecycle, QNN lib extraction, image file management
 */
class LocalDreamModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private fun safeReject(promise: Promise, code: String, message: String, throwable: Throwable? = null) =
        SafePromise(promise, TAG).reject(code, message, throwable)

    private fun safeResolve(promise: Promise, value: Any?) =
        SafePromise(promise, TAG).resolve(value)

    companion object {
        private const val TAG = "LocalDreamModule"
        private const val MODULE_NAME = "LocalDreamModule"
        private const val EXECUTABLE_NAME = "libstable_diffusion_core.so"
        private const val RUNTIME_DIR = "runtime_libs"
        private const val SERVER_PORT = 18081

        private const val MNN_OPENCL_TUNING_MODE = "WIDE"
        private const val QNN_BASE_SIZE = 512
        /** OpenCL (GPU VAE) is only sound at exactly 512x512 (see buildGenerationBody). */
        private const val OPENCL_SOUND_SIZE = 512
        /** The ONE native default step count: getConstants() and the request fallback both use it.
         *  JS `DEFAULT_IMAGE_STEPS` (utils/imageGenAdvice.ts) must equal it (jest-asserted). */
        private const val DEFAULT_STEPS = 20
        private const val EVENT_PROGRESS = "LocalDreamProgress"
        private const val EVENT_ERROR = "LocalDreamError"
        /** Server lifecycle changes JS did not ask for (cancel restart, NPU patch restart, fallback). */
        private const val EVENT_STATUS = "LocalDreamStatus"

        /** True when canonical [path] is [root] itself or lies under it. A bare startsWith would
         *  accept a sibling such as "/data/files_evil" for root "/data/files". Android paths always
         *  use '/', so the separator is literal: File.separator made this host-dependent ('\' on a
         *  Windows JVM running the unit tests). */
        internal fun isPathInside(path: String, root: String): Boolean =
            path == root || path.startsWith(root.trimEnd('/') + '/')

        /** Which qnnlibs asset a start needs: the QNN runtime only for the NPU backend, the core
         *  executable only when nativeLibraryDir has none. */
        internal fun shouldCopyRuntimeAsset(fileName: String, copyQnnLibs: Boolean, copyExecutable: Boolean): Boolean =
            if (fileName == EXECUTABLE_NAME) copyExecutable else copyQnnLibs

        /** [preferred] when no other process holds it on loopback, else a free ephemeral port. */
        internal fun pickServerPort(preferred: Int): Int {
            val loopback = InetAddress.getByName("127.0.0.1")
            val preferredFree = try {
                ServerSocket(preferred, 1, loopback).close()
                true
            } catch (e: IOException) {
                Log.w(TAG, "Port $preferred is already bound (${e.message}); picking a free port")
                false
            }
            if (preferredFree) return preferred
            val free = ServerSocket(0, 1, loopback).use { it.localPort }
            return free
        }

        // Mirrors local-dream's getChipsetSuffix: any SM-prefixed chip → supported
        internal fun isNpuSupportedInternal(): Boolean {
            val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                Build.SOC_MODEL
            } else {
                return false
            }
            return soc.startsWith("SM")
        }

        internal fun resolveModelDir(dir: File, isCpu: Boolean): File? {
            val markerFile = if (isCpu) "unet.mnn" else "unet.bin"

            if (File(dir, markerFile).exists()) return dir

            fun searchDir(current: File, depth: Int): File? {
                if (depth > 3) return null
                current.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
                    if (File(subDir, markerFile).exists()) {
                        Log.d(TAG, "Found $markerFile in: ${subDir.absolutePath}")
                        return subDir
                    }
                    val deeper = searchDir(subDir, depth + 1)
                    if (deeper != null) return deeper
                }
                return null
            }

            return searchDir(dir, 0)
        }

        /**
         * Detect text_embedding_size from the model files.
         * All SD1.5 models (which is everything in xororz/sd-mnn and sd-qnn) use
         * CLIP ViT-L/14 with 768-dimensional text embeddings.
         * Note: "clip_v2" refers to MNN model format v2 (separate weight files),
         * NOT CLIP architecture v2. The embedding dimension is still 768.
         */
        internal fun detectTextEmbeddingSize(modelDir: File, isCpu: Boolean): String {
            // SD1.5 models always use 768
            return "768"
        }

        /** NPU (QNN) UNets are compiled for ONE latent shape (512x512 → 64x64). Any other size needs a
         *  zstd resolution patch applied at server start (`--patch`), named like upstream local-dream
         *  (BackendService): square "<W>.patch" then "<W>x<H>.patch"; non-square "<W>x<H>.patch".
         *  Without it the core feeds a W/8 x H/8 latent into the fixed 64x64 graph → rainbow. */
        internal fun resolvePatchFile(modelDir: File, width: Int, height: Int): File? {
            if (width == QNN_BASE_SIZE && height == QNN_BASE_SIZE) return null
            val candidates = if (width == height) {
                listOf(File(modelDir, "$width.patch"), File(modelDir, "${width}x$height.patch"))
            } else {
                listOf(File(modelDir, "${width}x$height.patch"))
            }
            return candidates.firstOrNull { it.isFile }
        }

        /** Sizes an NPU model can actually render: 512 plus every patch in its directory. */
        internal fun availableQnnSizes(modelDir: File): List<Pair<Int, Int>> {
            val square = Regex("""^(\d+)\.patch$""")
            val rect = Regex("""^(\d+)x(\d+)\.patch$""")
            val found = modelDir.listFiles()?.mapNotNull { f ->
                square.matchEntire(f.name)?.let { m -> m.groupValues[1].toInt().let { it to it } }
                    ?: rect.matchEntire(f.name)?.let { m -> m.groupValues[1].toInt() to m.groupValues[2].toInt() }
            } ?: emptyList()
            return (listOf(QNN_BASE_SIZE to QNN_BASE_SIZE) + found).distinct()
        }

        /** MNN (CPU) SD1.5: upstream offers square 128..512 on a 64 grid (the UNet downsamples the
         *  W/8 latent 3x, so W must be a multiple of 64). */
        internal fun snapMnnSize(px: Int): Int = ((px + 32) / 64 * 64).coerceIn(128, 512)

        /** What the engine will really render for a request, decided BEFORE generation. */
        internal data class RenderPlan(
            val width: Int,
            val height: Int,
            val adjusted: Boolean,
            val reason: String?,
        )

        /** MNN is forced square (min side) on upstream's 128..512 / 64 grid. */
        internal fun planMnnSize(reqW: Int, reqH: Int): RenderPlan {
            val side = snapMnnSize(minOf(reqW, reqH))
            val adjusted = side != reqW || side != reqH
            return RenderPlan(
                side, side, adjusted,
                if (adjusted) "MNN renders square 128-512 px on a 64 px grid" else null,
            )
        }

        /** NPU renders 512x512, or a size whose resolution patch sits in the model directory. */
        internal fun planQnnSize(modelDir: File, reqW: Int, reqH: Int): RenderPlan {
            if (reqW == QNN_BASE_SIZE && reqH == QNN_BASE_SIZE) return RenderPlan(reqW, reqH, false, null)
            if (resolvePatchFile(modelDir, reqW, reqH) != null) return RenderPlan(reqW, reqH, false, null)
            val available = availableQnnSizes(modelDir).joinToString(", ") { "${it.first}x${it.second}" }
            return RenderPlan(
                QNN_BASE_SIZE, QNN_BASE_SIZE, true,
                "NPU model has no ${reqW}x$reqH resolution patch (available: $available)",
            )
        }

        /** OpenCL is requested AND not force-disabled AND the render size is the proven-clean 512x512. */
        internal fun effectiveUseOpenCL(
            requested: Boolean, forceCpuCompute: Boolean, width: Int, height: Int,
        ): Boolean = requested && !forceCpuCompute && width == OPENCL_SOUND_SIZE && height == OPENCL_SOUND_SIZE

        internal fun buildCommand(
            executable: File,
            modelDir: File,
            runtimeDir: File,
            isCpu: Boolean,
            patchFile: File? = null,
            port: Int = SERVER_PORT,
        ): List<String> {
            val embeddingSize = detectTextEmbeddingSize(modelDir, isCpu)
            Log.d(TAG, "Detected text_embedding_size: $embeddingSize")

            return if (isCpu) {
                // MNN backend — --cpu tells the binary to use MNN instead of QNN.
                // OpenCL GPU acceleration is requested per-request via "use_opencl": true in the
                // JSON body. Do NOT remove --cpu — without it the binary crashes on some devices.
                // IMPORTANT: Always pass "clip.mnn" even if only clip_v2.mnn exists.
                // The binary auto-detects clip_v2.mnn in the same directory when the
                // --clip path ends with "clip.mnn", and loads pos_emb.bin + token_emb.bin.
                // Passing clip_v2.mnn directly bypasses this and causes a segfault.
                mutableListOf(
                    executable.absolutePath,
                    "--clip", File(modelDir, "clip.mnn").absolutePath,
                    "--unet", File(modelDir, "unet.mnn").absolutePath,
                    "--vae_decoder", File(modelDir, "vae_decoder.mnn").absolutePath,
                    "--tokenizer", File(modelDir, "tokenizer.json").absolutePath,
                    "--port", port.toString(),
                    "--text_embedding_size", embeddingSize,
                    "--cpu",
                ).also { cmd ->
                    val vaeEncoder = File(modelDir, "vae_encoder.mnn")
                    if (vaeEncoder.exists()) {
                        cmd.addAll(listOf("--vae_encoder", vaeEncoder.absolutePath))
                    }
                }
            } else {
                // QNN NPU backend
                // Same clip.mnn rule applies for QNN — binary auto-detects clip_v2
                val hasMnnClip = File(modelDir, "clip.mnn").exists() || File(modelDir, "clip_v2.mnn").exists()
                val clipFile = if (hasMnnClip) "clip.mnn" else "clip.bin"

                mutableListOf(
                    executable.absolutePath,
                    "--clip", File(modelDir, clipFile).absolutePath,
                    "--unet", File(modelDir, "unet.bin").absolutePath,
                    "--vae_decoder", File(modelDir, "vae_decoder.bin").absolutePath,
                    "--tokenizer", File(modelDir, "tokenizer.json").absolutePath,
                    "--backend", File(runtimeDir, "libQnnHtp.so").absolutePath,
                    "--system_library", File(runtimeDir, "libQnnSystem.so").absolutePath,
                    "--port", port.toString(),
                    "--text_embedding_size", embeddingSize,
                ).also { cmd ->
                    if (patchFile != null) cmd.addAll(listOf("--patch", patchFile.absolutePath))
                    if (hasMnnClip) {
                        cmd.add("--use_cpu_clip")
                    }
                    val vaeEncoder = File(modelDir, "vae_encoder.bin")
                    if (vaeEncoder.exists()) {
                        cmd.addAll(listOf("--vae_encoder", vaeEncoder.absolutePath))
                    }
                }
            }
        }

        /**
         * Strip a `data:image/...;base64,` prefix and surrounding whitespace, then base64-decode.
         * A stray data-URL prefix or newline previously made the raw decode mis-read compressed
         * bytes as pixels (rainbow noise), so normalize before anything else.
         */
        internal fun decodeBase64Image(base64Image: String): ByteArray {
            val cleaned = base64Image.substringAfterLast("base64,", base64Image).trim()
            return try {
                Base64.decode(cleaned, Base64.DEFAULT)
            } catch (e: IllegalArgumentException) {
                Base64.decode(cleaned, Base64.NO_WRAP or Base64.URL_SAFE)
            }
        }

        /**
         * The ONE decoder for every image this module writes (final + preview + the exported
         * saveRgbAsPng bridge). Order matters — ENCODED-IMAGE MAGIC BYTES ARE CHECKED FIRST:
         *   1. JPEG (FF D8 FF) or PNG (89 50 4E 47) → BitmapFactory. CONFIRMED by `strings` on the
         *      SD core (CP-01): the server JPEG-encodes the final `complete.image` AND previews
         *      (preview_format=jpeg). Packing those compressed bytes as raw pixels is exactly the
         *      reported rainbow noise, so encoded payloads must win regardless of their length.
         *   2. length == w*h*3 → raw interleaved RGB24 (HWC), alpha forced opaque — only for a core
         *      build that emits uncompressed buffers.
         *   3. length == w*h*4 → interleaved RGBA32.
         *   4. otherwise → BitmapFactory (last resort) or throw.
         * Every call logs `[IMG-DECODE] size/wh/first8` so the real payload format is never guessed.
         * Uses the `complete` event's own width/height for the raw path (see buildFinalResult).
         */
        private fun looksLikeJpeg(b: ByteArray): Boolean =
            b.size >= 3 && (b[0].toInt() and 0xFF) == 0xFF && (b[1].toInt() and 0xFF) == 0xD8 &&
                (b[2].toInt() and 0xFF) == 0xFF
        private fun looksLikePng(b: ByteArray): Boolean =
            b.size >= 8 && (b[0].toInt() and 0xFF) == 0x89 && b[1].toInt() == 0x50 &&
                b[2].toInt() == 0x4E && b[3].toInt() == 0x47

        internal fun saveServerImageToPng(base64Image: String, width: Int, height: Int, outputPath: String) {
            val bytes = decodeBase64Image(base64Image)
            val n = width * height
            // Decisive diagnostic: exact payload the server sent, so the decode path is never a
            // guess again. If images still look wrong, this line reveals size vs w*h*3/4 and the
            // magic bytes → the true layout.
            val first = bytes.take(8).joinToString(" ") { "%02X".format(it.toInt() and 0xFF) }
            Log.d(TAG, "[IMG-DECODE] size=${bytes.size} wh=${width}x${height} wh3=${n * 3} wh4=${n * 4} first8=$first")
            // CONFIRMED (CP-01, `strings` on the SD core): the server JPEG-encodes the final image
            // and JPEG-encodes previews (preview_format=jpeg). So an ENCODED image must be detected
            // FIRST by its magic bytes and decoded with BitmapFactory — packing those bytes as raw
            // pixels is exactly the reported rainbow. Raw RGB/RGBA is only a last resort for a core
            // build that emits uncompressed buffers, and even then must not shadow an encoded payload
            // whose length happens to equal w*h*3/4.
            val bitmap: Bitmap = when {
                looksLikeJpeg(bytes) || looksLikePng(bytes) ->
                    BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        ?: throw IllegalArgumentException(
                            "Encoded image (${bytes.size} bytes, first8=$first) failed to decode"
                        )
                bytes.size == n * 3 -> Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bmp ->
                    val pixels = IntArray(n)
                    for (i in 0 until n) {
                        val idx = i * 3
                        val r = bytes[idx].toInt() and 0xFF
                        val g = bytes[idx + 1].toInt() and 0xFF
                        val b = bytes[idx + 2].toInt() and 0xFF
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                    bmp.setPixels(pixels, 0, width, 0, 0, width, height)
                }
                bytes.size == n * 4 -> Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bmp ->
                    val pixels = IntArray(n)
                    for (i in 0 until n) {
                        val idx = i * 4
                        val r = bytes[idx].toInt() and 0xFF
                        val g = bytes[idx + 1].toInt() and 0xFF
                        val b = bytes[idx + 2].toInt() and 0xFF
                        val a = bytes[idx + 3].toInt() and 0xFF
                        pixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
                    }
                    bmp.setPixels(pixels, 0, width, 0, 0, width, height)
                }
                else -> BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    ?: throw IllegalArgumentException(
                        "Image payload (${bytes.size} bytes, first8=$first) is not JPEG/PNG nor raw RGB (${n * 3}) / RGBA (${n * 4}) for ${width}x${height}"
                    )
            }
            File(outputPath).parentFile?.mkdirs()
            FileOutputStream(outputPath).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            bitmap.recycle()
        }

        internal fun buildEnvironment(runtimeDir: File): Map<String, String> {
            val env = mutableMapOf<String, String>()

            val systemLibPaths = mutableListOf(
                runtimeDir.absolutePath,
                "/system/lib64",
                "/vendor/lib64",
                "/vendor/lib64/egl",
            )

            try {
                val maliSymlink = File("/system/vendor/lib64/egl/libGLES_mali.so")
                if (maliSymlink.exists()) {
                    val realPath = maliSymlink.canonicalPath
                    val soc = realPath.split("/").getOrNull(realPath.split("/").size - 2)
                    if (soc != null) {
                        listOf("/vendor/lib64/$soc", "/vendor/lib64/egl/$soc").forEach { path ->
                            if (!systemLibPaths.contains(path)) systemLibPaths.add(path)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to resolve Mali paths: ${e.message}")
            }

            env["LD_LIBRARY_PATH"] = systemLibPaths.joinToString(":")
            env["DSP_LIBRARY_PATH"] = runtimeDir.absolutePath
            env["ADSP_LIBRARY_PATH"] = runtimeDir.absolutePath

            // MNN OpenCL tuning: request wider kernel search for better Adreno perf
            env["MNN_OPENCL_TUNING"] = MNN_OPENCL_TUNING_MODE

            return env
        }
    }

    // Server state below is mutated only under serverMutex (load / unload / restart / generate);
    // @Volatile so lock-free readers (isModelLoaded, cancelGeneration, invalidate) see the latest.
    private val serverMutex = Mutex()
    @Volatile private var serverProcess: Process? = null
    @Volatile private var currentModelPath: String? = null
    @Volatile private var currentBackend: String? = null
    /** Resolved model dir of the running server + the resolution patch it was started with. */
    @Volatile private var currentModelDir: File? = null
    @Volatile private var currentPatchFile: File? = null
    @Volatile private var isServerReady = false
    /** Port the running server listens on (SERVER_PORT unless it was already bound). */
    @Volatile private var serverPort = SERVER_PORT
    private val coroutineScope = CoroutineScope(Dispatchers.Default + Job())
    @Volatile private var monitorJob: Job? = null
    /** Server restart after a cancelled generation (see cancelGeneration). */
    @Volatile private var restartJob: Job? = null
    /** Which generations are cancelled, by run id (see GenerationCancelGate). */
    private val cancelGate = GenerationCancelGate()
    /** Run id of the generation holding serverMutex (set in generateImageLocked). */
    @Volatile private var currentRunId = 0L
    /** Unique per generation: preview file names embed it so a cached Image never shows a prior run. */
    @Volatile private var generationId: String = ""
    @Volatile private var activeGenerationConnection: HttpURLConnection? = null
    // Bounded ring of the most recent server stdout/stderr lines + last exit code, so a crash
    // rejection can carry the server's OWN error text to a details popup instead of a generic message.
    private val serverLogRing = java.util.ArrayDeque<String>()
    private val SERVER_LOG_RING_MAX = 60
    @Volatile private var lastServerExitCode: Int? = null

    private fun recordServerLine(line: String) {
        synchronized(serverLogRing) {
            serverLogRing.addLast(line)
            while (serverLogRing.size > SERVER_LOG_RING_MAX) serverLogRing.removeFirst()
        }
    }

    /** Compact, human-readable diagnostics for a crash popup: backend, exit code, recent server log. */
    private fun crashDiagnostics(lastN: Int = 14): String {
        val recent = synchronized(serverLogRing) { serverLogRing.toList() }.takeLast(lastN)
        val header = "backend=${currentBackend ?: "?"} port=$serverPort exit=${lastServerExitCode ?: "n/a"}"
        return if (recent.isEmpty()) "[server] $header (no server output captured)"
        else "[server] $header\n" + recent.joinToString("\n")
    }

    override fun getName(): String = MODULE_NAME

    override fun getConstants(): Map<String, Any> {
        return mapOf(
            "DEFAULT_STEPS" to DEFAULT_STEPS,
            "DEFAULT_GUIDANCE_SCALE" to 7.5,
            "DEFAULT_WIDTH" to 512,
            "DEFAULT_HEIGHT" to 512,
            "SUPPORTED_WIDTHS" to listOf(128, 192, 256, 320, 384, 448, 512),
            "SUPPORTED_HEIGHTS" to listOf(128, 192, 256, 320, 384, 448, 512),
            "SERVER_PORT" to SERVER_PORT,
        )
    }

    private fun sendEvent(eventName: String, params: WritableMap) {
        reactApplicationContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(eventName, params)
    }

    // =====================================================================
    // QNN Library Extraction
    // =====================================================================

    /**
     * Copies the qnnlibs assets a start needs into runtime_libs (an equal-size file is kept).
     * MNN/CPU never loads the QNN runtime (~165 MB), so it copies only the core executable, and
     * only when nativeLibraryDir has none.
     */
    private fun prepareRuntimeDir(copyQnnLibs: Boolean, copyExecutable: Boolean): File {
        val runtimeDir = File(reactApplicationContext.filesDir, RUNTIME_DIR).apply {
            if (!exists()) mkdirs()
        }

        try {
            val qnnLibs = reactApplicationContext.assets.list("qnnlibs")
            qnnLibs?.filter { shouldCopyRuntimeAsset(it, copyQnnLibs, copyExecutable) }?.forEach { fileName ->
                val targetLib = File(runtimeDir, fileName)

                val needsCopy = !targetLib.exists() || run {
                    val assetInputStream = reactApplicationContext.assets.open("qnnlibs/$fileName")
                    val assetSize = assetInputStream.use { it.available().toLong() }
                    targetLib.length() != assetSize
                }

                if (needsCopy) {
                    reactApplicationContext.assets.open("qnnlibs/$fileName").use { input ->
                        targetLib.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                    Log.d(TAG, "Copied $fileName to runtime directory")
                }

                targetLib.setReadable(true, true)
                targetLib.setExecutable(true, true)
            }
            Log.i(TAG, "Runtime libraries prepared in: ${runtimeDir.absolutePath} (qnn=$copyQnnLibs, executable=$copyExecutable)")
        } catch (e: IOException) {
            Log.w(TAG, "No QNN libraries found in assets (CPU-only mode): ${e.message}")
        }

        runtimeDir.setReadable(true, true)
        runtimeDir.setExecutable(true, true)
        return runtimeDir
    }

    // =====================================================================
    // Process Lifecycle
    // =====================================================================

    private fun normalizeBackend(params: ReadableMap): String {
        val requestedBackend = if (params.hasKey("backend")) params.getString("backend") else null
        return when (requestedBackend?.lowercase()) {
            "mnn", "cpu" -> "mnn"
            "qnn", "npu" -> "qnn"
            "auto", null, "" -> "auto"
            else -> "auto"
        }
    }

    private fun resolveBackendAndDir(
        normalizedBackend: String, rawModelDir: File,
    ): Pair<String, File>? {
        val cpuModelDir = resolveModelDir(rawModelDir, true)
        val qnnModelDir = resolveModelDir(rawModelDir, false)
        val npuSupported = isNpuSupportedInternal()
        return when (normalizedBackend) {
            "mnn" -> cpuModelDir?.let { "mnn" to it }
            "qnn" -> qnnModelDir?.let { "qnn" to it }
            else -> resolveAutoBackend(cpuModelDir, qnnModelDir, npuSupported)
        }
    }

    private fun resolveAutoBackend(
        cpuModelDir: File?, qnnModelDir: File?, npuSupported: Boolean,
    ): Pair<String, File>? = when {
        qnnModelDir != null && npuSupported -> "qnn" to qnnModelDir
        cpuModelDir != null -> "mnn" to cpuModelDir
        qnnModelDir != null -> "qnn" to qnnModelDir
        else -> null
    }

    private suspend fun startWithFallback(
        modelPath: String, backend: String, modelDir: File, cpuModelDir: File?,
    ): StartResult {
        val result = tryStartServer(modelPath, modelDir, backend, backend == "mnn")
        if (result.success) return result

        if (backend != "qnn" || cpuModelDir == null) return result

        Log.w(TAG, "QNN backend failed (${result.error}), falling back to MNN/CPU")
        stopServer()
        val fallbackResult = tryStartServer(modelPath, cpuModelDir, "mnn", true)
        if (fallbackResult.success) {
            Log.i(TAG, "Successfully fell back to MNN/CPU backend")
            return fallbackResult
        }
        return StartResult(false, "QNN failed: ${result.error}. MNN fallback also failed: ${fallbackResult.error}")
    }

    @ReactMethod
    fun loadModel(params: ReadableMap, promise: Promise) {
        // IO: stopServer blocks up to 5 s. The mutex keeps load, unload, restart and generate
        // from interleaving on serverProcess / currentModelPath / isServerReady.
        coroutineScope.launch(Dispatchers.IO) { serverMutex.withLock { loadModelLocked(params, promise) } }
    }

    private suspend fun loadModelLocked(params: ReadableMap, promise: Promise) {
        try {
            val modelPath = params.getString("modelPath")
            if (modelPath.isNullOrBlank()) {
                safeReject(promise, "INVALID_ARGS", "modelPath is required")
                return
            }

            val rawModelDir = File(modelPath)
            if (!rawModelDir.exists() || !rawModelDir.isDirectory) {
                safeReject(promise, "MODEL_NOT_FOUND", "Model directory not found: $modelPath")
                return
            }

            val normalizedBackend = normalizeBackend(params)
            val (backend, modelDir) = resolveBackendAndDir(normalizedBackend, rawModelDir) ?: run {
                val contents = rawModelDir.listFiles()?.map { it.name }?.joinToString(", ") ?: "empty"
                safeReject(promise,
                    "MODEL_FILES_NOT_FOUND",
                    "Could not find model files (unet.mnn or unet.bin) in $modelPath or its subdirectories. " +
                        "Directory contents: [$contents]"
                )
                return
            }

            Log.d(TAG, "Resolved model directory: ${modelDir.absolutePath}")
            Log.d(TAG, "Backend selection: requested=$normalizedBackend, selected=$backend")

            if (currentModelPath == modelPath && serverProcess?.isAlive == true && isServerReady) {
                Log.d(TAG, "Model already loaded: $modelPath")
                safeResolve(promise, true)
                return
            }

            stopServer()
            Log.d(TAG, "Loading model from: $modelPath, backend: $backend")

            val cpuModelDir = resolveModelDir(rawModelDir, true)
            val result = startWithFallback(modelPath, backend, modelDir, cpuModelDir)

            if (result.success) {
                safeResolve(promise, true)
            } else {
                safeReject(promise, "SERVER_FAILED", result.error ?: "Server failed to start")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            stopServer()
            safeReject(promise, "LOAD_ERROR", "Failed to load model: ${e.message}", e)
        }
    }

    private data class StartResult(val success: Boolean, val error: String? = null)

    /** Caller holds serverMutex. On any failure (timeout, early exit, cancellation) the started
     *  process is stopped, so a failed start never leaves a server holding the model in RAM. */
    private suspend fun tryStartServer(
        modelPath: String,
        modelDir: File,
        backend: String,
        isCpu: Boolean,
        patchFile: File? = null,
    ): StartResult {
        // Look for executable in nativeLibraryDir first (has execute permission),
        // then fall back to runtime_libs (extracted from assets)
        val nativeDir = reactApplicationContext.applicationInfo.nativeLibraryDir
        val nativeDirFile = File(nativeDir, EXECUTABLE_NAME)
        val runtimeDir = prepareRuntimeDir(copyQnnLibs = !isCpu, copyExecutable = !nativeDirFile.exists())
        val runtimeDirFile = File(runtimeDir, EXECUTABLE_NAME)

        val executableFile = when {
            nativeDirFile.exists() -> {
                Log.d(TAG, "Using executable from nativeLibraryDir: ${nativeDirFile.absolutePath}")
                nativeDirFile
            }
            runtimeDirFile.exists() -> {
                Log.d(TAG, "Using executable from runtime_libs: ${runtimeDirFile.absolutePath}")
                if (!runtimeDirFile.setExecutable(true, true)) {
                    Log.w(TAG, "Failed to set executable permission on ${runtimeDirFile.absolutePath}")
                }
                runtimeDirFile
            }
            else -> {
                return StartResult(false,
                    "Executable not found in nativeLibraryDir (${nativeDirFile.absolutePath}) " +
                    "or runtime_libs (${runtimeDirFile.absolutePath})")
            }
        }

        // A port another process already holds would answer /health for us (a stale server from a
        // killed app process, or an unrelated app): use a free port instead. The core takes --port.
        val port = pickServerPort(SERVER_PORT)
        if (port != SERVER_PORT) Log.w(TAG, "Port $SERVER_PORT busy; starting the server on port $port")

        // Build command based on backend
        val command = buildCommand(executableFile, modelDir, runtimeDir, isCpu, if (isCpu) null else patchFile, port)

        // Build environment
        val env = buildEnvironment(runtimeDir)

        // Log model directory contents for debugging
        val modelFiles = modelDir.listFiles()?.map { "${it.name} (${it.length()} bytes)" }?.joinToString(", ")
        Log.d(TAG, "Model dir contents: [$modelFiles]")
        Log.d(TAG, "COMMAND: ${command.joinToString(" ")}")
        Log.d(TAG, "LD_LIBRARY_PATH=${env["LD_LIBRARY_PATH"]}")

        val processBuilder = ProcessBuilder(command).apply {
            directory(executableFile.parentFile)
            redirectErrorStream(true)
            environment().putAll(env)
        }

        val proc = processBuilder.start()
        serverProcess = proc
        serverPort = port
        currentModelPath = modelPath
        currentBackend = backend
        currentModelDir = modelDir
        currentPatchFile = if (isCpu) null else patchFile
        isServerReady = false

        // Start monitoring stdout
        startMonitor(proc)

        // Wait for server to be ready (poll health endpoint)
        // Use 120s for QNN (first-time cache building can take a while)
        // Use 180s for MNN (CPU inference setup can be slow)
        val timeoutMs = if (isCpu) 180000L else 120000L
        val ready = try {
            waitForServer(proc, port, timeoutMs)
        } catch (e: CancellationException) {
            Log.w(TAG, "Server start cancelled; stopping the half-started server")
            stopServer()
            throw e
        }

        if (ready) {
            isServerReady = true
            Log.i(TAG, "Server is ready on port $port (backend: $backend)")
            return StartResult(true)
        }
        val failure = buildStartFailure(proc, timeoutMs)
        Log.w(TAG, "Server start failed, stopping it: ${failure.error}")
        stopServer()
        return failure
    }

    private fun buildStartFailure(proc: Process, timeoutMs: Long): StartResult {
        val alive = proc.isAlive
        if (alive) {
            return StartResult(false,
                "Server failed to start within ${timeoutMs/1000}s. " +
                "The model may be too large or the device is low on memory.")
        }
        val exitCode = try { proc.exitValue() } catch (e: IllegalThreadStateException) {
            Log.w(TAG, "exitValue unavailable: ${e.message}")
            null
        }
        val socModel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else "unknown"
        return StartResult(false,
            "Server process exited with code $exitCode. " +
            "Your device ($socModel) may not support this model's backend. " +
            "Try a CPU model instead.\n\n" + crashDiagnostics())
    }

    /**
     * Polls /health until our own process answers. Readiness needs BOTH a 200 and [proc] still
     * alive afterwards: a 200 from another listener while our process died is not our server.
     */
    private suspend fun waitForServer(proc: Process, port: Int, timeoutMs: Long): Boolean {
        val startTime = System.currentTimeMillis()
        while (System.currentTimeMillis() - startTime < timeoutMs) {
            // Bail early if the process has died
            if (!proc.isAlive) {
                Log.w(TAG, "Server process died while waiting for it to become ready")
                return false
            }

            try {
                val url = java.net.URL("http://127.0.0.1:$port/health")
                val conn = url.openConnection() as java.net.HttpURLConnection
                conn.connectTimeout = 1000
                conn.readTimeout = 1000
                conn.requestMethod = "GET"
                val code = conn.responseCode
                conn.disconnect()
                if (code == 200) {
                    if (proc.isAlive) return true
                    Log.w(TAG, "/health answered 200 but our server process is dead")
                    return false
                }
            } catch (e: Exception) {
                Log.v(TAG, "Server not ready yet: ${e.message}")
            }
            delay(500)
        }
        return false
    }

    /** Reads [proc]'s output. Captures the process, so a monitor of a stopped server never
     *  touches the state (isServerReady, error event) of the server that replaced it. */
    private fun startMonitor(proc: Process) {
        monitorJob?.cancel()
        monitorJob = coroutineScope.launch(Dispatchers.IO) {
            try {
                proc.inputStream.bufferedReader().use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        Log.i(TAG, "[server] $line")
                        line?.let { recordServerLine(it) }
                    }
                }

                val exitCode = proc.waitFor()
                lastServerExitCode = exitCode
                Log.i(TAG, "Server process exited with code: $exitCode")
                if (proc !== serverProcess) return@launch  // stopped on purpose / replaced
                isServerReady = false

                if (exitCode != 0 && exitCode != 143) { // 143 = SIGTERM (expected on stop)
                    withContext(Dispatchers.Main) {
                        val errorMap = Arguments.createMap().apply {
                            putString("error", "Server process exited unexpectedly (code: $exitCode)")
                        }
                        sendEvent(EVENT_ERROR, errorMap)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Monitor error", e)
            }
        }
    }

    private fun stopServer() {
        monitorJob?.cancel()
        monitorJob = null

        // Detach first: the monitor of this process then sees proc !== serverProcess and emits no
        // "exited unexpectedly" error for an intentional stop (SIGKILL exit 137 included).
        val proc = serverProcess
        serverProcess = null
        isServerReady = false
        proc?.let {
            try {
                it.destroy()
                if (!it.waitFor(5, TimeUnit.SECONDS)) {
                    it.destroyForcibly()
                }
                Log.i(TAG, "Server process stopped")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping server: ${e.message}")
            }
        }

        currentModelPath = null
        currentBackend = null
        currentModelDir = null
        currentPatchFile = null
    }

    @ReactMethod
    fun unloadModel(promise: Promise) {
        // Abort a running generation first, so it releases serverMutex instead of making unload wait
        // for the whole generation; a pending cancel-restart is cancelled (it stops its own process).
        // Run ids: only generations issued so far are cancelled, so a later load + generate is clean.
        cancelGate.cancelAll()
        restartJob?.cancel()
        activeGenerationConnection?.let { conn ->
            try { conn.disconnect() } catch (e: Exception) { Log.w(TAG, "disconnect on unload: ${e.message}") }
        }
        // Off the bridge thread: stopServer blocks up to 5 s waiting for the process to exit.
        coroutineScope.launch(Dispatchers.IO) {
            try {
                serverMutex.withLock { stopServer() }
                safeResolve(promise, true)
            } catch (e: Exception) {
                safeReject(promise, "UNLOAD_ERROR", "Failed to unload model: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun isModelLoaded(promise: Promise) {
        safeResolve(promise, serverProcess?.isAlive == true && isServerReady)
    }

    @ReactMethod
    fun getLoadedModelPath(promise: Promise) {
        safeResolve(promise, currentModelPath)
    }

    @ReactMethod
    fun isGenerating(promise: Promise) {
        safeResolve(promise, activeGenerationConnection != null)
    }

    @ReactMethod
    fun cancelGeneration(promise: Promise) {
        cancelGate.cancelAll()
        val conn = activeGenerationConnection
        activeGenerationConnection = null
        if (conn != null) {
            try { conn.disconnect() } catch (e: Exception) { Log.w(TAG, "disconnect on cancel: ${e.message}") }
            // The core exposes only /health and /generate (no cancel endpoint), and nothing shows it
            // stops sampling when the client disconnects. So a cancelled in-flight generation would
            // keep the CPU/NPU busy and the next request would queue behind it: restart the server.
            scheduleRestartAfterCancel()
        }
        safeResolve(promise, true)
    }

    /** Restarts the loaded model's server under serverMutex (after the cancelled generation has
     *  released it). Emits LocalDreamStatus restarting / ready / failed. */
    private fun scheduleRestartAfterCancel() {
        restartJob?.cancel()
        restartJob = coroutineScope.launch(Dispatchers.IO) {
            serverMutex.withLock {
                val modelPath = currentModelPath
                val dir = currentModelDir
                val backend = currentBackend
                val patch = currentPatchFile
                if (modelPath == null || dir == null || backend == null) return@withLock
                Log.i(TAG, "Restarting server after cancel (backend=$backend)")
                try {
                    emitStatus("restarting", "Stopping the cancelled generation")
                    stopServer()
                    val r = tryStartServer(modelPath, dir, backend, backend == "mnn", patch)
                    if (r.success) emitStatus("ready", null)
                    else emitStatus("failed", "Server restart after cancel failed: ${r.error}")
                } catch (e: CancellationException) {
                    throw e  // unload / invalidate: tryStartServer already stopped its process
                } catch (e: Exception) {
                    // An uncaught exception in this launch would crash the app process.
                    Log.e(TAG, "Server restart after cancel failed", e)
                    stopServer()
                    emitStatus("failed", "Server restart after cancel failed: ${e.message}")
                }
            }
        }
    }

    private suspend fun emitStatus(status: String, message: String?) {
        Log.i(TAG, "[STATUS] $status${if (message != null) ": $message" else ""}")
        withContext(Dispatchers.Main) {
            try {
                sendEvent(EVENT_STATUS, Arguments.createMap().apply {
                    putString("status", status)
                    if (message != null) putString("message", message) else putNull("message")
                })
            } catch (e: Exception) {
                // React context torn down: the status only informs JS, never blocks the caller.
                Log.w(TAG, "Could not emit $EVENT_STATUS: ${e.message}")
            }
        }
    }

    // =====================================================================
    // Image Generation (HTTP POST + SSE parsing on native side)
    // =====================================================================

    /**
     * Check if the server is alive and responsive before making a request.
     */
    private fun checkServerHealth(): Boolean {
        return try {
            val url = URL("http://127.0.0.1:$serverPort/health")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 3000
            conn.readTimeout = 3000
            conn.requestMethod = "GET"
            val code = conn.responseCode
            conn.disconnect()
            code == 200
        } catch (e: Exception) {
            Log.w(TAG, "Health check failed: ${e.message}")
            false
        }
    }

    /** Clamp an image dimension to a safe SD range and snap to a multiple of 8. Guards against a
     *  corrupt/NaN value (marshalled as a ~1e9 int) that would make the server OOM (std::bad_alloc). */
    private fun clampDimension(px: Int): Int {
        val safe = if (px in 64..1024) px else 512
        return (safe / 8) * 8
    }

    /**
     * KNOWN ISSUE — sub-512 rainbow on MNN (CPU/GPU): OPEN, see docs/notes/native-image-sub512-rainbow.md.
     * The bundled core (upstream local-dream c56cf6c) resizes MNN sessions per request, yet the
     * device shows rainbow at every size < 512. NPU cause is known and guarded below.
     *
     * The size the running backend can really render for this request, restarting an NPU server
     * with the matching resolution patch when needed. NPU without a patch for the size → 512x512
     * (a wrong-shaped latent into the fixed graph is the rainbow). MNN → upstream's 64 grid.
     */
    private fun requestedSize(params: ReadableMap): Pair<Int, Int> =
        clampDimension(if (params.hasKey("width")) params.getInt("width") else 512) to
            clampDimension(if (params.hasKey("height")) params.getInt("height") else 512)

    /** Pure plan (no server restart) of what the CURRENTLY loaded backend renders for this request. */
    private fun planRenderSize(params: ReadableMap): RenderPlan {
        val (reqW, reqH) = requestedSize(params)
        val dir = currentModelDir
        return if (currentBackend != "qnn" || dir == null || currentModelPath == null) {
            planMnnSize(reqW, reqH)
        } else {
            planQnnSize(dir, reqW, reqH)
        }
    }

    private suspend fun resolveRenderSize(params: ReadableMap): Pair<Int, Int> {
        val (reqW, reqH) = requestedSize(params)
        val plan = planRenderSize(params)
        val dir = currentModelDir
        val modelPath = currentModelPath
        if (plan.adjusted) Log.w(TAG, "[IMG-SIZE] ${reqW}x$reqH -> ${plan.width}x${plan.height}: ${plan.reason}")
        if (currentBackend != "qnn" || dir == null || modelPath == null) return plan.width to plan.height
        val w = plan.width
        val h = plan.height
        val patch = if (w == QNN_BASE_SIZE && h == QNN_BASE_SIZE) null else resolvePatchFile(dir, w, h)
        if (patch?.absolutePath != currentPatchFile?.absolutePath) {
            Log.i(TAG, "[IMG-SIZE] npu restart for ${w}x$h patch=${patch?.name ?: "none"}")
            emitStatus("restarting", "Reloading the NPU model for ${w}x$h")
            stopServer()
            val r = tryStartServer(modelPath, dir, "qnn", false, patch)
            if (r.success) {
                emitStatus("ready", null)
                return w to h
            }
            // tryStartServer already stopped the failed process. Fall back to the MNN/CPU files of
            // the same model, as loadModel does (startWithFallback), at MNN's own render size.
            val cpuModelDir = resolveModelDir(File(modelPath), true)
                ?: run {
                    emitStatus("failed", "NPU restart for ${w}x$h failed: ${r.error}")
                    throw java.io.IOException("NPU restart for ${w}x$h failed: ${r.error}")
                }
            Log.w(TAG, "[IMG-SIZE] NPU restart failed (${r.error}), falling back to MNN/CPU")
            emitStatus("fallback", "NPU restart for ${w}x$h failed; switching to CPU (MNN)")
            val fb = tryStartServer(modelPath, cpuModelDir, "mnn", true)
            if (!fb.success) {
                emitStatus("failed", "NPU restart failed: ${r.error}. MNN fallback also failed: ${fb.error}")
                throw java.io.IOException("NPU restart for ${w}x$h failed: ${r.error}. MNN fallback also failed: ${fb.error}")
            }
            emitStatus("ready", null)
            val mnn = planMnnSize(reqW, reqH)
            return mnn.width to mnn.height
        }
        return w to h
    }

    /**
     * Up-front answer to "what will this request really render?" - no generation, no restart.
     * Lets UI/callers show the effective size and OpenCL state instead of discovering them after.
     * With no model loaded the backend is unknown, so the request is echoed unadjusted.
     */
    @ReactMethod
    fun getEffectiveImageSize(params: ReadableMap, promise: Promise) {
        try {
            val (reqW, reqH) = requestedSize(params)
            val backend = currentBackend
            val plan = if (backend == null) RenderPlan(reqW, reqH, false, "no model loaded") else planRenderSize(params)
            val requestedOpenCL = params.hasKey("useOpenCL") && params.getBoolean("useOpenCL")
            safeResolve(promise, Arguments.createMap().apply {
                putInt("requestedWidth", reqW)
                putInt("requestedHeight", reqH)
                putInt("width", plan.width)
                putInt("height", plan.height)
                putBoolean("adjusted", plan.adjusted)
                if (plan.reason != null) putString("reason", plan.reason) else putNull("reason")
                if (backend != null) putString("backend", backend) else putNull("backend")
                putBoolean("openClRequested", requestedOpenCL)
                putBoolean(
                    "useOpenCL",
                    backend == "mnn" && effectiveUseOpenCL(requestedOpenCL, false, plan.width, plan.height),
                )
            })
        } catch (e: Exception) {
            safeReject(promise, "SIZE_PLAN_ERROR", "Failed to plan image size: ${e.message}", e)
        }
    }

    private fun buildGenerationBody(
        params: ReadableMap, forceCpuCompute: Boolean = false, size: Pair<Int, Int>? = null,
    ): JSONObject = JSONObject().apply {
        put("prompt", params.getString("prompt") ?: "")
        put("negative_prompt", params.getString("negativePrompt") ?: "")
        put("steps", (if (params.hasKey("steps")) params.getInt("steps") else DEFAULT_STEPS).coerceIn(1, 100))
        put("cfg", if (params.hasKey("guidanceScale")) params.getDouble("guidanceScale") else 7.5)
        put("seed", if (params.hasKey("seed")) params.getInt("seed") else (Math.random() * 2147483647).toInt())
        // Clamp dimensions and steps to sane bounds. A corrupt/NaN value marshalled from JS (seen
        // as a ~1e9 dimension) makes the server allocate gigabytes and crash with std::bad_alloc,
        // so a bad value must never reach it. 64..1024 px, multiple of 8 (SD requirement).
        val reqWidth = size?.first ?: clampDimension(if (params.hasKey("width")) params.getInt("width") else 512)
        val reqHeight = size?.second ?: clampDimension(if (params.hasKey("height")) params.getInt("height") else 512)
        put("width", reqWidth)
        put("height", reqHeight)
        // The SD core's sampler set is {dpm, euler_a}; default stays dpm when JS sends nothing.
        val scheduler = if (params.hasKey("scheduler")) params.getString("scheduler") else null
        put("scheduler", if (scheduler == "euler_a") "euler_a" else "dpm")
        // Preview contract: the known-good build requests JPEG previews; the preview decoder reads
        // them via BitmapFactory (carries its own dimensions, so it never shears). Restored here to
        // match the known-good request contract.
        put("preview_format", "jpeg")
        // OpenCL GPU acceleration is only SOUND at 512x512. The GPU VAE decode writes a
        // tiled/strided raw RGB24 buffer whose rows line up with the decoder's width-stride read
        // ONLY at 512x512; at any smaller size the rows shear into a diagonal rainbow
        // (device-confirmed: 512x512 clean, sub-512 rainbow). The known-good build shipped
        // use_opencl=false everywhere; commit 5e1c72b started forwarding the JS toggle and so
        // reintroduced the shear for every non-512 request. Fix: keep the GPU fast path at exactly
        // 512x512 (proven clean on device) and force the clean CPU VAE for every other size.
        // `forceCpuCompute` still forces it off for the OpenCL-fault retry (see generateImage);
        // the JS toggle also now defaults off (localDreamGenerator / appStore).
        val requestedOpenCL = if (params.hasKey("useOpenCL")) params.getBoolean("useOpenCL") else false
        val useOpenCL = effectiveUseOpenCL(requestedOpenCL, forceCpuCompute, reqWidth, reqHeight)
        put("use_opencl", useOpenCL)
        put("show_diffusion_process", true)
        put("show_diffusion_stride", if (params.hasKey("previewInterval")) params.getInt("previewInterval") else 2)
        // img2img: JS passes `initImage` (base64 PNG/JPG, no data-URL prefix) plus `denoiseStrength`
        // in 0..1. The SD core reads the keys `image`, `mask` and `denoise_strength`: its JSON key
        // strings are width/height/image/mask (libstable_diffusion_core.so), and upstream c56cf6c
        // sets request_img2img only `if (json.contains("image"))`. The old key `init_img` is not
        // in the binary at all, so every img2img used to run as plain txt2img.
        // denoise_strength is how MUCH the ref image is changed: 1.0 = ignore the ref, low = stay
        // close to it. The JS side already resolved the user's "keep" as (1 - keep).
        // The QNN path runs img2img on the NPU when the model has vae_encoder.bin (passed as
        // --vae_encoder); without it the core warns "VAE Encoder path missing. img2img disabled
        // unless --cpu". When no init image is sent the body is a plain txt2img request.
        putImg2ImgKeys(
            this,
            initImage = if (params.hasKey("initImage")) params.getString("initImage") else null,
            denoise = if (params.hasKey("denoiseStrength")) params.getDouble("denoiseStrength") else null,
            mask = if (params.hasKey("maskImage")) params.getString("maskImage") else null,
        )
    }

    private fun openGenerationConnection(): HttpURLConnection {
        val url = URL("http://127.0.0.1:$serverPort/generate")
        return (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Accept", "text/event-stream")
            connectTimeout = 10000
            readTimeout = 0 // Wait for generation or explicit cancellation, however long it takes.
        }
    }

    private fun previewDir(): File = File(reactApplicationContext.cacheDir, "preview")

    /** Deletes the step previews of the previous / finished generation (called at start and end). */
    private fun clearPreviewDir() {
        previewDir().listFiles()?.forEach { f ->
            if (!f.delete()) Log.w(TAG, "Could not delete preview ${f.name}")
        }
    }

    private fun savePreviewImage(
        previewBase64: String, step: Int, reqWidth: Int, reqHeight: Int,
    ): String? = try {
        val previewDir = previewDir().apply {
            if (!exists()) mkdirs()
        }
        val previewPath = File(previewDir, "preview_${generationId}_step_$step.png").absolutePath
        saveServerImageToPng(previewBase64, reqWidth, reqHeight, previewPath)
        previewPath
    } catch (e: Exception) {
        Log.w(TAG, "Failed to save preview: ${e.message}")
        null
    }

    private suspend fun handleProgressEvent(data: JSONObject, body: JSONObject) {
        val step = data.getInt("step")
        val totalSteps = data.getInt("total_steps")
        val progressMap = Arguments.createMap().apply {
            putInt("step", step)
            putInt("totalSteps", totalSteps)
            putDouble("progress", step.toDouble() / totalSteps.toDouble())
        }

        val previewBase64 = data.optString("image", "")
        if (previewBase64.isNotEmpty()) {
            val previewPath = savePreviewImage(previewBase64, step, body.getInt("width"), body.getInt("height"))
            if (previewPath != null) progressMap.putString("previewPath", previewPath)
        }

        withContext(Dispatchers.Main) { sendEvent(EVENT_PROGRESS, progressMap) }
    }

    private sealed class SseParseResult {
        data class Complete(val data: JSONObject, val body: JSONObject) : SseParseResult()
        object Cancelled : SseParseResult()
        object NoResult : SseParseResult()
    }

    private suspend fun parseSseStream(
        connection: HttpURLConnection, body: JSONObject,
    ): SseParseResult {
        var completeData: JSONObject? = null
        var currentEventType = ""

        BufferedReader(InputStreamReader(connection.inputStream)).use { reader ->
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                if (isRunCancelled()) return SseParseResult.Cancelled

                val trimmed = line!!.trim()
                if (trimmed.startsWith("event: ")) {
                    currentEventType = trimmed.substring(7).trim()
                    continue
                }
                if (!trimmed.startsWith("data: ")) continue

                try {
                    val data = JSONObject(trimmed.substring(6))
                    when (data.optString("type", currentEventType)) {
                        "progress" -> handleProgressEvent(data, body)
                        "complete" -> completeData = data
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to parse SSE data: ${e.message}")
                }
                currentEventType = ""
            }
        }

        if (isRunCancelled()) return SseParseResult.Cancelled
        return if (completeData != null) SseParseResult.Complete(completeData!!, body) else SseParseResult.NoResult
    }

    private fun buildFinalResult(completeData: JSONObject, body: JSONObject, params: ReadableMap): WritableMap {
        val imageBase64 = completeData.getString("image")
        val width = completeData.getInt("width")
        val height = completeData.getInt("height")
        val seed = completeData.optInt("seed", 0)
        val generationTimeMs = completeData.optLong("generation_time_ms", 0)

        val imageId = UUID.randomUUID().toString()
        val outputDir = File(reactApplicationContext.filesDir, "generated_images").apply {
            if (!exists()) mkdirs()
        }
        val outputPath = File(outputDir, "$imageId.png").absolutePath
        saveServerImageToPng(imageBase64, width, height, outputPath)

        return Arguments.createMap().apply {
            putString("id", imageId)
            putString("imagePath", outputPath)
            putInt("width", width)
            putInt("height", height)
            // What the caller asked for vs what was rendered, and whether OpenCL really ran, so a
            // silent adjustment (NPU no patch, MNN square, OpenCL off below 512) is visible to JS.
            val (reqW, reqH) = requestedSize(params)
            putInt("requestedWidth", reqW)
            putInt("requestedHeight", reqH)
            putBoolean("useOpenCL", body.optBoolean("use_opencl", false))
            putInt("seed", seed)
            putDouble("generationTimeMs", generationTimeMs.toDouble())
        }
    }

    private fun handleGeneralException(e: Exception, promise: Promise) {
        if (isRunCancelled()) {
            safeReject(promise, "CANCELLED", "Generation cancelled")
        } else {
            Log.e(TAG, "Generation error: ${e.javaClass.simpleName}", e)
            safeReject(promise, "GENERATION_ERROR",
                "Failed to generate image: [${e.javaClass.simpleName}] ${e.message ?: "unknown error"}\n\n" +
                crashDiagnostics(), e)
        }
    }

    @ReactMethod
    fun generateImage(params: ReadableMap, promise: Promise) {
        // Run id issued on the bridge thread, in call order: a cancel issued after this call but
        // before the generation gets serverMutex (e.g. behind a cancel-restart) is still honoured,
        // and this call can no longer clear a cancel aimed at an earlier, still-queued generation
        // (the old shared flag was reset here, so that generation ran despite the cancel).
        val runId = cancelGate.issue()
        coroutineScope.launch(Dispatchers.IO) {
            serverMutex.withLock { generateImageLocked(runId, params, promise) }
        }
    }

    /** The generation holding serverMutex was cancelled (user cancel or unload). */
    private fun isRunCancelled(): Boolean = cancelGate.isCancelled(currentRunId)

    /** Caller holds serverMutex, so load / unload / restart cannot swap the server mid-generation. */
    private suspend fun generateImageLocked(runId: Long, params: ReadableMap, promise: Promise) {
        currentRunId = runId
        if (isRunCancelled()) {
            safeReject(promise, "CANCELLED", "Generation cancelled")
            return
        }
        if (!isServerReady || serverProcess?.isAlive != true) {
            safeReject(promise, "SERVER_NOT_READY", "Server is not running. Load a model first.")
            return
        }
        if (!checkServerHealth()) {
            isServerReady = false
            safeReject(promise, "SERVER_NOT_READY",
                "Server process is not responsive. Try unloading and reloading the model.")
            return
        }

        generationId = System.currentTimeMillis().toString(36) + "_" + java.util.UUID.randomUUID().toString().take(8)
        clearPreviewDir()
        try {
            resolveGenerationResult(runOneGeneration(params, forceCpuCompute = false), params, promise)
        } catch (e: java.io.EOFException) {
            // The server tore down the stream mid-inference. With the request now validated
            // (dimensions clamped, steps bounded) a genuine EOF is a real server fault, not a
            // bad-parameter crash we can paper over. Surface it with diagnostics and let the
            // JS popup show the server's own output - no silent restart/CPU-fallback juggling.
            if (isRunCancelled()) {
                safeReject(promise, "CANCELLED", "Generation cancelled")
                return
            }
            isServerReady = false
            safeReject(promise, "SERVER_CRASHED",
                "The image server closed the stream during inference. " +
                "Reload the model and try again.\n\n" + crashDiagnostics())
        } catch (e: Exception) {
            handleGeneralException(e, promise)
        } finally {
            activeGenerationConnection = null
            clearPreviewDir()
        }
    }

    /**
     * Run a single /generate round-trip. Opens its own connection (tracked as the active
     * generation connection so cancellation can abort it) and parses the SSE stream. Throws
     * EOFException when the server tears the stream down mid-inference so the caller can decide
     * whether to retry on CPU compute. Rejects inline only for a non-200 response.
     */
    private suspend fun runOneGeneration(params: ReadableMap, forceCpuCompute: Boolean): SseParseResult {
        var connection: HttpURLConnection? = null
        try {
            val body = buildGenerationBody(params, forceCpuCompute, resolveRenderSize(params))
            Log.d(TAG, "Starting generation (forceCpuCompute=$forceCpuCompute): ${body.toString().take(200)}...")

            connection = openGenerationConnection()
            activeGenerationConnection = connection
            // Checked AFTER publishing the connection: a cancel either sees it (and disconnects) or
            // set the flag before this read, so the request is never sent after a cancel.
            if (isRunCancelled()) return SseParseResult.Cancelled
            OutputStreamWriter(connection.outputStream).use { it.write(body.toString()); it.flush() }

            val responseCode = connection.responseCode
            if (responseCode != 200) {
                val errorBody = try {
                    connection.errorStream?.bufferedReader()?.readText() ?: "no error body"
                } catch (_: Exception) { "could not read error" }
                throw java.io.IOException(
                    "Server returned $responseCode: ${connection.responseMessage}. Body: $errorBody")
            }
            return parseSseStream(connection, body)
        } finally {
            connection?.disconnect()
        }
    }

    private fun resolveGenerationResult(result: SseParseResult, params: ReadableMap, promise: Promise) {
        when (result) {
            is SseParseResult.Complete -> safeResolve(promise, buildFinalResult(result.data, result.body, params))
            is SseParseResult.Cancelled -> safeReject(promise, "CANCELLED", "Generation cancelled")
            is SseParseResult.NoResult -> safeReject(promise, "NO_RESULT", "Server did not return a complete event")
        }
    }

    // =====================================================================
    // Image File Management (RGB → PNG conversion and file operations)
    // =====================================================================

    @ReactMethod
    fun saveRgbAsPng(params: ReadableMap, promise: Promise) {
        coroutineScope.launch {
            try {
                val base64Rgb = params.getString("base64Rgb") ?: ""
                val width = params.getInt("width")
                val height = params.getInt("height")
                val outputPath = params.getString("outputPath") ?: ""

                if (base64Rgb.isEmpty() || outputPath.isEmpty()) {
                    safeReject(promise, "INVALID_ARGS", "base64Rgb and outputPath are required")
                    return@launch
                }

                // ONE decoder for every image the module writes: encoded (JPEG/PNG/…) OR raw
                // (planar RGB / interleaved RGBA), data-URL prefix stripped. Same path as the
                // preview + final SSE images, so any decode fix applies everywhere at once.
                saveServerImageToPng(base64Rgb, width, height, outputPath)

                safeResolve(promise, true)
            } catch (e: Exception) {
                Log.e(TAG, "Error saving RGB as PNG", e)
                safeReject(promise, "SAVE_ERROR", "Failed to save image: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun getGeneratedImages(promise: Promise) {
        try {
            val outputDir = File(reactApplicationContext.filesDir, "generated_images")
            if (!outputDir.exists()) {
                safeResolve(promise, Arguments.createArray())
                return
            }

            val images = Arguments.createArray()
            outputDir.listFiles()?.filter { it.extension == "png" }?.forEach { file ->
                val imageMap = Arguments.createMap().apply {
                    putString("id", file.nameWithoutExtension)
                    putString("imagePath", file.absolutePath)
                    putDouble("size", file.length().toDouble())
                    // ISO-8601, matching iOS and the screenshot watcher. Epoch milliseconds as text
                    // satisfies the TypeScript `string` and nothing else: every sync peer reads this
                    // with Date.parse, which answers NaN, and refuses the file.
                    putString(
                        "createdAt",
                        Instant.ofEpochMilli(file.lastModified()).toString(),
                    )
                }
                images.pushMap(imageMap)
            }

            safeResolve(promise, images)
        } catch (e: Exception) {
            safeReject(promise, "LIST_ERROR", "Failed to list generated images: ${e.message}", e)
        }
    }

    @ReactMethod
    fun deleteGeneratedImage(imageId: String, promise: Promise) {
        try {
            val outputDir = File(reactApplicationContext.filesDir, "generated_images")
            val imageFile = File(outputDir, "$imageId.png")

            if (imageFile.exists()) {
                imageFile.delete()
                safeResolve(promise, true)
            } else {
                safeReject(promise, "NOT_FOUND", "Image not found: $imageId")
            }
        } catch (e: Exception) {
            safeReject(promise, "DELETE_ERROR", "Failed to delete image: ${e.message}", e)
        }
    }

    /**
     * Get the server port for the TypeScript layer to connect to.
     */
    @ReactMethod
    fun getServerPort(promise: Promise) {
        safeResolve(promise, serverPort)
    }

    /**
     * Check if the device has a supported Qualcomm NPU.
     */
    @ReactMethod
    fun isNpuSupported(promise: Promise) {
        safeResolve(promise, isNpuSupportedInternal())
    }

    @ReactMethod
    fun getSoCModel(promise: Promise) {
        val soc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL
        } else {
            ""
        }
        safeResolve(promise, soc)
    }

    /** Rejects (CACHE_ERROR) unless [modelPath] resolves inside filesDir. canonicalPath can throw
     *  IOException, so it runs inside the try instead of escaping the bridge call. */
    private fun isInsideAppFiles(modelPath: String, promise: Promise): Boolean {
        val inside = try {
            isPathInside(File(modelPath).canonicalPath, reactApplicationContext.filesDir.canonicalPath)
        } catch (e: IOException) {
            safeReject(promise, "CACHE_ERROR", "Invalid model path: ${e.message}", e)
            return false
        }
        if (!inside) safeReject(promise, "CACHE_ERROR", "Model path is outside the app directory")
        return inside
    }

    /**
     * Clear OpenCL kernel cache files (.mnnc) from a model directory.
     * Forces MNN to retune OpenCL kernels on the next generation,
     * which may find better kernels for the current GPU.
     */
    @ReactMethod
    fun clearOpenCLCache(modelPath: String, promise: Promise) {
        if (!isInsideAppFiles(modelPath, promise)) return
        coroutineScope.launch(Dispatchers.IO) {
            try {
                val modelDir = File(modelPath)
                val cpuModelDir = resolveModelDir(modelDir, true)
                if (cpuModelDir == null) {
                    safeResolve(promise, 0)
                    return@launch
                }

                var cleared = 0
                val cachePattern = Regex(".*\\.mnnc(\\..+)?$")
                cpuModelDir.listFiles()?.filter { it.name.matches(cachePattern) }?.forEach { file ->
                    Log.d(TAG, "Deleting OpenCL cache: ${file.name}")
                    if (file.delete()) cleared++
                }
                Log.i(TAG, "Cleared $cleared OpenCL cache file(s) from ${cpuModelDir.absolutePath}")
                safeResolve(promise, cleared)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to clear OpenCL cache", e)
                safeReject(promise, "CACHE_ERROR", "Failed to clear OpenCL cache: ${e.message}", e)
            }
        }
    }

    /**
     * Check if OpenCL kernel cache (.mnnc files) exists for the given model.
     * Returns false on first run, indicating GPU kernel compilation will be needed.
     */
    @ReactMethod
    fun hasOpenCLCache(modelPath: String, promise: Promise) {
        if (!isInsideAppFiles(modelPath, promise)) return
        coroutineScope.launch(Dispatchers.IO) {
            try {
                val modelDir = File(modelPath)
                val cpuModelDir = resolveModelDir(modelDir, true)
                if (cpuModelDir == null) {
                    safeResolve(promise, false)
                    return@launch
                }

                val cachePattern = Regex(".*\\.mnnc(\\..+)?$")
                val hasCache = cpuModelDir.listFiles()?.any { it.name.matches(cachePattern) } == true
                safeResolve(promise, hasCache)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to check OpenCL cache", e)
                safeReject(promise, "CACHE_ERROR", "Failed to check OpenCL cache: ${e.message}", e)
            }
        }
    }

    @ReactMethod
    fun addListener(eventName: String) {
        // Required for RN event emitter
    }

    @ReactMethod
    fun removeListeners(count: Int) {
        // Required for RN event emitter
    }

    override fun invalidate() {
        super.invalidate()
        coroutineScope.cancel()
        stopServer()
    }
}

/**
 * The img2img keys the bundled SD core reads: `image` (base64 PNG/JPG), `denoise_strength` (0..1,
 * default 0.6) and `mask`. Nothing is added without an init image, so the body stays txt2img.
 */
internal fun putImg2ImgKeys(body: JSONObject, initImage: String?, denoise: Double?, mask: String?) {
    if (initImage.isNullOrBlank()) return
    body.put("image", initImage)
    body.put("denoise_strength", (denoise ?: 0.6).coerceIn(0.0, 1.0))
    if (!mask.isNullOrBlank()) body.put("mask", mask)
}
