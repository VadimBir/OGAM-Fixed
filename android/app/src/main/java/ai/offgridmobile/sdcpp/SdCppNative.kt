package ai.offgridmobile.sdcpp

/** Receives sampling progress from libsdcpp_jni (called on the generating thread). */
interface SdCppProgressListener {
    fun onProgress(step: Int, steps: Int, seconds: Float)
}

/** JNI surface of libsdcpp_jni.so (stable-diffusion.cpp). See android/app/src/main/cpp/sdcpp. */
object SdCppNative {
    @Volatile
    private var loadError: Throwable? = null

    val isLoaded: Boolean by lazy {
        try {
            System.loadLibrary("sdcpp_jni")
            true
        } catch (t: Throwable) {
            loadError = t
            false
        }
    }

    fun loadErrorMessage(): String = loadError?.message ?: "libsdcpp_jni not loaded"

    @JvmStatic external fun nativeSystemInfo(): String

    @JvmStatic external fun nativeLoad(
        modelPath: String,
        vaePath: String,
        taesdPath: String,
        nThreads: Int,
        wtype: String,
        flashAttn: Boolean,
        mmap: Boolean,
    ): Long

    /**
     * nativeLoad plus the memory options (sdcpp_jni.cpp nativeLoadEx). Absent from a prebuilt .so that
     * predates it: calling it then throws UnsatisfiedLinkError (callers fall back to nativeLoad).
     * paramsBackend: "" = weights resident, "disk" / "te=disk,vae=disk,diffusion=disk" = per-run reload.
     * loraApplyMode: "" / "auto" / "immediately" / "at_runtime".
     */
    @JvmStatic external fun nativeLoadEx(
        modelPath: String,
        vaePath: String,
        taesdPath: String,
        nThreads: Int,
        wtype: String,
        flashAttn: Boolean,
        mmap: Boolean,
        paramsBackend: String,
        loraApplyMode: String,
        vaeConvDirect: Boolean,
    ): Long

    @JvmStatic external fun nativeModelVersion(handle: Long): String

    @JvmStatic external fun nativeFree(handle: Long)

    @JvmStatic external fun nativeCancel(handle: Long)

    /** Returns [w,h,c as int32 LE][pixels], or throws. outNames receives resolved sampler/scheduler. */
    @JvmStatic external fun nativeGenerate(
        handle: Long,
        prompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Float,
        seed: Long,
        sampler: String,
        scheduler: String,
        clipSkip: Int,
        vaeTiling: Boolean,
        loraPaths: Array<String>,
        loraMultipliers: FloatArray,
        outNames: Array<String?>,
        listener: SdCppProgressListener?,
    ): ByteArray?
}
