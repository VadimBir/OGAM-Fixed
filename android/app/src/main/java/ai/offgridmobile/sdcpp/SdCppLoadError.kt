package ai.offgridmobile.sdcpp

/**
 * Maps the message of a failed nativeLoad to the promise reject code. The Vulkan build has no CPU
 * fallback (app policy: GPU/NPU only): libsdcpp_jni.cpp throws "ERR_NO_GPU: ..." when no Vulkan GPU is
 * usable, and JS shows that instead of a generic load failure.
 */
internal object SdCppLoadError {
    const val NO_GPU = "ERR_NO_GPU"
    const val LOAD = "ERR_LOAD"

    fun codeFor(message: String?): String =
        if (message != null && message.startsWith("$NO_GPU:")) NO_GPU else LOAD
}
