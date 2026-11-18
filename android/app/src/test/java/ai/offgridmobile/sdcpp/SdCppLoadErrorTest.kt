package ai.offgridmobile.sdcpp

import org.junit.Assert.assertEquals
import org.junit.Test

/** Falsify: make codeFor always return LOAD -> the ERR_NO_GPU case goes red. */
@Suppress("kotlin:S100") // Backtick test names are idiomatic Kotlin
class SdCppLoadErrorTest {
    @Test
    fun `native no-gpu message maps to ERR_NO_GPU`() {
        assertEquals("ERR_NO_GPU", SdCppLoadError.codeFor("ERR_NO_GPU: libvulkan.so could not be loaded (this build runs on the GPU only)"))
    }

    @Test
    fun `other native failures stay ERR_LOAD`() {
        assertEquals("ERR_LOAD", SdCppLoadError.codeFor("stable-diffusion.cpp failed to load the checkpoint"))
        assertEquals("ERR_LOAD", SdCppLoadError.codeFor("ERR_OUT_OF_MEMORY: native allocation failed in load"))
    }

    @Test
    fun `null and empty messages stay ERR_LOAD`() {
        assertEquals("ERR_LOAD", SdCppLoadError.codeFor(null))
        assertEquals("ERR_LOAD", SdCppLoadError.codeFor(""))
    }

    @Test
    fun `code must be a prefix not a substring`() {
        assertEquals("ERR_LOAD", SdCppLoadError.codeFor("failed: ERR_NO_GPU: nope"))
    }
}
