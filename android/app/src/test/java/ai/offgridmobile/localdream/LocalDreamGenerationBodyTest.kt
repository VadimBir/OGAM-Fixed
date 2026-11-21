package ai.offgridmobile.localdream

import android.app.Application
import com.facebook.react.bridge.JavaOnlyMap
import org.json.JSONException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The /generate body and SSE stream for the local-dream v2.8.1 core (RequestParser.hpp, main.cpp).
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], application = Application::class)
class LocalDreamGenerationBodyTest {

    private fun params(vararg pairs: Pair<String, Any>): JavaOnlyMap =
        JavaOnlyMap().apply {
            putString("prompt", "a cat")
            for ((k, v) in pairs) {
                when (v) {
                    is String -> putString(k, v)
                    is Int -> putInt(k, v)
                    is Double -> putDouble(k, v)
                    is Boolean -> putBoolean(k, v)
                }
            }
        }

    // ── output_format / preview_format ───────────────────────────────────────

    @Test
    fun `body asks for jpeg output and previews (the core default is raw)`() {
        val body = LocalDreamModule.buildGenerationBody(params())
        assertEquals("jpeg", body.getString("output_format"))
        assertEquals("jpeg", body.getString("preview_format"))
    }

    // ── aspect_ratio ─────────────────────────────────────────────────────────

    @Test
    fun `body carries aspect_ratio only when one is given`() {
        val sdxl = LocalDreamModule.buildGenerationBody(params(), size = 1024 to 768, aspectRatio = "4:3")
        assertEquals("4:3", sdxl.getString("aspect_ratio"))
        assertEquals(1024, sdxl.getInt("width"))
        assertEquals(768, sdxl.getInt("height"))

        val sd15 = LocalDreamModule.buildGenerationBody(params(), size = 512 to 512)
        assertFalse(sd15.has("aspect_ratio"))
    }

    // ── scheduler map ────────────────────────────────────────────────────────

    @Test
    fun `every v2_8_1 scheduler is passed through, anything else becomes dpm`() {
        for (s in listOf(
            "dpm", "dpm_karras", "euler_a", "euler_a_karras", "euler", "euler_karras", "lcm", "dpm_sde", "dpm_sde_karras",
        )) {
            assertEquals(s, LocalDreamModule.buildGenerationBody(params("scheduler" to s)).getString("scheduler"))
        }
        assertEquals("dpm", LocalDreamModule.buildGenerationBody(params("scheduler" to "dpm++2m")).getString("scheduler"))
        assertEquals("dpm", LocalDreamModule.buildGenerationBody(params()).getString("scheduler"))
    }

    // ── img2img keys kept ────────────────────────────────────────────────────

    @Test
    fun `img2img keys still ride along`() {
        val body = LocalDreamModule.buildGenerationBody(
            params("initImage" to "QUJD", "denoiseStrength" to 0.3, "maskImage" to "TUFTSw=="),
        )
        assertEquals("QUJD", body.getString("image"))
        assertEquals(0.3, body.getDouble("denoise_strength"), 1e-9)
        assertEquals("TUFTSw==", body.getString("mask"))
    }

    // ── SSE error event ──────────────────────────────────────────────────────

    @Test
    fun `an error event yields its message`() {
        val ev = parseSseData("""{"type":"error","message":"Client disconnected, generation aborted"}""", "error")
        assertEquals(SseEvent.Error("Client disconnected, generation aborted"), ev)
    }

    @Test
    fun `an error payload without type is classified by the event line, blank message gets a fallback`() {
        val ev = parseSseData("""{"message":""}""", "error")
        assertTrue(ev is SseEvent.Error)
        assertTrue((ev as SseEvent.Error).message.isNotBlank())
    }

    @Test
    fun `progress and complete are classified, other types ignored`() {
        assertTrue(parseSseData("""{"type":"progress","step":1,"total_steps":20}""", "") is SseEvent.Progress)
        assertTrue(parseSseData("""{"type":"complete","image":"x","width":1,"height":1}""", "") is SseEvent.Complete)
        assertEquals(SseEvent.Other, parseSseData("""{"type":"heartbeat"}""", ""))
    }

    @Test(expected = JSONException::class)
    fun `a malformed payload throws for the caller to log and skip`() {
        parseSseData("not json", "progress")
    }
}
