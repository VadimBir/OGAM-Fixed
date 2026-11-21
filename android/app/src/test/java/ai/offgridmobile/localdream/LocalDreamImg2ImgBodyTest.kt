package ai.offgridmobile.localdream

import android.app.Application
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The bundled SD core reads `image` / `mask` / `denoise_strength` (its JSON key strings; upstream
 * c56cf6c `if (json.contains("image"))`). The old `init_img` key made every img2img run as txt2img.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], application = Application::class)
class LocalDreamImg2ImgBodyTest {
    @Test
    fun `an init image is sent as image with the denoise strength, never init_img`() {
        val body = JSONObject()
        putImg2ImgKeys(body, initImage = "QUJD", denoise = 0.4, mask = null)
        assertEquals("QUJD", body.getString("image"))
        assertEquals(0.4, body.getDouble("denoise_strength"), 1e-9)
        assertFalse(body.has("init_img"))
        assertFalse(body.has("mask"))
    }

    @Test
    fun `denoise defaults to 0_6 and is clamped to 0 to 1, mask rides along`() {
        val dflt = JSONObject().also { putImg2ImgKeys(it, "QUJD", null, "TUFTSw==") }
        assertEquals(0.6, dflt.getDouble("denoise_strength"), 1e-9)
        assertEquals("TUFTSw==", dflt.getString("mask"))
        val high = JSONObject().also { putImg2ImgKeys(it, "QUJD", 3.0, null) }
        assertEquals(1.0, high.getDouble("denoise_strength"), 1e-9)
    }

    @Test
    fun `no init image keeps the body txt2img, even with a mask`() {
        val body = JSONObject()
        putImg2ImgKeys(body, initImage = " ", denoise = 0.5, mask = "TUFTSw==")
        assertEquals(0, body.length())
    }
}
