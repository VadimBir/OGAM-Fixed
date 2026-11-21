package ai.offgridmobile.localdream

import android.app.Application
import android.graphics.BitmapFactory
import android.os.Build
import android.util.Base64
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowBitmapFactory
import org.robolectric.util.ReflectionHelpers
import java.io.File

/**
 * Tests for pure helper functions in LocalDreamModule.
 * All methods under test live in the companion object and have no instance state.
 */
@Suppress("kotlin:S100") // Backtick test names are idiomatic Kotlin
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33], application = Application::class)
class LocalDreamModuleTest {

    @get:Rule
    val tmp = TemporaryFolder()

    // ── isNpuSupportedInternal ────────────────────────────────────────────────

    @Test
    fun `isNpuSupportedInternal returns true for SM8650 (SD 8 Gen 3)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "SM8650")
        assertTrue(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns true for SM8550 (SD 8 Gen 2)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "SM8550")
        assertTrue(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns true for SM8350 (SD 888)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "SM8350")
        assertTrue(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns true for SM8635 (SD 8s Gen 3)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "SM8635")
        assertTrue(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns true for SM7450 (SD 7 Gen 1)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "SM7450")
        assertTrue(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns true for SM8250 (SD 870 — any SM is supported)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "SM8250")
        assertTrue(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns true for SM7225 (SD 750G — any SM is supported)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "SM7225")
        assertTrue(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns false for QCS prefix (not SM)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "QCS8550")
        assertFalse(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns false for QCM prefix (not SM)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "QCM6490")
        assertFalse(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns false for non-Qualcomm SoC`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "Exynos2400")
        assertFalse(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns false for Tensor (Google)`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "Tensor G4")
        assertFalse(LocalDreamModule.isNpuSupportedInternal())
    }

    @Test
    fun `isNpuSupportedInternal returns false for empty SoC model`() {
        ReflectionHelpers.setStaticField(Build::class.java, "SOC_MODEL", "")
        assertFalse(LocalDreamModule.isNpuSupportedInternal())
    }

    // ── resolveModelDir ───────────────────────────────────────────────────────

    @Test
    fun `resolveModelDir returns root dir when marker is at root for CPU`() {
        val root = tmp.newFolder("model")
        root.resolve("unet.mnn").createNewFile()

        assertEquals(root, LocalDreamModule.resolveModelDir(root, isCpu = true))
    }

    @Test
    fun `resolveModelDir returns root dir when marker is at root for QNN`() {
        val root = tmp.newFolder("model")
        root.resolve("unet.bin").createNewFile()

        assertEquals(root, LocalDreamModule.resolveModelDir(root, isCpu = false))
    }

    @Test
    fun `resolveModelDir finds marker one level deep`() {
        val root = tmp.newFolder("model")
        val sub = root.resolve("inner").also { it.mkdir() }
        sub.resolve("unet.mnn").createNewFile()

        assertEquals(sub, LocalDreamModule.resolveModelDir(root, isCpu = true))
    }

    @Test
    fun `resolveModelDir finds marker three levels deep`() {
        val root = tmp.newFolder("model")
        val deep = root.resolve("a/b/c").also { it.mkdirs() }
        deep.resolve("unet.bin").createNewFile()

        assertEquals(deep, LocalDreamModule.resolveModelDir(root, isCpu = false))
    }

    @Test
    fun `resolveModelDir finds marker four levels deep (boundary of search depth)`() {
        // searchDir is called with depth=3 for the 4th level directory — depth > 3 is false,
        // so children are still checked. The limit only cuts off at depth=4 (5 levels below root).
        val root = tmp.newFolder("model")
        val deep = root.resolve("a/b/c/d").also { it.mkdirs() }
        deep.resolve("unet.mnn").createNewFile()

        assertEquals(deep, LocalDreamModule.resolveModelDir(root, isCpu = true))
    }

    @Test
    fun `resolveModelDir returns null when marker is five levels deep (beyond limit)`() {
        val root = tmp.newFolder("model")
        root.resolve("a/b/c/d/e").also { it.mkdirs() }.resolve("unet.mnn").createNewFile()

        assertNull(LocalDreamModule.resolveModelDir(root, isCpu = true))
    }

    @Test
    fun `resolveModelDir returns null when no marker file exists`() {
        val root = tmp.newFolder("model")
        root.resolve("some_other_file.bin").createNewFile()

        assertNull(LocalDreamModule.resolveModelDir(root, isCpu = true))
    }

    @Test
    fun `resolveModelDir does not confuse CPU and QNN markers`() {
        val root = tmp.newFolder("model")
        root.resolve("unet.bin").createNewFile() // QNN marker only

        // CPU search should not match unet.bin
        assertNull(LocalDreamModule.resolveModelDir(root, isCpu = true))
        // QNN search should match
        assertNotNull(LocalDreamModule.resolveModelDir(root, isCpu = false))
    }

    // ── buildCommand — v2.8.1 core CLI (main.cpp:141-159) ─────────────────────

    private fun makeCpuModelDir(withEncoder: Boolean = false): java.io.File = tmp.newFolder("cpu_model").also { dir ->
        listOf("tokenizer.json", "clip_v2.mnn", "pos_emb.bin", "token_emb.bin", "unet.mnn", "vae_decoder.mnn")
            .forEach { dir.resolve(it).createNewFile() }
        if (withEncoder) dir.resolve("vae_encoder.mnn").createNewFile()
    }

    private fun makeQnnModelDir(withEncoder: Boolean = false): java.io.File =
        tmp.newFolder("qnn_model").also { dir ->
            listOf("tokenizer.json", "clip_v2.mnn", "pos_emb.bin", "token_emb.bin", "unet.bin", "vae_decoder.bin")
                .forEach { dir.resolve(it).createNewFile() }
            if (withEncoder) dir.resolve("vae_encoder.bin").createNewFile()
        }

    private fun makeSdxlModelDir(withEncoder: Boolean = true): java.io.File =
        tmp.newFolder("sdxl_model").also { dir ->
            listOf(
                "tokenizer.json", "clip.mnn", "clip_2.mnn", "clip_2.mnn.weight", "pos_emb.bin", "token_emb.bin",
                "pos_emb_2.bin", "token_emb_2.bin", "unet.bin", "vae_decoder.bin", "SDXL",
            ).forEach { dir.resolve(it).createNewFile() }
            if (withEncoder) dir.resolve("vae_encoder.bin").createNewFile()
        }

    private fun makeExecutable(): java.io.File = tmp.newFile("libstable_diffusion_core.so")
    private fun makeRuntimeDir(): java.io.File = tmp.newFolder("runtime")

    /** Flags the v2.8.1 core rejects (main.cpp:226-227 exits on an unknown option). */
    private val removedFlags = listOf(
        "--clip", "--unet", "--vae_decoder", "--vae_encoder", "--tokenizer", "--backend", "--system_library",
        "--text_embedding_size", "--use_cpu_clip", "--cpu",
    )

    private fun valueOf(cmd: List<String>, flag: String): String? =
        cmd.indexOf(flag).takeIf { it >= 0 }?.let { cmd[it + 1] }

    @Test
    fun `buildCommand sd15cpu passes type, model_dir and port, no lib_dir`() {
        val dir = makeCpuModelDir(withEncoder = true)
        val exe = makeExecutable()
        val cmd = LocalDreamModule.buildCommand(exe, dir, makeRuntimeDir(), isCpu = true)
        assertEquals(
            listOf(exe.absolutePath, "--type", "sd15cpu", "--model_dir", dir.absolutePath, "--port", "18081"),
            cmd,
        )
        assertFalse(cmd.contains("--lib_dir"))
    }

    @Test
    fun `buildCommand sd15cpu adds --no_img2img without vae_encoder mnn`() {
        val cmd = LocalDreamModule.buildCommand(makeExecutable(), makeCpuModelDir(), makeRuntimeDir(), isCpu = true)
        assertTrue(cmd.contains("--no_img2img"))
    }

    @Test
    fun `buildCommand sd15npu passes type, model_dir, lib_dir and port`() {
        val dir = makeQnnModelDir(withEncoder = true)
        val rt = makeRuntimeDir()
        val cmd = LocalDreamModule.buildCommand(makeExecutable(), dir, rt, isCpu = false)
        assertEquals("sd15npu", valueOf(cmd, "--type"))
        assertEquals(dir.absolutePath, valueOf(cmd, "--model_dir"))
        assertEquals(rt.absolutePath, valueOf(cmd, "--lib_dir"))
        assertEquals("18081", valueOf(cmd, "--port"))
        assertFalse(cmd.contains("--no_img2img"))
        assertFalse(cmd.contains("--lowram"))
    }

    @Test
    fun `buildCommand sd15npu adds --no_img2img without vae_encoder bin`() {
        val cmd = LocalDreamModule.buildCommand(makeExecutable(), makeQnnModelDir(), makeRuntimeDir(), isCpu = false)
        assertTrue(cmd.contains("--no_img2img"))
    }

    @Test
    fun `buildCommand adds --use_v_pred only when a V_PRED file exists`() {
        val exe = makeExecutable()
        val rt = makeRuntimeDir()
        val dir = makeQnnModelDir(withEncoder = true)
        assertFalse(LocalDreamModule.buildCommand(exe, dir, rt, isCpu = false).contains("--use_v_pred"))
        dir.resolve("V_PRED").createNewFile()
        assertTrue(LocalDreamModule.buildCommand(exe, dir, rt, isCpu = false).contains("--use_v_pred"))
    }

    @Test
    fun `buildCommand sdxl passes type sdxl, lib_dir and --lowram, never --patch`() {
        val dir = makeSdxlModelDir()
        val rt = makeRuntimeDir()
        val patch = File(dir, "768.patch").apply { writeText("x") }
        val cmd = LocalDreamModule.buildCommand(
            makeExecutable(), dir, rt, isCpu = false, patchFile = patch, port = 40125, isSdxl = true,
        )
        assertEquals("sdxl", valueOf(cmd, "--type"))
        assertEquals(dir.absolutePath, valueOf(cmd, "--model_dir"))
        assertEquals(rt.absolutePath, valueOf(cmd, "--lib_dir"))
        assertEquals("40125", valueOf(cmd, "--port"))
        assertTrue(cmd.contains("--lowram"))
        assertFalse(cmd.contains("--patch"))
        assertFalse(cmd.contains("--no_img2img"))
    }

    @Test
    fun `buildCommand sdxl without vae_encoder adds --no_img2img`() {
        val cmd = LocalDreamModule.buildCommand(
            makeExecutable(), makeSdxlModelDir(withEncoder = false), makeRuntimeDir(), isCpu = false, isSdxl = true,
        )
        assertTrue(cmd.contains("--no_img2img"))
        assertTrue(cmd.contains("--lowram"))
    }

    @Test
    fun `buildCommand never passes --safety_checker nor a removed pre-v2_8 flag, for any kind`() {
        val exe = makeExecutable()
        val rt = makeRuntimeDir()
        val cmds = listOf(
            LocalDreamModule.buildCommand(exe, makeCpuModelDir(), rt, isCpu = true),
            LocalDreamModule.buildCommand(exe, makeQnnModelDir(), rt, isCpu = false),
            LocalDreamModule.buildCommand(exe, makeSdxlModelDir(), rt, isCpu = false, isSdxl = true),
        )
        for (cmd in cmds) {
            assertFalse(cmd.toString(), cmd.any { it.startsWith("--safety_checker") })
            for (flag in removedFlags) assertFalse("$flag in $cmd", cmd.contains(flag))
        }
    }

    @Test
    fun `buildCommand isSdxl is ignored on the CPU backend`() {
        val cmd = LocalDreamModule.buildCommand(
            makeExecutable(), makeCpuModelDir(), makeRuntimeDir(), isCpu = true, isSdxl = true,
        )
        assertEquals("sd15cpu", valueOf(cmd, "--type"))
        assertFalse(cmd.contains("--lowram"))
    }

    // ── SDXL detection / SoC gate ────────────────────────────────────────────

    @Test
    fun `resolveModelDir finds the SDXL unet under output qnn_models_sdxl_8gen3 and isSdxlModelDir tells it apart`() {
        val root = tmp.newFolder("sdxl_root")
        val inner = root.resolve("output/qnn_models_sdxl_8gen3").also { it.mkdirs() }
        listOf("unet.bin", "clip.mnn", "clip_2.mnn", "SDXL").forEach { inner.resolve(it).createNewFile() }

        val resolved = LocalDreamModule.resolveModelDir(root, isCpu = false)
        assertEquals(inner, resolved)
        assertTrue(LocalDreamModule.isSdxlModelDir(requireNotNull(resolved), root))
    }

    @Test
    fun `isSdxlModelDir accepts the SDXL marker, clip_2 mnn or a root marker, rejects SD1_5`() {
        val sd15 = makeQnnModelDir()
        assertFalse(LocalDreamModule.isSdxlModelDir(sd15))

        val markerOnly = tmp.newFolder("marker_only").also { it.resolve("SDXL").createNewFile() }
        assertTrue(LocalDreamModule.isSdxlModelDir(markerOnly))

        val clip2Only = tmp.newFolder("clip2_only").also { it.resolve("clip_2.mnn").createNewFile() }
        assertTrue(LocalDreamModule.isSdxlModelDir(clip2Only))

        val root = tmp.newFolder("root_marker").also { it.resolve("SDXL").createNewFile() }
        val inner = root.resolve("inner").also { it.mkdirs() }
        assertTrue(LocalDreamModule.isSdxlModelDir(inner, root))
    }

    @Test
    fun `isSdxlCapableSoc matches upstream's set, including a suffixed SM8850`() {
        listOf("SM8650", "SM8750", "SM8750P", "SM8850", "SM8850P", "SM8845", "SM8850-AC").forEach {
            assertTrue(it, LocalDreamModule.isSdxlCapableSoc(it))
        }
        listOf("SM8550", "SM8475", "SM7675", "", "Tensor G4").forEach {
            assertFalse(it, LocalDreamModule.isSdxlCapableSoc(it))
        }
    }

    // ── SDXL size plan / aspect ratio ────────────────────────────────────────

    @Test
    fun `sdxlAspectRatio reduces a non-square request and is null when square or without encoder`() {
        assertEquals("4:3", LocalDreamModule.sdxlAspectRatio(1024, 768, hasVaeEncoder = true))
        assertEquals("9:16", LocalDreamModule.sdxlAspectRatio(576, 1024, hasVaeEncoder = true))
        assertNull(LocalDreamModule.sdxlAspectRatio(512, 512, hasVaeEncoder = true))
        assertNull(LocalDreamModule.sdxlAspectRatio(1024, 768, hasVaeEncoder = false))
    }

    @Test
    fun `planSdxlSize mirrors the core crop of the 1024 canvas`() {
        assertEquals(
            LocalDreamModule.Companion.RenderPlan(1024, 1024, true, "SDXL renders on a fixed 1024 px canvas"),
            LocalDreamModule.planSdxlSize(512, 512, hasVaeEncoder = true),
        )
        assertEquals(
            LocalDreamModule.Companion.RenderPlan(1024, 1024, false, null),
            LocalDreamModule.planSdxlSize(1024, 1024, hasVaeEncoder = true),
        )
        // RequestParser.hpp:118-127: short side = floor(1024 * short / long), floored to a multiple of 8.
        assertEquals(1024 to 728, LocalDreamModule.planSdxlSize(700, 500, true).let { it.width to it.height })
        assertEquals(576 to 1024, LocalDreamModule.planSdxlSize(576, 1024, true).let { it.width to it.height })
        val noEncoder = LocalDreamModule.planSdxlSize(1024, 768, hasVaeEncoder = false)
        assertEquals(1024 to 1024, noEncoder.width to noEncoder.height)
        assertEquals("SDXL without a VAE encoder renders only 1024x1024", noEncoder.reason)
    }

    // ── buildEnvironment ──────────────────────────────────────────────────────

    @Test
    fun `buildEnvironment always sets all three env vars`() {
        val runtimeDir = makeRuntimeDir()
        val env = LocalDreamModule.buildEnvironment(runtimeDir)

        assertTrue(env.containsKey("LD_LIBRARY_PATH"))
        assertTrue(env.containsKey("DSP_LIBRARY_PATH"))
        assertTrue(env.containsKey("ADSP_LIBRARY_PATH"))
    }

    @Test
    fun `buildEnvironment sets DSP and ADSP paths to runtimeDir`() {
        val runtimeDir = makeRuntimeDir()
        val env = LocalDreamModule.buildEnvironment(runtimeDir)

        assertEquals(runtimeDir.absolutePath, env["DSP_LIBRARY_PATH"])
        assertEquals(runtimeDir.absolutePath, env["ADSP_LIBRARY_PATH"])
    }

    @Test
    fun `buildEnvironment includes runtimeDir as first entry in LD_LIBRARY_PATH`() {
        val runtimeDir = makeRuntimeDir()
        val env = LocalDreamModule.buildEnvironment(runtimeDir)

        // Prefix check, not split(":"): on a Windows test host the absolute path itself holds a colon (C:\).
        assertTrue(requireNotNull(env["LD_LIBRARY_PATH"]).startsWith(runtimeDir.absolutePath + ":"))
    }

    @Test
    fun `buildEnvironment includes standard system library paths`() {
        val env = LocalDreamModule.buildEnvironment(makeRuntimeDir())
        val ldPath = requireNotNull(env["LD_LIBRARY_PATH"])

        assertTrue(ldPath.contains("/system/lib64"))
        assertTrue(ldPath.contains("/vendor/lib64"))
        assertTrue(ldPath.contains("/vendor/lib64/egl"))
    }

    // ── saveServerImageToPng: the ONE decoder (interleaved RGB24 / RGBA / encoded) ──
    // Matches the known-good reference build: raw w*h*3 → interleaved RGB24 (HWC),
    // raw w*h*4 → interleaved RGBA, otherwise → BitmapFactory (encoded, e.g. JPEG preview).

    private fun rgbBase64(vararg bytes: Int): String =
        Base64.encodeToString(ByteArray(bytes.size) { bytes[it].toByte() }, Base64.DEFAULT)

    @Test
    fun `raw decode throws when byte count is neither RGB nor RGBA`() {
        val base64 = Base64.encodeToString(ByteArray(6), Base64.DEFAULT) // 6 bytes; 2x2 needs 12 or 16
        // Legacy-graphics Robolectric decodes ANY bytes to a fake bitmap; a device returns null for garbage.
        ShadowBitmapFactory.setAllowInvalidImageData(false)
        try {
            LocalDreamModule.saveServerImageToPng(base64, 2, 2, tmp.newFile("out.png").absolutePath)
            fail("Expected IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("12"))
        }
    }

    @Test
    fun `raw decode creates a PNG file at the given path`() {
        val base64 = rgbBase64(0xFF, 0x00, 0x00) // 1x1 red (planar==interleaved at n=1)
        val out = tmp.newFile("out.png")
        LocalDreamModule.saveServerImageToPng(base64, 1, 1, out.absolutePath)
        assertTrue(out.exists())
        assertTrue("file should not be empty", out.length() > 0)
    }

    @Test
    fun `raw decode creates parent directories when they do not exist`() {
        val out = File(tmp.root, "a/b/c/out.png")
        assertFalse(out.parentFile!!.exists())
        LocalDreamModule.saveServerImageToPng(rgbBase64(0, 0, 0), 1, 1, out.absolutePath)
        assertTrue(out.exists())
    }

    @Test
    fun `raw decode encodes red channel correctly`() {
        val base64 = rgbBase64(0xFF, 0x00, 0x00) // 1x1 red -> ARGB 0xFFFF0000
        val out = tmp.newFile("red.png")
        LocalDreamModule.saveServerImageToPng(base64, 1, 1, out.absolutePath)
        val pixel = BitmapFactory.decodeFile(out.absolutePath).getPixel(0, 0)
        assertEquals(0xFFFF0000.toInt(), pixel)
    }

    @Test
    fun `raw decode encodes blue channel correctly`() {
        val base64 = rgbBase64(0x00, 0x00, 0xFF) // 1x1 blue -> ARGB 0xFF0000FF
        val out = tmp.newFile("blue.png")
        LocalDreamModule.saveServerImageToPng(base64, 1, 1, out.absolutePath)
        val pixel = BitmapFactory.decodeFile(out.absolutePath).getPixel(0, 0)
        assertEquals(0xFF0000FF.toInt(), pixel)
    }

    @Test
    fun `raw decode reads multi-pixel as interleaved RGB`() {
        // 2x1 image [red | blue] in INTERLEAVED layout: pixel0 R,G,B = FF,00,00; pixel1 = 00,00,FF
        val base64 = rgbBase64(0xFF, 0x00, 0x00,  0x00, 0x00, 0xFF)
        val out = tmp.newFile("2x1.png")
        LocalDreamModule.saveServerImageToPng(base64, 2, 1, out.absolutePath)
        val bmp = BitmapFactory.decodeFile(out.absolutePath)
        assertEquals(0xFFFF0000.toInt(), bmp.getPixel(0, 0))
        assertEquals(0xFF0000FF.toInt(), bmp.getPixel(1, 0))
    }

    @Test
    fun `raw decode reads RGBA (w_h_4) as interleaved`() {
        // 1x1 with explicit alpha byte -> interleaved RGBA path
        val base64 = rgbBase64(0x12, 0x34, 0x56, 0x80)
        val out = tmp.newFile("rgba.png")
        LocalDreamModule.saveServerImageToPng(base64, 1, 1, out.absolutePath)
        val pixel = BitmapFactory.decodeFile(out.absolutePath).getPixel(0, 0)
        assertEquals(0x80, (pixel ushr 24) and 0xFF)
    }

    @Test
    fun `raw RGB decode preserves alpha as fully opaque`() {
        val base64 = rgbBase64(0x12, 0x34, 0x56) // 1x1 RGB
        val out = tmp.newFile("alpha.png")
        LocalDreamModule.saveServerImageToPng(base64, 1, 1, out.absolutePath)
        val pixel = BitmapFactory.decodeFile(out.absolutePath).getPixel(0, 0)
        assertEquals(0xFF, (pixel ushr 24) and 0xFF)
    }

    // ── looksLikeEncodedImage / saveServerImageToPng (rainbow-bug fix) ──────────

    /** Encode a 1x1 solid-color bitmap to a real compressed image, base64-encoded. */
    private fun encodedImageBase64(
        format: android.graphics.Bitmap.CompressFormat,
        argb: Int,
    ): String {
        val bmp = android.graphics.Bitmap.createBitmap(1, 1, android.graphics.Bitmap.Config.ARGB_8888)
        bmp.setPixel(0, 0, argb)
        val bos = java.io.ByteArrayOutputStream()
        bmp.compress(format, 100, bos)
        return Base64.encodeToString(bos.toByteArray(), Base64.DEFAULT)
    }

    @Test
    fun `decodeBase64Image strips a data-URL prefix before decoding`() {
        // "data:image/png;base64,<...>" must decode to the same bytes as the bare base64.
        val raw = rgbBase64(0x01, 0x02, 0x03)
        val prefixed = "data:image/png;base64,$raw"
        assertArrayEquals(
            LocalDreamModule.decodeBase64Image(raw),
            LocalDreamModule.decodeBase64Image(prefixed),
        )
    }

    @Test
    fun `saveServerImageToPng decodes an encoded PNG payload instead of packing bytes`() {
        // Regression for the rainbow bug: an encoded (PNG/JPEG) payload must be image-decoded,
        // not hand-packed as raw RGB. Byte length here != width*height*3, so the raw path would throw.
        val base64 = encodedImageBase64(android.graphics.Bitmap.CompressFormat.PNG, 0xFF00FF00.toInt())
        val out = tmp.newFile("encoded.png")
        LocalDreamModule.saveServerImageToPng(base64, 1, 1, out.absolutePath)
        val pixel = BitmapFactory.decodeFile(out.absolutePath).getPixel(0, 0)
        assertEquals(0xFF00FF00.toInt(), pixel)
    }

    @Test
    fun `saveServerImageToPng decodes an encoded JPEG payload`() {
        val base64 = encodedImageBase64(android.graphics.Bitmap.CompressFormat.JPEG, 0xFFFF0000.toInt())
        val out = tmp.newFile("encoded.jpg.png")
        LocalDreamModule.saveServerImageToPng(base64, 1, 1, out.absolutePath)
        val bmp = BitmapFactory.decodeFile(out.absolutePath)
        assertNotNull(bmp)
        // JPEG is lossy; assert it is red-dominant and opaque rather than an exact value.
        val pixel = bmp.getPixel(0, 0)
        assertEquals(0xFF, (pixel ushr 24) and 0xFF)
        assertTrue("red channel should dominate", ((pixel ushr 16) and 0xFF) > 200)
    }

    @Test
    fun `saveServerImageToPng falls back to raw RGB for a non-encoded payload`() {
        val base64 = rgbBase64(0x00, 0x00, 0xFF) // 1x1 raw blue
        val out = tmp.newFile("raw-fallback.png")
        LocalDreamModule.saveServerImageToPng(base64, 1, 1, out.absolutePath)
        assertEquals(0xFF0000FF.toInt(), BitmapFactory.decodeFile(out.absolutePath).getPixel(0, 0))
    }


    // ── NPU resolution patches (sub-512 rainbow) ─────────────────────────────

    @Test
    fun `resolvePatchFile returns null at 512 and when no patch exists`() {
        val dir = tmp.newFolder("npu1")
        assertNull(LocalDreamModule.resolvePatchFile(dir, 512, 512))
        assertNull(LocalDreamModule.resolvePatchFile(dir, 384, 384))
    }

    @Test
    fun `resolvePatchFile prefers square W patch then WxH, non-square only WxH`() {
        val dir = tmp.newFolder("npu2")
        File(dir, "384x384.patch").writeText("x")
        assertEquals("384x384.patch", LocalDreamModule.resolvePatchFile(dir, 384, 384)?.name)
        File(dir, "384.patch").writeText("x")
        assertEquals("384.patch", LocalDreamModule.resolvePatchFile(dir, 384, 384)?.name)
        File(dir, "256x512.patch").writeText("x")
        assertEquals("256x512.patch", LocalDreamModule.resolvePatchFile(dir, 256, 512)?.name)
        assertNull(LocalDreamModule.resolvePatchFile(dir, 512, 256))
    }

    @Test
    fun `availableQnnSizes lists 512 plus patches`() {
        val dir = tmp.newFolder("npu3")
        File(dir, "768.patch").writeText("x")
        File(dir, "256x512.patch").writeText("x")
        File(dir, "unet.bin").writeText("x")
        val sizes = LocalDreamModule.availableQnnSizes(dir).toSet()
        assertEquals(setOf(512 to 512, 768 to 768, 256 to 512), sizes)
    }

    @Test
    fun `buildCommand sd15npu passes --patch only when given, CPU never`() {
        val npu = makeQnnModelDir()
        val patch = File(npu, "384.patch").apply { writeText("x") }
        val exe = makeExecutable()
        val rt = makeRuntimeDir()
        val withPatch = LocalDreamModule.buildCommand(exe, npu, rt, isCpu = false, patchFile = patch)
        assertEquals(patch.absolutePath, withPatch[withPatch.indexOf("--patch") + 1])
        assertFalse(LocalDreamModule.buildCommand(exe, npu, rt, isCpu = false).contains("--patch"))
        assertFalse(
            LocalDreamModule.buildCommand(exe, makeCpuModelDir(), rt, isCpu = true, patchFile = patch).contains("--patch"),
        )
    }

    @Test
    fun `snapMnnSize keeps upstream square 128 to 512 on the 64 grid`() {
        assertEquals(128, LocalDreamModule.snapMnnSize(64))
        assertEquals(256, LocalDreamModule.snapMnnSize(256))
        assertEquals(384, LocalDreamModule.snapMnnSize(360))
        assertEquals(512, LocalDreamModule.snapMnnSize(1024))
    }

    // -- buildCommand port / isPathInside / shouldCopyRuntimeAsset -------------

    @Test
    fun `buildCommand passes the given port to both backends`() {
        val exe = makeExecutable()
        val rt = makeRuntimeDir()
        val cpu = LocalDreamModule.buildCommand(exe, makeCpuModelDir(), rt, isCpu = true, port = 40123)
        assertEquals("40123", cpu[cpu.indexOf("--port") + 1])
        val npu = LocalDreamModule.buildCommand(exe, makeQnnModelDir(), rt, isCpu = false, port = 40124)
        assertEquals("40124", npu[npu.indexOf("--port") + 1])
        val sdxl = LocalDreamModule.buildCommand(exe, makeSdxlModelDir(), rt, isCpu = false, port = 40126, isSdxl = true)
        assertEquals("40126", sdxl[sdxl.indexOf("--port") + 1])
    }

    @Test
    fun `isPathInside accepts root and children, rejects sibling prefix`() {
        val root = "/data/user/0/ai.offgridmobile/files"
        assertTrue(LocalDreamModule.isPathInside(root, root))
        assertTrue(LocalDreamModule.isPathInside("$root/models/sd", root))
        assertFalse(LocalDreamModule.isPathInside("${root}_evil/models", root))
        assertFalse(LocalDreamModule.isPathInside("/data/user/0/other", root))
    }

    @Test
    fun `shouldCopyRuntimeAsset copies QNN libs only for NPU and executable only when asked`() {
        assertTrue(LocalDreamModule.shouldCopyRuntimeAsset("libQnnHtp.so", copyQnnLibs = true, copyExecutable = false))
        assertFalse(LocalDreamModule.shouldCopyRuntimeAsset("libQnnHtp.so", copyQnnLibs = false, copyExecutable = true))
        assertTrue(LocalDreamModule.shouldCopyRuntimeAsset("libstable_diffusion_core.so", copyQnnLibs = false, copyExecutable = true))
        assertFalse(LocalDreamModule.shouldCopyRuntimeAsset("libstable_diffusion_core.so", copyQnnLibs = true, copyExecutable = false))
    }
}
