package ai.offgridmobile.download

import okhttp3.Request
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.ServerSocket
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread

/**
 * Civitai gated downloads: the key rides as `?token=` on the civitai.com URL. A 401 / 403 / sign-in
 * page must fail with a key-specific reason (WorkerDownload.civitaiAuthFailureReason) instead of a
 * generic failure or a junk "model" file.
 */
class CivitaiDownloadAuthTest {

    private val dl = "https://civitai.com/api/download/models/101"
    private val dlKey = "$dl?token=abc"
    private val signed = "https://civitai-delivery.example-storage.com/file?X-Amz-Signature=s"

    @Test
    fun noKeyAnd401IsAuthRequired() {
        assertEquals(DownloadReason.CIVITAI_AUTH_REQUIRED, WorkerDownload.civitaiAuthFailureReason(dl, 401, dl, "application/json"))
    }

    @Test
    fun noKeyAnd403IsAuthRequired() {
        assertEquals(DownloadReason.CIVITAI_AUTH_REQUIRED, WorkerDownload.civitaiAuthFailureReason(dl, 403, dl, null))
    }

    @Test
    fun noKeyAndFollowedLoginRedirectIsAuthRequired() {
        val login = "https://civitai.com/login?returnUrl=%2Fapi%2Fdownload%2Fmodels%2F101"
        assertEquals(DownloadReason.CIVITAI_AUTH_REQUIRED, WorkerDownload.civitaiAuthFailureReason(dl, 200, login, null))
    }

    @Test
    fun noKeyAndHtmlBodyIsAuthRequired() {
        assertEquals(DownloadReason.CIVITAI_AUTH_REQUIRED, WorkerDownload.civitaiAuthFailureReason(dl, 200, dl, "text/html; charset=utf-8"))
    }

    @Test
    fun keyAnd401OrLoginPageIsKeyRejected() {
        assertEquals(DownloadReason.CIVITAI_KEY_REJECTED, WorkerDownload.civitaiAuthFailureReason(dlKey, 401, dlKey, null))
        assertEquals(DownloadReason.CIVITAI_KEY_REJECTED, WorkerDownload.civitaiAuthFailureReason("$dl?type=Model&token=abc", 200, "https://civitai.com/login", "text/html"))
    }

    @Test
    fun keyAnd403IsForbidden() {
        assertEquals(DownloadReason.CIVITAI_FORBIDDEN, WorkerDownload.civitaiAuthFailureReason(dlKey, 403, dlKey, null))
    }

    @Test
    fun emptyTokenCountsAsNoKey() {
        assertEquals(DownloadReason.CIVITAI_AUTH_REQUIRED, WorkerDownload.civitaiAuthFailureReason("$dl?token=", 401, dl, null))
    }

    @Test
    fun signedStorageFileIsNotAFailure() {
        assertNull(WorkerDownload.civitaiAuthFailureReason(dlKey, 200, signed, "application/octet-stream"))
        assertNull(WorkerDownload.civitaiAuthFailureReason(dl, 206, signed, null))
    }

    @Test
    fun otherCivitaiErrorsKeepTheGenericPath() {
        assertNull(WorkerDownload.civitaiAuthFailureReason(dlKey, 404, dlKey, null))
        assertNull(WorkerDownload.civitaiAuthFailureReason(dlKey, 429, dlKey, null))
        assertNull(WorkerDownload.civitaiAuthFailureReason(dlKey, 500, dlKey, "text/html"))
    }

    @Test
    fun nonCivitaiDownloadsAreNeverClassified() {
        val hf = "https://huggingface.co/x/resolve/main/m.gguf"
        assertNull(WorkerDownload.civitaiAuthFailureReason(hf, 401, hf, null))
        assertNull(WorkerDownload.civitaiAuthFailureReason("https://notcivitai.com/a?token=x", 200, "https://notcivitai.com/login", "text/html"))
    }

    @Test
    fun civitaiCodesHaveKeySpecificMessagesAndAreNotRetried() {
        assertTrue(DownloadReason.messageFor(DownloadReason.CIVITAI_AUTH_REQUIRED)!!.startsWith("Civitai needs your API key"))
        assertTrue(DownloadReason.messageFor(DownloadReason.CIVITAI_KEY_REJECTED)!!.contains("rejected your API key"))
        assertTrue(DownloadReason.messageFor(DownloadReason.CIVITAI_FORBIDDEN)!!.contains("does not permit"))
        for (c in listOf(DownloadReason.CIVITAI_AUTH_REQUIRED, DownloadReason.CIVITAI_KEY_REJECTED, DownloadReason.CIVITAI_FORBIDDEN)) {
            assertFalse(DownloadReason.isRetryable(c))
            assertEquals(c, DownloadReason.toUiState(DownloadStatus.FAILED, c).reasonCode)
        }
    }

    /**
     * The worker's real OkHttp client on a cross-host redirect (127.0.0.1 -> localhost, two loopback
     * servers): the redirect target is requested exactly as the Location says, so a `?token=` on the
     * first URL never reaches the storage host, and an Authorization header is dropped. This is why
     * the key must be in the first request's query, not only in a header the redirect would lose.
     */
    @Test
    fun workerClientDropsAuthorizationOnCrossHostRedirectAndRequestsLocationAsIs() {
        val storage = OneShotServer { "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 4\r\nConnection: close\r\n\r\nSAFE" }
        val origin = OneShotServer {
            "HTTP/1.1 307 Temporary Redirect\r\nLocation: http://localhost:${storage.port}/file?sig=1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        }
        try {
            val req = Request.Builder()
                .url("http://127.0.0.1:${origin.port}/api/download/models/101?token=abc")
                .header("Authorization", "Bearer abc")
                .build()
            WorkerDownload.httpClient.newCall(req).execute().use { resp ->
                assertEquals(200, resp.code)
                assertEquals("SAFE", resp.body!!.string())
            }
            val first = origin.awaitRequest()
            val second = storage.awaitRequest()
            assertTrue(first.lines.first().startsWith("GET /api/download/models/101?token=abc "))
            assertTrue(first.lines.any { it.equals("Authorization: Bearer abc", ignoreCase = true) })
            assertTrue(second.lines.first().startsWith("GET /file?sig=1 "))
            assertFalse(second.lines.any { it.startsWith("Authorization:", ignoreCase = true) })
            assertFalse(second.lines.first().contains("token="))
        } finally {
            origin.close()
            storage.close()
        }
    }

    private class CapturedRequest(val lines: List<String>)

    /** Loopback HTTP/1.1 server that answers one request with a fixed response and records its head. */
    private class OneShotServer(response: () -> String) {
        private val socket = ServerSocket(0)
        val port: Int = socket.localPort
        private val captured = CopyOnWriteArrayList<CapturedRequest>()
        private val worker = thread(isDaemon = true) {
            runCatching {
                socket.accept().use { s ->
                    val reader = BufferedReader(InputStreamReader(s.getInputStream()))
                    val lines = mutableListOf<String>()
                    while (true) {
                        val line = reader.readLine() ?: break
                        if (line.isEmpty()) break
                        lines.add(line)
                    }
                    captured.add(CapturedRequest(lines))
                    s.getOutputStream().apply { write(response().toByteArray()); flush() }
                }
            }
        }

        fun awaitRequest(): CapturedRequest {
            worker.join(5_000)
            return captured.firstOrNull() ?: throw AssertionError("no request reached port $port")
        }

        fun close() = socket.close()
    }
}
