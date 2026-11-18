package ai.offgridmobile.pdf

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import ai.offgridmobile.SafePromise
import io.legere.pdfiumandroid.PdfDocument
import io.legere.pdfiumandroid.PdfiumCore
import android.os.ParcelFileDescriptor
import android.util.Log
import java.io.File

class PDFExtractorModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val NAME = "PDFExtractorModule"
    }

    private fun safeReject(promise: Promise, code: String, message: String, throwable: Throwable? = null) =
        SafePromise(promise, NAME).reject(code, message, throwable)

    private fun safeResolve(promise: Promise, value: Any?) =
        SafePromise(promise, NAME).resolve(value)

    override fun getName(): String = NAME

    private fun extractPageText(doc: PdfDocument, pageIndex: Int, sb: StringBuilder) {
        val page = doc.openPage(pageIndex)
        try {
            val textPage = page.openTextPage()
            try {
                val charCount = textPage.textPageCountChars()
                if (charCount > 0) {
                    val text = textPage.textPageGetText(0, charCount)
                    if (text != null) sb.append(text).append("\n\n")
                }
            } finally {
                textPage.close()
            }
        } finally {
            page.close()
        }
    }

    @ReactMethod
    fun extractText(filePath: String, maxChars: Double, promise: Promise) {
        Thread {
            try {
                val file = File(filePath)
                if (!file.exists()) {
                    safeReject(promise, "PDF_ERROR", "File not found: $filePath")
                    return@Thread
                }

                val limit = maxChars.toInt()
                val core = PdfiumCore(reactApplicationContext)
                var fd: ParcelFileDescriptor? = null
                var doc: PdfDocument? = null
                val sb = StringBuilder()
                // finally: a corrupt PDF or an exception mid-page must not leak the native
                // document, the file descriptor or the Pdfium library handle.
                try {
                    fd = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
                    doc = core.newDocument(fd)
                    val pageCount = doc.getPageCount()

                    for (i in 0 until pageCount) {
                        extractPageText(doc, i, sb)

                        if (sb.length >= limit) {
                            sb.setLength(limit)
                            sb.append("\n\n... [Extracted ${i + 1} of $pageCount pages]")
                            break
                        }
                    }
                } finally {
                    runCatching { doc?.close() }.onFailure { Log.w(NAME, "doc.close failed: ${it.message}") }
                    runCatching { fd?.close() }.onFailure { Log.w(NAME, "fd.close failed: ${it.message}") }
                    // PdfiumCore's API is not vendored here (io.legere:pdfiumandroid:1.0.35); close it
                    // only if this version is Closeable, so the build never depends on the guess.
                    runCatching { (core as Any as? java.io.Closeable)?.close() }
                        .onFailure { Log.w(NAME, "core.close failed: ${it.message}") }
                }
                safeResolve(promise, sb.toString())
            } catch (e: Exception) {
                safeReject(promise, "PDF_ERROR", "Failed to extract text: ${e.message}", e)
            }
        }.start()
    }
}
