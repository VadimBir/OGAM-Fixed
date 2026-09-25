package ai.offgridmobile.keepalive

import android.os.Build
import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.uimanager.ViewManager

/** JS switch for ModelKeepAliveService (see src/services/modelKeepAlive.ts). */
class ModelKeepAliveModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {
    override fun getName(): String = "ModelKeepAliveModule"

    @ReactMethod
    fun start(label: String, promise: Promise) {
        try {
            ModelKeepAliveService.start(reactContext, label)
            promise.resolve(true)
        } catch (e: RuntimeException) {
            // Background-start denial (Android 12+) or missing permission: report, never crash.
            val denied = e is SecurityException || e is IllegalStateException ||
                (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
                    e is android.app.ForegroundServiceStartNotAllowedException)
            if (denied) promise.resolve(false) else promise.reject("keepalive_failed", e)
        }
    }

    @ReactMethod
    fun stop(promise: Promise) {
        ModelKeepAliveService.stop(reactContext)
        promise.resolve(null)
    }

    override fun invalidate() {
        ModelKeepAliveService.stop(reactContext)
        super.invalidate()
    }
}

class ModelKeepAlivePackage : ReactPackage {
    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> =
        listOf(ModelKeepAliveModule(reactContext))

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> = emptyList()
}
