package ai.offgridmobile.keepalive

import android.os.Build
import android.util.Log
import com.facebook.react.ReactPackage
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.facebook.react.uimanager.ViewManager

/** JS switch for ModelKeepAliveService (see src/services/modelKeepAlive.ts). */
class ModelKeepAliveModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {
    override fun getName(): String = "ModelKeepAliveModule"

    private val stopListener: (String) -> Unit = { reason -> emitStopRequested(reason) }

    init {
        // Notification "Stop" / swipe-away reach JS through this, so JS unloads the models.
        ModelKeepAliveService.stopListener = stopListener
    }

    /** Event EVENT_STOP_REQUESTED {reason: "notification" | "task_removed"}: JS must unload the
     *  resident models (the service is already stopping). */
    private fun emitStopRequested(reason: String) {
        if (!reactContext.hasActiveReactInstance()) {
            Log.w(TAG, "Stop requested ($reason) but no active React instance to unload models")
            return
        }
        try {
            reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit(EVENT_STOP_REQUESTED, Arguments.createMap().apply { putString("reason", reason) })
        } catch (e: RuntimeException) {
            Log.e(TAG, "Could not emit $EVENT_STOP_REQUESTED", e)
        }
    }

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

    @ReactMethod
    fun addListener(eventName: String) { /* required by NativeEventEmitter */ }

    @ReactMethod
    fun removeListeners(count: Int) { /* required by NativeEventEmitter */ }

    override fun invalidate() {
        if (ModelKeepAliveService.stopListener === stopListener) ModelKeepAliveService.stopListener = null
        ModelKeepAliveService.stop(reactContext)
        super.invalidate()
    }

    companion object {
        private const val TAG = "ModelKeepAlive"
        const val EVENT_STOP_REQUESTED = "ModelKeepAliveStopRequested"
    }
}

class ModelKeepAlivePackage : ReactPackage {
    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> =
        listOf(ModelKeepAliveModule(reactContext))

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> = emptyList()
}
