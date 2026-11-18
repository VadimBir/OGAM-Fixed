package ai.offgridmobile.devicememory

import android.app.ActivityManager
import android.content.ComponentCallbacks2
import android.content.Context
import android.content.res.Configuration
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.DeviceEventManagerModule

/**
 * Exposes the REAL per-process memory headroom to JS — not a fraction-of-total-RAM
 * guess. iOS uses os_proc_available_memory(); Android has no exact twin, so we use
 * ActivityManager's system available memory, which is the practical ceiling for
 * large native allocations before the low-memory killer applies pressure. Same TS
 * contract on both platforms (getMemoryInfo → { processAvailableBytes, ... }).
 *
 * Also forwards the OS memory-pressure callbacks (onTrimMemory / onLowMemory) to JS as
 * TRIM_EVENT, because React Native's AppState 'memoryWarning' is iOS-only. JS side:
 * src/services/modelResidency/memoryPressure.ts (classification + release order).
 */
class DeviceMemoryModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val trimCallbacks = object : ComponentCallbacks2 {
        override fun onTrimMemory(level: Int) {
            emitTrim(level, "trim")
        }

        @Deprecated("Deprecated in Java")
        override fun onLowMemory() {
            emitTrim(LOW_MEMORY_LEVEL, "lowMemory")
        }

        override fun onConfigurationChanged(newConfig: Configuration) {
            // Not a memory signal.
        }
    }

    init {
        reactContext.applicationContext.registerComponentCallbacks(trimCallbacks)
    }

    override fun getName(): String = "DeviceMemoryModule"

    @ReactMethod
    fun getMemoryInfo(promise: Promise) {
        try {
            val mi = readMemoryInfo()
            val map = Arguments.createMap()
            map.putDouble("processAvailableBytes", mi.availMem.toDouble())
            map.putDouble("footprintBytes", (mi.totalMem - mi.availMem).toDouble())
            map.putBoolean("lowMemory", mi.lowMemory)
            promise.resolve(map)
        } catch (e: Exception) {
            promise.reject("MEM_ERROR", e)
        }
    }

    @ReactMethod
    fun addListener(eventName: String) {
        // Required by React Native's NativeEventEmitter contract.
    }

    @ReactMethod
    fun removeListeners(count: Int) {
        // Callbacks stay registered for the module lifetime; emit is a no-op without JS.
    }

    override fun invalidate() {
        runCatching {
            reactApplicationContext.applicationContext.unregisterComponentCallbacks(trimCallbacks)
        }
        super.invalidate()
    }

    private fun readMemoryInfo(): ActivityManager.MemoryInfo {
        val am = reactApplicationContext
            .getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        return mi
    }

    private fun emitTrim(level: Int, source: String) {
        val context = reactApplicationContext
        if (!context.hasActiveReactInstance()) return
        runCatching {
            val mi = readMemoryInfo()
            val payload = Arguments.createMap().apply {
                putInt("level", level)
                putString("source", source)
                putBoolean("lowMemory", mi.lowMemory)
                putDouble("availMB", (mi.availMem / (1024L * 1024L)).toDouble())
            }
            context
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit(TRIM_EVENT, payload)
        }
    }

    companion object {
        /** Must match TRIM_EVENT in src/services/modelResidency/memoryPressure.ts. */
        const val TRIM_EVENT = "DeviceMemoryTrim"
        /** onLowMemory has no ComponentCallbacks2 level; JS classifies by source. */
        const val LOW_MEMORY_LEVEL = -1
    }
}
