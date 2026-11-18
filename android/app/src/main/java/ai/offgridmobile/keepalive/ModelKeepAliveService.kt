package ai.offgridmobile.keepalive

import android.app.ForegroundServiceStartNotAllowedException
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat

/**
 * Holds Off Grid at foreground-service priority while a model is resident in RAM.
 *
 * lmkd picks victims by oom_score_adj: a backgrounded app without a service sits at cached
 * priority (900+) and is among the first killed once RAM tightens, which drops the loaded model
 * (seconds-to-minutes to reload). A foreground service lifts the process to perceptible (~200),
 * so every cached and background app is reclaimed before this one. It does NOT make the app
 * unkillable: when the device is truly out of memory lmkd still kills it, so the model budgets
 * stay the first line of defence. In the foreground (adj 0) this changes nothing.
 *
 * The user can always release it: the notification's "Stop" action and swiping the app away from
 * Recents both stop the service and ask JS (via [stopListener]) to unload the models.
 */
class ModelKeepAliveService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            // Notification "Stop": delivered by startService on an already running service, so no
            // startForeground obligation is pending for this command.
            Log.i(TAG, "Stop requested from the notification")
            requestUnloadAndStop("notification")
            return START_NOT_STICKY
        }
        // Promote first and unconditionally: a service started with startForegroundService that
        // stops (or is stopped) before calling startForeground crashes the whole app process.
        val promoted = promote(intent?.getStringExtra(EXTRA_LABEL))
        val stopNow = synchronized(lock) {
            startPending = false
            val s = stopRequested
            stopRequested = false
            s
        }
        if (!promoted || stopNow) {
            if (stopNow) Log.i(TAG, "stop() arrived while the start was pending; stopping now")
            stopSelfSafely()
        }
        // The model lives in this process; after a kill there is nothing to keep alive.
        return START_NOT_STICKY
    }

    /** Swipe-away from Recents: release the model instead of holding it at perceptible priority. */
    override fun onTaskRemoved(rootIntent: Intent?) {
        Log.i(TAG, "Task removed; stopping keep-alive and unloading models")
        requestUnloadAndStop("task_removed")
        super.onTaskRemoved(rootIntent)
    }

    private fun promote(label: String?): Boolean {
        return try {
            val notification = NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Model loaded")
                .setContentText(label ?: "Keeping the model in memory.")
                .setSmallIcon(android.R.drawable.stat_notify_sync_noanim)
                .setOngoing(true)
                .setSilent(true)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .addAction(0, "Stop", stopPendingIntent(this))
                .build()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
            true
        } catch (e: RuntimeException) {
            // Never rethrow from onStartCommand: that kills the app (and the loaded model) for a
            // service whose only job is to protect it.
            val denied = e is SecurityException ||
                (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && e is ForegroundServiceStartNotAllowedException)
            if (denied) Log.w(TAG, "keep-alive promotion denied: ${e.message}")
            else Log.e(TAG, "keep-alive promotion failed", e)
            false
        }
    }

    private fun requestUnloadAndStop(reason: String) {
        val listener = stopListener
        if (listener != null) {
            try {
                listener(reason)
            } catch (e: RuntimeException) {
                Log.e(TAG, "stop listener failed", e)
            }
        } else {
            Log.w(TAG, "No JS listener registered; stopping the service without unloading models")
        }
        stopSelfSafely()
    }

    private fun stopSelfSafely() {
        try {
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        } catch (e: RuntimeException) {
            Log.w(TAG, "stopForeground failed: ${e.message}")
        }
        stopSelf()
    }

    companion object {
        private const val TAG = "ModelKeepAlive"
        const val CHANNEL_ID = "offgrid-model-keepalive"
        const val NOTIFICATION_ID = 4712
        private const val EXTRA_LABEL = "label"
        private const val ACTION_STOP = "ai.offgridmobile.keepalive.STOP"

        /** Called (main thread) when the user stops keep-alive from outside JS: "notification"
         *  or "task_removed". ModelKeepAliveModule forwards it to JS to unload the models. */
        @Volatile
        var stopListener: ((reason: String) -> Unit)? = null

        // start() -> onStartCommand window. stop() inside it must not stopService (the system then
        // kills the app with "did not then call startForeground"); it is deferred to onStartCommand.
        private val lock = Any()
        private var startPending = false
        private var stopRequested = false

        private fun stopPendingIntent(context: Context): PendingIntent {
            val intent = Intent(context, ModelKeepAliveService::class.java).setAction(ACTION_STOP)
            return PendingIntent.getService(
                context, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
        }

        private fun ensureChannel(context: Context) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
            val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (manager.getNotificationChannel(CHANNEL_ID) != null) return
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Loaded model", NotificationManager.IMPORTANCE_LOW).apply {
                    description = "Shown while a model is kept in memory so Android does not reclaim it."
                    setShowBadge(false)
                },
            )
        }

        /** Must be called while the app is in the foreground (Android 12+ background-start rule). */
        fun start(context: Context, label: String) {
            ensureChannel(context)
            val intent = Intent(context, ModelKeepAliveService::class.java).putExtra(EXTRA_LABEL, label)
            synchronized(lock) {
                startPending = true
                stopRequested = false
            }
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) context.startForegroundService(intent)
                else context.startService(intent)
            } catch (e: RuntimeException) {
                synchronized(lock) { startPending = false }
                throw e  // ModelKeepAliveModule.start maps it to resolve(false) / reject
            }
        }

        fun stop(context: Context) {
            val deferred = synchronized(lock) {
                if (startPending) stopRequested = true
                startPending
            }
            if (deferred) {
                Log.i(TAG, "stop() while start is pending; onStartCommand will stop after promotion")
                return
            }
            context.stopService(Intent(context, ModelKeepAliveService::class.java))
        }
    }
}
