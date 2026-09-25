package ai.offgridmobile.keepalive

import android.app.ForegroundServiceStartNotAllowedException
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat

/**
 * Holds Off Grid at foreground-service priority while a model is resident in RAM.
 *
 * lmkd picks victims by oom_score_adj: a backgrounded app without a service sits at cached
 * priority (900+) and is among the first killed once RAM tightens, which drops the loaded model
 * (seconds-to-minutes to reload). A foreground service lifts the process to perceptible (~200),
 * so every cached and background app is reclaimed before this one. It does NOT make the app
 * unkillable: when the device is truly out of memory lmkd still kills it, so the model budgets
 * stay the first line of defence. In the foreground (adj 0) this changes nothing.
 */
class ModelKeepAliveService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        try {
            val notification = NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Model loaded")
                .setContentText(intent?.getStringExtra(EXTRA_LABEL) ?: "Keeping the model in memory.")
                .setSmallIcon(android.R.drawable.stat_notify_sync_noanim)
                .setOngoing(true)
                .setSilent(true)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .build()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
        } catch (e: RuntimeException) {
            val denied = e is SecurityException ||
                (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && e is ForegroundServiceStartNotAllowedException)
            if (!denied) throw e
            Log.w(TAG, "keep-alive promotion denied: ${e.message}")
            stopSelf()
        }
        // The model lives in this process; after a kill there is nothing to keep alive.
        return START_NOT_STICKY
    }

    companion object {
        private const val TAG = "ModelKeepAlive"
        const val CHANNEL_ID = "offgrid-model-keepalive"
        const val NOTIFICATION_ID = 4712
        private const val EXTRA_LABEL = "label"

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
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) context.startForegroundService(intent)
            else context.startService(intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, ModelKeepAliveService::class.java))
        }
    }
}
