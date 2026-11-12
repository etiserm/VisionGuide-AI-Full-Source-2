package org.visionguide.app

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Size
import android.util.SizeF
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.AspectRatioStrategy
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.ResolutionSelector
import androidx.camera.core.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors
import kotlin.math.atan

private data class CameraOptics(
    val sensorWidth: Float,
    val sensorHeight: Float,
    val focalLength: Float
)

class CameraRunner(context: Context) {
    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    @Volatile private var active = false
    @Volatile private var generation = 0
    @Volatile private var currentDetections: List<Detection> = emptyList()
    private var provider: ProcessCameraProvider? = null
    private var analysisUseCase: ImageAnalysis? = null
    private var analysisExecutor: ExecutorService? = null
    private var worker: AnalysisWorker? = null
    @Volatile private var previousWorkerClose = CompletableFuture.completedFuture(Unit)

    fun start(
        previewView: PreviewView,
        lifecycleOwner: LifecycleOwner,
        sensitivity: () -> Float,
        uncertaintyAlertsEnabled: () -> Boolean,
        onCountChanged: (Int) -> Unit,
        onStatus: (String) -> Unit,
        onHazard: (String) -> Unit
    ) {
        if (active) return
        active = true
        val sessionGeneration = ++generation
        val closeBarrier = previousWorkerClose
        currentDetections = emptyList()
        onStatus("Starting camera…")
        val future = try {
            ProcessCameraProvider.getInstance(appContext)
        } catch (error: Exception) {
            active = false
            onStatus("Camera could not start: ${error.message ?: "check camera permission"}")
            return
        }
        try {
            future.addListener({
                if (!isCurrent(sessionGeneration)) return@addListener
                try {
                    val cameraProvider = future.get()
                    provider = cameraProvider
                    val selector = ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(640, 480),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                            )
                        )
                        .build()
                    val preview = Preview.Builder()
                        .setResolutionSelector(selector)
                        .build()
                        .also { it.surfaceProvider = previewView.surfaceProvider }
                    val analysis = ImageAnalysis.Builder()
                        .setResolutionSelector(selector)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    cameraProvider.unbindAll()
                    val camera = cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis
                    )
                    val cameraOptics = getCameraOptics(camera.cameraInfo)
                    val executor = Executors.newSingleThreadExecutor { runnable ->
                        Thread(runnable, "VisionGuide-inference").apply { priority = Thread.NORM_PRIORITY - 1 }
                    }
                    val analysisWorker = AnalysisWorker(
                        appContext,
                        closeBarrier,
                        canContinue = { isCurrent(sessionGeneration) }
                    )
                    analysisExecutor = executor
                    analysisUseCase = analysis
                    worker = analysisWorker
                    var lastInferenceTime = 0L
                    var lastReportedCount = -1
                    var lastReportedInferenceError = 0L
                    analysis.setAnalyzer(executor) { imageProxy ->
                        try {
                            if (!isCurrent(sessionGeneration)) return@setAnalyzer
                            val now = SystemClock.elapsedRealtime()
                            if (now - lastInferenceTime < 400L) return@setAnalyzer
                            lastInferenceTime = now
                            val mediaImage = imageProxy.image ?: return@setAnalyzer
                            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                            val crop = imageProxy.cropRect
                            val uncroppedVerticalPixels = if (rotationDegrees == 90 || rotationDegrees == 270) {
                                imageProxy.width
                            } else {
                                imageProxy.height
                            }
                            val croppedVerticalPixels = if (rotationDegrees == 90 || rotationDegrees == 270) {
                                crop.width()
                            } else {
                                crop.height()
                            }
                            val cropHeightRatio = if (uncroppedVerticalPixels > 0) {
                                croppedVerticalPixels.toFloat() / uncroppedVerticalPixels
                            } else {
                                1f
                            }
                            val verticalFieldOfView = getVerticalFieldOfViewDegrees(
                                cameraOptics,
                                rotationDegrees,
                                cropHeightRatio
                            )

                            val result = analysisWorker.analyze(
                                mediaImage,
                                rotationDegrees,
                                verticalFieldOfView,
                                sensitivity(),
                                uncertaintyAlertsEnabled(),
                                onStatus = { status ->
                                    mainHandler.post { if (isCurrent(sessionGeneration)) onStatus(status) }
                                },
                                onHazard = { message ->
                                    mainHandler.post { if (isCurrent(sessionGeneration)) onHazard(message) }
                                }
                            )

                            if (result != null && isCurrent(sessionGeneration)) {
                                currentDetections = result
                                if (result.size != lastReportedCount) {
                                    lastReportedCount = result.size
                                    mainHandler.post {
                                        if (isCurrent(sessionGeneration)) onCountChanged(result.size)
                                    }
                                }
                            } else if (isCurrent(sessionGeneration)) {
                                currentDetections = emptyList()
                                if (lastReportedCount != 0) {
                                    lastReportedCount = 0
                                    mainHandler.post {
                                        if (isCurrent(sessionGeneration)) onCountChanged(0)
                                    }
                                }
                            }
                        } catch (error: Exception) {
                            val now = SystemClock.elapsedRealtime()
                            if (isCurrent(sessionGeneration) && now - lastReportedInferenceError >= 5_000L) {
                                lastReportedInferenceError = now
                                currentDetections = emptyList()
                                if (lastReportedCount != 0) {
                                    lastReportedCount = 0
                                    mainHandler.post {
                                        if (isCurrent(sessionGeneration)) onCountChanged(0)
                                    }
                                }
                                val message = "Recognition issue: ${error.message ?: "retrying"}"
                                mainHandler.post {
                                    if (isCurrent(sessionGeneration)) onStatus(message)
                                }
                            }
                        } finally {
                            imageProxy.close()
                        }
                    }
                    if (isCurrent(sessionGeneration)) onStatus("Camera is on. Loading object recognition…")
                } catch (error: Exception) {
                    stop()
                    onStatus("Camera could not start: ${error.message ?: "check camera permission"}")
                }
            }, ContextCompat.getMainExecutor(appContext))
        } catch (error: Exception) {
            stop()
            onStatus("Camera could not start: ${error.message ?: "check camera permission"}")
        }
    }

    fun stop() {
        active = false
        generation += 1
        currentDetections = emptyList()
        if (provider == null && analysisExecutor == null && worker == null) return
        runCatching { analysisUseCase?.clearAnalyzer() }
        analysisUseCase = null
        runCatching { provider?.unbindAll() }
        provider = null
        val executor = analysisExecutor
        val oldWorker = worker
        analysisExecutor = null
        worker = null
        val closeBarrier = CompletableFuture<Unit>()
        previousWorkerClose = closeBarrier
        val releaseWorker = Runnable {
            runCatching { oldWorker?.close() }
            closeBarrier.complete(Unit)
        }
        if (executor != null) {
            try {
                executor.execute(releaseWorker)
            } catch (_: Exception) {
                releaseWorker.run()
            }
            executor.shutdown()
        } else {
            releaseWorker.run()
        }
    }

    private fun isCurrent(sessionGeneration: Int): Boolean =
        active && generation == sessionGeneration

    fun describeCurrentView(): String = SceneDescription.make(currentDetections)

    private fun getVerticalFieldOfViewDegrees(
        optics: CameraOptics?,
        rotationDegrees: Int,
        cropHeightRatio: Float
    ): Float? {
        if (optics == null || cropHeightRatio <= 0f) return null
        val sensorDimension = if (rotationDegrees == 90 || rotationDegrees == 270) {
            optics.sensorWidth
        } else {
            optics.sensorHeight
        }
        val croppedSensorDimension = sensorDimension * cropHeightRatio.coerceIn(0f, 1f)
        return Math.toDegrees(2.0 * atan(croppedSensorDimension / (2.0 * optics.focalLength))).toFloat()
            .takeIf { it in 20f..100f }
    }

    private fun getCameraOptics(cameraInfo: androidx.camera.core.CameraInfo): CameraOptics? {
        return runCatching {
            val camera2 = Camera2CameraInfo.from(cameraInfo)
            val physicalSize: SizeF = camera2.getCameraCharacteristic(
                CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE
            ) ?: return null
            val focalLength = camera2.getCameraCharacteristic(
                CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS
            )?.firstOrNull() ?: return null
            if (physicalSize.width <= 0f || physicalSize.height <= 0f || focalLength <= 0f) return null
            CameraOptics(physicalSize.width, physicalSize.height, focalLength)
        }.getOrNull()
    }
}

private class AnalysisWorker(
    context: Context,
    private val previousWorkerClose: CompletableFuture<Unit>,
    private val canContinue: () -> Boolean
) : AutoCloseable {
    private val appContext = context.applicationContext
    private var detector: OnnxDetector? = null
    private var loadingFailed = false
    private var consecutiveInferenceErrors = 0
    private var lastInferenceErrorReport = 0L
    private val recentHazards = HashMap<String, Int>()
    private val announcedHazards = HashSet<String>()
    private val hazardLastSeen = HashMap<String, Long>()
    private var lastHazardAnnouncement = 0L

    fun analyze(
        image: android.media.Image,
        rotationDegrees: Int,
        verticalFieldOfViewDegrees: Float?,
        sensitivity: Float,
        uncertaintyAlertsEnabled: Boolean,
        onStatus: (String) -> Unit,
        onHazard: (String) -> Unit
    ): List<Detection>? {
        if (!canContinue()) return null
        if (loadingFailed) return null
        if (detector == null) {
            onStatus("Loading object recognition…")
            try {
                previousWorkerClose.join()
                if (!canContinue()) return null
                detector = OnnxDetector(appContext)
                onStatus("Object recognition is ready.")
            } catch (error: Exception) {
                loadingFailed = true
                onStatus("Model could not load: ${error.message ?: "check the app files"}")
                return null
            }
        }
        val detections = try {
            val recovered = consecutiveInferenceErrors > 0
            detector!!.detect(image, rotationDegrees, verticalFieldOfViewDegrees, sensitivity)
                .also {
                    consecutiveInferenceErrors = 0
                    if (recovered) onStatus("Object recognition resumed.")
                }
        } catch (error: Exception) {
            consecutiveInferenceErrors += 1
            val now = SystemClock.elapsedRealtime()
            if (now - lastInferenceErrorReport >= 5_000L) {
                lastInferenceErrorReport = now
                onStatus("Recognition issue: ${error.message ?: "retrying"}")
            }
            if (consecutiveInferenceErrors >= 3) {
                loadingFailed = true
                runCatching { detector?.close() }
                detector = null
                onStatus("Recognition could not continue: ${error.message ?: "restart the camera"}")
            }
            return null
        }
        announceImportantObjects(detections, sensitivity, uncertaintyAlertsEnabled, onHazard)
        return detections
    }

    private fun announceImportantObjects(
        detections: List<Detection>,
        sensitivity: Float,
        alertsEnabled: Boolean,
        onHazard: (String) -> Unit
    ) {
        if (!alertsEnabled) {
            recentHazards.clear()
            announcedHazards.clear()
            hazardLastSeen.clear()
            return
        }
        val now = SystemClock.elapsedRealtime()
        val expired = hazardLastSeen.filterValues { now - it >= 5_000L }.keys
        expired.forEach { key ->
            hazardLastSeen.remove(key)
            recentHazards.remove(key)
            announcedHazards.remove(key)
        }
        val minimumHazardConfidence = maxOf(sensitivity, 0.45f)
        val urgent = detections.filter {
            it.priority <= 2 && it.confidence >= minimumHazardConfidence
        }
        val seen = HashSet<String>()
        urgent.forEach { detection ->
            val key = "${detection.label}|${detection.zone}"
            seen += key
            hazardLastSeen[key] = now
        }
        seen.forEach { key ->
            recentHazards[key] = (recentHazards[key] ?: 0) + 1
        }
        recentHazards.keys.retainAll(seen)
        if (now - lastHazardAnnouncement < 2_500L) return

        val candidate = urgent.firstOrNull { detection ->
            val key = "${detection.label}|${detection.zone}"
            (recentHazards[key] ?: 0) >= 3 && key !in announcedHazards
        } ?: return
        val objectName = when (candidate.label) {
            "person" -> "A person"
            "stairs" -> "Stairs"
            else -> {
                val first = candidate.label.firstOrNull()?.lowercaseChar()
                val article = if (first != null && first in "aeiou") "An" else "A"
                "$article ${candidate.label}"
            }
        }
        val distance = candidate.steps?.let { ", roughly ${it.first} to ${it.last} steps away" }.orEmpty()
        val verb = if (candidate.label == "stairs") "have appeared" else "has appeared"
        val message = "$objectName $verb ${SceneDescription.location(candidate)}$distance."
        val key = "${candidate.label}|${candidate.zone}"
        announcedHazards += key
        lastHazardAnnouncement = now
        onHazard(message)
    }

    override fun close() {
        detector?.close()
        detector = null
    }
}
