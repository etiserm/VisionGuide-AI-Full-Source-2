package org.visionguide.app

import android.content.Context
import android.graphics.RectF
import android.media.Image
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class OnnxDetector(context: Context) : AutoCloseable {
    private val inputSize = 640
    private val env = OrtEnvironment.getEnvironment()
    private val sessionOptions = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(2)
        setInterOpNumThreads(1)
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
    private val session: OrtSession
    private val inputName: String
    private val outputName: String
    private val labels: List<String>
    private val inputValues = FloatArray(3 * inputSize * inputSize)
    private val inputBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(inputValues.size * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    init {
        labels = context.assets.open("classes.json").bufferedReader().use { reader ->
            val json = JSONArray(reader.readText())
            List(json.length()) { index -> json.getString(index) }
        }
        val modelFile = File(context.cacheDir, "visionguide.onnx")
        val expectedLength = context.assets.openFd("visionguide.onnx").use { it.length }
        if (!modelFile.exists() || modelFile.length() != expectedLength) {
            context.assets.open("visionguide.onnx").use { source ->
                modelFile.outputStream().buffered().use { destination ->
                    source.copyTo(destination, bufferSize = 1024 * 1024)
                }
            }
        }
        val loadedSession = try {
            env.createSession(modelFile.absolutePath, sessionOptions)
        } catch (error: Throwable) {
            sessionOptions.close()
            throw error
        }
        try {
            val loadedInputName = loadedSession.inputNames.firstOrNull()
                ?: throw IllegalStateException("The object model has no input tensor.")
            val loadedOutputName = loadedSession.outputNames.firstOrNull()
                ?: throw IllegalStateException("The object model has no output tensor.")
            session = loadedSession
            inputName = loadedInputName
            outputName = loadedOutputName
        } catch (error: Throwable) {
            runCatching { loadedSession.close() }
            sessionOptions.close()
            throw error
        }
    }

    fun detect(
        image: Image,
        rotationDegrees: Int,
        verticalFieldOfViewDegrees: Float?,
        sensitivity: Float
    ): List<Detection> {
        val rotation = ((rotationDegrees % 360) + 360) % 360
        val crop = image.cropRect
        val sourceWidth = crop.width()
        val sourceHeight = crop.height()
        if (sourceWidth <= 0 || sourceHeight <= 0) return emptyList()
        val orientedWidth = if (rotation == 90 || rotation == 270) sourceHeight else sourceWidth
        val orientedHeight = if (rotation == 90 || rotation == 270) sourceWidth else sourceHeight
        val scale = min(inputSize.toFloat() / orientedWidth, inputSize.toFloat() / orientedHeight)
        val contentWidth = orientedWidth * scale
        val contentHeight = orientedHeight * scale
        val padX = (inputSize - contentWidth) / 2f
        val padY = (inputSize - contentHeight) / 2f

        preprocessYuv(image, crop, rotation, orientedWidth, orientedHeight, scale, padX, padY)
        inputBuffer.clear()
        inputBuffer.put(inputValues)
        inputBuffer.flip()

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val inputTensor = OnnxTensor.createTensor(env, inputBuffer, shape)
        try {
            session.run(mapOf(inputName to inputTensor)).use { result ->
                val output = result.get(outputName).get() as OnnxTensor
                val values = output.floatBuffer
                val firstValue = values.position()
                val numberOfRows = values.remaining() / 6
                val detections = ArrayList<Detection>(min(numberOfRows, 24))
                for (index in 0 until numberOfRows) {
                    val offset = firstValue + index * 6
                    val x1 = values.get(offset)
                    val y1 = values.get(offset + 1)
                    val x2 = values.get(offset + 2)
                    val y2 = values.get(offset + 3)
                    val confidence = values.get(offset + 4)
                    val classValue = values.get(offset + 5)
                    if (!confidence.isFinite() || confidence !in 0f..1f || !classValue.isFinite()) continue
                    val classIndex = classValue.toInt()
                    if (classValue != classIndex.toFloat() || confidence < sensitivity || classIndex !in labels.indices) continue
                    if (!x1.isFinite() || !y1.isFinite() || !x2.isFinite() || !y2.isFinite()) continue

                    val left = ((x1 - padX) / scale / orientedWidth).coerceIn(0f, 1f)
                    val top = ((y1 - padY) / scale / orientedHeight).coerceIn(0f, 1f)
                    val right = ((x2 - padX) / scale / orientedWidth).coerceIn(0f, 1f)
                    val bottom = ((y2 - padY) / scale / orientedHeight).coerceIn(0f, 1f)
                    if (right <= left || bottom <= top) continue

                    val label = labels[classIndex]
                    val box = RectF(left, top, right, bottom)
                    val steps = DistanceEstimator.estimateSteps(
                        label = label,
                        box = box,
                        frameHeightPixels = orientedHeight,
                        verticalFieldOfViewDegrees = verticalFieldOfViewDegrees,
                        confidence = confidence
                    )
                    detections += Detection(label, confidence, box, steps)
                }
                return detections.sortedWith(compareBy<Detection> { it.priority }
                    .thenByDescending { it.confidence })
            }
        } finally {
            inputTensor.close()
        }
    }

    private fun preprocessYuv(
        image: Image,
        crop: android.graphics.Rect,
        rotation: Int,
        orientedWidth: Int,
        orientedHeight: Int,
        scale: Float,
        padX: Float,
        padY: Float
    ) {
        val fill = 114f / 255f
        inputValues.fill(fill)
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuffer = yPlane.buffer.duplicate()
        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()
        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride
        val contentWidth = orientedWidth * scale
        val contentHeight = orientedHeight * scale
        val offsetX = (inputSize - contentWidth) / 2f
        val offsetY = (inputSize - contentHeight) / 2f
        val sourceXForOutput = IntArray(inputSize) { dx ->
            val coordinate = (dx - offsetX) / scale
            if (coordinate >= 0f && coordinate < orientedWidth) coordinate.toInt() else -1
        }
        val sourceYForOutput = IntArray(inputSize) { dy ->
            val coordinate = (dy - offsetY) / scale
            if (coordinate >= 0f && coordinate < orientedHeight) coordinate.toInt() else -1
        }

        for (dy in 0 until inputSize) {
            val oy = sourceYForOutput[dy]
            if (oy < 0) continue
            for (dx in 0 until inputSize) {
                val ox = sourceXForOutput[dx]
                if (ox < 0) continue
                val sourceX: Int
                val sourceY: Int
                when (rotation) {
                    90 -> {
                        sourceX = crop.left + oy
                        sourceY = crop.top + crop.height() - 1 - ox
                    }
                    180 -> {
                        sourceX = crop.left + crop.width() - 1 - ox
                        sourceY = crop.top + crop.height() - 1 - oy
                    }
                    270 -> {
                        sourceX = crop.left + crop.width() - 1 - oy
                        sourceY = crop.top + ox
                    }
                    else -> {
                        sourceX = crop.left + ox
                        sourceY = crop.top + oy
                    }
                }
                val yy = yBuffer.get(sourceY * yRowStride + sourceX * yPixelStride).toInt() and 0xff
                val uIndex = (sourceY / 2) * uRowStride + (sourceX / 2) * uPixelStride
                val vIndex = (sourceY / 2) * vRowStride + (sourceX / 2) * vPixelStride
                val y = max(0, yy - 16)
                val u = (uBuffer.get(uIndex).toInt() and 0xff) - 128
                val v = (vBuffer.get(vIndex).toInt() and 0xff) - 128
                val red = ((298 * y + 409 * v + 128) shr 8).coerceIn(0, 255) / 255f
                val green = ((298 * y - 100 * u - 208 * v + 128) shr 8).coerceIn(0, 255) / 255f
                val blue = ((298 * y + 516 * u + 128) shr 8).coerceIn(0, 255) / 255f
                val pixel = dy * inputSize + dx
                inputValues[pixel] = red
                inputValues[inputSize * inputSize + pixel] = green
                inputValues[2 * inputSize * inputSize + pixel] = blue
            }
        }
    }

    override fun close() {
        try {
            session.close()
        } finally {
            sessionOptions.close()
        }
    }
}
