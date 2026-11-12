package org.visionguide.app

import android.graphics.RectF
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.tan

data class Detection(
    val label: String,
    val confidence: Float,
    val box: RectF,
    val steps: IntRange?
) {
    val zone: String
        get() = when {
            box.centerX() < 1f / 3f -> "left"
            box.centerX() > 2f / 3f -> "right"
            else -> "center"
        }

    val priority: Int
        get() = when (label) {
            "car", "truck", "bus", "motorcycle" -> 0
            "stairs", "staircase", "pothole", "curb", "kerb", "construction barrier",
            "barrier", "traffic cone", "bollard", "pole", "manhole cover" -> 1
            "person", "pedestrian", "wheelchair", "bicycle", "scooter" -> 2
            "traffic light", "traffic sign", "crosswalk" -> 3
            "dog", "cat", "bird", "horse" -> 5
            else -> 6
        }
}

object DistanceEstimator {
    // Typical object heights are reference values, not exact measurements.
    private val typicalHeightsMeters = mapOf(
        "person" to 1.70f, "wheelchair" to 0.95f, "bicycle" to 1.05f,
        "scooter" to 0.95f, "motorcycle" to 1.20f, "car" to 1.45f,
        "bus" to 3.00f, "truck" to 2.60f, "train" to 3.60f, "tram" to 3.00f,
        "dog" to 0.55f, "cat" to 0.28f, "bench" to 0.85f, "chair" to 0.85f,
        "table" to 0.75f, "trash can" to 0.75f, "bicycle rack" to 0.95f,
        "fire hydrant" to 0.75f, "parking meter" to 1.40f, "bus shelter" to 2.20f,
        "shopping cart" to 1.00f, "traffic light" to 0.80f, "traffic sign" to 0.55f,
        "traffic cone" to 0.70f, "bollard" to 0.80f, "pole" to 2.00f,
        "construction barrier" to 0.80f, "manhole cover" to 0.70f,
        "fence" to 1.20f, "handrail" to 0.90f, "door" to 2.00f, "window" to 1.00f,
        "couch" to 0.90f, "bed" to 0.65f, "pillow" to 0.25f, "cabinet" to 1.40f,
        "bookshelf" to 1.70f, "lamp" to 0.50f, "television" to 0.60f, "laptop" to 0.22f,
        "cell phone" to 0.16f, "remote control" to 0.18f, "keyboard" to 0.15f,
        "headphones" to 0.18f, "bottle" to 0.28f, "cup" to 0.11f, "glass" to 0.12f,
        "bowl" to 0.12f, "kettle" to 0.28f, "microwave" to 0.35f, "oven" to 0.90f,
        "stove" to 0.90f, "refrigerator" to 1.70f, "sink" to 0.90f,
        "toilet" to 0.75f, "bathtub" to 0.60f, "shower" to 2.00f,
        "washing machine" to 0.85f, "dishwasher" to 0.85f, "book" to 0.23f,
        "clock" to 0.25f, "mirror" to 0.80f, "vase" to 0.35f,
        "potted plant" to 0.60f, "backpack" to 0.50f, "handbag" to 0.30f,
        "suitcase" to 0.60f, "umbrella" to 0.85f, "shoes" to 0.15f,
        "box" to 0.50f, "medicine bottle" to 0.12f, "toothbrush" to 0.19f,
        "towel" to 0.30f, "fire extinguisher" to 0.55f, "baby stroller" to 1.05f
    )

    fun estimateSteps(
        label: String,
        box: RectF,
        frameHeightPixels: Int,
        verticalFieldOfViewDegrees: Float?,
        confidence: Float
    ): IntRange? {
        val objectHeight = typicalHeightsMeters[label] ?: return null
        val fov = verticalFieldOfViewDegrees ?: return null
        if (confidence < 0.45f || fov !in 35f..100f || frameHeightPixels <= 0) return null
        if (box.height() < 24f / frameHeightPixels || box.top <= 0.01f || box.bottom >= 0.99f) {
            return null
        }

        val focalLengthPixels = frameHeightPixels / (2f * tan(Math.toRadians((fov / 2f).toDouble()).toFloat()))
        val distanceMeters = objectHeight * focalLengthPixels / (box.height() * frameHeightPixels)
        if (distanceMeters !in 0.4f..15f) return null

        val typicalSteps = distanceMeters / 0.68f
        val low = maxOf(1, floor(typicalSteps * 0.65f).toInt())
        val high = maxOf(low + 1, ceil(typicalSteps * 1.35f).toInt())
        return low..high
    }
}

object SceneDescription {
    fun make(detections: List<Detection>, maxObjects: Int = 4): String {
        if (detections.isEmpty()) return "I can't clearly identify any objects in the current view."

        val grouped = detections
            .groupBy { "${it.label}|${it.zone}" }
            .values
            .map { it.first() to it.size }
            .sortedWith(compareBy<Pair<Detection, Int>> { it.first.priority }
                .thenByDescending { it.first.confidence })
            .take(maxObjects)

        val phrases = grouped.map { (detection, count) ->
            val name = objectName(detection.label, count)
            val location = when (detection.zone) {
                "left" -> "on the left"
                "right" -> "on the right"
                else -> "near the center"
            }
            val distance = detection.steps?.let {
                ", roughly ${it.first} to ${it.last} steps away"
            }.orEmpty()
            "$name $location$distance"
        }

        val list = when (phrases.size) {
            1 -> phrases[0]
            2 -> "${phrases[0]} and ${phrases[1]}"
            else -> phrases.dropLast(1).joinToString(", ") + ", and ${phrases.last()}"
        }
        return "I can see $list."
    }

    fun location(detection: Detection): String = when (detection.zone) {
        "left" -> "on the left"
        "right" -> "on the right"
        else -> "near the center"
    }

    private fun objectName(label: String, count: Int): String {
        if (count == 1) {
            when (label) {
                "person" -> return "a person"
                "stairs", "shoes", "headphones", "clothing" -> return label
                "keys" -> return "a set of keys"
            }
            val first = label.firstOrNull()?.lowercaseChar()
            val article = if (first != null && first in "aeiou") "an" else "a"
            return "$article $label"
        }
        return when (label) {
            "person" -> "$count people"
            "stairs" -> "$count staircases"
            "shoes" -> "$count pairs of shoes"
            "headphones" -> "$count pairs of headphones"
            "keys" -> "$count sets of keys"
            "clothing" -> "clothing"
            "knife" -> "$count knives"
            "computer mouse" -> "$count computer mice"
            else -> "$count ${pluralize(label)}"
        }
    }

    private fun pluralize(label: String): String {
        val lastWordStart = label.lastIndexOf(' ') + 1
        val lastWord = label.substring(lastWordStart)
        val plural = when {
            lastWord.endsWith("ch") || lastWord.endsWith("sh") ||
                lastWord.endsWith("s") || lastWord.endsWith("x") || lastWord.endsWith("z") -> "${lastWord}es"
            lastWord.endsWith("y") && lastWord.length > 1 && lastWord[lastWord.lastIndex - 1] !in "aeiou" ->
                "${lastWord.dropLast(1)}ies"
            else -> "${lastWord}s"
        }
        return label.substring(0, lastWordStart) + plural
    }
}
