package org.visionguide.app

import android.content.Context
import android.media.AudioAttributes
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.util.Locale

enum class Speaker(val key: String, val title: String) {
    FEMALE("female", "Female"),
    MALE("male", "Male")
}

class VoiceController(context: Context) : TextToSpeech.OnInitListener {
    private val preferences = context.getSharedPreferences("visionguide", Context.MODE_PRIVATE)
    private val engine = TextToSpeech(context.applicationContext, this)
    private var initialized = false
    private var initializationFailed = false
    private var pendingSpeech: Pair<String, Boolean>? = null
    private var closed = false

    var status by mutableStateOf("Preparing voice…")
        private set
    var speaker by mutableStateOf(
        if (preferences.getString("speaker", Speaker.FEMALE.key) == Speaker.MALE.key) {
            Speaker.MALE
        } else Speaker.FEMALE
    )
        private set
    var volume by mutableFloatStateOf(preferences.getFloat("speechVolume", 0.8f).coerceIn(0.1f, 1f))
        private set
    var voiceName by mutableStateOf("System English voice")
        private set
    var isMuted by mutableStateOf(false)
        private set

    override fun onInit(result: Int) {
        if (closed) return
        if (result != TextToSpeech.SUCCESS) {
            initializationFailed = true
            pendingSpeech = null
            status = "Speech engine could not start."
            return
        }
        try {
            engine.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            engine.setLanguage(Locale.US)
            initialized = true
            chooseVoice(speaker)
            if (initializationFailed) return
        } catch (error: Exception) {
            initialized = false
            initializationFailed = true
            pendingSpeech = null
            status = "Speech setup failed: ${error.message ?: "check the text-to-speech engine"}"
            return
        }
        if (pendingSpeech == null) status = "Voice is ready."
        pendingSpeech?.let { (text, interrupt) ->
            pendingSpeech = null
            speak(text, interrupt)
        }
    }

    fun speak(text: String, interrupt: Boolean = false) {
        if (closed || isMuted) return
        if (initializationFailed) {
            status = "Speech is unavailable. Check Android text-to-speech settings."
            return
        }
        if (!initialized) {
            pendingSpeech = text to interrupt
            status = "Voice is loading…"
            return
        }
        val queueMode = if (interrupt) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        val parameters = android.os.Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, volume)
            putFloat(TextToSpeech.Engine.KEY_PARAM_PAN, 0f)
        }
        val result = engine.speak(text, queueMode, parameters, "visionguide-${System.nanoTime()}")
        status = if (result == TextToSpeech.SUCCESS) {
            "Speaking with $voiceName."
        } else {
            "Speech failed. Check the installed text-to-speech engine and voice data."
        }
    }

    fun repeatInstructions() {
        speak(
            "Welcome to VisionGuide. Volume up is in the upper left corner. " +
                "Volume down is in the upper right. Female voice is in the lower left. " +
                "Male voice is in the lower right. Tap the center to start the camera. " +
                "Tap the center again to hear a description.",
            interrupt = true
        )
    }

    fun adjustVolume(amount: Float) {
        volume = (volume + amount).coerceIn(0.1f, 1f)
        preferences.edit().putFloat("speechVolume", volume).apply()
        speak("Volume ${(volume * 100).toInt()} percent.", interrupt = true)
    }

    fun selectSpeaker(selection: Speaker) {
        speaker = selection
        preferences.edit().putString("speaker", selection.key).apply()
        chooseVoice(selection)
        speak("Hello. This is your ${selection.title.lowercase()} voice.", interrupt = true)
    }

    fun toggleMute() {
        isMuted = !isMuted
        if (isMuted) {
            pendingSpeech = null
            engine.stop()
            status = "Voice is muted."
        } else {
            speak("Voice is on.", interrupt = true)
        }
    }

    private fun chooseVoice(selection: Speaker) {
        if (!initialized) return
        try {
            val englishVoices = engine.voices.orEmpty()
                .filter { it.locale.language == "en" }
                .sortedWith(compareBy<Voice> { it.isNetworkConnectionRequired }.thenByDescending { it.quality })

            val genderTokens = if (selection == Speaker.FEMALE) {
                listOf("female", "woman", "samantha", "ava", "zira", "susan")
            } else {
                listOf("male", "man", "daniel", "alex", "david", "mark")
            }
            val selected = englishVoices.firstOrNull { voice ->
                voiceMatchesHint(voice, genderTokens)
            } ?: englishVoices.firstOrNull { it.locale.country == "US" }
                ?: englishVoices.firstOrNull()

            if (selected != null) {
                engine.voice = selected
                voiceName = selected.name
                status = if (voiceMatchesHint(selected, genderTokens)) {
                    "${selection.title} English voice selected."
                } else {
                    "Using the installed English voice; this phone may not label voice gender."
                }
            } else {
                voiceName = "System default"
                status = "No English voice is installed. Download one in Android text-to-speech settings."
            }
        } catch (error: Exception) {
            initializationFailed = true
            initialized = false
            pendingSpeech = null
            status = "Could not select a voice: ${error.message ?: "check text-to-speech settings"}"
        }
    }

    private fun voiceMatchesHint(voice: Voice, hints: List<String>): Boolean {
        val tokens = Regex("[a-z0-9]+")
            .findAll("${voice.name} ${voice.locale}".lowercase(Locale.ROOT))
            .map { it.value }
            .toSet()
        return hints.any(tokens::contains)
    }

    fun close() {
        if (closed) return
        closed = true
        pendingSpeech = null
        engine.stop()
        engine.shutdown()
    }

    fun stop() {
        pendingSpeech = null
        engine.stop()
        status = "Voice is ready."
    }
}
