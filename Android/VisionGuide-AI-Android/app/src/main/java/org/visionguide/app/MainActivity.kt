package org.visionguide.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.camera.view.PreviewView
import kotlinx.coroutines.delay
import java.util.concurrent.atomic.AtomicReference

private val Beige = Color(0xFFF5F0E6)
private val Sand = Color(0xFFE5DED1)
private val Charcoal = Color(0xFF383A3D)
private val Ink = Color(0xFF2E3033)
private data class CameraSettings(val sensitivity: Float, val automaticAlerts: Boolean)

class MainActivity : ComponentActivity() {
    private lateinit var voice: VoiceController

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        voice = VoiceController(this)
        setContent { VisionGuideApp(voice) }
    }

    override fun onDestroy() {
        voice.close()
        super.onDestroy()
    }
}

@Composable
private fun VisionGuideApp(voice: VoiceController) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraRunner = remember(context) { CameraRunner(context) }
    val preferences = remember(context) { context.getSharedPreferences("visionguide", 0) }

    var cameraActive by remember { mutableStateOf(false) }
    var cameraPaused by remember { mutableStateOf(false) }
    var cameraStatus by remember { mutableStateOf("Camera is off.") }
    var cameraPermissionDenied by remember { mutableStateOf(false) }
    var detectedCount by remember { mutableIntStateOf(0) }
    var sensitivity by remember { mutableFloatStateOf(preferences.getFloat("sensitivity", 0.25f).coerceIn(0.2f, 0.5f)) }
    var automaticAlerts by remember { mutableStateOf(preferences.getBoolean("automaticAlerts", true)) }
    var settingsExpanded by remember { mutableStateOf(false) }

    val cameraSettings = remember {
        AtomicReference(CameraSettings(sensitivity, automaticAlerts))
    }
    SideEffect {
        cameraSettings.set(CameraSettings(sensitivity, automaticAlerts))
    }

    val cameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            cameraPermissionDenied = false
            cameraActive = true
            cameraPaused = false
            cameraStatus = "Starting camera…"
        } else {
            cameraPermissionDenied = true
            cameraStatus = "Camera permission was not allowed. Enable it in Android Settings."
            voice.speak(cameraStatus, interrupt = true)
        }
    }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP) {
                cameraActive = false
                cameraPaused = false
                cameraRunner.stop()
                voice.stop()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }
    DisposableEffect(cameraRunner) {
        onDispose { cameraRunner.stop() }
    }
    LaunchedEffect(Unit) {
        delay(700)
        voice.repeatInstructions()
    }

    val startCamera = {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            cameraPermissionDenied = false
            cameraActive = true
            cameraPaused = false
            cameraStatus = "Starting camera…"
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    MaterialTheme(colorScheme = lightColorScheme(
        primary = Charcoal,
        onPrimary = Color.White,
        background = Beige,
        onBackground = Ink,
        surface = Sand,
        onSurface = Ink
    )) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Beige)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 14.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Header(cameraStatus, cameraActive)

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(18.dp))
                    .background(Sand)
                    .padding(8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                ActionButton("Volume +", Modifier.weight(1f)) { voice.adjustVolume(0.1f) }
                Text(
                    "${(voice.volume * 100).toInt()}%",
                    modifier = Modifier.width(48.dp),
                    textAlign = TextAlign.Center,
                    fontWeight = FontWeight.Bold
                )
                ActionButton("Volume −", Modifier.weight(1f)) { voice.adjustVolume(-0.1f) }
            }

            CameraPanel(
                active = cameraActive && !cameraPaused,
                paused = cameraPaused,
                status = cameraStatus,
                onTap = {
                    when {
                        !cameraActive -> startCamera()
                        cameraPaused -> {
                            cameraPaused = false
                            cameraStatus = "Resuming camera…"
                        }
                        else -> voice.speak(cameraRunner.describeCurrentView(), interrupt = true)
                    }
                },
                runner = cameraRunner,
                sensitivity = { cameraSettings.get().sensitivity },
                automaticAlerts = { cameraSettings.get().automaticAlerts },
                onCountChanged = { detectedCount = it },
                onStatus = { message ->
                    val isFatal = message.startsWith("Camera could not") ||
                        message.startsWith("Model could not") ||
                        message.startsWith("Recognition could not")
                    if (isFatal) {
                        cameraActive = false
                        cameraPaused = false
                        detectedCount = 0
                    }
                    cameraStatus = message
                    if (isFatal) {
                        voice.speak(message, interrupt = true)
                    } else if (message == "Object recognition is ready.") {
                        voice.speak("Object recognition is ready.")
                    }
                },
                onHazard = { voice.speak(it, interrupt = true) }
            )

            Text(
                when {
                    !cameraActive -> "Tap the center when you are ready."
                    cameraPaused -> "Camera paused. Resume to check the view."
                    else -> "Objects currently detected: $detectedCount"
                },
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center,
                color = Ink.copy(alpha = 0.72f),
                fontSize = 14.sp
            )

            Text(
                voice.status,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center,
                color = Ink.copy(alpha = 0.72f),
                fontSize = 13.sp
            )

            if (cameraPermissionDenied) {
                ActionButton("Open camera settings", Modifier.fillMaxWidth()) {
                    val settingsIntent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                        data = Uri.fromParts("package", context.packageName, null)
                    }
                    context.startActivity(settingsIntent)
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                ActionButton("Repeat guide", Modifier.weight(1f)) { voice.repeatInstructions() }
                ActionButton(
                    if (voice.isMuted) "Enable voice" else "Mute voice",
                    Modifier.weight(1f)
                ) { voice.toggleMute() }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(18.dp))
                    .background(Sand)
                    .padding(8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                VoiceButton(Speaker.FEMALE, selected = voice.speaker == Speaker.FEMALE, modifier = Modifier.weight(1f)) {
                    voice.selectSpeaker(Speaker.FEMALE)
                }
                Column(modifier = Modifier.width(102.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(voice.speaker.title.uppercase(), fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    Text(voice.voiceName, fontSize = 10.sp, maxLines = 1)
                }
                VoiceButton(Speaker.MALE, selected = voice.speaker == Speaker.MALE, modifier = Modifier.weight(1f)) {
                    voice.selectSpeaker(Speaker.MALE)
                }
            }

            Surface(
                color = Sand,
                shape = RoundedCornerShape(18.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        if (settingsExpanded) "Hide settings" else "More settings",
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable(role = Role.Button) { settingsExpanded = !settingsExpanded }
                            .semantics {
                                contentDescription = if (settingsExpanded) "Hide settings" else "More settings"
                            },
                        fontWeight = FontWeight.SemiBold
                    )
                    if (settingsExpanded) {
                        HorizontalDivider(color = Charcoal.copy(alpha = 0.16f))
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Sensitivity", modifier = Modifier.weight(1f))
                            Text("${"%.2f".format(sensitivity)}", fontWeight = FontWeight.Bold)
                        }
                        Slider(
                            value = sensitivity,
                            onValueChange = {
                                sensitivity = it
                                preferences.edit().putFloat("sensitivity", it).apply()
                            },
                            valueRange = 0.2f..0.5f,
                            steps = 5
                        )
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Automatic important-object alerts", modifier = Modifier.weight(1f))
                            Switch(
                                checked = automaticAlerts,
                                onCheckedChange = {
                                    automaticAlerts = it
                                    preferences.edit().putBoolean("automaticAlerts", it).apply()
                                }
                            )
                        }
                        Text(
                            "Distance is a rough estimate from typical object sizes. It may be inaccurate; use it only as an additional cue.",
                            fontSize = 12.sp,
                            color = Ink.copy(alpha = 0.70f)
                        )
                        if (cameraActive) {
                            ActionButton(if (cameraPaused) "Resume camera" else "Pause camera", Modifier.fillMaxWidth()) {
                                cameraPaused = !cameraPaused
                                cameraStatus = if (cameraPaused) "Camera is paused." else "Resuming camera…"
                            }
                            ActionButton("Stop camera", Modifier.fillMaxWidth()) {
                                cameraActive = false
                                cameraPaused = false
                                detectedCount = 0
                                cameraStatus = "Camera is off."
                                cameraRunner.stop()
                            }
                        }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
private fun Header(status: String, cameraActive: Boolean) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.weight(1f)) {
            Text("VisionGuide AI", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = Ink)
            Text(status, fontSize = 12.sp, maxLines = 1, color = Ink.copy(alpha = 0.7f))
        }
        Box(
            Modifier
                .width(14.dp)
                .height(14.dp)
                .clip(RoundedCornerShape(50))
                .background(if (cameraActive) Color(0xFF41955C) else Color(0xFFFF8A00))
        )
    }
}

@Composable
private fun CameraPanel(
    active: Boolean,
    paused: Boolean,
    status: String,
    onTap: () -> Unit,
    runner: CameraRunner,
    sensitivity: () -> Float,
    automaticAlerts: () -> Boolean,
    onCountChanged: (Int) -> Unit,
    onStatus: (String) -> Unit,
    onHazard: (String) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember(context) {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }
    val currentSensitivity by rememberUpdatedState(sensitivity)
    val currentAlerts by rememberUpdatedState(automaticAlerts)
    val currentCountChanged by rememberUpdatedState(onCountChanged)
    val currentStatus by rememberUpdatedState(onStatus)
    val currentHazard by rememberUpdatedState(onHazard)

    DisposableEffect(active, lifecycleOwner) {
        if (active) {
            runner.start(
                previewView = previewView,
                lifecycleOwner = lifecycleOwner,
                sensitivity = currentSensitivity,
                uncertaintyAlertsEnabled = currentAlerts,
                onCountChanged = { currentCountChanged(it) },
                onStatus = { currentStatus(it) },
                onHazard = { currentHazard(it) }
            )
        } else {
            runner.stop()
        }
        onDispose { runner.stop() }
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 300.dp, max = 430.dp)
            .height(370.dp)
            .clip(RoundedCornerShape(24.dp))
            .background(Charcoal)
    ) {
        if (active) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Brush.verticalGradient(listOf(Color.Transparent, Color.Black.copy(alpha = 0.70f))))
                    .clickable(role = Role.Button, onClick = onTap)
            ) {
                Text(
                    "Tap the center to describe this view",
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp, vertical = 20.dp),
                    textAlign = TextAlign.Center,
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }
        } else {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clickable(role = Role.Button, onClick = onTap)
                    .padding(26.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text(if (paused) "Camera paused" else "Camera", fontSize = 38.sp, color = Color.White)
                    Text(
                        when {
                            paused -> "Tap the center to resume camera analysis"
                            status.contains("permission", ignoreCase = true) -> "Tap to allow camera access"
                            else -> "Tap the center to start the camera"
                        },
                        textAlign = TextAlign.Center,
                        color = Color.White.copy(alpha = 0.82f),
                        fontSize = 16.sp
                    )
                }
            }
        }
    }
}

@Composable
private fun ActionButton(title: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        modifier = modifier.heightIn(min = 58.dp),
        shape = RoundedCornerShape(16.dp),
        colors = ButtonDefaults.buttonColors(containerColor = Charcoal, contentColor = Color.White)
    ) {
        Text(title, fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center)
    }
}

@Composable
private fun VoiceButton(speaker: Speaker, selected: Boolean, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        modifier = modifier.heightIn(min = 58.dp),
        shape = RoundedCornerShape(15.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) Color(0xFF55585B) else Charcoal,
            contentColor = Color.White
        )
    ) {
        Text(speaker.title, fontWeight = FontWeight.SemiBold)
    }
}
