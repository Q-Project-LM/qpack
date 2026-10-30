package ai.qproject.wear

import ai.qproject.qpack.GenStats
import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File

/** One button, one text area. The model file is looked up in the app's files dir (adb push it there, see android/README.md). */
class MainActivity : Activity(), VoiceAssistant.Listener {
    private lateinit var heard: TextView
    private lateinit var reply: TextView
    private lateinit var button: Button
    private var assistant: VoiceAssistant? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        heard = TextView(this).apply { textSize = 12f; gravity = Gravity.CENTER }
        reply = TextView(this).apply { textSize = 14f }
        button = Button(this).apply { text = "Speak"; isEnabled = false; setOnClickListener { assistant?.listen() } }
        setContentView(ScrollView(this).apply {
            addView(LinearLayout(this@MainActivity).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 24, 24, 24); addView(heard); addView(reply); addView(button) })
        })
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1) else start()
    }

    override fun onRequestPermissionsResult(code: Int, p: Array<out String>, r: IntArray) { if (r.firstOrNull() == PackageManager.PERMISSION_GRANTED) start() else reply.text = "Microphone permission is required" }

    private fun start() {
        val model = File(filesDir, "model.qpack")
        if (!model.exists()) { reply.text = "Put a model at ${model.path}"; return }
        reply.text = "Loading…"
        Thread {
            try { val a = VoiceAssistant(this, model.path, this); runOnUiThread { assistant = a; reply.text = ""; button.isEnabled = true } }
            catch (e: RuntimeException) { runOnUiThread { reply.text = "Cannot load model: ${e.message}" } }
        }.start()
    }

    override fun onHeard(text: String) { heard.text = text; reply.text = "" }
    override fun onReply(textSoFar: String) { reply.text = textSoFar }
    override fun onDone(stats: GenStats?) { if (stats != null) heard.text = "%.1f tok/s".format(stats.decodeTps) }
    override fun onError(message: String) { reply.text = message }
    override fun onDestroy() { assistant?.close(); super.onDestroy() }
}
