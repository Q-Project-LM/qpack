package ai.qproject.chat

import ai.qproject.qpack.ChatMessage
import ai.qproject.qpack.QPack
import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.view.Gravity
import android.view.Menu
import android.view.MenuItem
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors

/**
 * A local chat with any .qpack model. Models are read in place (mmap): put them in the app's models folder
 * (Android/data/ai.qproject.chat/files/models/, adb push works) or pick a file anywhere with the system file picker.
 * The model runs on a single background thread; the reply streams into the last bubble. Voice in (SpeechRecognizer,
 * offline preferred) and voice out (TextToSpeech) are optional and use the device's own engines.
 */
class ChatActivity : Activity() {
    private lateinit var list: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var input: EditText
    private lateinit var send: Button
    private lateinit var status: TextView
    private val worker = Executors.newSingleThreadExecutor()
    private val history = ArrayList<ChatMessage>()
    private var model: QPack? = null
    private var pfd: ParcelFileDescriptor? = null   // keeps a picked file open: the model is mmap'd through /proc/self/fd
    @Volatile private var generating = false
    private var think = false
    private var speak = false
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var recognizer: SpeechRecognizer? = null
    private val prefs by lazy { getSharedPreferences("qchat", MODE_PRIVATE) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(8), dp(8), dp(8), dp(8)) }
        scroll = ScrollView(this).apply { addView(list) }
        status = TextView(this).apply { textSize = 12f; setTextColor(Color.GRAY); setPadding(dp(12), dp(2), dp(12), dp(2)); text = "No model loaded" }
        input = EditText(this).apply { hint = "Message"; maxLines = 4; imeOptions = EditorInfo.IME_ACTION_SEND; isEnabled = false }
        send = Button(this).apply { text = "Send"; isEnabled = false; setOnClickListener { if (generating) model?.cancel() else submit(input.text.toString()) } }
        val mic = Button(this).apply { text = "Mic"; setOnClickListener { listen() } }
        val bar = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL; addView(input, LinearLayout.LayoutParams(0, -2, 1f)); addView(mic); addView(send) }
        setContentView(LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f)); addView(status); addView(bar) })
        tts = TextToSpeech(this) { ttsReady = it == TextToSpeech.SUCCESS; if (ttsReady) tts?.language = Locale.getDefault() }
        prefs.getString("model", null)?.let { openModel(it) }
    }

    // ---- menu: model, reasoning, voice replies, new chat
    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menu.add(0, 1, 0, "Model…"); menu.add(0, 2, 0, "Reasoning (think)").isCheckable = true
        menu.add(0, 3, 0, "Speak replies").isCheckable = true; menu.add(0, 4, 0, "New chat"); return true
    }
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            1 -> chooseModel()
            2 -> { think = !think; item.isChecked = think }
            3 -> { speak = !speak; item.isChecked = speak; if (!speak) tts?.stop() }
            4 -> { model?.cancel(); history.clear(); list.removeAllViews(); tts?.stop() }
        }
        return true
    }

    private fun modelsDir(): File = (getExternalFilesDir("models") ?: File(filesDir, "models")).also { it.mkdirs() }

    private fun chooseModel() {
        val files = modelsDir().listFiles { f -> f.name.endsWith(".qpack") }?.sortedBy { it.name } ?: emptyList()
        val names = files.map { "${it.name}  (${it.length() shr 20} MiB)" } + "Browse for a .qpack file…"
        AlertDialog.Builder(this).setTitle("Model (folder: ${modelsDir().path})").setItems(names.toTypedArray()) { _, i ->
            if (i < files.size) openModel(files[i].path) else startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"), PICK)
        }.show()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri: Uri = data?.data ?: return
        if (requestCode != PICK || resultCode != RESULT_OK) return
        try { pfd?.close(); pfd = contentResolver.openFileDescriptor(uri, "r"); openModel("/proc/self/fd/${pfd!!.fd}", remember = false) }
        catch (e: Exception) { status.text = "Cannot open the file: ${e.message}" }
    }

    /** Threads = the fast cores only: on big.LITTLE phones the efficiency cores make every parallel step wait for the slowest one. */
    private fun fastCores(): Int {
        val f = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { File("/sys/devices/system/cpu/cpu$it/cpufreq/cpuinfo_max_freq").takeIf { p -> p.canRead() }?.readText()?.trim()?.toLongOrNull() }
        if (f.isEmpty()) return 0
        val top = f.max(); return f.count { it >= top * 7 / 10 }.coerceAtLeast(1)
    }

    private fun openModel(path: String, remember: Boolean = true) {
        status.text = "Loading…"; input.isEnabled = false; send.isEnabled = false; model?.cancel()
        worker.execute {
            try {
                model?.close(); model = null
                val m = QPack(path, contextTokens = 2048, threads = fastCores()); model = m
                if (remember) prefs.edit().putString("model", path).apply()
                runOnUiThread { status.text = "Model: ${m.info}"; input.isEnabled = true; send.isEnabled = true }
            } catch (e: RuntimeException) { runOnUiThread { status.text = "Cannot load model: ${e.message}" } }
        }
    }

    // ---- chat
    private fun bubble(text: String, mine: Boolean): TextView = TextView(this).apply {
        this.text = text; textSize = 15f; setTextColor(if (mine) Color.WHITE else Color.BLACK); setPadding(dp(12), dp(8), dp(12), dp(8)); setTextIsSelectable(true)
        background = GradientDrawable().apply { cornerRadius = dp(14).toFloat(); setColor(if (mine) Color.parseColor("#2f6fed") else Color.parseColor("#e9ebef")) }
        layoutParams = LinearLayout.LayoutParams(-2, -2).apply { gravity = if (mine) Gravity.END else Gravity.START; setMargins(dp(4), dp(4), dp(4), dp(4)) }
        list.addView(this); scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
    }

    private fun submit(text: String) {
        val t = text.trim(); val m = model ?: return
        if (t.isEmpty() || generating) return
        input.setText(""); bubble(t, true); history += ChatMessage("user", t); val out = bubble("…", false)
        generating = true; send.text = "Stop"; tts?.stop()
        val turn = history.toList(); val th = think
        worker.execute {
            val reply = StringBuilder(); val pending = ByteArrayOutputStream(); var spoken = 0
            fun say(final: Boolean) {   // speak finished sentences while the model is still generating
                if (!speak || !ttsReady) return
                val s = reply.toString(); val end = if (final) s.length else s.lastIndexOfAny(charArrayOf('.', '!', '?', '\n')) + 1
                if (end > spoken) { val sentence = s.substring(spoken, end).trim(); spoken = end; if (sentence.isNotEmpty()) tts?.speak(sentence, TextToSpeech.QUEUE_ADD, null, "s$end") }
            }
            val stats = try {
                m.chat(turn, maxNewTokens = 512, temperature = 0.7f, think = th, sink = { b ->
                    pending.write(b); reply.append(pending.toString(Charsets.UTF_8)); pending.reset()
                    val s = reply.toString(); runOnUiThread { out.text = s; scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) } }; say(false); true
                })
            } catch (e: RuntimeException) { runOnUiThread { out.text = "Error: ${e.message}" }; null }
            say(true)
            runOnUiThread {
                if (reply.isNotEmpty()) history += ChatMessage("assistant", reply.toString()) else if (history.lastOrNull()?.role == "user") history.removeAt(history.size - 1)
                generating = false; send.text = "Send"
                if (stats != null) status.text = "%d tokens, %.1f tok/s (prompt %d, %d cached) — %s".format(stats.generatedTokens, stats.decodeTps, stats.promptTokens, stats.reusedTokens, stats.finish)
            }
        }
    }

    // ---- voice input
    private fun listen() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) { requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1); return }
        if (!SpeechRecognizer.isRecognitionAvailable(this)) { status.text = "No speech recogniser on this device"; return }
        recognizer?.destroy()
        val r = if (android.os.Build.VERSION.SDK_INT >= 33 && SpeechRecognizer.isOnDeviceRecognitionAvailable(this)) SpeechRecognizer.createOnDeviceSpeechRecognizer(this) else SpeechRecognizer.createSpeechRecognizer(this)
        recognizer = r; status.text = "Listening…"
        r.setRecognitionListener(object : RecognitionListener {
            override fun onResults(b: Bundle) { b.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.let { if (model != null) submit(it) else input.setText(it) } }
            override fun onError(e: Int) { status.text = "Speech recognition error $e" }
            override fun onReadyForSpeech(p: Bundle?) {}; override fun onBeginningOfSpeech() {}; override fun onRmsChanged(v: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}; override fun onEndOfSpeech() {}; override fun onPartialResults(p: Bundle?) {}; override fun onEvent(t: Int, p: Bundle?) {}
        })
        r.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM).putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true))
    }

    override fun onRequestPermissionsResult(code: Int, p: Array<out String>, r: IntArray) { if (r.firstOrNull() == PackageManager.PERMISSION_GRANTED) listen() }

    override fun onDestroy() {
        model?.cancel(); recognizer?.destroy(); tts?.shutdown()
        worker.execute { model?.close(); pfd?.close() }; worker.shutdown(); super.onDestroy()
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private companion object { const val PICK = 7 }
}
