package ai.qproject.wear

import ai.qproject.qpack.ChatMessage
import ai.qproject.qpack.GenStats
import ai.qproject.qpack.QPack
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.io.ByteArrayOutputStream
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Fully on-device voice loop: SpeechRecognizer (on-device recogniser when the watch has one, offline-preferred otherwise)
 * -> QPack LLM (streaming) -> TextToSpeech, one sentence at a time so speech starts while the model is still generating.
 * All three stages run locally; no network permission is declared. Call everything except [close] from the main thread.
 */
class VoiceAssistant(private val ctx: Context, modelPath: String, private val listener: Listener, locale: Locale = Locale.getDefault()) {
    interface Listener {
        fun onHeard(text: String)
        fun onReply(textSoFar: String)
        fun onDone(stats: GenStats?)
        fun onError(message: String)
    }

    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor()   // the model runs here
    private val history = ArrayList<ChatMessage>()
    private val llm: QPack = QPack(modelPath, contextTokens = 1024)
    private var recognizer: SpeechRecognizer? = null
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private val recLocale = locale

    init {
        history += ChatMessage("system", "You are a voice assistant on a smartwatch. Answer in one or two short sentences, plain text, no lists, no markdown.")
        tts = TextToSpeech(ctx) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) tts?.language = recLocale else listener.onError("TextToSpeech is not available")
        }
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String?) {}
            override fun onDone(id: String?) { if (id == LAST) main.post { listener.onDone(null) } }
            @Deprecated("deprecated in API") override fun onError(id: String?) {}
        })
    }

    /** Listen once; the recognised text is sent to the model and the reply is spoken. */
    fun listen() {
        if (!SpeechRecognizer.isRecognitionAvailable(ctx)) { listener.onError("No speech recogniser on this device"); return }
        tts?.stop(); llm.cancel()
        recognizer?.destroy()
        val r = if (android.os.Build.VERSION.SDK_INT >= 33 && SpeechRecognizer.isOnDeviceRecognitionAvailable(ctx)) SpeechRecognizer.createOnDeviceSpeechRecognizer(ctx)
                else SpeechRecognizer.createSpeechRecognizer(ctx)
        recognizer = r
        r.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: Bundle) {
                val text = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                if (text.isNullOrBlank()) listener.onError("Did not catch that") else { listener.onHeard(text); ask(text) }
            }
            override fun onError(error: Int) { listener.onError("Speech recognition error $error") }
            override fun onReadyForSpeech(p: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(v: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(p: Bundle?) {}
            override fun onEvent(t: Int, p: Bundle?) {}
        })
        r.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE, recLocale.toLanguageTag()).putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true))
    }

    fun ask(text: String) {
        history += ChatMessage("user", text)
        val turn = history.toList()
        worker.execute {
            val reply = StringBuilder(); val pending = ByteArrayOutputStream(); var spoken = 0
            // speak every finished sentence as soon as it is complete
            fun flush(final: Boolean) {
                val s = reply.toString()
                var end = if (final) s.length else s.lastIndexOfAny(charArrayOf('.', '!', '?', '\n', '。', '!', '?')) + 1
                if (end > spoken) {
                    val sentence = s.substring(spoken, end).trim(); spoken = end
                    if (sentence.isNotEmpty() && ttsReady) tts?.speak(sentence, TextToSpeech.QUEUE_ADD, null, if (final) LAST else "s$spoken")
                }
            }
            val stats = try {
                llm.chat(turn, maxNewTokens = 160, temperature = 0.6f, think = false, sink = { bytes ->
                    pending.write(bytes); reply.append(pending.toString(Charsets.UTF_8)); pending.reset()
                    val so = reply.toString(); main.post { listener.onReply(so) }
                    flush(false); true
                })
            } catch (e: RuntimeException) { main.post { listener.onError(e.message ?: "model error") }; null }
            flush(true)
            main.post { history += ChatMessage("assistant", reply.toString()); if (stats != null) listener.onDone(stats) }
        }
    }

    fun close() { llm.cancel(); recognizer?.destroy(); tts?.shutdown(); worker.shutdown(); worker.execute { llm.close() } }

    private companion object { const val LAST = "last" }
}
