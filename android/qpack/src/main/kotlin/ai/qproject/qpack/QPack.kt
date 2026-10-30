package ai.qproject.qpack

/** Receives generated text as UTF-8 bytes (a delta never ends inside a character). Return false to stop generation. */
fun interface TextSink { fun onBytes(bytes: ByteArray): Boolean }

data class ChatMessage(val role: String, val content: String)   // role: "system" | "user" | "assistant"

data class GenStats(val finish: String, val promptTokens: Int, val reusedTokens: Int, val generatedTokens: Int, val prefillSeconds: Double, val decodeSeconds: Double) {
    val decodeTps: Double get() = if (decodeSeconds > 0) generatedTokens / decodeSeconds else 0.0
}

/**
 * A model loaded from a .qpack file (mmap, so the weights are paged in on demand and shared with the page cache).
 * chat() blocks the calling thread: call it from a background thread / coroutine; cancel() may be called from any thread.
 */
class QPack(path: String, contextTokens: Int = 2048, threads: Int = 0) : AutoCloseable {
    private var handle: Long = nativeLoad(path, contextTokens, threads)
    val info: String get() = nativeInfo(handle)   // "arch,vocab,max_ctx"

    fun chat(
        messages: List<ChatMessage>, maxNewTokens: Int = 256, temperature: Float = 0.7f, topP: Float = 0.9f, topK: Int = 40,
        repeatPenalty: Float = 1.1f, seed: Long = 0, think: Boolean = false, sink: TextSink,
    ): GenStats {
        check(handle != 0L) { "closed" }
        val r = nativeChat(handle, messages.map { it.role }.toTypedArray(), messages.map { it.content }.toTypedArray(), maxNewTokens, temperature, topP, topK,
            repeatPenalty, seed, think, sink)
        val p = r.split(",")
        return GenStats(p[0], p[1].toInt(), p[2].toInt(), p[3].toInt(), p[4].toDouble(), p[5].toDouble())
    }

    fun cancel() { if (handle != 0L) nativeCancel(handle) }

    override fun close() { if (handle != 0L) { nativeFree(handle); handle = 0 } }

    private companion object {
        init { System.loadLibrary("qpack_jni") }
        @JvmStatic external fun nativeLoad(path: String, ctx: Int, threads: Int): Long
        @JvmStatic external fun nativeFree(h: Long)
        @JvmStatic external fun nativeCancel(h: Long)
        @JvmStatic external fun nativeInfo(h: Long): String
        @JvmStatic external fun nativeChat(h: Long, roles: Array<String>, contents: Array<String>, maxNew: Int, temp: Float, topP: Float, topK: Int, repeat: Float,
                                           seed: Long, think: Boolean, sink: TextSink): String
    }
}
