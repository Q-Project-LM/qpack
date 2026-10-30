// JNI bridge: ai.qproject.qpack.QPack (Kotlin) -> qp::Session. One handle = one loaded model with its KV cache.
#include <jni.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "qpack/engine.h"
#include "qpack/qpack.h"

namespace {

struct Handle { std::unique_ptr<qp::Session> s; std::atomic<bool> cancel{false}; };

std::string str(JNIEnv *e, jstring j) {
  if (!j) return {};
  const jclass sc = e->FindClass("java/lang/String"); const jmethodID gb = e->GetMethodID(sc, "getBytes", "(Ljava/lang/String;)[B");
  jstring cs = e->NewStringUTF("UTF-8"); jbyteArray a = (jbyteArray)e->CallObjectMethod(j, gb, cs);
  const jsize n = e->GetArrayLength(a); std::string r(n, '\0'); e->GetByteArrayRegion(a, 0, n, (jbyte *)r.data());
  e->DeleteLocalRef(a); e->DeleteLocalRef(cs); e->DeleteLocalRef(sc); return r;
}
void throw_rt(JNIEnv *e, const std::string &m) { jclass c = e->FindClass("java/lang/RuntimeException"); e->ThrowNew(c, m.c_str()); }

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_qproject_qpack_QPack_nativeLoad(JNIEnv *e, jclass, jstring path, jint ctx, jint threads) {
  try {
    if (threads > 0) qp::set_threads(threads);   // only effective before the first model use in the process
    auto h = new Handle(); h->s.reset(new qp::Session(qp::load_model(str(e, path), ctx))); return (jlong)h;
  } catch (const std::exception &ex) { throw_rt(e, ex.what()); return 0; }
}

JNIEXPORT void JNICALL Java_ai_qproject_qpack_QPack_nativeFree(JNIEnv *, jclass, jlong h) { delete (Handle *)h; }
JNIEXPORT void JNICALL Java_ai_qproject_qpack_QPack_nativeCancel(JNIEnv *, jclass, jlong h) { ((Handle *)h)->cancel = true; }

JNIEXPORT jstring JNICALL Java_ai_qproject_qpack_QPack_nativeInfo(JNIEnv *e, jclass, jlong h) {
  qp::Model &m = ((Handle *)h)->s->model();
  return e->NewStringUTF((m.arch() + "," + std::to_string(m.vocab()) + "," + std::to_string(m.max_ctx())).c_str());
}

// messages: roles[i] / contents[i]; sink: ai.qproject.qpack.TextSink.onBytes(byte[]): Boolean (false = stop).
// Returns "finish,prompt_tokens,reused,gen_tokens,prefill_s,decode_s".
JNIEXPORT jstring JNICALL Java_ai_qproject_qpack_QPack_nativeChat(JNIEnv *e, jclass, jlong hp, jobjectArray roles, jobjectArray contents, jint maxNew, jfloat temp,
                                                                  jfloat topP, jint topK, jfloat repeat, jlong seed, jboolean think, jobject sink) {
  Handle *h = (Handle *)hp; h->cancel = false;
  try {
    std::vector<qp::Message> msgs; const jsize n = e->GetArrayLength(roles);
    for (jsize i = 0; i < n; i++) {
      jstring r = (jstring)e->GetObjectArrayElement(roles, i), c = (jstring)e->GetObjectArrayElement(contents, i);
      msgs.push_back({str(e, r), str(e, c)}); e->DeleteLocalRef(r); e->DeleteLocalRef(c);
    }
    qp::SamplerParams sp; sp.temperature = temp; sp.top_p = topP; sp.top_k = topK; sp.repeat_penalty = repeat; sp.seed = (uint64_t)seed;
    qp::Sampler sm(sp);
    const jmethodID cb = e->GetMethodID(e->GetObjectClass(sink), "onBytes", "([B)Z");
    auto on_text = [&](const std::string &t) {
      if (h->cancel) return false;
      jbyteArray a = e->NewByteArray((jsize)t.size()); e->SetByteArrayRegion(a, 0, (jsize)t.size(), (const jbyte *)t.data());
      const jboolean go = e->CallBooleanMethod(sink, cb, a); e->DeleteLocalRef(a);
      return go == JNI_TRUE && !e->ExceptionCheck();
    };
    const qp::GenResult r = h->s->chat_turn(msgs, maxNew, sm, on_text, think == JNI_TRUE);
    const std::string out = (h->cancel ? std::string("cancelled") : r.finish) + "," + std::to_string(r.prompt_tokens) + "," + std::to_string(r.reused) + "," +
                            std::to_string(r.ids.size()) + "," + std::to_string(r.prefill_s) + "," + std::to_string(r.decode_s);
    return e->NewStringUTF(out.c_str());
  } catch (const std::exception &ex) { if (!e->ExceptionCheck()) throw_rt(e, ex.what()); return nullptr; }
}

}  // extern "C"
