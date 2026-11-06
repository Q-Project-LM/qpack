// qpack-server: OpenAI-compatible HTTP server for a .qpack model (POSIX sockets, no dependencies).
//   GET  /health, /v1/models, /            (a minimal chat page)
//   POST /v1/chat/completions             messages, max_tokens, temperature, top_p, top_k, min_p, repeat_penalty, seed,
//                                         stop, stream (SSE); N-Q: "think": true starts the answer with <THINK>, and the
//                                         <think>..</think> part is returned as reasoning_content
//   POST /v1/completions                  prompt (raw text), same sampling fields
//   POST /tokenize, /detokenize
// One model, one generation at a time (requests queue on a mutex); the KV cache / state is reused across requests that
// share a prefix, so a growing chat only evaluates the new turn.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include "qpack/engine.h"

using qp::Json;

namespace {

std::mutex g_mu;
qp::Session *g_s = nullptr;
std::string g_model_id;
std::atomic<long> g_counter{0};

struct Req { std::string method, path, body; bool ok = false; };

bool send_all(int fd, const std::string &s) {
  size_t off = 0;
  while (off < s.size()) { ssize_t n = send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL); if (n <= 0) return false; off += (size_t)n; }
  return true;
}
Req read_req(int fd) {
  Req r; std::string buf; char tmp[65536];
  size_t hend = std::string::npos;
  while ((hend = buf.find("\r\n\r\n")) == std::string::npos) {
    ssize_t n = recv(fd, tmp, sizeof tmp, 0); if (n <= 0) return r; buf.append(tmp, n);
    if (buf.size() > (1 << 20)) return r;
  }
  std::string head = buf.substr(0, hend); size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) return r;
  r.method = head.substr(0, sp1); r.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
  size_t q = r.path.find('?'); if (q != std::string::npos) r.path.resize(q);
  size_t clen = 0; std::string low; for (char c : head) low += (char)tolower((unsigned char)c);
  size_t cl = low.find("\r\ncontent-length:"); if (cl != std::string::npos) clen = strtoul(head.c_str() + cl + 17, nullptr, 10);
  if (clen > (64u << 20)) return r;
  r.body = buf.substr(hend + 4);
  while (r.body.size() < clen) { ssize_t n = recv(fd, tmp, sizeof tmp, 0); if (n <= 0) return r; r.body.append(tmp, n); }
  r.body.resize(clen); r.ok = true; return r;
}
const char *kCors = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
void respond(int fd, int code, const std::string &body, const char *type = "application/json") {
  const char *msg = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : code == 204 ? "No Content" : "Internal Server Error";
  send_all(fd, "HTTP/1.1 " + std::to_string(code) + " " + msg + "\r\nContent-Type: " + type + "\r\nContent-Length: " + std::to_string(body.size()) +
                   "\r\n" + kCors + "Connection: close\r\n\r\n" + body);
}
void error(int fd, int code, const std::string &m) {
  Json e = Json::object(), d = Json::object(); d.set("message", m); d.set("type", code == 400 ? "invalid_request_error" : "server_error"); e.set("error", d);
  respond(fd, code, e.dump());
}

qp::SamplerParams sampling(const Json &b) {
  qp::SamplerParams p;
  if (b.has("temperature")) p.temperature = (float)b["temperature"].as_num();
  if (b.has("top_p")) p.top_p = (float)b["top_p"].as_num();
  if (b.has("top_k")) p.top_k = (int)b["top_k"].as_int();
  if (b.has("min_p")) p.min_p = (float)b["min_p"].as_num();
  if (b.has("repeat_penalty")) p.repeat_penalty = (float)b["repeat_penalty"].as_num();
  else if (b.has("frequency_penalty")) p.repeat_penalty = 1.f + (float)b["frequency_penalty"].as_num() * 0.5f;
  if (b.has("seed")) p.seed = (uint64_t)b["seed"].as_int();
  return p;
}
std::vector<std::string> stops(const Json &b) {
  std::vector<std::string> o; const Json &s = b["stop"];
  if (s.type == Json::String) o.push_back(s.str); else for (auto &x : s.arr) o.push_back(x.as_str());
  return o;
}
std::string content_text(const Json &c) {   // "content": string, or [{type: "text", text}, ...]
  if (c.type == Json::String) return c.str;
  std::string o; for (auto &p : c.arr) if (p["type"].as_str() == "text") o += p["text"].as_str();
  return o;
}
int max_tokens(const Json &b, int def) {
  long long n = b.has("max_completion_tokens") ? b["max_completion_tokens"].as_int() : b["max_tokens"].as_int(def);
  return n > 0 ? (int)n : def;
}
std::string now_id(const char *p) { return std::string(p) + std::to_string((long)time(nullptr)) + "-" + std::to_string(g_counter++); }

// splits streamed text into reasoning (inside <think>..</think>) and content
struct ThinkSplit {
  bool in = false; std::string pend;
  void feed(const std::string &t, std::string &reason, std::string &content) {
    pend += t;
    for (;;) {
      const std::string tag = in ? "</think>" : "<think>";
      size_t f = pend.find(tag);
      if (f != std::string::npos) { (in ? reason : content) += pend.substr(0, f); pend = pend.substr(f + tag.size()); in = !in; continue; }
      size_t keep = 0; for (size_t k = std::min(tag.size() - 1, pend.size()); k > 0; k--) if (pend.compare(pend.size() - k, k, tag, 0, k) == 0) { keep = k; break; }
      (in ? reason : content) += pend.substr(0, pend.size() - keep); pend = pend.substr(pend.size() - keep); return;
    }
  }
  void flush(std::string &reason, std::string &content) { (in ? reason : content) += pend; pend.clear(); }
};

Json usage(const qp::GenResult &r) {
  Json u = Json::object(); u.set("prompt_tokens", r.prompt_tokens); u.set("completion_tokens", (int)r.ids.size()); u.set("total_tokens", r.prompt_tokens + (int)r.ids.size());
  Json t = Json::object(); t.set("cached_tokens", r.reused); u.set("prompt_tokens_details", t);
  Json tm = Json::object(); tm.set("prompt_ms", r.prefill_s * 1e3); tm.set("predicted_ms", r.decode_s * 1e3);
  tm.set("predicted_per_second", r.ids.size() / std::max(r.decode_s, 1e-9)); u.set("timings", tm);
  if (r.circuit_calls) { u.set("circuit_calls", r.circuit_calls); u.set("circuit_failed", r.circuit_failed); }
  if (r.drafted) { u.set("draft_tokens", r.drafted); u.set("draft_accepted", r.accepted); }
  return u;
}

void chat(int fd, const Json &b, bool completion) {
  const bool stream = b["stream"].as_bool();
  qp::Sampler sampler(sampling(b));
  const std::string id = now_id(completion ? "cmpl-" : "chatcmpl-"); const long created = (long)time(nullptr);
  std::vector<int> prompt; std::vector<qp::Message> msgs;
  const bool think = b["think"].as_bool() || b["chat_template_kwargs"]["enable_thinking"].as_bool();
  if (completion) {
    prompt = g_s->tok().encode(content_text(b["prompt"]));
    if (g_s->chat().name() == "nq") prompt.insert(prompt.begin(), g_s->tok().token_id("<bos>"));
  } else {
    for (auto &m : b["messages"].arr) msgs.push_back({m["role"].as_str() == "developer" ? "system" : m["role"].as_str(), content_text(m["content"])});
    if (msgs.empty()) return error(fd, 400, "messages is empty");
  }
  if (stream) send_all(fd, std::string("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n") + kCors + "Connection: close\r\n\r\n");
  auto chunk = [&](const Json &delta, const Json &finish, const Json *use) {
    Json c = Json::object(); c.set("id", id); c.set("object", completion ? "text_completion" : "chat.completion.chunk"); c.set("created", created); c.set("model", g_model_id);
    Json ch = Json::object(); ch.set("index", 0);
    if (completion) ch.set("text", delta["content"]); else ch.set("delta", delta);
    ch.set("finish_reason", finish); Json arr = Json::array(); arr.push(ch); c.set("choices", arr);
    if (use) c.set("usage", *use);
    return send_all(fd, "data: " + c.dump() + "\n\n");
  };
  ThinkSplit ts; ts.in = think && g_s->chat().name() == "nq"; bool alive = true, first = true;
  auto on_text = [&](const std::string &t) {
    if (!stream) return true;
    std::string reason, content; if (completion) content = t; else ts.feed(t, reason, content);
    Json d = Json::object(); if (first && !completion) d.set("role", "assistant");
    if (!reason.empty()) d.set("reasoning_content", reason);
    if (!content.empty()) d.set("content", content);
    if (d.size() == 0) return true;
    first = false; alive = chunk(d, nullptr, nullptr); return alive;
  };
  qp::GenResult r;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    try {
      int n = max_tokens(b, 1024);
      r = completion ? g_s->generate(prompt, std::min(n, g_s->model().max_ctx() - (int)prompt.size()), sampler, on_text, stops(b))
                     : g_s->chat_turn(msgs, n, sampler, on_text, think);
    } catch (const std::exception &e) {
      if (stream) { Json err = Json::object(); err.set("error", e.what()); send_all(fd, "data: " + err.dump() + "\n\n"); return; }
      return error(fd, 400, e.what());
    }
  }
  fprintf(stderr, "%s: prompt %d (%d cached), gen %zu tok, %.1f tok/s, %s\n", completion ? "completion" : "chat", r.prompt_tokens, r.reused, r.ids.size(),
          r.ids.size() / std::max(r.decode_s, 1e-9), r.finish.c_str());
  if (stream) {
    if (!alive) return;
    std::string reason, content; ts.flush(reason, content);
    if (!reason.empty() || !content.empty()) { Json d = Json::object(); if (!reason.empty()) d.set("reasoning_content", reason); if (!content.empty()) d.set("content", content); chunk(d, nullptr, nullptr); }
    Json u = usage(r); chunk(Json::object(), r.finish, &u);
    send_all(fd, "data: [DONE]\n\n");
    return;
  }
  std::string reason, content;
  if (completion) content = r.text; else { ts.feed(r.text, reason, content); ts.flush(reason, content); }
  Json res = Json::object(); res.set("id", id); res.set("object", completion ? "text_completion" : "chat.completion"); res.set("created", created); res.set("model", g_model_id);
  Json ch = Json::object(); ch.set("index", 0);
  if (completion) ch.set("text", content);
  else {
    Json m = Json::object(); m.set("role", "assistant"); m.set("content", content); if (!reason.empty()) m.set("reasoning_content", reason); ch.set("message", m);
  }
  ch.set("finish_reason", r.finish); Json arr = Json::array(); arr.push(ch); res.set("choices", arr); res.set("usage", usage(r));
  respond(fd, 200, res.dump());
}

const char *kPage = R"HTML(<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>qpack-server</title><style>
:root{--bg:#fff;--fg:#1b1b1f;--mut:#6b6b76;--bd:#e3e3e8;--me:#eef1ff}
@media (prefers-color-scheme:dark){:root{--bg:#141417;--fg:#e9e9ee;--mut:#9a9aa5;--bd:#2c2c33;--me:#22263a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}main{max-width:760px;margin:0 auto;padding:16px}
#log div{padding:10px 12px;border:1px solid var(--bd);border-radius:10px;margin:8px 0;white-space:pre-wrap}#log .u{background:var(--me)}
#log .t{color:var(--mut);font-size:13px}form{display:flex;gap:8px}textarea{flex:1;resize:vertical;min-height:44px;padding:10px;border-radius:10px;border:1px solid var(--bd);background:var(--bg);color:var(--fg);font:inherit}
button{padding:0 18px;border-radius:10px;border:1px solid var(--bd);background:var(--fg);color:var(--bg);font:inherit;cursor:pointer}small{color:var(--mut)}</style></head>
<body><main><h3 id="h">qpack-server</h3><div id="log"></div><form id="f"><textarea id="q" placeholder="Message"></textarea><button>Send</button></form>
<small><label><input type="checkbox" id="th"> think</label> · <a href="#" id="clr">clear</a></small></main><script>
const log=document.getElementById('log'),msgs=[];fetch('/v1/models').then(r=>r.json()).then(j=>{document.getElementById('h').textContent=j.data[0].id});
document.getElementById('clr').onclick=e=>{e.preventDefault();msgs.length=0;log.innerHTML=''};
function add(c,t){const d=document.createElement('div');d.className=c;d.textContent=t;log.appendChild(d);return d}
document.getElementById('q').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit()}};
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q'),text=q.value.trim();if(!text)return;q.value='';
add('u',text);msgs.push({role:'user',content:text});const th=add('t',''),a=add('a','…');let ans='',re='';
const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:msgs,stream:true,think:document.getElementById('th').checked})});
const rd=r.body.getReader(),dec=new TextDecoder();let buf='';for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
let i;while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+2);if(!line.startsWith('data: ')||line==='data: [DONE]')continue;const j=JSON.parse(line.slice(6));
const d=j.choices&&j.choices[0].delta||{};if(d.reasoning_content){re+=d.reasoning_content;th.textContent=re}if(d.content){ans+=d.content;a.textContent=ans}
if(j.usage)th.textContent=(re?re+'\n':'')+`${j.usage.completion_tokens} tok · ${j.usage.timings.predicted_per_second.toFixed(1)} tok/s`}}
if(!th.textContent)th.remove();msgs.push({role:'assistant',content:ans})}</script></body></html>)HTML";

void handle(int fd) {
  Req r = read_req(fd);
  if (!r.ok) { close(fd); return; }
  try {
    if (r.method == "OPTIONS") respond(fd, 204, "");
    else if (r.method == "GET" && (r.path == "/" || r.path == "/index.html")) respond(fd, 200, kPage, "text/html; charset=utf-8");
    else if (r.method == "GET" && r.path == "/health") respond(fd, 200, "{\"status\":\"ok\"}");
    else if (r.method == "GET" && r.path == "/v1/models") {
      Json m = Json::object(); m.set("id", g_model_id); m.set("object", "model"); m.set("owned_by", "qpack");
      Json meta = Json::object(); meta.set("arch", g_s->model().arch()); meta.set("n_ctx", g_s->model().max_ctx()); meta.set("chat_format", g_s->chat().name()); m.set("meta", meta);
      Json arr = Json::array(); arr.push(m); Json o = Json::object(); o.set("object", "list"); o.set("data", arr); respond(fd, 200, o.dump());
    } else if (r.method == "POST") {
      Json b = Json::parse(r.body);
      if (r.path == "/v1/chat/completions" || r.path == "/chat/completions") chat(fd, b, false);
      else if (r.path == "/v1/completions" || r.path == "/completions") chat(fd, b, true);
      else if (r.path == "/tokenize") { Json o = Json::object(), a = Json::array(); for (int t : g_s->tok().encode(b["content"].as_str())) a.push(t); o.set("tokens", a); respond(fd, 200, o.dump()); }
      else if (r.path == "/detokenize") { std::vector<int> ids; for (auto &t : b["tokens"].arr) ids.push_back((int)t.as_int()); Json o = Json::object(); o.set("content", g_s->tok().decode(ids, false)); respond(fd, 200, o.dump()); }
      else error(fd, 404, "no such endpoint: " + r.path);
    } else error(fd, 404, "no such endpoint: " + r.path);
  } catch (const std::exception &e) { error(fd, 400, e.what()); }
  close(fd);
}

}  // namespace

int main(int argc, char **argv) {
  std::string path, host = "127.0.0.1", alias, draft; int port = 8080, ctx = 0, lookup = 0, draft_k = 4;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
    if (a == "-m" || a == "--model") path = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "-c" || a == "--ctx") ctx = std::stoi(next());
    else if (a == "-t" || a == "--threads") qp::set_threads(std::stoi(next()));
    else if (a == "--device") setenv("QPACK_DEVICE", next().c_str(), 1);
    else if (a == "-a" || a == "--alias") alias = next();
    else if (a == "--lookup") lookup = std::stoi(next());
    else if (a == "--draft") draft = next();
    else if (a == "--draft-k") draft_k = std::stoi(next());
    else if (path.empty() && a[0] != '-') path = a;
    else { fprintf(stderr, "usage: qpack-server -m model.qpack [--host 127.0.0.1] [--port 8080] [-c ctx] [-t threads] [-a alias] [--lookup N] [--draft model.qpack [--draft-k N]]\n"); return 2; }
  }
  if (path.empty()) { fprintf(stderr, "usage: qpack-server -m model.qpack [--host 127.0.0.1] [--port 8080] [-c ctx] [-t threads] [-a alias] [--lookup N] [--draft model.qpack [--draft-k N]]\n"); return 2; }
  signal(SIGPIPE, SIG_IGN);
  std::unique_ptr<qp::Session> s;
  try {
    s.reset(new qp::Session(qp::load_model(path, ctx)));
    if (lookup > 0) s->set_lookup(lookup);
    if (!draft.empty()) s->set_draft(qp::load_model(draft, s->model().max_ctx()), draft_k);
  } catch (const std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
  g_s = s.get();
  if (alias.empty()) {   // file name without .qpack; "model.q8.qpack" in a release folder takes the folder name
    size_t sl = path.find_last_of('/'); alias = sl == std::string::npos ? path : path.substr(sl + 1);
    size_t d = alias.find(".qpack"); if (d != std::string::npos) alias.resize(d);
    if (alias.compare(0, 5, "model") == 0 && sl != std::string::npos && sl > 0) { std::string dir = path.substr(0, sl); alias = dir.substr(dir.find_last_of('/') + 1) + alias.substr(5); }
  }
  g_model_id = alias;
  int srv = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) { fprintf(stderr, "bad host %s\n", host.c_str()); return 1; }
  if (bind(srv, (sockaddr *)&addr, sizeof addr) < 0 || listen(srv, 64) < 0) { perror("bind/listen"); return 1; }
  fprintf(stderr, "qpack-server: %s (%s, chat %s, ctx %d, %d threads) on http://%s:%d\n", g_model_id.c_str(), s->model().arch().c_str(), s->chat().name().c_str(),
          s->model().max_ctx(), qp::pool().size(), host.c_str(), port);
  for (;;) {
    int fd = accept(srv, nullptr, nullptr); if (fd < 0) continue;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    std::thread(handle, fd).detach();
  }
}
