// qpack-cli: run a .qpack model from the terminal — one-shot completion (-p), interactive chat (default), self-test.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "qpack/engine.h"

static void usage() {
  fprintf(stderr,
          "usage: qpack-cli -m model.qpack [options]\n"
          "  -p, --prompt TEXT      complete TEXT and exit (raw, no chat template)\n"
          "  -q, --ask TEXT         one chat turn with TEXT as the user message and exit\n"
          "  -sys, --system TEXT    system message for chat\n"
          "  -n, --n-predict N      max new tokens (default 512)\n"
          "  -c, --ctx N            context window (default min(4096, model limit))\n"
          "  -t, --threads N        CPU threads (default: hardware, at most 16)\n"
          "  --temp F --top-k N --top-p F --min-p F --repeat-penalty F --repeat-last-n N --seed N\n"
          "  --think                start the N-Q answer with <THINK>\n"
          "  --selftest             run the container's self-test and exit\n"
          "  --verbose              print timings after every reply\n"
          "  --lookup N             speculative decoding by prompt lookup, up to N tokens per step (helps models of a few\n"
          "                         hundred MB and up on repetitive output: code edits, quoting, summaries)\n"
          "  --draft model.qpack    speculative decoding with a smaller model of the same tokenizer; --draft-k N (4)\n");
}

int main(int argc, char **argv) {
  std::string path, prompt, ask, system; int n_predict = 512, ctx = 0; bool selftest = false, verbose = false, think = false, have_prompt = false, have_ask = false;
  int lookup = 0, draft_k = 4; std::string draft;
  qp::SamplerParams sp;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
    if (a == "-m" || a == "--model") path = next();
    else if (a == "-p" || a == "--prompt") { prompt = next(); have_prompt = true; }
    else if (a == "-q" || a == "--ask") { ask = next(); have_ask = true; }
    else if (a == "-sys" || a == "--system") system = next();
    else if (a == "-n" || a == "--n-predict") n_predict = std::stoi(next());
    else if (a == "-c" || a == "--ctx") ctx = std::stoi(next());
    else if (a == "-t" || a == "--threads") qp::set_threads(std::stoi(next()));
    else if (a == "--device") setenv("QPACK_DEVICE", next().c_str(), 1);
    else if (a == "--temp") sp.temperature = std::stof(next());
    else if (a == "--top-k") sp.top_k = std::stoi(next());
    else if (a == "--top-p") sp.top_p = std::stof(next());
    else if (a == "--min-p") sp.min_p = std::stof(next());
    else if (a == "--repeat-penalty") sp.repeat_penalty = std::stof(next());
    else if (a == "--repeat-last-n") sp.repeat_last_n = std::stoi(next());
    else if (a == "--seed") sp.seed = std::stoull(next());
    else if (a == "--think") think = true;
    else if (a == "--lookup") lookup = std::stoi(next());
    else if (a == "--draft") draft = next();
    else if (a == "--draft-k") draft_k = std::stoi(next());
    else if (a == "--selftest") selftest = true;
    else if (a == "--verbose") verbose = true;
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else if (path.empty() && a[0] != '-') path = a;
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
  }
  if (path.empty()) { usage(); return 2; }
  try {
    auto t0 = std::chrono::steady_clock::now();
    auto model = qp::load_model(path, ctx);
    double load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (selftest) {
      auto st = qp::run_selftest(*model);
      printf("%s: %s, vocab %d, ctx %d, loaded in %.2f s, self-test %d/%d (max |dlogit| %.2e), threads %d\n", path.c_str(), model->arch().c_str(),
             model->vocab(), model->max_ctx(), load, st.ok, st.n, st.max_err, qp::pool().size());
      return st.ok == st.n ? 0 : 1;
    }
    qp::Session s(std::move(model));
    if (lookup > 0) s.set_lookup(lookup);
    if (!draft.empty()) s.set_draft(qp::load_model(draft, s.model().max_ctx()), draft_k);
    qp::Sampler sampler(sp);
    fprintf(stderr, "%s: %s, chat format %s, ctx %d, %d threads, loaded in %.2f s\n", path.c_str(), s.model().arch().c_str(), s.chat().name().c_str(),
            s.model().max_ctx(), qp::pool().size(), load);
    auto print = [](const std::string &t) { fwrite(t.data(), 1, t.size(), stdout); fflush(stdout); return true; };
    auto stats = [&](const qp::GenResult &r) {
      fprintf(stderr, "\n[prompt %d tok (%d cached) %.1f tok/s | gen %zu tok %.1f tok/s | %s]\n", r.prompt_tokens, r.reused,
              (r.prompt_tokens - r.reused) / std::max(r.prefill_s, 1e-9), r.ids.size(), r.ids.size() / std::max(r.decode_s, 1e-9), r.finish.c_str());
      if (r.drafted) fprintf(stderr, "[speculative: %d drafted, %d accepted, %d passes]\n", r.drafted, r.accepted, r.verify_passes);
    };
    if (have_prompt) {
      std::vector<int> ids = s.tok().encode(prompt);
      if (s.chat().name() == "nq") ids.insert(ids.begin(), s.tok().token_id("<bos>"));
      print(prompt); auto r = s.generate(ids, n_predict, sampler, print); printf("\n"); stats(r);
      return 0;
    }
    std::vector<qp::Message> msgs;
    if (!system.empty()) msgs.push_back({"system", system});
    auto turn = [&](const std::string &user) {
      msgs.push_back({"user", user});
      auto r = s.chat_turn(msgs, n_predict, sampler, print, think);
      printf("\n"); if (verbose || have_ask) stats(r);
      msgs.push_back({"assistant", r.raw});
    };
    if (have_ask) { turn(ask); return 0; }
    fprintf(stderr, "chat mode: empty line or /exit to quit, /reset to clear the history\n");
    std::string line;
    for (;;) {
      fprintf(stderr, "\n> "); if (!std::getline(std::cin, line) || line == "/exit" || line.empty()) break;
      if (line == "/reset") { msgs.resize(system.empty() ? 0 : 1); continue; }
      turn(line);
    }
  } catch (const std::exception &e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
  return 0;
}
