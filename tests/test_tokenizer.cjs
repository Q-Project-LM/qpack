// node tests/test_tokenizer.cjs <tokenizer.json> <reference.json>
const fs = require('fs'); const { HFTokenizer } = require('../runtime/hf_tokenizer.js');
const [tjp, refp] = process.argv.slice(2);
let t0 = Date.now(); const tok = new HFTokenizer(JSON.parse(fs.readFileSync(tjp))); console.log('load', Date.now() - t0, 'ms');
const ref = JSON.parse(fs.readFileSync(refp)); let ok = 0, dec = 0; const bad = []; t0 = Date.now();
for (const r of ref) { let ids; try { ids = tok.encode(r.text); } catch (e) { ids = ['ERR ' + e.message]; }
  if (JSON.stringify(ids) === JSON.stringify(r.ids)) ok++; else bad.push(JSON.stringify(r.text.slice(0, 50)));
  if (tok.decode(r.ids, false) === r.text.normalize('NFC') || tok.decode(r.ids, false) === r.text) dec++; }
console.log(`encode identical ${ok}/${ref.length} (${Date.now() - t0} ms), decode round-trip ${dec}/${ref.length}`); if (bad.length) console.log('mismatches:', bad.slice(0, 6));
