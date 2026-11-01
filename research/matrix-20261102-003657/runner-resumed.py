from pathlib import Path
import datetime, hashlib, json, os, platform, re, shlex, subprocess, time, urllib.request, urllib.parse, traceback, csv
from zoneinfo import ZoneInfo
os.chdir('/workspace/qpack')
r=Path('research'); d=max([p for p in r.glob('matrix-*') if (p/'manifest.json').exists()],key=lambda p:p.stat().st_mtime)
m=json.loads((d/'manifest.json').read_text()); plan=m['plan']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
commands=json.loads((d/'commands.json').read_text()); results=[]
import shutil
for n in ['results.json','SUMMARY.md','commands.json']:
 if (d/n).exists() and not (d/('parser-attempt1-'+n)).exists(): shutil.copy2(d/n,d/('parser-attempt1-'+n))
base=max([p for p in r.glob('baseline-*') if p.is_dir()],key=lambda p:p.stat().st_mtime)
m['resume_script_sha256']=sha(r/'resume_matrix.py'); (d/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False))
with (r/'HISTORY.md').open('a') as f: f.write('\nMatrix preparation initially hit HTTP429; snapshots cached and requests retried. Initial matrix runner then misparsed successful self-test output (expected selftest, actual self-test). Original results and logs retained; resume_matrix.py fixes parsing and reuses completed quantizations.\n')
def execute(label,args,timeout=900):
 print('RUN',label,flush=True); start=time.monotonic(); log=d/'logs'/(label+'.log')
 attempt=0
 while log.exists():
  attempt+=1; log=d/'logs'/(label+'-retry'+str(attempt)+'.log')
 status=0
 try:
  with log.open('w') as f: status=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,text=True,timeout=timeout).returncode
 except subprocess.TimeoutExpired: status=124
 rec=dict(label=label,command=shlex.join(args),log=str(log),returncode=status,seconds=time.monotonic()-start)
 commands.append(rec); (d/'commands.json').write_text(json.dumps(commands,indent=2))
 print(label,'exit',status,'seconds',round(rec['seconds'],1),flush=True)
 if status: raise RuntimeError(label+' failed: '+log.read_text()[-1000:])
 return log.read_text()
def write_results():
 (d/'results.json').write_text(json.dumps(results,indent=2))
 lines=['# Quantization matrix','','Source SHA256: '+m['source_sha256'],'','18 configurations: f16, q8, and q4/q4m x plain/calib/GPTQ/GPTQ-act-order x head q8/q4. Embedding follows tool defaults. No algorithm changes.','','Primary PPL: separate Wikipedia articles for calibration/evaluation, context256, up to8 chunks, threads8. Calibration cap8192, balanced RU/EN article order; actual counts in logs. Pilot PPL uses saved baseline prose, 2 chunks. Corpora are preliminary and narrow; not comparable with README scores.','','Speed: synthetic tokens, prompt64/decode128, 1/8/16 threads, separate best-of3 per phase. Selftest checks recalculated quantized references, not original quality.','','| Variant | MiB | PPL RU | PPL EN | Decode 1 | Decode 8 | Decode 16 | Status |','|---|---:|---:|---:|---:|---:|---:|---|']
 for x in results:
  if x['status']=='ok':
   lines.append('| '+x['name']+' | '+f"{x['bytes']/1048576:.2f}"+' | '+f"{x['ppl']['ru']['perplexity']:.4f}"+' | '+f"{x['ppl']['en']['perplexity']:.4f}"+' | '+' | '.join(str(x['bench'][str(t)]['decode']) for t in [1,8,16])+' | '+x['selftest']+' |')
  else: lines.append('| '+x['name']+' | - | - | - | - | - | - | FAILED: '+x.get('error','').replace('\n',' ')[:160]+' |')
 lines+=['','See results.json for pilot PPL, all prompt speeds, token counts, sizes and model hashes; commands.json and logs/ preserve all commands and errors. sources.json records article URLs, revisions and attribution.']
 (d/'SUMMARY.md').write_text('\n'.join(lines))
for item in plan:
 name=item['name']; x=dict(name=name,flags=item['flags'],status='running')
 try:
  if item['flags'] is None: model='model.qpack'
  else:
   model=str(d/'artifacts'/(name+'.qpack'))
   if any(c['label']==name+'-quantize' and c['returncode']==0 for c in commands) and Path(model).exists():
    print('REUSE',name,flush=True)
   else:
    model=str(d/'artifacts'/(name+'-retry.qpack'))
    execute(name+'-quantize',['./build/qpack-quantize','model.qpack',model,'-t','8']+item['flags'])
  x.update(bytes=Path(model).stat().st_size,sha256=sha(model),path=model)
  output=execute(name+'-selftest',['./build/qpack-cli','-m',model,'--selftest','-t','8'],timeout=120)
  match=re.search(r'self[- ]?test\s+(\d+/\d+)',output)
  x['selftest']=match[1] if match else output.strip()
  if not match or match[1]!='5/5': raise RuntimeError('Selftest did not pass 5/5: '+output)
  x['ppl']={}; x['pilot_ppl']={}
  for scope,folder,chunks in [('ppl',d/'data',8),('pilot_ppl',base,2)]:
   for lang in ['ru','en']:
    output=execute(name+'-'+scope+'-'+lang,['./build/qpack-perplexity','-m',model,'-f',str(folder/('eval-'+lang+'.txt')),'-c','256','--chunks',str(chunks),'-t','8'],timeout=180)
    match=re.search(r'(\d+) tokens, mean nll ([0-9.]+), perplexity ([0-9.]+)',output)
    if not match: raise RuntimeError('PPL parse failed: '+output)
    x[scope][lang]=dict(tokens=int(match[1]),nll=float(match[2]),perplexity=float(match[3]))
  output=execute(name+'-bench',['./build/qpack-bench','-m',model,'-p','64','-n','128','-t','1,8,16','-r','3'],timeout=180)
  x['bench']={}
  for line in output.splitlines():
   cells=[c.strip() for c in line.split('|')]
   if len(cells)>=7 and cells[4] in ['1','8','16']:
    x['bench'][cells[4]]=dict(prompt=float(cells[5]),decode=float(cells[6]))
  if len(x['bench'])!=3: raise RuntimeError('Bench parse failed: '+output)
  x['status']='ok'
 except Exception as e:
  x['status']='failed'; x['error']=str(e); (d/'logs'/(name+'-failure.log')).write_text(traceback.format_exc())
 results.append(x); write_results(); print('VARIANT',name,x['status'],flush=True)
print('DONE',d,flush=True)
with (r/'HISTORY.md').open('a') as f: f.write('\nFinished '+d.name+': '+str(sum(x['status']=='ok' for x in results))+'/'+str(len(plan))+' variants completed successfully; see SUMMARY.md and failure logs.\n')
print((d/'SUMMARY.md').read_text(),flush=True)

