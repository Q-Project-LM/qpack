from pathlib import Path
import datetime, hashlib, json, os, platform, re, shlex, subprocess, time, urllib.request, urllib.parse, traceback, csv
from zoneinfo import ZoneInfo
os.chdir('/workspace/qpack')
r=Path('research'); stamp=datetime.datetime.now(ZoneInfo('Europe/Moscow')).strftime('%Y%m%d-%H%M%S')
d=r/('matrix-'+stamp); d.mkdir(); (d/'logs').mkdir(); (d/'artifacts').mkdir(); (d/'data').mkdir()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
commands=[]; results=[]
def execute(label,args,timeout=900):
 print('RUN',label,flush=True); start=time.monotonic(); log=d/'logs'/(label+'.log')
 status=0
 try:
  with log.open('w') as f: status=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,text=True,timeout=timeout).returncode
 except subprocess.TimeoutExpired: status=124
 rec=dict(label=label,command=shlex.join(args),returncode=status,seconds=time.monotonic()-start)
 commands.append(rec); (d/'commands.json').write_text(json.dumps(commands,indent=2))
 print(label,'exit',status,'seconds',round(rec['seconds'],1),flush=True)
 if status: raise RuntimeError(label+' failed: '+log.read_text()[-1000:])
 return log.read_text()
def fetch_article(lang,title):
 url='https://'+lang+'.wikipedia.org/wiki/'+urllib.parse.quote(title.replace(' ','_'))
 cached=cache_map.get((lang,title))
 if cached:
  text=cached.read_text(); filename=cached.name; (d/'data'/filename).write_text(text,encoding='utf-8')
  sources.append(dict(language=lang,title=title,url=url,revision=None,file='data/'+filename,sha256=sha(d/'data'/filename),note='Cached snapshot from initial API download; revision metadata unavailable.',license='Wikipedia text: CC BY-SA; see source article history'))
  return text
 query=urllib.parse.urlencode(dict(action='query',prop='extracts|info|revisions',explaintext='1',format='json',titles=title,rvprop='ids',inprop='url',redirects='1'))
 api_url='https://'+lang+'.wikipedia.org/w/api.php?'+query
 for attempt in range(5):
  try:
   time.sleep(5)
   with urllib.request.urlopen(urllib.request.Request(api_url,headers={'User-Agent':'QpackResearch/1.0 (quantization comparison)'}),timeout=60) as f: obj=json.load(f)
   break
  except urllib.error.HTTPError as e:
   if e.code!=429 or attempt==4: raise
   delay=max(15,int(e.headers.get('Retry-After','15'))); print('RATE_LIMIT',title,delay,flush=True); time.sleep(delay)
 p=next(iter(obj['query']['pages'].values())); text=p.get('extract','')
 if len(text)<3000: raise RuntimeError('Article too short: '+title)
 filename=lang+'-'+str(p['pageid'])+'.txt'; (d/'data'/filename).write_text(text,encoding='utf-8')
 sources.append(dict(language=lang,title=p['title'],url=p.get('fullurl',url),revision=p.get('revisions',[{}])[0].get('revid'),api_url=api_url,file='data/'+filename,sha256=sha(d/'data'/filename),license='Wikipedia text: CC BY-SA; see article history'))
 return text
sources=[]
cache_dir=Path('research/matrix-20261102-003503/data')
cache_map={}
for lang,titles in {'ru':['Физика','Биология','География','Компьютер','История','Астрономия'],'en':['Physics','Biology','Geography','Computer']}.items():
 files=sorted(cache_dir.glob(lang+'-*.txt'),key=lambda p:p.stat().st_mtime)
 if len(files)!=len(titles): raise RuntimeError('Unexpected cache layout')
 cache_map.update({(lang,title):p for title,p in zip(titles,files)})

cal_titles={'ru':['Физика','Биология','География','Компьютер'],'en':['Physics','Biology','Geography','Computer']}
eval_titles={'ru':['История','Астрономия'],'en':['History','Astronomy']}
cal={}
for lang in ['ru','en']:
 cal[lang]=[fetch_article(lang,t)[:6000] for t in cal_titles[lang]]
 text='\n\n'.join(fetch_article(lang,t)[:12000] for t in eval_titles[lang])
 (d/'data'/('eval-'+lang+'.txt')).write_text(text,encoding='utf-8')
calib='\n\n'.join(cal[lang][i] for i in range(4) for lang in ['ru','en'])
(d/'data'/'calib.txt').write_text(calib,encoding='utf-8')
(d/'data'/'sources.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False))
plan=[dict(name='f16',flags=None),dict(name='q8',flags=['--type','q8'])]
for typ in ['q4','q4m']:
 for method in ['plain','calib','gptq','gptq-act']:
  for head in [False,True]:
   flags=['--type',typ]
   if method!='plain': flags+=['--calib',str(d/'data'/'calib.txt'),'--calib-tokens','8192']
   if method.startswith('gptq'): flags+=['--gptq']
   if method=='gptq-act': flags+=['--act-order']
   if head: flags+=['--q4-head']
   plan.append(dict(name=typ+'-'+method+('-head4' if head else ''),flags=flags))
m=dict(created_moscow=datetime.datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),source='model.qpack',source_bytes=Path('model.qpack').stat().st_size,source_sha256=sha('model.qpack'),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),platform=platform.platform(),cpus=os.cpu_count(),plan=plan,calibration_token_cap=8192,evaluation='Separate Wikipedia topics; ctx=256, up to 8 chunks per language, 8 threads. Public snapshots; not a broad benchmark. Existing pilot texts additionally checked.',benchmark='Synthetic prompt64/decode128; threads1,8,16; 3 repetitions; independent best-of-3 phase rates.',script_sha256=sha(r/'run_matrix.py'),binary_sha256={p:sha('build/'+p) for p in ['qpack-cli','qpack-quantize','qpack-bench','qpack-perplexity']},data_sha256={p.name:sha(p) for p in (d/'data').glob('*.txt')})
(d/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)); (d/'logs'/'lscpu.log').write_text(subprocess.check_output(['lscpu'],text=True))
with (r/'HISTORY.md').open('a') as f: f.write('\n## '+m['created_moscow']+'\nStarted '+d.name+': 18 main configurations; separate RU/EN Wikipedia calibration and evaluation pages. Original f16 unchanged.\n')
base=max([p for p in r.glob('baseline-*') if p.is_dir()],key=lambda p:p.stat().st_mtime)
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

