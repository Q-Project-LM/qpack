from pathlib import Path
import json,hashlib,subprocess,re,time
r=Path('research'); base=r/'matrix-20261102-003657'; out=r/'extended-holdout'; out.mkdir(exist_ok=False); (out/'logs').mkdir()
sources=json.loads((base/'data/sources.json').read_text())
selected={'ru':['История','Астрономия'],'en':['History','Astronomy']}; manifest={}
for lang,titles in selected.items():
 parts=[]; src=[]
 for title in titles:
  e=next(x for x in sources if x['language']==lang and x['title'].lower()==title.lower())
  text=(base/e['file']).read_text(); assert len(text)>15000
  parts.append(text[12000:]); src.append({'title':title,'file':str(base/e['file']),'offset_chars':12000,'source_sha256':e['sha256']})
 target=out/(lang+'.txt'); target.write_text('\n\n'.join(parts)); manifest[lang]={'sources':src,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
old=json.loads((base/'results.json').read_text()); baseline=next(x['path'] for x in old if x['name']=='q4-gptq-act')
models={'f16':'model.qpack','baseline':baseline,'damp005':'research/attempt-damp005/candidate.qpack','wide':'research/attempt-wide-scales/candidate.qpack','medium':'research/attempt-medium-scales/candidate.qpack'}
results={}
for name,model in models.items():
 results[name]={}
 for lang in ['ru','en']:
  cmd=['build/qpack-perplexity','-m',model,'-f',str(out/(lang+'.txt')),'-c','256','--chunks','20','-t','8']; print('RUN',name,lang,flush=True); start=time.time(); p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); (out/'logs'/(name+'-'+lang+'.log')).write_text(p.stdout); assert p.returncode==0,(name,lang,p.stdout)
  match=re.search(r'(\d+) tokens, mean nll ([0-9.]+), perplexity ([0-9.]+)',p.stdout); assert match
  results[name][lang]={'tokens':int(match[1]),'nll':float(match[2]),'ppl':float(match[3]),'seconds':time.time()-start,'command':cmd,'model_sha256':hashlib.sha256(Path(model).read_bytes()).hexdigest()}; (out/'results.json').write_text(json.dumps(results,indent=2)); print('DONE',name,lang,match[1],match[3],flush=True)
for lang in ['ru','en']: assert len({results[m][lang]['tokens'] for m in models})==1
with (r/'HISTORY.md').open('a') as f: f.write('\n## Расширенная отложенная выборка\nПродолжения статей после первых 12000 символов, без статей калибровки; контекст 256, до 20 блоков, 8 потоков. Это те же статьи, что ранее, поэтому выборка не полностью независима по тематике. См. research/extended-holdout/results.json и manifest.json.\n')
print('COMPLETE',flush=True)

