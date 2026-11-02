from pathlib import Path
import subprocess,json,time,hashlib,re,os
r=Path('research/qwen-reference-quant'); r.mkdir(exist_ok=True); a=Path('research/qwen-validation/artifacts'); base=Path('research/matrix-20261102-003657/data'); ext=Path('research/extended-holdout'); f32=str(a/'qwen-reference.f32.qpack'); cmds=[]; results={}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def run(label,args):
 print('RUN',label,flush=True); start=time.time()
 with (r/(label+'.log')).open('w') as out: p=subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,text=True)
 cmds.append({'label':label,'args':args,'code':p.returncode,'seconds':time.time()-start}); (r/'commands.json').write_text(json.dumps(cmds,indent=2)); print('EXIT',label,p.returncode,'seconds',round(time.time()-start,1),flush=True)
 if p.returncode: raise RuntimeError((r/(label+'.log')).read_text()[-1000:])
 return (r/(label+'.log')).read_text()
common=['--type','q4m','--calib',str(base/'calib.txt'),'--calib-tokens','4096','--gptq','--act-order','-t','8']
for name,extra in [('reference-quant',[])]:
 model=str(a/(name+'.qpack')); run(name+'-quantize',['build/qpack-quantize',f32,model]+common+extra); results[name]={'bytes':Path(model).stat().st_size,'sha256':sha(model),'ppl':{}}
 out=run(name+'-selftest',['build/qpack-cli','-m',model,'--selftest','-t','8']); results[name]['selftest_output']=out.strip()
 for scope,folder,prefix in [('wiki',base,'eval-'),('extended',ext,'')]:
  for lang in ['ru','en']:
   out=run(name+'-'+scope+'-'+lang,['build/qpack-perplexity','-m',model,'-f',str(folder/(prefix+lang+'.txt')),'-c','256','--chunks','4','-t','8']); m=re.search(r'(\d+) tokens, mean nll ([0-9.]+), perplexity ([0-9.]+)',out); assert m; results[name]['ppl'][scope+'-'+lang]={'tokens':int(m[1]),'nll':float(m[2]),'ppl':float(m[3])}
 (r/'results.json').write_text(json.dumps(results,indent=2))
assert all(v['tokens'] == 1020 for v in results['reference-quant']['ppl'].values())
(r/'DONE').write_text('Comparison complete; original f32 model has no selftest reference.\n'); print(json.dumps(results,indent=2)); print('COMPLETE',flush=True)
