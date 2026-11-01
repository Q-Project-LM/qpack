from pathlib import Path
import subprocess,json,hashlib,re,time
r=Path('research'); d=r/'q4m-last-half-validation'; d.mkdir(exist_ok=False); base=r/'matrix-20261102-003657'; ext=r/'extended-holdout'; prior=json.loads((base/'results.json').read_text()); ref=next(x['path'] for x in prior if x['name']=='q4m-gptq-act'); proto=r/'attempt-q4m-lasthalf/candidate.qpack'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
commands=[]
def run(label,args,expected=0):
 print('RUN',label,flush=True); t=time.time(); p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); (d/(label+'.log')).write_text(p.stdout); commands.append(dict(label=label,args=args,code=p.returncode,seconds=time.time()-t)); (d/'commands.json').write_text(json.dumps(commands,indent=2)); print('EXIT',label,p.returncode,flush=True); assert p.returncode==expected,(label,p.stdout[-1000:]); return p.stdout
common=['--type','q4m','--calib',str(base/'data/calib.txt'),'--calib-tokens','8192','--gptq','--act-order','-t','8']
run('invalid',['build/qpack-quantize','model.qpack',str(d/'invalid.qpack'),'--type','q4','--q4m-last-half'],2); assert not (d/'invalid.qpack').exists()
default=str(d/'default.qpack'); candidate=str(d/'candidate.qpack')
run('default-quantize',['build/qpack-quantize','model.qpack',default]+common); assert sha(default)==sha(ref),'default output changed'
run('candidate-quantize',['build/qpack-quantize','model.qpack',candidate]+common+['--q4m-last-half'])
assert Path(candidate).stat().st_size==Path(ref).stat().st_size
run('selftest',['build/qpack-cli','-m',candidate,'--selftest','-t','8'])
for model_name,model in [('candidate',candidate),('reference',ref)]:
 for lang in ['ru','en']:
  run(model_name+'-extended-'+lang,['build/qpack-perplexity','-m',model,'-f',str(ext/(lang+'.txt')),'-c','256','--chunks','20','-t','8'])
 run(model_name+'-bench',['build/qpack-bench','-m',model,'-p','64','-n','128','-t','1,8,16','-r','3'])
for lang in ['ru','en']:
 t=(d/('candidate-extended-'+lang+'.log')).read_text(); p=(r/'attempt-q4m-lasthalf'/('candidate-extended-'+lang+'.log')).read_text(); pat=r'(\d+) tokens, mean nll ([0-9.]+), perplexity ([0-9.]+)'; a=re.search(pat,t); b=re.search(pat,p); assert a and b and a.groups()==b.groups(),(lang,a,b)
(d/'DONE').write_text('Default SHA matches reference; candidate PPL matches prototype; selftest and benchmarks completed.\n'); print('DONE',flush=True)

