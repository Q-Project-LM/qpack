from pathlib import Path
import subprocess,json,time,re,hashlib
r=Path('research'); base=r/'matrix-20261102-003657'; pilot=r/'baseline-20261102-001640'; ext=r/'extended-holdout'; d=r/'attempt-q4m-firsthalf'; d.mkdir(exist_ok=False)
source=Path('tools/qpack-quantize.cpp').read_text(); old='return i < n_layers / 8 || i >= 7 * n_layers / 8 || (i - n_layers / 8) % 3 == 2;'; new='return i < n_layers / 2;'; assert source.count(old)==1
(d/'quantize.cpp').write_text(source.replace(old,new)); models=json.loads((base/'results.json').read_text()); ref=next(x['path'] for x in models if x['name']=='q4m-gptq-act'); candidate=str(d/'candidate.qpack'); cmds=[]
def run(label,args):
 print('RUN',label,flush=True); t=time.time(); p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); (d/(label+'.log')).write_text(p.stdout); cmds.append(dict(label=label,args=args,code=p.returncode,seconds=time.time()-t)); (d/'commands.json').write_text(json.dumps(cmds,indent=2)); print('EXIT',label,p.returncode,flush=True)
 if p.returncode: raise RuntimeError(p.stdout[-1500:])
run('build',['g++','-std=c++17','-O3','-march=native','-Iinclude',str(d/'quantize.cpp'),'build/libqpack.a','-pthread','-o',str(d/'quantize')])
run('quantize',[str(d/'quantize'),'model.qpack',candidate,'--type','q4m','--calib',str(base/'data/calib.txt'),'--calib-tokens','8192','--gptq','--act-order','-t','8'])
run('selftest',['build/qpack-cli','-m',candidate,'--selftest','-t','8'])
for name,model in [('candidate',candidate)]:
 for scope,folder,chunks,prefix in [('wiki',base/'data','8','eval-'),('pilot',pilot,'2','eval-'),('extended',ext,'20','')]:
  for lang in ['ru','en']: run(name+'-'+scope+'-'+lang,['build/qpack-perplexity','-m',model,'-f',str(folder/(prefix+lang+'.txt')),'-c','256','--chunks',chunks,'-t','8'])
print('SIZES',Path(ref).stat().st_size,Path(candidate).stat().st_size,flush=True)
print('COMPLETE',flush=True)

