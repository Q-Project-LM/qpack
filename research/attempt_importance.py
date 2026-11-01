from pathlib import Path
import subprocess,json,time
r=Path('research'); base=r/'matrix-20261102-003657'; pilot=r/'baseline-20261102-001640'; ext=r/'extended-holdout'; source=Path('tools/qpack-quantize.cpp').read_text()
old='imp[i] = (float)Hf[(size_t)i * n + i] + (float)damp;'
assert source.count(old)==1
for name,new in [('sqrtimp','imp[i] = std::sqrt(std::max(0.0f, (float)Hf[(size_t)i * n + i] + (float)damp));'),('uniformimp','imp[i] = 1.0f;')]:
 d=r/('attempt-'+name); d.mkdir(exist_ok=False); (d/'quantize.cpp').write_text(source.replace(old,new)); model=str(d/'candidate.qpack'); cmds=[]
 def run(label,args):
  print('RUN',name,label,flush=True); t=time.time(); p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); (d/(label+'.log')).write_text(p.stdout); cmds.append(dict(label=label,args=args,code=p.returncode,seconds=time.time()-t)); (d/'commands.json').write_text(json.dumps(cmds,indent=2)); print('EXIT',name,label,p.returncode,flush=True)
  if p.returncode: raise RuntimeError(p.stdout[-1500:])
 run('build',['g++','-std=c++17','-O3','-march=native','-Iinclude',str(d/'quantize.cpp'),'build/libqpack.a','-pthread','-o',str(d/'quantize')])
 run('quantize',[str(d/'quantize'),'model.qpack',model,'--type','q4','--calib',str(base/'data/calib.txt'),'--calib-tokens','8192','--gptq','--act-order','-t','8'])
 run('selftest',['build/qpack-cli','-m',model,'--selftest','-t','8'])
 for scope,folder,chunks,prefix in [('wiki',base/'data','8','eval-'),('pilot',pilot,'2','eval-'),('extended',ext,'20','')]:
  for lang in ['ru','en']: run(scope+'-'+lang,['build/qpack-perplexity','-m',model,'-f',str(folder/(prefix+lang+'.txt')),'-c','256','--chunks',chunks,'-t','8'])
 print('DONE',name,flush=True)

