from pathlib import Path
import subprocess,shlex,json,time
r=Path('research'); d=r/'attempt-dense-scales'; d.mkdir(exist_ok=False)
s=Path('tools/qpack-quantize.cpp').read_text(); old='k <= 12; k++) try_scale(m / (7.f + 0.125f * k))'; new='k <= 24; k++) try_scale(m / (7.f + 0.0625f * k))'; old2='k <= 12; k++) try_scale(-m / (7.5f + 0.125f * k))'; new2='k <= 24; k++) try_scale(-m / (7.5f + 0.0625f * k))'; assert s.count(old)==1 and s.count(old2)==1
(d/'quantize.cpp').write_text(s.replace(old,new).replace(old2,new2))
base=Path('research/matrix-20261102-003657'); model=str(d/'candidate.qpack'); commands=[]
def run(name,args):
 print('RUN',name,flush=True); start=time.time(); p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); (d/(name+'.log')).write_text(p.stdout); commands.append(dict(name=name,args=args,code=p.returncode,seconds=time.time()-start)); (d/'commands.json').write_text(json.dumps(commands,indent=2)); print(name,p.returncode,flush=True)
 if p.returncode: raise RuntimeError(p.stdout[-2000:])
run('build',['g++','-std=c++17','-O3','-march=native','-Iinclude',str(d/'quantize.cpp'),'build/libqpack.a','-pthread','-o',str(d/'quantize')])
run('quantize',[str(d/'quantize'),'model.qpack',model,'--type','q4','--calib',str(base/'data/calib.txt'),'--calib-tokens','8192','--gptq','--act-order','-t','8'])
run('selftest',['build/qpack-cli','-m',model,'--selftest','-t','8'])
for lang in ['ru','en']:
 run('ppl-'+lang,['build/qpack-perplexity','-m',model,'-f',str(base/('data/eval-'+lang+'.txt')),'-c','256','--chunks','8','-t','8'])
print('DONE',flush=True)

