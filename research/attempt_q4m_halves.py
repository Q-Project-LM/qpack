from pathlib import Path
import subprocess
r=Path('research'); original=(r/'attempt_q4m_middle.py').read_text(); marker='return !(i < n_layers / 8 || i >= 7 * n_layers / 8 || (i - n_layers / 8) % 3 == 2);'; assert original.count(marker)==1
for name,policy in [('firsthalf','return i < n_layers / 2;'),('lasthalf','return i >= n_layers / 2;')]:
 text=original.replace('attempt-q4m-middle','attempt-q4m-'+name).replace(marker,policy).replace("[('reference',ref),('candidate',candidate)]","[('candidate',candidate)]")
 p=r/('attempt_q4m_'+name+'.py'); p.write_text(text); print('RUN',name,flush=True)
 with (r/('attempt-q4m-'+name+'-console.log')).open('w') as f: result=subprocess.run(['python3',str(p)],stdout=f,stderr=subprocess.STDOUT)
 print('EXIT',name,result.returncode,flush=True)
 if result.returncode: raise RuntimeError(name)
print('COMPLETE',flush=True)

