from pathlib import Path
import urllib.request, hashlib, subprocess, json, re, time
r=Path('research/qwen-fiction-holdout'); r.mkdir(exist_ok=True)
u='https://www.gutenberg.org/files/11/11-0.txt'
req=urllib.request.Request(u,headers={'User-Agent':'qpack-research/1.0'})
b=urllib.request.urlopen(req,timeout=45).read()
s=b.decode('utf-8-sig'); start=s.find('CHAPTER I. Down the Rabbit-Hole', s.find('CHAPTER I. Down the Rabbit-Hole')+1)
if start < 0: start=s.find('Alice was beginning to get very tired')
assert start >= 0
text=s[start:start+60000]
assert len(text)>20000
(r/'alice.txt').write_text(text)
(r/'source.json').write_text(json.dumps({'url':u,'raw_sha256':hashlib.sha256(b).hexdigest(),'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'extraction_start':start,'characters':len(text)},indent=2))
results={}
for name in ['default','calib-4096']:
 cmd=['build/qpack-perplexity','-m',f'research/qwen-validation/artifacts/{name}.qpack','-f',str(r/'alice.txt'),'-c','256','--chunks','16','-t','8']
 print('RUN',name,flush=True); t=time.time(); p=subprocess.run(cmd,capture_output=True,text=True); (r/(name+'.log')).write_text(p.stdout+p.stderr); print('EXIT',name,p.returncode,'seconds',round(time.time()-t,1),flush=True)
 if p.returncode: raise RuntimeError(p.stderr)
 m=re.search(r'(\d+) tokens, mean nll ([0-9.]+), perplexity ([0-9.]+)',p.stdout+p.stderr); assert m, p.stdout+p.stderr
 results[name]={'tokens':int(m[1]),'nll':float(m[2]),'ppl':float(m[3]),'command':cmd}
(r/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)
