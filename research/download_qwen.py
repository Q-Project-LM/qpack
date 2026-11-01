from pathlib import Path
import urllib.request,json,hashlib,os
root=Path('research/qwen-validation'); root.mkdir(exist_ok=True); hf=root/'hf'; hf.mkdir(exist_ok=True)
headers={'User-Agent':'qpack-quantization-research/1.0'}
def get(url): return urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120)
url='https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct?blobs=true'
with get(url) as f: meta=json.load(f)
revision=meta['sha']; print('REVISION',revision,flush=True)
names=['config.json','tokenizer.json','model.safetensors']; items={x['rfilename']:x for x in meta['siblings']}; assert all(n in items for n in names)
manifest={'repository':'Qwen/Qwen2.5-0.5B-Instruct','revision':revision,'source_api':url,'files':{}}
for n in names:
 p=hf/n; part=hf/(n+'.part'); item=items[n]; expected=item.get('lfs',{}).get('sha256'); expected_size=item.get('size'); print('DOWNLOAD',n,expected_size,flush=True); h=hashlib.sha256(); total=0
 with get('https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/'+revision+'/'+n) as src,part.open('wb') as dst:
  while True:
   chunk=src.read(1024*1024)
   if not chunk: break
   dst.write(chunk); h.update(chunk); total+=len(chunk)
 if expected_size is not None: assert total==expected_size,(n,total,expected_size)
 if expected: assert h.hexdigest()==expected,(n,'sha mismatch')
 os.replace(part,p); manifest['files'][n]={'bytes':total,'sha256':h.hexdigest(),'hub_sha256':expected}; (root/'source.json').write_text(json.dumps(manifest,indent=2)); print('DONE',n,total,flush=True)
print('COMPLETE',flush=True)
