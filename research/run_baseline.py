from pathlib import Path
import datetime, hashlib, json, os, platform, re, shlex, subprocess, time
from zoneinfo import ZoneInfo
os.chdir('/workspace/qpack')
r=Path('research'); r.mkdir(exist_ok=True)
d=r/('baseline-'+datetime.datetime.now(ZoneInfo('Europe/Moscow')).strftime('%Y%m%d-%H%M%S')); d.mkdir(); (d/'logs').mkdir(); (d/'artifacts').mkdir()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
commands=[]
def run(label,args):
 print('RUN',label,flush=True); start=time.time()
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (d/'logs'/(label+'.log')).write_text(p.stdout)
 commands.append(dict(label=label,command=shlex.join(args),returncode=p.returncode,seconds=time.time()-start))
 (d/'commands.json').write_text(json.dumps(commands,indent=2))
 print(label,'exit',p.returncode,flush=True)
 if p.returncode: raise RuntimeError(p.stdout[-2000:])
 return p.stdout
ru='''Городская библиотека открывается в девять часов утра. В читальном зале стоят большие столы, а возле окон расположены полки с книгами по истории и естественным наукам. Посетитель может выбрать книгу самостоятельно или обратиться к сотруднику. Для поиска нужного издания важно знать имя автора и название. Если книга находится в другом отделении, её можно заказать заранее и получить через несколько дней.
Вода в природе постоянно перемещается между океаном, сушей и атмосферой. Солнечное тепло вызывает испарение с поверхности водоёмов. Поднимаясь вверх, влажный воздух охлаждается, и водяной пар превращается в маленькие капли. Из этих капель образуются облака. Когда капли становятся достаточно тяжёлыми, они выпадают в виде дождя. Часть воды впитывается в почву, а другая часть возвращается в реки.
Чтобы приготовить овощной суп, сначала нужно вымыть и нарезать картофель, морковь и лук. Затем овощи кладут в кастрюлю с водой и варят на небольшом огне. В конце добавляют соль и зелень. Время приготовления зависит от размера кусочков. Готовый суп оставляют под крышкой на несколько минут, после чего разливают по тарелкам.
Компьютерная программа выполняет последовательность инструкций. Некоторые инструкции работают с числами, другие читают текст или сохраняют результаты в файл. Перед изменением программы полезно записать её исходное поведение. После изменения следует повторить проверку в тех же условиях. Если сравнивать разные входные данные или использовать другое оборудование, будет трудно определить причину полученной разницы.
В школьном саду ученики посадили молодые деревья. Весной они поливают саженцы и следят за состоянием листьев. Осенью собирают опавшую листву и укрывают почву вокруг стволов. Наблюдения записывают в общий журнал. Через несколько лет можно сравнить высоту деревьев и узнать, какие условия оказались наиболее благоприятными для роста.
Перед поездкой на поезде путешественник проверяет время отправления и номер платформы. Билет и документ лучше приготовить заранее. На вокзале необходимо следить за объявлениями, поскольку расписание иногда меняется. В дороге удобно читать книгу или смотреть в окно. По прибытии стоит убедиться, что все личные вещи взяты с собой.
'''
en='''A public library gives people a quiet place to read, study, and discover new subjects. Books are arranged by topic so that visitors can find related material on nearby shelves. A librarian can help someone search for a particular title or choose a useful introduction to an unfamiliar subject. Some books may be borrowed, while rare editions remain in the reading room. Returning borrowed items on time allows other readers to use them as well.
Water moves through the environment in a continuous cycle. Heat from the sun causes liquid water to evaporate from rivers, lakes, and oceans. As warm air rises and cools, water vapor forms tiny droplets. These droplets gather into clouds. Eventually rain or snow carries the water back to the ground. Some water enters the soil, and some flows across the surface into streams. This process supplies fresh water to plants and animals.
Growing a small vegetable garden requires regular attention. The gardener chooses a sunny location, prepares the soil, and plants seeds at the recommended depth. Young plants need water, but too much water can damage their roots. Removing weeds reduces competition for light and nutrients. Keeping a notebook helps the gardener remember when seeds were planted and which methods produced healthy vegetables. Results can differ from one season to the next because weather conditions also matter.
A computer program follows instructions to transform input into output. To evaluate a change, an engineer first records how the original program behaves. The same input is then used with the modified version. Measuring both versions under similar conditions makes the comparison easier to interpret. A single successful example does not establish that every possible case works correctly. Clear records of commands, settings, and results help another person repeat the experiment.
Before taking a train journey, a traveler checks the departure time and keeps the ticket ready. Stations may have several platforms, so it is useful to arrive early and read the information boards. Announcements can explain unexpected delays or changes. During the journey, passengers should keep the passage clear and place large bags in the designated space. Before leaving the train, they check the seat and overhead shelf for belongings.
'''
for lang,text in [('ru',ru),('en',en)]: (d/('eval-'+lang+'.txt')).write_text(text,encoding='utf-8')
m=dict(created_moscow=datetime.datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),source='model.qpack',source_sha256=sha('model.qpack'),source_bytes=Path('model.qpack').stat().st_size,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),platform=platform.platform(),cpus=os.cpu_count(),evaluation='Assistant-authored RU/EN smoke prose. No calibration. ctx=256, max 2 chunks, 8 threads. Preliminary, not representative.',text_sha256={l:sha(d/('eval-'+l+'.txt')) for l in ['ru','en']})
(d/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False))
(d/'logs'/'lscpu.log').write_text(subprocess.check_output(['lscpu'],text=True))
with (r/'HISTORY.md').open('a') as f: f.write('\n## '+m['created_moscow']+'\nStarted '+d.name+'. Baseline f16 versus existing q4 (no GPTQ). Source untouched.\n')
q4=str(d/'artifacts'/'model.q4.qpack')
run('quantize-q4',['./build/qpack-quantize','model.qpack',q4])
results=[]
for label,model in [('f16','model.qpack'),('q4',q4)]:
 run(label+'-selftest',['./build/qpack-cli','-m',model,'--selftest'])
 run(label+'-bench',['./build/qpack-bench','-m',model,'-p','64','-n','128','-t','1,8','-r','3'])
 ppls={}
 for lang in ['ru','en']:
  text=run(label+'-ppl-'+lang,['./build/qpack-perplexity','-m',model,'-f',str(d/('eval-'+lang+'.txt')),'-c','256','--chunks','2','-t','8'])
  match=re.search(r'(\d+) tokens, mean nll ([0-9.]+), perplexity ([0-9.]+)',text)
  if not match: raise RuntimeError('PPL parse failed: '+text)
  ppls[lang]=dict(tokens=int(match[1]),nll=float(match[2]),perplexity=float(match[3]))
 results.append(dict(model=label,bytes=Path(model).stat().st_size,sha256=sha(model),ppl=ppls))
(d/'results.json').write_text(json.dumps(results,indent=2))
lines=['# Baseline f16 vs q4','','Date (Moscow): '+m['created_moscow'],'Branch: '+m['branch']+'; commit: '+m['commit'],'','Preliminary smoke test on saved assistant-authored prose. No calibration or GPTQ. Do not treat this as general quality or compare directly with README corpus scores.','','Perplexity: ctx 256, up to 2 chunks per language, 8 threads. Speed: prompt 64, generation 128, threads 1 and 8, 3 repetitions. See raw benchmark logs for aggregation.','','| Model | MiB | PPL RU | PPL EN |','|---|---:|---:|---:|']
for x in results: lines.append(f"| {x['model']} | {x['bytes']/1048576:.2f} | {x['ppl']['ru']['perplexity']:.4f} | {x['ppl']['en']['perplexity']:.4f} |")
for lang in ['ru','en']:
 a,b=[x['ppl'][lang] for x in results]
 if a['tokens']!=b['tokens']: raise RuntimeError('Token counts differ')
 lines.append(f"{lang.upper()}: {a['tokens']} tokens per model; q4 PPL change {(b['perplexity']/a['perplexity']-1)*100:+.2f}%.")
lines+=['','## Speed f16',(d/'logs/f16-bench.log').read_text(),'','## Speed q4',(d/'logs/q4-bench.log').read_text(),'','commands.json records all commands and exit codes. logs/ holds complete output. Source model not overwritten.']
(d/'SUMMARY.md').write_text('\n'.join(lines))
with (r/'HISTORY.md').open('a') as f: f.write('\nCompleted: ['+d.name+']('+d.name+'/SUMMARY.md). Both selftests passed; see raw logs. No inference code changes.\n')
print('DONE',d,flush=True); print((d/'SUMMARY.md').read_text(),flush=True)

