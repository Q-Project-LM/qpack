from pathlib import Path
import json,hashlib,html,collections
r=Path('research'); d=max([p for p in r.glob('matrix-*') if (p/'manifest.json').exists()],key=lambda p:p.stat().st_mtime)
x=json.loads((d/'results.json').read_text()); m=json.loads((d/'manifest.json').read_text())
assert len(x)==18,(len(x),'not finished')
assert {b['name'] for b in x}=={b['name'] for b in m['plan']}
assert all(b['status']=='ok' and b['selftest']=='5/5' for b in x)
assert hashlib.sha256(Path('model.qpack').read_bytes()).hexdigest()==m['source_sha256']
for b in x:
 for lang in ['ru','en']:
  assert b['ppl'][lang]['tokens']==2040
  assert b['pilot_ppl'][lang]['tokens']==510
 assert set(b['bench'])=={'1','8','16'}
 assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256']
a=x[0]; rows=[]; plain=[]
plain.append('variant MiB dRU% dEN% gen8 gen16')
for b in x:
 dr,de=[(b['ppl'][l]['perplexity']/a['ppl'][l]['perplexity']-1)*100 for l in ['ru','en']]
 vals=[b['name'],f"{b['bytes']/1048576:.2f}",f'{dr:+.2f}%',f'{de:+.2f}%',str(b['bench']['8']['decode']),str(b['bench']['16']['decode'])]
 rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
 plain.append(' '.join(vals))
report='<!doctype html><meta charset="utf-8"><title>qpack — результаты квантизации</title><style>body{font:14px system-ui;margin:24px;color:#17212e;background:#f6f8fa}h1{font-size:22px}table{border-collapse:collapse;background:white;font-size:12px}td,th{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}p{max-width:850px}a{color:#1262be}</style><h1>qpack: 18 из 18 конфигураций проверены</h1><p>У всех selftest 5/5. Размер — МиБ. ΔPPL относительно f16: меньше лучше. Генерация — токенов/с на 8 и 16 потоках.</p><table><tr>'+''.join('<th>'+h+'</th>' for h in ['Вариант','МиБ','ΔPPL RU','ΔPPL EN','Ген. 8','Ген. 16'])+'</tr>'+''.join(rows)+'</table><p>Wikipedia: 2040 оценённых токенов на язык; отдельно проверены прежние тексты (510 токенов на язык). Калибровка: 8176 токенов из других статей RU/EN. Бенчмарк: синтетические токены, prompt 64, generation 128, лучший из трёх запусков для каждой фазы. Результаты предварительные; небольшое снижение PPL не доказывает общего улучшения модели.</p><p><a href="SUMMARY.md">Полный отчёт</a> · <a href="results.json">Данные JSON</a> · <a href="commands.json">Команды</a> · <a href="data/sources.json">Источники текстов</a></p>'
(d/'REPORT.html').write_text(report); (d/'COMPACT.txt').write_text('\n'.join(plain))
audit=dict(configurations=18,selftests_passed=18,primary_tokens_per_language=2040,pilot_tokens_per_language=510,source_unchanged=True,all_artifact_hashes_verified=True,threads=[1,8,16],script_snapshots_verified=dict(initial=hashlib.sha256((d/'runner-initial.py').read_bytes()).hexdigest()==m['script_sha256'],resumed=hashlib.sha256((d/'runner-resumed.py').read_bytes()).hexdigest()==m['resume_script_sha256']))
assert all(audit['script_snapshots_verified'].values())
(d/'AUDIT.json').write_text(json.dumps(audit,indent=2)); print('AUDIT PASSED',d)
with (r/'HISTORY.md').open('a') as f: f.write('\nFinal audit passed: all 18 configurations, 18 selftests 5/5, equal 2040 primary and 510 pilot tokens per language, all weight hashes and source identity verified. See '+str(d/'REPORT.html')+'.\n')
with (r/'README.md').open('a') as f: f.write('\n## Полная матрица\nСерия '+d.name+': 18 конфигураций (f16, q8, q4/q4m × plain/calib/GPTQ/GPTQ-act-order × head q8/q4). Итог: REPORT.html, SUMMARY.md, results.json и AUDIT.json. Данные и атрибуция Wikipedia сохранены в data/. Произвольные пользовательские списки --keep-q8 не перебирались. Ошибки подготовки и парсера архивированы отдельно; итоговые статусы находятся в results.json.\n')
print('\n'.join(plain))

