
## 2026-11-02T00:16:40.883826+03:00
Started baseline-20261102-001640. Baseline f16 versus existing q4 (no GPTQ). Source untouched.

Completed: [baseline-20261102-001640](baseline-20261102-001640/SUMMARY.md). Both selftests passed; see raw logs. No inference code changes.

Technical note: initial runner creation had a newline escaping SyntaxError before any model command ran. Output preserved in baseline-attempt1.log; fixed runner completed all commands successfully. Benchmark aggregation and both selftests subsequently inspected.

## 2026-11-02T00:37:08.605482+03:00
Started matrix-20261102-003657: 18 main configurations; separate RU/EN Wikipedia calibration and evaluation pages. Original f16 unchanged.

Matrix preparation initially hit HTTP429; snapshots cached and requests retried. Initial matrix runner then misparsed successful self-test output (expected selftest, actual self-test). Original results and logs retained; resume_matrix.py fixes parsing and reuses completed quantizations.

Matrix preparation initially hit HTTP429; snapshots cached and requests retried. Initial matrix runner then misparsed successful self-test output (expected selftest, actual self-test). Original results and logs retained; resume_matrix.py fixes parsing and reuses completed quantizations.

Finished matrix-20261102-003657: 18/18 variants completed successfully; see SUMMARY.md and failure logs.

Final audit passed: all 18 configurations, 18 selftests 5/5, equal 2040 primary and 510 pilot tokens per language, all weight hashes and source identity verified. See research/matrix-20261102-003657/REPORT.html.

Verified final numeric conclusions saved to research/matrix-20261102-003657/CONCLUSIONS.md. The definitive q4m GPTQ deltas are -0.33% RU / +0.75% EN; any differing preliminary chat readings are superseded by audited results.json. All calibration logs show 8176 tokens. Binary hashes unchanged.

## Плотный поиск масштаба
Отклонён: q4 GPTQ act-order RU PPL 23.5632 против 23.5392, EN 11.8903 против 11.8974; selftest 5/5. Код в основном квантизаторе не менялся. Следующая попытка: расширенный диапазон.

## Расширенный диапазон масштаба
Отклонён: основная выборка RU/EN 23.4790/11.8670 лучше эталона 23.5392/11.8974, но независимая короткая выборка ухудшилась: 16.3175/12.1479 против 16.1425/12.1263. Selftest 5/5. Следующая попытка: умеренное расширение.

## Умеренно расширенный диапазон
Отклонён: основная выборка RU/EN 23.4906/11.8663 улучшилась относительно 23.5392/11.8974, но короткая RU ухудшилась до 16.2614 против 16.1425; EN 12.1001 против 12.1263. Selftest 5/5. Основной исходник не изменён.

## 2026-11-02T01:34:15.864676+03:00
Started matrix-20261102-013404: 18 main configurations; separate RU/EN Wikipedia calibration and evaluation pages. Original f16 unchanged.

Finished matrix-20261102-013404: 18/18 variants completed successfully; see SUMMARY.md and failure logs.

## Повтор 18 конфигураций после трёх отклонённых попыток
Результаты: research/matrix-20261102-013404/SUMMARY.md; аудит: FINAL_AUDIT.json. Все selftest 5/5. Совпадение текстов: {'calib.txt': True, 'eval-ru.txt': True, 'eval-en.txt': True}. Основной алгоритм не менялся.

## Регуляризация GPTQ: 0,5% и 2%
Эталон (1%): Wikipedia RU/EN 23.5392/11.8974; короткие RU/EN 16.1425/12.1263. 0,5%: 23.5714/11.8714 и 16.0769/12.0626; 2%: 23.5803/11.9067 и 16.2978/12.0449. Оба selftest завершились успешно. Оба варианта ухудшили минимум одну метрику; основной код не изменён. Тексты короткие; более широкий независимый корпус нужен для оценки практической значимости.

## Расширенная отложенная выборка
Продолжения статей после первых 12000 символов, без статей калибровки; контекст 256, до 20 блоков, 8 потоков. Это те же статьи, что ранее, поэтому выборка не полностью независима по тематике. См. research/extended-holdout/results.json и manifest.json.

На продолжениях статей три кандидата (damp005, wide и medium) уступили сохранённой версии на обоих языках; итоги в extended-holdout/SUMMARY.md.

## Взвешивание масштаба GPTQ
На расширенной выборке эталон q4 GPTQ act-order RU/EN 26.6559/9.8755; sqrt(Hii) 26.7799/9.9003; равномерные веса 26.8102/9.9227. Оба кандидата хуже на обоих языках, selftest успешен; основной алгоритм не изменён. Полные команды и три выборки — в attempt-sqrtimp и attempt-uniformimp.
