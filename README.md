# MARS — Multi-Agent Research System

Автономная система глубокого исследования. На вход — сложный вопрос (научный, технический, бизнес или
междисциплинарный). На выход — структурированный отчёт с источниками, разбором противоречий, гипотезами,
графом знаний и полной трассой шагов для воспроизведения.

```
Вопрос ─► Planner ─► Searcher ×N (web search) + Reader (ваши файлы) ─► Analyst ─► Critic ─┐
                          ▲                                                       │ пробелы?
                          └────────────── доисследование (раунд 2..max) ◄─────────┘
                                                                                  ▼
                               Hypothesis ─► Knowledge Graph ─► Writer ─► report.md
```

## Агенты

| Агент | Задача | Выход |
|---|---|---|
| **Planner** | Разбивает вопрос на 3–6 подвопросов, формулирует поисковые запросы и критерии успеха | `ResearchPlan` |
| **Searcher** | Ищет и читает источники (серверные `web_search` / `web_fetch` Claude), извлекает атомарные утверждения | `SearchFindings` |
| **Reader** | Читает ваши файлы (PDF, TXT, MD, CSV, JSON) и извлекает из них утверждения как из источника | `SearchFindings` |
| **Analyst** | Синтезирует знания между подвопросами и дисциплинами, оценивает силу доказательств | `Synthesis` |
| **Critic** | Находит противоречия, слабые утверждения, пробелы; решает, нужен ли ещё раунд | `Critique` |
| **Hypothesis** | Генерирует проверяемые гипотезы с методом проверки и условием опровержения | `HypothesisSet` |
| **Writer** | Пишет отчёт со ссылками `[id]` на источники | `ReportDraft` |

Все обмены между агентами — валидируемые Pydantic-схемы (`mars/schemas.py`), модель отвечает через
structured outputs, поэтому ответы всегда парсятся.

## Быстрый старт

```bash
pip install -e ".[dev]"

# 1. Демо без ключа и сети (детерминированный mock)
mars run "Как микропластик влияет на здоровье человека?" --mock

# 2. Реальное исследование
export ANTHROPIC_API_KEY=sk-ant-...      # или `ant auth login`
mars run "Сможет ли натрий-ионная технология вытеснить LFP в стационарных накопителях к 2030?"

# Свои документы как источники, лимит раундов и потолок стоимости
mars run "..." --doc report.pdf --doc notes.md --rounds 3 --max-cost 10

# Оценить стоимость, не вызывая модель
mars estimate --sub-questions 5 --docs 2

# Проверить качество готового запуска без вызова модели (ссылки, источники, покрытие)
mars check runs/<папка_запуска> --min-score 80

# Веб-интерфейс: форма, прогресс шагов, отчёт с графом — http://127.0.0.1:8000
pip install -e ".[web]" && mars serve

# Пересобрать отчёт из артефактов без вызова модели
mars rebuild runs/<папка_запуска>
```

Docker: `docker build -t mars . && docker run -e ANTHROPIC_API_KEY -v $PWD/runs:/app/runs mars run "вопрос"`

## Что получается на выходе

Каждый запуск — папка `runs/<время>-<вопрос>/`:

| Файл | Содержимое |
|---|---|
| `report.md` | Итоговый отчёт: резюме, выводы, разделы, таблица противоречий, гипотезы, граф (Mermaid), источники, шаги |
| `report.html` | Тот же отчёт одним файлом с интерактивным графом знаний (узлы можно двигать, наведение показывает связи) |
| `manifest.json` | Версия MARS, вопрос, полный конфиг (модели, effort, лимиты) |
| `trace.jsonl` | Каждый вызов агента: промпт, SHA-256, модель, длительность, ответ, ошибки |
| `plan.json` … `draft.json` | Промежуточные артефакты каждого шага |
| `graph.json` / `graph.mmd` / `graph.graphml` | Граф знаний (GraphML открывается в Gephi, yEd, Cytoscape) |
| `quality.json` | Автопроверка: оценка 0–100 и список проверок (битые ссылки, утверждения без источника, непокрытые подвопросы) |
| `meta.json` | Раунды, шаги, расход токенов и стоимость в долларах |

Пример готового прогона (mock): [`examples/sample_run/report.md`](examples/sample_run/report.md).

## Конфигурация

`config/mars.yaml` — модель, `effort` и `max_tokens` для каждого агента, число раундов, параллельность,
лимит веб-поисков. Переменные окружения: `ANTHROPIC_API_KEY`, `MARS_CONFIG`, `MARS_BACKEND=mock`.

По умолчанию все агенты работают на `claude-fable-5-2` (самая сильная модель, $8 / $40 за 1M токенов).
Включены:

* **серверные фолбэки** (`use_server_fallbacks`): если модель откажется отвечать, API сам повторит запрос
  на подходящей модели;
* **кэширование промптов** (`prompt_caching`): повторяющийся контекст читается из кэша примерно за 10% цены.

**Экономия:** самый объёмный агент — Searcher. Для него и Reader можно поставить `claude-opus-5-5`
($4 / $20) или `claude-sonnet-5-5` ($2 / $10), оставив Fable для Analyst, Critic и Hypothesis.
Фактическая стоимость каждого запуска пишется в `meta.json` и в последний шаг лога.

## Использование как библиотеки

```python
import asyncio
from mars.config import load_settings
from mars.orchestrator import Orchestrator

result = asyncio.run(Orchestrator(load_settings()).run("Ваш вопрос"))
print(result.report_path, len(result.critique.contradictions))
```

## Структура

```
mars/
  schemas.py          # контракты между агентами (Pydantic)
  llm.py              # ClaudeLLM (стриминг, structured outputs, web search, фолбэки) + MockLLM
  agents/             # planner, searcher, reader, analyst, critic, hypothesis, writer
  orchestrator.py     # пайплайн и цикл раундов
  knowledge_graph.py  # граф знаний → JSON / Mermaid / GraphML
  report.py           # детерминированная сборка report.md
  report_html.py      # report.html с интерактивным графом
  quality.py          # автопроверка качества запуска
  costs.py            # цены моделей, оценка и учёт стоимости
  web/app.py          # веб-интерфейс (FastAPI + SSE)
  provenance.py       # папка запуска, trace.jsonl
  cli.py              # mars run / estimate / check / serve / rebuild
config/mars.yaml
docs/ARCHITECTURE.md
tests/                # офлайн-тесты на MockLLM
examples/
```

## Тесты

```bash
pytest -q        # 11 тестов, без сети
ruff check mars tests
```

Подробности архитектуры, решения и план развития — в [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
