"""Сборка финального Markdown-отчёта из артефактов (детерминированно — можно пересобрать без модели)."""
from __future__ import annotations

from .schemas import Critique, HypothesisSet, ReportDraft, ResearchPlan, SearchFindings, Synthesis


def render_report(plan: ResearchPlan, findings: list[SearchFindings], synthesis: Synthesis, critique: Critique,
                  hypotheses: HypothesisSet, draft: ReportDraft, mermaid: str, meta: dict) -> str:
    sources = {s.id: s for f in findings for s in f.sources}
    claims = {c.id: c for f in findings for c in f.claims}
    out: list[str] = [f"# {draft.title}", ""]
    out += [f"> **Вопрос:** {plan.main_question}", ">",
            f"> Запуск `{meta.get('run_id', '')}` · раундов: {meta.get('rounds', '?')} · "
            f"источников: {len(sources)} · утверждений: {len(claims)} · "
            f"сила доказательств: **{synthesis.evidence_strength}**", ""]
    out += ["## Резюме", "", draft.executive_summary, ""]
    out += ["## Ключевые выводы", ""] + [f"- {k}" for k in synthesis.key_findings] + [""]
    for s in draft.sections:
        out += [f"## {s.title}", "", s.body_markdown, ""]

    out += ["## Противоречия", ""]
    if critique.contradictions:
        out += ["| # | Утверждения | Суть | Причина | Разрешение | Важность |", "|---|---|---|---|---|---|"]
        for x in critique.contradictions:
            out.append(f"| {x.id} | {', '.join(x.claim_ids)} | {x.description} | {x.likely_cause} | "
                       f"{x.resolution} | {x.severity} |")
    else:
        out.append("Существенных противоречий не найдено.")
    out.append("")

    out += ["## Гипотезы", ""]
    for h in hypotheses.hypotheses:
        out += [f"### {h.id}. {h.statement}", "",
                f"- Новизна: {h.novelty:.2f} · Правдоподобие: {h.plausibility:.2f}",
                f"- Опора: {', '.join(h.supporting_claim_ids)}",
                f"- Как проверить: {h.test_method}",
                f"- Опровергается, если: {h.falsifiable_by}", ""]

    out += ["## Граф знаний", "", "```mermaid", mermaid, "```", ""]
    out += ["## Открытые вопросы и пробелы", ""]
    out += [f"- {q}" for q in synthesis.open_questions + critique.gaps] or ["- —"]
    out += ["", "## Ограничения", ""] + [f"- {x}" for x in draft.limitations] + [""]
    out += ["## Заключение", "", draft.conclusion, ""]

    out += ["## Источники", "", "| id | Источник | Тип | Год | Надёжность |", "|---|---|---|---|---|"]
    for s in sources.values():
        out.append(f"| {s.id} | [{s.title}]({s.url}) {('— ' + s.publisher) if s.publisher else ''} | "
                   f"{s.source_type} | {s.year or '—'} | {s.reliability:.2f} |")
    out.append("")

    if meta.get("quality_markdown"):
        out += ["## Проверка качества", "", meta["quality_markdown"], ""]

    out += ["## Воспроизводимость", "",
            "Все шаги записаны в папке запуска:", "",
            "| Файл | Что внутри |", "|---|---|",
            "| `manifest.json` | версия MARS, вопрос, полный конфиг (модели, effort, лимиты) |",
            "| `trace.jsonl` | каждый вызов агента: промпт, хэши, модель, длительность, ответ |",
            "| `plan.json`, `findings.json`, `synthesis.json`, `critique.json`, `hypotheses.json` | артефакты шагов |",
            "| `graph.json`, `graph.mmd`, `graph.graphml` | граф знаний |", "",
            f"Пересобрать отчёт без вызова модели: `mars rebuild {meta.get('run_dir', '<run_dir>')}`", "",
            "Шаги пайплайна:", ""]
    out += [f"{i}. {s}" for i, s in enumerate(meta.get("steps", []), 1)]
    out.append("")
    return "\n".join(out)
