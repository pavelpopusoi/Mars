"""Оркестратор MARS: план → параллельный поиск → синтез → критика (цикл) → гипотезы → отчёт."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .agents import Analyst, Critic, HypothesisGenerator, Planner, Reader, Searcher, Writer
from .agents.reader import SUPPORTED
from .config import Settings
from .knowledge_graph import KnowledgeGraph
from .llm import LLM, LLMError, make_llm
from .provenance import RunRecorder
from .quality import evaluate
from .report import render_report
from .report_html import render_html
from .schemas import Critique, HypothesisSet, ReportDraft, ResearchPlan, SearchFindings, SubQuestion, Synthesis

log = logging.getLogger("mars")


@dataclass
class RunResult:
    run_dir: Path
    report_path: Path
    report_markdown: str
    plan: ResearchPlan
    findings: list[SearchFindings]
    critique: Critique
    hypotheses: HypothesisSet


class Orchestrator:
    def __init__(self, settings: Settings, llm: LLM | None = None,
                 on_event: Callable[[str], None] | None = None):
        self.settings = settings
        self.llm = llm or make_llm(settings)
        self.on_event = on_event or (lambda msg: log.info(msg))

    async def run(self, question: str, context: str = "",
                  documents: list[str | Path] | None = None) -> RunResult:
        docs = self._check_documents(documents or [])
        rec = RunRecorder(self.settings.runs_dir, question, self.settings)
        self.last_run_dir = rec.dir
        args = (self.llm, self.settings, rec)
        planner, searcher, analyst = Planner(*args), Searcher(*args), Analyst(*args)
        critic, hyp_gen, writer = Critic(*args), HypothesisGenerator(*args), Writer(*args)
        steps: list[str] = []

        def step(msg: str) -> None:
            steps.append(msg)
            self.on_event(msg)

        # 1. План
        plan = await planner.run(question, context)
        rec.save("plan", plan)
        step(f"Planner: {len(plan.sub_questions)} подвопросов")

        # 2-4. Раунды поиск → синтез → критика
        web_task = asyncio.create_task(self._search_many(searcher, plan.sub_questions))
        doc_findings = await self._read_many(Reader(*args), question, docs)
        findings: list[SearchFindings] = doc_findings + await web_task
        if docs:
            step(f"Reader: документов прочитано — {len(doc_findings)} из {len(docs)}")
        step(f"Searcher: раунд 1, утверждений: {sum(len(f.claims) for f in findings)}")
        if not findings:
            raise LLMError("Не удалось собрать ни одного источника — проверьте сеть, ключ и лимиты")
        round_no = 1
        while True:
            synthesis = await analyst.run(plan, findings)
            step(f"Analyst: раунд {round_no}, сила доказательств — {synthesis.evidence_strength}")
            critique = await critic.run(plan, findings, synthesis, round_no, self.settings.max_rounds)
            step(f"Critic: раунд {round_no}, противоречий — {len(critique.contradictions)}, "
                 f"пробелов — {len(critique.gaps)}")
            if not (critique.needs_another_round and critique.follow_up_queries):
                break
            round_no += 1
            follow = SubQuestion(
                id=f"r{round_no}", question="Доисследование пробелов: " + "; ".join(critique.gaps)[:500],
                rationale="Закрыть пробелы, найденные критиком", search_queries=critique.follow_up_queries[:4],
                domain="other")
            findings += await self._search_many(searcher, [follow])
            step(f"Searcher: раунд {round_no} (доисследование), всего утверждений: "
                 f"{sum(len(f.claims) for f in findings)}")

        rec.save("findings", [f.model_dump() for f in findings])
        rec.save("synthesis", synthesis)
        rec.save("critique", critique)

        # 5. Гипотезы
        hypotheses = await hyp_gen.run(plan, findings, synthesis, critique)
        rec.save("hypotheses", hypotheses)
        step(f"Hypothesis: {len(hypotheses.hypotheses)} гипотез")

        # 6. Граф знаний
        graph = KnowledgeGraph.build(findings, critique.contradictions, hypotheses.hypotheses)
        rec.write_text("graph.json", graph.to_json())
        rec.write_text("graph.mmd", graph.to_mermaid())
        rec.write_text("graph.graphml", graph.to_graphml())
        step(f"Knowledge graph: {len(graph.nodes)} узлов, {len(graph.edges)} рёбер")

        # 7. Отчёт
        draft = await writer.run(plan, findings, synthesis, critique, hypotheses)
        rec.save("draft", draft)
        step("Writer: отчёт готов")
        quality = evaluate(plan, findings, critique, hypotheses, draft)
        rec.save("quality", quality.to_dict())
        step(f"Проверка качества: {quality.score}/100")
        meta = {"run_id": rec.dir.name, "run_dir": str(rec.dir), "rounds": round_no, "steps": steps,
                "documents": [p.name for p in docs], "usage": vars(self.llm.usage),
                "quality_score": quality.score, "quality_markdown": quality.markdown()}
        rec.save("meta", meta)
        md = render_report(plan, findings, synthesis, critique, hypotheses, draft, graph.to_mermaid(), meta)
        report_path = rec.write_text("report.md", md)
        rec.write_text("report.html", render_html(md, graph, draft.title))
        u = self.llm.usage
        step(f"Готово: вызовов — {u.calls}, стоимость — ${u.cost_usd:.2f}")
        return RunResult(rec.dir, report_path, md, plan, findings, critique, hypotheses)

    def _check_documents(self, documents: list[str | Path]) -> list[Path]:
        paths = [Path(d) for d in documents]
        if len(paths) > self.settings.max_documents:
            raise LLMError(f"Слишком много документов: {len(paths)} (лимит {self.settings.max_documents})")
        for p in paths:
            if not p.is_file():
                raise LLMError(f"Файл не найден: {p}")
            if p.suffix.lower() not in SUPPORTED:
                raise LLMError(f"Формат {p.suffix} не поддерживается ({', '.join(sorted(SUPPORTED))})")
        return paths

    async def _read_many(self, reader: Reader, question: str, docs: list[Path]) -> list[SearchFindings]:
        async def one(i: int, path: Path) -> SearchFindings | None:
            try:
                return await reader.run(question, path, f"d{i}")
            except LLMError as e:
                log.warning("Документ %s не прочитан: %s", path.name, e)
                return None

        results = await asyncio.gather(*(one(i, p) for i, p in enumerate(docs, 1)))
        return [r for r in results if r is not None]

    async def _search_many(self, searcher: Searcher, sqs: list[SubQuestion]) -> list[SearchFindings]:
        sem = asyncio.Semaphore(self.settings.max_parallel_searches)

        async def one(sq: SubQuestion) -> SearchFindings | None:
            async with sem:
                try:
                    return await searcher.run(sq)
                except LLMError as e:  # один упавший подвопрос не роняет всё исследование
                    log.warning("Поиск по %s не удался: %s", sq.id, e)
                    return None

        results = await asyncio.gather(*(one(sq) for sq in sqs))
        return [r for r in results if r is not None]


def rebuild_report(run_dir: str | Path) -> Path:
    """Пересобрать report.md из сохранённых артефактов без вызова модели."""
    d = Path(run_dir)

    def load(name: str):
        return json.loads((d / f"{name}.json").read_text(encoding="utf-8"))

    plan = ResearchPlan.model_validate(load("plan"))
    findings = [SearchFindings.model_validate(f) for f in load("findings")]
    synthesis = Synthesis.model_validate(load("synthesis"))
    critique = Critique.model_validate(load("critique"))
    hypotheses = HypothesisSet.model_validate(load("hypotheses"))
    draft = ReportDraft.model_validate(load("draft"))
    meta = load("meta")
    graph = KnowledgeGraph.build(findings, critique.contradictions, hypotheses.hypotheses)
    md = render_report(plan, findings, synthesis, critique, hypotheses, draft, graph.to_mermaid(), meta)
    path = d / "report.md"
    path.write_text(md, encoding="utf-8")
    (d / "report.html").write_text(render_html(md, graph, draft.title), encoding="utf-8")
    return path
