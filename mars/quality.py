"""Автопроверка качества запуска без вызова модели: целостность ссылок, покрытие, опора на источники."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .schemas import Critique, HypothesisSet, ReportDraft, ResearchPlan, SearchFindings


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    weight: int = 1


@dataclass
class QualityReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def score(self) -> int:
        total = sum(c.weight for c in self.checks) or 1
        return round(100 * sum(c.weight for c in self.checks if c.passed) / total)

    def to_dict(self) -> dict:
        return {"score": self.score, "checks": [asdict(c) for c in self.checks]}

    def table(self) -> str:
        lines = [f"Оценка качества: {self.score}/100", ""]
        lines += [f"[{'OK' if c.passed else '!!'}] {c.name}: {c.detail}" for c in self.checks]
        return "\n".join(lines)

    def markdown(self) -> str:
        rows = ["| Проверка | Результат | Детали |", "|---|---|---|"]
        rows += [f"| {c.name} | {'пройдена' if c.passed else '**не пройдена**'} | {c.detail} |" for c in self.checks]
        return "\n".join([f"Автоматическая оценка: **{self.score}/100**", "", *rows])


def evaluate(plan: ResearchPlan, findings: list[SearchFindings], critique: Critique,
             hypotheses: HypothesisSet, draft: ReportDraft) -> QualityReport:
    sources = {s.id: s for f in findings for s in f.sources}
    claims = {c.id: c for f in findings for c in f.claims}
    q = QualityReport()

    def add(name: str, bad: list[str], ok_detail: str, weight: int = 1) -> None:
        shown = ", ".join(bad[:5]) + (f" и ещё {len(bad) - 5}" if len(bad) > 5 else "")
        q.checks.append(Check(name, not bad, ok_detail if not bad else shown, weight))

    # 1. Ссылки в тексте отчёта ведут на существующие источники
    text = " ".join([draft.executive_summary, draft.conclusion, *(s.body_markdown for s in draft.sections)])
    cited = set(re.findall(r"\[([\w\-]+-s\d+)\]", text))
    add("Ссылки в тексте существуют", sorted(cited - sources.keys()), f"{len(cited)} ссылок, все найдены", 3)
    q.checks.append(Check("В тексте есть ссылки на источники", bool(cited),
                          f"процитировано источников: {len(cited & sources.keys())} из {len(sources)}", 2))

    # 2. Утверждения опираются на существующие источники
    orphan = [c.id for c in claims.values() if not c.source_ids or set(c.source_ids) - sources.keys()]
    add("У каждого утверждения есть источник", orphan, f"{len(claims)} утверждений", 3)

    # 3. У источников настоящие адреса
    bad_url = [s.id for s in sources.values() if not re.match(r"(https?|file)://\S+", s.url)]
    add("У источников корректные адреса", bad_url, f"{len(sources)} источников", 2)

    # 4. Каждый подвопрос плана получил утверждения
    covered = {f.sub_question_id for f in findings if f.claims}
    add("Все подвопросы покрыты", [s.id for s in plan.sub_questions if s.id not in covered],
        f"{len(plan.sub_questions)} подвопросов", 2)

    # 5. Противоречия и гипотезы ссылаются на существующие утверждения
    add("Противоречия ссылаются на утверждения",
        [x.id for x in critique.contradictions if len(set(x.claim_ids) & claims.keys()) < 2],
        f"{len(critique.contradictions)} противоречий")
    add("Гипотезы опираются на утверждения",
        [h.id for h in hypotheses.hypotheses if not set(h.supporting_claim_ids) & claims.keys()],
        f"{len(hypotheses.hypotheses)} гипотез")
    add("Гипотезы проверяемы", [h.id for h in hypotheses.hypotheses
                               if len(h.test_method) < 15 or len(h.falsifiable_by) < 15],
        "у каждой есть метод проверки и условие опровержения")

    # 6. Разнообразие и надёжность источников
    domains = {re.sub(r"^www\.", "", re.sub(r"^\w+://([^/]+).*", r"\1", s.url)) for s in sources.values()}
    q.checks.append(Check("Источники из разных мест", len(domains) >= min(3, len(plan.sub_questions)),
                          f"независимых сайтов и файлов: {len(domains)}"))
    strong = [s for s in sources.values() if s.reliability >= 0.7]
    q.checks.append(Check("Есть надёжные источники", len(strong) * 3 >= len(sources) > 0,
                          f"с надёжностью от 0.7: {len(strong)} из {len(sources)}"))
    return q


def evaluate_run(run_dir: str | Path) -> QualityReport:
    d = Path(run_dir)

    def load(name: str):
        return json.loads((d / f"{name}.json").read_text(encoding="utf-8"))

    return evaluate(ResearchPlan.model_validate(load("plan")),
                    [SearchFindings.model_validate(f) for f in load("findings")],
                    Critique.model_validate(load("critique")),
                    HypothesisSet.model_validate(load("hypotheses")),
                    ReportDraft.model_validate(load("draft")))
