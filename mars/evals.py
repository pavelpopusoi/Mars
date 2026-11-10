"""Оценка качества по содержанию: судья для одного запуска и прогон набора эталонных вопросов."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .agents.analyst import findings_digest
from .agents.judge import Judge
from .config import Settings
from .llm import LLM, LLMError, make_llm
from .orchestrator import Orchestrator
from .quality import evaluate_run
from .schemas import ClaimCheck, JudgeVerdict, SearchFindings

CRITERIA = {
    "answers_question": "Отвечает на вопрос",
    "evidence_support": "Опора на доказательства",
    "contradiction_handling": "Разбор противоречий",
    "hypothesis_quality": "Качество гипотез",
    "clarity": "Ясность",
}
SUPPORT_WEIGHT = {"supported": 1.0, "partially_supported": 0.5, "not_supported": 0.0}


@dataclass
class JudgeResult:
    verdict: JudgeVerdict
    claim_checks: list[ClaimCheck] = field(default_factory=list)

    @property
    def score(self) -> int:
        """0..100: среднее по пяти критериям."""
        v = self.verdict
        return round(sum(getattr(v, k) for k in CRITERIA) / len(CRITERIA) * 20)

    @property
    def claim_support(self) -> float | None:
        """Доля проверенных утверждений, подтверждённых источником (недоступные источники не считаются)."""
        rated = [SUPPORT_WEIGHT[c.verdict] for c in self.claim_checks if c.verdict in SUPPORT_WEIGHT]
        return round(sum(rated) / len(rated), 2) if rated else None

    def to_dict(self) -> dict:
        return {"score": self.score, "claim_support": self.claim_support, "verdict": self.verdict.model_dump(),
                "claim_checks": [c.model_dump() for c in self.claim_checks]}

    def table(self) -> str:
        v = self.verdict
        lines = [f"Оценка судьи: {self.score}/100", ""]
        lines += [f"  {label}: {getattr(v, key)}/5" for key, label in CRITERIA.items()]
        if self.claim_checks:
            lines += ["", f"Сверка с источниками: подтверждено {self.claim_support}"
                          f" (проверено утверждений: {len(self.claim_checks)})"]
            lines += [f"  [{c.verdict}] {c.claim_id}: {c.explanation}" for c in self.claim_checks]
        for title, items in [("Сильные стороны", v.strengths), ("Слабые стороны", v.weaknesses),
                             ("Не подтверждено", v.unsupported_statements)]:
            if items:
                lines += ["", f"{title}:"] + [f"  - {x}" for x in items]
        lines += ["", v.verdict]
        return "\n".join(lines)


async def judge_run(run_dir: str | Path, settings: Settings, llm: LLM | None = None,
                    verify_claims: int = 0) -> JudgeResult:
    """Оценить готовый запуск. verify_claims — сколько утверждений сверить с первоисточниками в вебе."""
    d = Path(run_dir)
    for name in ("report.md", "findings.json", "manifest.json"):
        if not (d / name).is_file():
            raise LLMError(f"В папке {d} нет {name} — это не папка запуска MARS")
    findings = [SearchFindings.model_validate(f)
                for f in json.loads((d / "findings.json").read_text(encoding="utf-8"))]
    question = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["question"]
    judge = Judge(llm or make_llm(settings), settings)
    verdict = await judge.review(question, (d / "report.md").read_text(encoding="utf-8"), findings_digest(findings))

    sources = {s.id: s for f in findings for s in f.sources}
    checkable = [(c, sources[c.source_ids[0]]) for f in findings for c in f.claims
                 if c.source_ids and c.source_ids[0] in sources and sources[c.source_ids[0]].url.startswith("http")]
    checkable.sort(key=lambda pair: -pair[0].confidence)  # самые уверенные утверждения важнее всего проверить

    async def one(claim, source) -> ClaimCheck:
        try:
            return await judge.check_claim(claim, source)
        except LLMError as e:
            return ClaimCheck(claim_id=claim.id, verdict="source_unavailable", explanation=str(e))

    sem = asyncio.Semaphore(settings.max_parallel_searches)

    async def limited(pair) -> ClaimCheck:
        async with sem:
            return await one(*pair)

    checks = list(await asyncio.gather(*(limited(p) for p in checkable[:max(0, verify_claims)])))
    result = JudgeResult(verdict, checks)
    (d / "judge.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


async def run_eval(questions_file: str | Path, settings: Settings, out_dir: str | Path,
                   llm: LLM | None = None, verify_claims: int = 0, on_event=print) -> dict:
    """Прогнать набор эталонных вопросов и собрать сводку."""
    spec = yaml.safe_load(Path(questions_file).read_text(encoding="utf-8")) or {}
    questions = spec.get("questions") or []
    if not questions:
        raise LLMError(f"В {questions_file} нет списка questions")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = settings.model_copy(update={"runs_dir": str(out / "runs")})
    llm = llm or make_llm(settings)
    rows: list[dict] = []
    for i, q in enumerate(questions, 1):
        qid, text = q.get("id", f"e{i}"), q["question"]
        on_event(f"[{i}/{len(questions)}] {qid}: {text}")
        row: dict = {"id": qid, "question": text}
        try:
            result = await Orchestrator(settings, llm=llm, on_event=lambda m: None).run(text)
            judged = await judge_run(result.run_dir, settings, llm, verify_claims)
            lowered = result.report_markdown.lower()
            must = [m for m in q.get("must_mention", [])]
            hit = [m for m in must if m.lower() in lowered]
            row.update(run_dir=str(result.run_dir), structure=evaluate_run(result.run_dir).score,
                       judge=judged.score, claim_support=judged.claim_support,
                       coverage=round(len(hit) / len(must), 2) if must else None,
                       missing=[m for m in must if m not in hit], weaknesses=judged.verdict.weaknesses)
        except LLMError as e:
            row["error"] = str(e)
        rows.append(row)

    ok = [r for r in rows if "error" not in r]

    def mean(key: str) -> float | None:
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    summary = {"questions": len(rows), "failed": len(rows) - len(ok), "mean_structure": mean("structure"),
               "mean_judge": mean("judge"), "mean_claim_support": mean("claim_support"),
               "mean_coverage": mean("coverage"), "cost_usd": llm.usage.cost_usd, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def cell(v) -> str:
        return "—" if v is None else str(v)

    md = ["# Сводка оценки MARS", "",
          f"Вопросов: {len(rows)}, с ошибкой: {len(rows) - len(ok)}, стоимость: ${llm.usage.cost_usd:.2f}", "",
          "| id | Структура /100 | Судья /100 | Подтверждено источниками | Покрытие тем | Чего не хватает |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['id']} | ошибка: {r['error']} | | | | |")
        else:
            md.append(f"| {r['id']} | {r['structure']} | {r['judge']} | {cell(r['claim_support'])} | "
                      f"{cell(r['coverage'])} | {', '.join(r['missing']) or '—'} |")
    md += [f"| **среднее** | {cell(summary['mean_structure'])} | {cell(summary['mean_judge'])} | "
           f"{cell(summary['mean_claim_support'])} | {cell(summary['mean_coverage'])} | |", ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return summary
