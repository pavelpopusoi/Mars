from __future__ import annotations

from ..schemas import ResearchPlan, SearchFindings, Synthesis
from .base import Agent


def findings_digest(findings: list[SearchFindings]) -> str:
    lines = []
    for f in findings:
        lines.append(f"## {f.sub_question_id}: {f.summary}")
        for s in f.sources:
            lines.append(f"  [{s.id}] {s.title} ({s.publisher}, {s.year}, {s.source_type}, rel={s.reliability})")
        for c in f.claims:
            lines.append(f"  ({c.id}) {c.statement} — src={c.source_ids} conf={c.confidence}")
    return "\n".join(lines)


class Analyst(Agent):
    name = "analyst"
    role = (
        "You are the Synthesis Analyst. Integrate evidence across sub-questions and disciplines: identify key "
        "findings, points of consensus, and what remains open. Weigh evidence by source reliability and "
        "independence, not by count. Rate overall evidence strength."
    )

    async def run(self, plan: ResearchPlan, findings: list[SearchFindings]) -> Synthesis:
        prompt = (
            f"ГЛАВНЫЙ ВОПРОС: {plan.main_question}\n\nДОКАЗАТЕЛЬСТВА:\n{findings_digest(findings)}\n\n"
            "Синтезируй знания."
        )
        return await self.ask(prompt, Synthesis)
