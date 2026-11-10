from __future__ import annotations

from ..schemas import Critique, HypothesisSet, ReportDraft, ResearchPlan, SearchFindings, Synthesis
from .analyst import findings_digest
from .base import Agent


class Writer(Agent):
    name = "writer"
    role = (
        "You are the Report Writer. Write a clear, well-structured research report for an expert reader. "
        "Cite sources inline as [source_id] using only ids that exist. State uncertainty explicitly, discuss "
        "contradictions honestly, and keep claims proportional to evidence. 4-7 sections."
    )

    async def run(self, plan: ResearchPlan, findings: list[SearchFindings], synthesis: Synthesis,
                  critique: Critique, hypotheses: HypothesisSet) -> ReportDraft:
        prompt = (
            f"ГЛАВНЫЙ ВОПРОС: {plan.main_question}\nИнтерпретация: {plan.interpretation}\n\n"
            f"ДОКАЗАТЕЛЬСТВА:\n{findings_digest(findings)}\n\nСИНТЕЗ:\n{synthesis.model_dump_json(indent=1)}\n\n"
            f"ПРОТИВОРЕЧИЯ И ПРОБЕЛЫ:\n{critique.model_dump_json(indent=1)}\n\n"
            f"ГИПОТЕЗЫ:\n{hypotheses.model_dump_json(indent=1)}\n\n"
            "Напиши отчёт. Разделы про противоречия, гипотезы и источники будут добавлены автоматически — "
            "не дублируй их, сосредоточься на анализе."
        )
        return await self.ask(prompt, ReportDraft)
