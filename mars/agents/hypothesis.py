from __future__ import annotations

from ..schemas import Critique, HypothesisSet, ResearchPlan, SearchFindings, Synthesis
from .analyst import findings_digest
from .base import Agent


class HypothesisGenerator(Agent):
    name = "hypothesis"
    role = (
        "You are the Hypothesis Generator. From the evidence, contradictions and gaps, propose 3-5 novel, "
        "falsifiable hypotheses. Each must reference supporting claim ids, give a concrete test method "
        "(experiment, dataset, metric) and state what result would falsify it. Prefer hypotheses that explain "
        "contradictions or bridge disciplines. Ids h1, h2, ..."
    )

    async def run(self, plan: ResearchPlan, findings: list[SearchFindings], synthesis: Synthesis,
                  critique: Critique) -> HypothesisSet:
        prompt = (
            f"ГЛАВНЫЙ ВОПРОС: {plan.main_question}\n\nДОКАЗАТЕЛЬСТВА:\n{findings_digest(findings)}\n\n"
            f"СИНТЕЗ:\n{synthesis.model_dump_json(indent=1)}\n\nКРИТИКА:\n{critique.model_dump_json(indent=1)}\n\n"
            "Сгенерируй гипотезы."
        )
        return await self.ask(prompt, HypothesisSet)
