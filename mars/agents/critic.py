from __future__ import annotations

from ..schemas import Critique, ResearchPlan, SearchFindings, Synthesis
from .analyst import findings_digest
from .base import Agent


class Critic(Agent):
    name = "critic"
    role = (
        "You are the Critic and Contradiction Detector. Find claims that conflict (directly or via "
        "incompatible numbers, definitions, time periods or populations), explain the likely cause and how to "
        "resolve it. Flag weakly supported claims (single low-reliability source, no primary data). List gaps "
        "and concrete follow-up search queries. Set needs_another_round=true only if another search round "
        "would materially change the answer."
    )

    async def run(self, plan: ResearchPlan, findings: list[SearchFindings], synthesis: Synthesis,
                  round_no: int, max_rounds: int) -> Critique:
        prompt = (
            f"ГЛАВНЫЙ ВОПРОС: {plan.main_question}\nКритерии успеха: {plan.success_criteria}\n"
            f"Раунд {round_no} из {max_rounds}.\n\nДОКАЗАТЕЛЬСТВА:\n{findings_digest(findings)}\n\n"
            f"СИНТЕЗ:\n{synthesis.model_dump_json(indent=1)}\n\nПроведи критический разбор. "
            "Используй только существующие id утверждений."
        )
        critique = await self.ask(prompt, Critique)
        if round_no >= max_rounds:
            critique.needs_another_round = False
        return critique
