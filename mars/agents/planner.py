from __future__ import annotations

from ..schemas import ResearchPlan
from .base import Agent


class Planner(Agent):
    name = "planner"
    role = (
        "You are the Research Planner. Decompose a complex research question into 3-6 independent, "
        "answerable sub-questions that together fully answer it. Cover mechanisms, evidence, counter-evidence, "
        "economics/practice and open problems when relevant. For each sub-question give 2-4 precise web "
        "search queries (mix English and the question's language). Use ids q1, q2, ..."
    )

    async def run(self, question: str, context: str = "") -> ResearchPlan:
        prompt = f"ВОПРОС: {question}\n"
        if context:
            prompt += f"\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ:\n{context}\n"
        prompt += "\nСоставь исследовательский план."
        return await self.ask(prompt, ResearchPlan)
