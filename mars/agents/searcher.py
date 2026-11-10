from __future__ import annotations

import time

from ..schemas import SearchFindings, SubQuestion
from .base import Agent


class Searcher(Agent):
    name = "searcher"
    role = (
        "You are the Evidence Searcher. Find high-quality, recent, primary sources (peer-reviewed papers, "
        "official statistics, standards, filings) and extract atomic, checkable claims. Every claim must cite "
        "source ids that exist in your sources list. Rate source reliability honestly (0..1). For each claim "
        "give subject / predicate / object for a knowledge graph."
    )

    async def run(self, sq: SubQuestion, extra_queries: list[str] | None = None) -> SearchFindings:
        queries = sq.search_queries + (extra_queries or [])
        # Шаг 1: веб-поиск (серверные инструменты Claude)
        started = time.monotonic()
        research_prompt = (
            f"Подвопрос {sq.id}: {sq.question}\nЗачем: {sq.rationale}\n"
            f"Поисковые запросы: {queries}\n"
            "Найди и прочитай лучшие источники. Перечисли ключевые факты с цифрами и ссылками."
        )
        web = await self.llm.web_research(self.name, self.system, research_prompt)
        self._trace(research_prompt, {"text": web.text[:4000], "sources": web.sources}, started)

        # Шаг 2: структурирование найденного в Source/Claim
        src_list = "\n".join(f"- {s['title']} — {s['url']}" for s in web.sources) or "(нет)"
        prompt = (
            f"Подвопрос id={sq.id}: {sq.question}\n\n"
            f"НАЙДЕННЫЙ МАТЕРИАЛ:\n{web.text}\n\nURL ИСТОЧНИКОВ:\n{src_list}\n\n"
            f"Оформи результат. id источников: '{sq.id}-s1', '{sq.id}-s2'...; id утверждений: "
            f"'{sq.id}-c1'...; sub_question_id='{sq.id}'. Используй только URL из списка выше."
        )
        findings = await self.ask(prompt, SearchFindings)
        findings.sub_question_id = sq.id
        return findings
