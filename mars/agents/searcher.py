from __future__ import annotations

import time

from ..schemas import SearchFindings, SubQuestion
from ..sources import Paper, search_academic
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
        # Шаг 1: веб-поиск (у Claude — серверный, у локальной модели — DuckDuckGo + Википедия)
        started = time.monotonic()
        research_prompt = (
            f"Подвопрос {sq.id}: {sq.question}\nЗачем: {sq.rationale}\n"
            f"Поисковые запросы: {queries}\n"
            "Найди и прочитай лучшие источники. Перечисли ключевые факты с цифрами и ссылками."
        )
        web = await self.llm.web_research(self.name, self.system, research_prompt, queries=queries)
        self._trace(research_prompt, {"text": web.text[:4000], "sources": web.sources}, started)

        # Шаг 1б: научные базы (обычные HTTP-запросы, ссылки гарантированно настоящие)
        papers: list[Paper] = []
        if self.settings.backend != "mock" and self.settings.academic_sources:
            started = time.monotonic()
            papers = await search_academic(queries, self.settings.academic_sources,
                                           self.settings.academic_results_per_source, self.settings.contact_email)
            self._trace(f"academic: {queries}", [p.title for p in papers], started)
        known = {s["url"] for s in web.sources}
        sources = web.sources + [{"title": f"{p.title} ({p.database}, {p.year})", "url": p.url}
                                 for p in papers if p.url not in known]
        academic = "\n\n".join(p.digest() for p in papers)

        # Шаг 2: структурирование найденного в Source/Claim
        src_list = "\n".join(f"- {s['title']} — {s['url']}" for s in sources) or "(нет)"
        prompt = (
            f"Подвопрос id={sq.id}: {sq.question}\n\n"
            f"НАЙДЕННЫЙ МАТЕРИАЛ (ВЕБ):\n{web.text}\n\n"
            f"СТАТЬИ ИЗ НАУЧНЫХ БАЗ (аннотации):\n{academic or '(нет)'}\n\nURL ИСТОЧНИКОВ:\n{src_list}\n\n"
            f"Оформи результат. id источников: '{sq.id}-s1', '{sq.id}-s2'...; id утверждений: "
            f"'{sq.id}-c1'...; sub_question_id='{sq.id}'. Используй только URL из списка выше."
        )
        findings = await self.ask(prompt, SearchFindings)
        findings.sub_question_id = sq.id
        return findings
