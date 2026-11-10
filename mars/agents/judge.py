from __future__ import annotations

import time

from ..schemas import Claim, ClaimCheck, JudgeVerdict, Source
from .base import Agent


class Judge(Agent):
    """Независимый судья: оценивает готовый отчёт и сверяет утверждения с первоисточниками."""

    name = "judge"
    role = (
        "You are an independent, strict research reviewer. You did not write this report. Score it against "
        "the evidence it cites, not against your own beliefs. A 5 means publishable, 3 means usable with "
        "caveats, 1 means misleading. Quote the exact report phrases that the listed claims do not support."
    )

    async def review(self, question: str, report_md: str, evidence_digest: str) -> JudgeVerdict:
        prompt = (
            f"ГЛАВНЫЙ ВОПРОС: {question}\n\nУТВЕРЖДЕНИЯ И ИСТОЧНИКИ, НА КОТОРЫЕ ОПИРАЕТСЯ ОТЧЁТ:\n{evidence_digest}\n\n"
            f"ОТЧЁТ:\n{report_md}\n\nОцени отчёт."
        )
        return await self.ask(prompt, JudgeVerdict)

    async def check_claim(self, claim: Claim, source: Source) -> ClaimCheck:
        started = time.monotonic()
        fetch_prompt = (
            f"Открой источник {source.url} и найди, что в нём сказано по поводу утверждения:\n"
            f"«{claim.statement}»\nПроцитируй относящиеся к делу места. Если страница недоступна, скажи об этом."
        )
        web = await self.llm.web_research(self.name, self.system, fetch_prompt)
        self._trace(fetch_prompt, {"text": web.text[:3000]}, started)
        prompt = (
            f"Утверждение id={claim.id}: «{claim.statement}»\nИсточник: {source.title} — {source.url}\n\n"
            f"ЧТО НАЙДЕНО В ИСТОЧНИКЕ:\n{web.text}\n\n"
            "Вынеси вердикт: подтверждает ли источник утверждение. Если источник открыть не удалось — "
            "source_unavailable."
        )
        check = await self.ask(prompt, ClaimCheck)
        check.claim_id = claim.id
        return check
