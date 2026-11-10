from __future__ import annotations

from pathlib import Path

from ..schemas import SearchFindings
from .base import Agent

SUPPORTED = {".pdf", ".txt", ".md", ".csv", ".json"}


class Reader(Agent):
    """Читает файл пользователя (PDF, текст) и извлекает из него утверждения как из источника."""

    name = "reader"
    role = (
        "You are the Document Reader. Read the attached user document and extract atomic, checkable claims "
        "relevant to the research question, with numbers and conditions. The document is the single source: "
        "do not add facts that are not in it. Give subject / predicate / object for each claim."
    )

    async def run(self, question: str, path: Path, doc_id: str) -> SearchFindings:
        prompt = (
            f"ГЛАВНЫЙ ВОПРОС: {question}\n\nДокумент id={doc_id}: {path.name}\n"
            f"Оформи результат: один источник с id '{doc_id}-s1', title='{path.name}', "
            f"url='file://{path.name}', source_type='user_document'. id утверждений: '{doc_id}-c1'...; "
            f"sub_question_id='{doc_id}'. В summary — краткое содержание документа по теме вопроса."
        )
        findings = await self.ask(prompt, SearchFindings, document=path)
        findings.sub_question_id = doc_id
        return findings
