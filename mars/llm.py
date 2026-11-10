"""Слой доступа к модели.

* `ClaudeLLM`  — реальный бэкенд (Anthropic SDK, async, стриминг, structured outputs,
                 серверные web_search/web_fetch, серверные фолбэки при отказе модели).
* `MockLLM`    — детерминированный офлайн-бэкенд для тестов и демо без API-ключа.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .config import AgentConfig, Settings
from .costs import WEB_SEARCH_USD, call_cost

log = logging.getLogger("mars.llm")
T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, u: Any, model: str = "") -> None:
        self.calls += 1
        if u is None:
            return
        inp = getattr(u, "input_tokens", 0) or 0
        out = getattr(u, "output_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        ws = getattr(getattr(u, "server_tool_use", None), "web_search_requests", 0) or 0
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_read_tokens += cr
        self.cache_write_tokens += cw
        self.web_searches += ws
        self.cost_usd = round(self.cost_usd + call_cost(model, inp, out, cr, cw) + ws * WEB_SEARCH_USD, 4)


@dataclass
class WebResult:
    text: str
    sources: list[dict[str, str]] = field(default_factory=list)  # [{url, title}]


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    usage: Usage

    async def structured(self, agent: str, system: str, prompt: str, schema: type[T],
                         document: Path | None = None) -> T: ...

    async def web_research(self, agent: str, system: str, prompt: str) -> WebResult: ...


# ---------------------------------------------------------------- Claude
class ClaudeLLM:
    def __init__(self, settings: Settings):
        import anthropic  # импорт здесь, чтобы mock-режим работал без ключа

        self._anthropic = anthropic
        self.settings = settings
        self.client = anthropic.AsyncAnthropic(max_retries=4)
        self.usage = Usage()

    def _common(self, cfg: AgentConfig, system: str) -> dict[str, Any]:
        kw: dict[str, Any] = dict(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": cfg.effort},
        )
        if self.settings.prompt_caching:
            # Автокэш: повторяющийся префикс (system, документ) читается из кэша за ~10% цены.
            kw["cache_control"] = {"type": "ephemeral"}
        if self.settings.use_server_fallbacks:
            # При отказе классификатора безопасности API сам перезапустит запрос на подходящей модели.
            kw["betas"] = [FALLBACK_BETA]
            kw["fallbacks"] = "default"
        return kw

    async def _stream(self, **kw: Any) -> Any:
        a = self._anthropic
        try:
            async with self.client.beta.messages.stream(**kw) as stream:
                msg = await stream.get_final_message()
        except a.BadRequestError as e:
            raise LLMError(f"Некорректный запрос: {e.message}") from e
        except a.AuthenticationError as e:
            raise LLMError("Неверный ANTHROPIC_API_KEY") from e
        except a.RateLimitError as e:
            raise LLMError("Превышен лимит запросов (после ретраев SDK)") from e
        except a.APIStatusError as e:
            raise LLMError(f"Ошибка API {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError("Нет соединения с API") from e
        self.usage.add(msg.usage, kw.get("model", ""))
        if msg.stop_reason == "refusal":
            cat = getattr(getattr(msg, "stop_details", None), "category", None)
            raise LLMError(f"Модель отказалась отвечать (категория: {cat})")
        if msg.stop_reason == "max_tokens":
            log.warning("Ответ обрезан по max_tokens (%s)", kw.get("max_tokens"))
        return msg

    async def _document_block(self, path: Path) -> dict[str, Any]:
        """PDF загружается через Files API, текстовые файлы передаются как есть."""
        if path.suffix.lower() == ".pdf":
            try:
                uploaded = await self.client.files.upload(file=path)
            except self._anthropic.APIError as e:
                raise LLMError(f"Не удалось загрузить {path.name}: {e}") from e
            source: dict[str, Any] = {"type": "file", "file_id": uploaded.id}
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            source = {"type": "text", "media_type": "text/plain", "data": text}
        return {"type": "document", "source": source, "title": path.name}

    async def structured(self, agent: str, system: str, prompt: str, schema: type[T],
                         document: Path | None = None) -> T:
        cfg = self.settings.agent(agent)
        content: Any = prompt
        if document is not None:
            content = [await self._document_block(document), {"type": "text", "text": prompt}]
        msg = await self._stream(
            **self._common(cfg, system),
            messages=[{"role": "user", "content": content}],
            output_format=schema,
        )
        parsed = getattr(msg, "parsed_output", None)
        if parsed is not None:
            return parsed
        text = "".join(b.text for b in msg.content if b.type == "text")
        try:
            return schema.model_validate_json(text)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"{agent}: не удалось разобрать ответ как {schema.__name__}") from e

    async def web_research(self, agent: str, system: str, prompt: str) -> WebResult:
        cfg = self.settings.agent(agent)
        tools = [
            {"type": "web_search_20260209", "name": "web_search",
             "max_uses": self.settings.max_web_searches_per_question},
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 3},
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        texts: list[str] = []
        sources: dict[str, str] = {}
        for _ in range(5):  # защита от бесконечного pause_turn
            msg = await self._stream(**self._common(cfg, system), tools=tools, messages=messages)
            for block in msg.content:
                if block.type == "text":
                    texts.append(block.text)
                    for c in getattr(block, "citations", None) or []:
                        if getattr(c, "url", None):
                            sources.setdefault(c.url, getattr(c, "title", "") or c.url)
                elif block.type == "web_search_tool_result" and isinstance(block.content, list):
                    for r in block.content:
                        if getattr(r, "url", None):
                            sources.setdefault(r.url, getattr(r, "title", "") or r.url)
            if msg.stop_reason != "pause_turn":
                break
            messages = [messages[0], {"role": "assistant", "content": msg.content}]
        return WebResult(text="\n".join(texts), sources=[{"url": u, "title": t} for u, t in sources.items()])


# ---------------------------------------------------------------- Mock
class MockLLM:
    """Возвращает правдоподобные детерминированные данные — пайплайн проверяется целиком без сети."""

    def __init__(self, settings: Settings | None = None):
        self.usage = Usage()
        self._round = 0

    async def web_research(self, agent: str, system: str, prompt: str) -> WebResult:
        self.usage.calls += 1
        await asyncio.sleep(0)
        return WebResult(
            text=f"[mock] найдено по запросу: {prompt[:80]}",
            sources=[{"url": "https://example.org/study-a", "title": "Study A"},
                     {"url": "https://example.org/report-b", "title": "Report B"}],
        )

    async def structured(self, agent: str, system: str, prompt: str, schema: type[T],
                         document: Path | None = None) -> T:
        self.usage.calls += 1
        await asyncio.sleep(0)
        name = schema.__name__
        factory = getattr(self, f"_mock_{name}", None)
        if factory is None:
            raise LLMError(f"MockLLM не умеет {name}")
        data = factory(prompt)
        if document is not None and name == "SearchFindings":
            for src in data["sources"]:
                src.update(title=document.name, url=f"file://{document.name}", source_type="user_document")
        return schema.model_validate(data)

    @staticmethod
    def _qid(prompt: str) -> str:
        for token in prompt.replace('"', " ").replace("=", " ").replace(":", " ").replace("'", " ").split():
            if token[:1] in ("q", "d", "r") and token[1:].isdigit():
                return token
        return "q1"

    def _mock_ResearchPlan(self, prompt: str) -> dict:
        main = prompt.split("ВОПРОС:", 1)[-1].strip().splitlines()[0] if "ВОПРОС:" in prompt else prompt[:120]
        return {
            "main_question": main,
            "interpretation": "Демонстрационный план (mock).",
            "sub_questions": [
                {"id": "q1", "question": "Каково текущее состояние знаний?", "rationale": "База",
                 "search_queries": ["state of the art", "review 2026"], "domain": "science"},
                {"id": "q2", "question": "Какие есть экономические последствия?", "rationale": "Эффект",
                 "search_queries": ["economic impact"], "domain": "business"},
            ],
            "success_criteria": ["Есть ответ на каждый подвопрос", "Противоречия разобраны"],
        }

    def _mock_SearchFindings(self, prompt: str) -> dict:
        q = self._qid(prompt)
        return {
            "sub_question_id": q,
            "summary": f"Сводка по {q}",
            "sources": [
                {"id": f"{q}-s1", "title": "Study A", "url": "https://example.org/study-a",
                 "publisher": "Journal X", "year": 2025, "source_type": "peer_reviewed", "reliability": 0.85},
                {"id": f"{q}-s2", "title": "Report B", "url": "https://example.org/report-b",
                 "publisher": "Agency Y", "year": 2024, "source_type": "official", "reliability": 0.7},
            ],
            "claims": [
                {"id": f"{q}-c1", "sub_question_id": q, "statement": "Фактор A увеличивает показатель B на 12%",
                 "source_ids": [f"{q}-s1"], "confidence": 0.8, "subject": "Фактор A",
                 "predicate": "увеличивает", "object": "Показатель B"},
                {"id": f"{q}-c2", "sub_question_id": q, "statement": "Фактор A не влияет на показатель B",
                 "source_ids": [f"{q}-s2"], "confidence": 0.5, "subject": "Фактор A",
                 "predicate": "не влияет на", "object": "Показатель B"},
            ],
        }

    def _mock_Synthesis(self, prompt: str) -> dict:
        return {"key_findings": ["Фактор A, вероятно, влияет на B"], "consensus_points": ["B растёт"],
                "open_questions": ["Размер эффекта"], "evidence_strength": "moderate"}

    def _mock_Critique(self, prompt: str) -> dict:
        self._round += 1
        return {
            "contradictions": [{
                "id": "x1", "claim_ids": ["q1-c1", "q1-c2"],
                "description": "Источники расходятся во влиянии A на B",
                "likely_cause": "methodology",
                "resolution": "Рецензируемое исследование с контрольной группой весомее отчёта",
                "severity": "medium"}],
            "weak_claim_ids": ["q1-c2"],
            "gaps": ["Нет данных за 2026 год"] if self._round == 1 else [],
            "follow_up_queries": ["A effect on B 2026 meta-analysis"] if self._round == 1 else [],
            "needs_another_round": self._round == 1,
        }

    def _mock_HypothesisSet(self, prompt: str) -> dict:
        return {"hypotheses": [{
            "id": "h1", "statement": "Эффект A на B проявляется только при условии C",
            "supporting_claim_ids": ["q1-c1", "q1-c2"], "novelty": 0.6, "plausibility": 0.7,
            "test_method": "Стратифицированный анализ по C на открытых данных",
            "falsifiable_by": "Одинаковый эффект при наличии и отсутствии C"}]}

    def _mock_ReportDraft(self, prompt: str) -> dict:
        return {
            "title": "Демонстрационный отчёт MARS",
            "executive_summary": "Фактор A, вероятно, влияет на B, но данные противоречивы [q1-s1][q1-s2].",
            "sections": [{"title": "Основные выводы",
                          "body_markdown": "Рецензируемые данные показывают рост B на 12% [q1-s1]."}],
            "conclusion": "Нужна проверка гипотезы h1.",
            "limitations": ["Mock-режим: данные синтетические"],
        }


def make_llm(settings: Settings) -> LLM:
    if settings.backend == "mock":
        return MockLLM(settings)
    return ClaudeLLM(settings)


def dumps(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
