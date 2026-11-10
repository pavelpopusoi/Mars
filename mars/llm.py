"""Слой доступа к модели.

* `ClaudeLLM`  — реальный бэкенд (Anthropic SDK, async, стриминг, structured outputs,
                 серверные web_search/web_fetch, серверные фолбэки при отказе модели).
* `OllamaLLM`  — бесплатная локальная модель через Ollama + бесплатный веб-поиск.
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

    async def web_research(self, agent: str, system: str, prompt: str,
                           queries: list[str] | None = None) -> WebResult: ...


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

    async def web_research(self, agent: str, system: str, prompt: str,
                           queries: list[str] | None = None) -> WebResult:
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


# ---------------------------------------------------------------- Ollama (локальная модель)
class OllamaLLM:
    """Бесплатная локальная модель через Ollama (http://localhost:11434).

    Структурированный ответ — через `format` (JSON-схема), веб-поиск — DuckDuckGo + Википедия
    (`mars.websearch`), после чего модель сама выбирает факты из скачанных страниц.
    """

    def __init__(self, settings: Settings):
        import httpx

        self._httpx = httpx
        self.settings = settings
        self.usage = Usage()
        self.base = settings.ollama_url.rstrip("/")
        # Локальная модель отвечает медленно: таймаут большой, а запросы идут по одному
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(settings.ollama_timeout, connect=10.0))
        self._lock = asyncio.Semaphore(1)
        self._think_supported = True

    def model_for(self, agent: str) -> str:
        m = self.settings.agent(agent).model
        return self.settings.ollama_model if m.startswith("claude") else m

    async def _chat(self, agent: str, system: str, user: str, fmt: dict | None = None) -> str:
        cfg = self.settings.agent(agent)
        ctx = self.settings.ollama_num_ctx
        answer = min(cfg.max_tokens, ctx // 2)
        # Ollama молча отрезает начало слишком длинного запроса. Режем сами — середину,
        # чтобы сохранить и вопрос в начале, и инструкции в конце (~3 символа на токен).
        budget = (ctx - answer) * 3 - len(system)
        if len(user) > budget > 2000:
            log.warning("%s: запрос длиннее окна модели, середина сокращена (%d → %d символов)",
                        agent, len(user), budget)
            head = budget // 3
            user = user[:head] + "\n…[сокращено]…\n" + user[-(budget - head):]
        body: dict[str, Any] = {
            "model": self.model_for(agent),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_ctx": ctx, "temperature": 0.2, "num_predict": answer},
        }
        if fmt is not None:
            body["format"] = fmt
        if self._think_supported:
            body["think"] = self.settings.ollama_think
        async with self._lock:
            try:
                r = await self.client.post(f"{self.base}/api/chat", json=body)
                if r.status_code == 400 and "think" in r.text and "think" in body:
                    self._think_supported = False  # модель без режима рассуждений
                    body.pop("think")
                    r = await self.client.post(f"{self.base}/api/chat", json=body)
            except self._httpx.ConnectError as e:
                raise LLMError(f"Ollama не запущена ({self.base}). Запустите приложение Ollama.") from e
            except self._httpx.TimeoutException as e:
                raise LLMError(f"Локальная модель не ответила за {self.settings.ollama_timeout:.0f} с") from e
        if r.status_code == 404:
            raise LLMError(f"Модель {body['model']} не скачана. Выполните: ollama pull {body['model']}")
        if r.status_code != 200:
            raise LLMError(f"Ошибка Ollama {r.status_code}: {r.text[:300]}")
        data = r.json()
        self.usage.calls += 1
        self.usage.input_tokens += data.get("prompt_eval_count", 0) or 0
        self.usage.output_tokens += data.get("eval_count", 0) or 0
        if data.get("done_reason") == "length":
            log.warning("%s: ответ обрезан по длине", agent)
        return (data.get("message") or {}).get("content", "")

    @staticmethod
    def _read_document(path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as e:
                raise LLMError("Для PDF с локальной моделью установите: pip install pypdf") from e
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        return path.read_text(encoding="utf-8", errors="replace")

    async def structured(self, agent: str, system: str, prompt: str, schema: type[T],
                         document: Path | None = None) -> T:
        if document is not None:
            limit = int(self.settings.ollama_num_ctx * 1.5)  # ~3 символа на токен, остаток окна — под ответ
            text = self._read_document(document)[:limit]
            prompt = f"ДОКУМЕНТ «{document.name}»:\n<<<\n{text}\n>>>\n\n{prompt}"
        system = (f"{system}\n\nОтвечай строго одним JSON-объектом по заданной схеме. "
                  f"Тексты пиши на языке: {self.settings.language}.")
        json_schema = schema.model_json_schema()
        last_err = ""
        for attempt in range(2):
            user = prompt if not last_err else (
                f"{prompt}\n\nПредыдущий ответ не прошёл проверку: {last_err[:500]}\nИсправь и верни корректный JSON.")
            raw = await self._chat(agent, system, user, fmt=json_schema)
            try:
                return schema.model_validate_json(raw)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                log.warning("%s: ответ не по схеме %s (попытка %d)", agent, schema.__name__, attempt + 1)
        raise LLMError(f"{agent}: локальная модель не смогла вернуть {schema.__name__}: {last_err[:200]}")

    async def web_research(self, agent: str, system: str, prompt: str,
                           queries: list[str] | None = None) -> WebResult:
        import re

        from .websearch import digest, fetch_url, web_search

        urls = re.findall(r"https?://[^\s«»\"'<>]+", prompt)
        if not queries and urls:  # просьба открыть конкретный источник (проверка утверждений судьёй)
            pages = [await fetch_url(urls[0].rstrip(".,);"), max_chars=self.settings.ollama_num_ctx)]
            pages = [p for p in pages if p.text]
            if not pages:
                return WebResult(text="Страница недоступна.", sources=[])
        else:
            queries = queries or [prompt[:200]]
            pages = await web_search(queries, max_results=self.settings.max_web_searches_per_question,
                                     fetch_top=self.settings.web_pages_to_read, language=self.settings.language)
            self.usage.web_searches += len(queries[:4])
        if not pages:
            return WebResult(text="(веб-поиск ничего не нашёл или недоступен)", sources=[])
        material = digest(pages)[: int(self.settings.ollama_num_ctx * 1.5)]
        text = await self._chat(agent, system, (
            f"{prompt}\n\nНИЖЕ — РЕЗУЛЬТАТЫ ВЕБ-ПОИСКА. Опирайся только на них, ничего не выдумывай.\n\n{material}\n\n"
            "Выпиши ключевые факты с цифрами. После каждого факта укажи URL источника из списка выше."))
        return WebResult(text=text, sources=[{"url": p.url, "title": p.title} for p in pages])


# ---------------------------------------------------------------- Mock
class MockLLM:
    """Возвращает правдоподобные детерминированные данные — пайплайн проверяется целиком без сети."""

    def __init__(self, settings: Settings | None = None):
        self.usage = Usage()
        self._round = 0

    async def web_research(self, agent: str, system: str, prompt: str,
                           queries: list[str] | None = None) -> WebResult:
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

    def _mock_JudgeVerdict(self, prompt: str) -> dict:
        return {"answers_question": 4, "evidence_support": 3, "contradiction_handling": 4,
                "hypothesis_quality": 3, "clarity": 4, "strengths": ["Противоречие разобрано явно"],
                "weaknesses": ["Мало независимых источников"], "unsupported_statements": [],
                "verdict": "Пригодно с оговорками (mock)."}

    def _mock_ClaimCheck(self, prompt: str) -> dict:
        cid = prompt.split("id=", 1)[-1].split(":", 1)[0].strip() if "id=" in prompt else "c1"
        return {"claim_id": cid, "verdict": "supported" if cid.endswith("c1") else "partially_supported",
                "explanation": "Источник содержит это утверждение (mock)."}

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
    if settings.backend == "ollama":
        return OllamaLLM(settings)
    return ClaudeLLM(settings)


def dumps(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
