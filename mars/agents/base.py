from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ..config import Settings
from ..llm import LLM, LLMError
from ..provenance import RunRecorder

T = TypeVar("T", bound=BaseModel)


class Agent:
    """Базовый агент: системный промпт + вызов модели + запись в трассу."""

    name: str = "agent"
    role: str = ""

    def __init__(self, llm: LLM, settings: Settings, recorder: RunRecorder | None = None):
        self.llm = llm
        self.settings = settings
        self.recorder = recorder

    @property
    def system(self) -> str:
        return (
            f"{self.role}\n\n"
            "You are one agent inside MARS, a multi-agent research system. Be rigorous: separate facts "
            "from interpretation, never invent sources, URLs, numbers or citations. If evidence is missing, say so.\n"
            f"Write every human-readable field in this language: {self.settings.language}."
        )

    async def ask(self, prompt: str, schema: type[T], document: Path | None = None) -> T:
        started = time.monotonic()
        try:
            out = await self.llm.structured(self.name, self.system, prompt, schema, document)
        except LLMError as e:
            self._trace(prompt, None, started, str(e))
            raise
        self._trace(prompt, out, started)
        return out

    def _trace(self, prompt: str, out: Any, started: float, error: str | None = None) -> None:
        if self.recorder:
            cfg = self.settings.agent(self.name)
            self.recorder.trace(self.name, system=self.system, prompt=prompt, output=out, started=started,
                                model=cfg.model, effort=cfg.effort, error=error)
