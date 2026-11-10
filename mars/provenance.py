"""Воспроизводимость: каждый запуск — отдельная папка с конфигом, трассой вызовов и артефактами."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from . import __version__


def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^\w]+", "-", text.lower(), flags=re.UNICODE).strip("-")
    return s[:n] or "run"


class RunRecorder:
    def __init__(self, runs_dir: str, question: str, settings: BaseModel):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.dir = Path(runs_dir) / f"{ts}-{slugify(question)}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._trace = self.dir / "trace.jsonl"
        self._step = 0
        self.save("manifest", {
            "mars_version": __version__,
            "question": question,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "settings": settings.model_dump(),
        })

    def save(self, name: str, obj: Any) -> Path:
        path = self.dir / f"{name}.json"
        if isinstance(obj, BaseModel):
            text = obj.model_dump_json(indent=2)
        else:
            text = json.dumps(_plain(obj), ensure_ascii=False, indent=2, default=str)
        path.write_text(text, encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def trace(self, agent: str, *, system: str, prompt: str, output: Any, started: float, model: str = "",
              effort: str = "", error: str | None = None) -> None:
        self._step += 1
        rec = {
            "step": self._step,
            "agent": agent,
            "model": model,
            "effort": effort,
            "system_sha256": hashlib.sha256(system.encode()).hexdigest()[:16],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "prompt": prompt,
            "duration_s": round(time.monotonic() - started, 2),
            "error": error,
            "output": _plain(output),
        }
        with self._trace.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _plain(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, (list, tuple)):
        return [_plain(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float)):
        return {k: _plain(v) for k, v in vars(obj).items()}
    return obj
