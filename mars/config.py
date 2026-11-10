"""Конфигурация: YAML-файл + переменные окружения."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    model: str = "claude-fable-5-2"
    effort: str = "high"  # low | medium | high | xhigh | max
    max_tokens: int = 16000


class Settings(BaseModel):
    backend: str = Field(default="anthropic", description="anthropic | mock")
    max_rounds: int = 2
    max_parallel_searches: int = 4
    max_web_searches_per_question: int = 5
    use_server_fallbacks: bool = True
    prompt_caching: bool = True
    max_documents: int = 10
    runs_dir: str = "runs"
    language: str = "ru"
    agents: dict[str, AgentConfig] = Field(default_factory=lambda: {
        "planner": AgentConfig(effort="high"),
        "searcher": AgentConfig(effort="medium"),
        "reader": AgentConfig(effort="medium"),
        "analyst": AgentConfig(effort="high"),
        "critic": AgentConfig(effort="high"),
        "hypothesis": AgentConfig(effort="high"),
        "writer": AgentConfig(effort="medium", max_tokens=64000),
    })

    def agent(self, name: str) -> AgentConfig:
        return self.agents.get(name, AgentConfig())


def load_settings(path: str | Path | None = None) -> Settings:
    data: dict = {}
    path = path or os.getenv("MARS_CONFIG", "config/mars.yaml")
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if os.getenv("MARS_BACKEND"):
        data["backend"] = os.environ["MARS_BACKEND"]
    return Settings(**data)
