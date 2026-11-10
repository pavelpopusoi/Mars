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
    backend: str = Field(default="anthropic", description="anthropic | ollama | mock")
    # Локальная модель (backend: ollama)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_num_ctx: int = 8192
    ollama_think: bool = False
    ollama_timeout: float = 900.0
    web_pages_to_read: int = 4
    max_rounds: int = 2
    max_parallel_searches: int = 4
    max_web_searches_per_question: int = 5
    use_server_fallbacks: bool = True
    prompt_caching: bool = True
    max_documents: int = 10
    academic_sources: list[str] = Field(default_factory=lambda: ["openalex", "arxiv", "pubmed"])
    academic_results_per_source: int = 4
    contact_email: str = Field(default="", description="Необязательно: e-mail для OpenAlex и PubMed")
    runs_dir: str = "runs"
    db_path: str = Field(default="", description="SQLite веб-сервера; пусто — <runs_dir>/mars.db")
    max_concurrent_runs: int = 2
    language: str = "ru"
    agents: dict[str, AgentConfig] = Field(default_factory=lambda: {
        "planner": AgentConfig(effort="high"),
        "searcher": AgentConfig(effort="medium"),
        "reader": AgentConfig(effort="medium"),
        "analyst": AgentConfig(effort="high"),
        "critic": AgentConfig(effort="high"),
        "hypothesis": AgentConfig(effort="high"),
        "writer": AgentConfig(effort="medium", max_tokens=64000),
        "judge": AgentConfig(effort="high"),
    })

    @property
    def database(self) -> str:
        return self.db_path or str(Path(self.runs_dir) / "mars.db")

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
