"""Pydantic-схемы: общий «язык» между агентами. Всё, что передаётся между шагами, — валидируемые объекты."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------- Планирование ----------
class SubQuestion(BaseModel):
    id: str = Field(description="Короткий id, например 'q1'")
    question: str
    rationale: str = Field(description="Зачем этот подвопрос нужен для ответа на главный")
    search_queries: list[str] = Field(description="2-4 поисковых запроса")
    domain: Literal["science", "technology", "business", "policy", "other"]


class ResearchPlan(BaseModel):
    main_question: str
    interpretation: str = Field(description="Как система поняла вопрос, допущения")
    sub_questions: list[SubQuestion]
    success_criteria: list[str]


# ---------- Источники и доказательства ----------
class Source(BaseModel):
    id: str = Field(description="Например 's1'")
    title: str
    url: str
    publisher: str = ""
    year: int | None = None
    source_type: Literal["peer_reviewed", "preprint", "official", "industry", "news", "blog", "user_document",
                         "other"] = "other"
    reliability: float = Field(ge=0, le=1, description="Оценка надёжности 0..1")


class Claim(BaseModel):
    id: str = Field(description="Например 'c1'")
    sub_question_id: str
    statement: str = Field(description="Одно проверяемое утверждение")
    source_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    subject: str = Field(description="Сущность-субъект для графа знаний")
    predicate: str = Field(description="Связь, глагол: 'увеличивает', 'зависит от'...")
    object: str = Field(description="Сущность-объект для графа знаний")


class SearchFindings(BaseModel):
    sub_question_id: str
    summary: str
    sources: list[Source]
    claims: list[Claim]


# ---------- Анализ ----------
class Synthesis(BaseModel):
    key_findings: list[str]
    consensus_points: list[str]
    open_questions: list[str]
    evidence_strength: Literal["weak", "moderate", "strong"]


# ---------- Критика ----------
class Contradiction(BaseModel):
    id: str
    claim_ids: list[str] = Field(description="Утверждения, которые противоречат друг другу")
    description: str
    likely_cause: Literal["methodology", "data", "definitions", "time_period", "bias", "error", "unknown"]
    resolution: str = Field(description="Как разрешить или почему остаётся открытым")
    severity: Literal["low", "medium", "high"]


class Critique(BaseModel):
    contradictions: list[Contradiction]
    weak_claim_ids: list[str] = Field(description="Слабо подтверждённые утверждения")
    gaps: list[str] = Field(description="Пробелы, которые стоит доисследовать")
    follow_up_queries: list[str] = Field(description="Поисковые запросы для следующего раунда")
    needs_another_round: bool


# ---------- Гипотезы ----------
class Hypothesis(BaseModel):
    id: str
    statement: str
    supporting_claim_ids: list[str]
    novelty: float = Field(ge=0, le=1)
    plausibility: float = Field(ge=0, le=1)
    test_method: str = Field(description="Как проверить: эксперимент, данные, метрика")
    falsifiable_by: str = Field(description="Какой результат опровергнет гипотезу")


class HypothesisSet(BaseModel):
    hypotheses: list[Hypothesis]


# ---------- Отчёт ----------
class ReportSection(BaseModel):
    title: str
    body_markdown: str = Field(description="Текст с ссылками вида [s1]")


class ReportDraft(BaseModel):
    title: str
    executive_summary: str
    sections: list[ReportSection]
    conclusion: str
    limitations: list[str]
