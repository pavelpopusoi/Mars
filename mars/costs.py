"""Цены и оценка стоимости. Цены — USD за 1M токенов (обновляйте по https://www.anthropic.com/pricing)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings

PRICES: dict[str, tuple[float, float]] = {  # model: (input, output)
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
CACHE_READ_MULT = 0.1     # чтение из кэша ≈ 10% цены входа
CACHE_WRITE_MULT = 1.25   # запись в кэш (5 мин) ≈ 125%
WEB_SEARCH_USD = 0.01     # за один веб-поиск ($10 / 1000)


def price(model: str) -> tuple[float, float]:
    return PRICES.get(model, PRICES["claude-fable-5-2"])


def call_cost(model: str, inp: int, out: int, cache_read: int = 0, cache_write: int = 0) -> float:
    pin, pout = price(model)
    return (inp * pin + cache_read * pin * CACHE_READ_MULT + cache_write * pin * CACHE_WRITE_MULT
            + out * pout) / 1_000_000


# Типичный объём одного вызова (вход, выход), включая thinking. Грубые средние по прогонам.
TYPICAL_TOKENS = {
    "planner": (1_500, 3_000),
    "searcher_web": (25_000, 4_000),   # результаты поиска входят во вход
    "searcher_struct": (6_000, 4_000),
    "reader": (30_000, 5_000),         # на документ ~20 страниц
    "analyst": (8_000, 4_000),
    "critic": (10_000, 4_000),
    "hypothesis": (10_000, 4_000),
    "writer": (14_000, 10_000),
}


@dataclass
class Estimate:
    calls: int
    input_tokens: int
    output_tokens: int
    web_searches: int
    usd_low: float
    usd_high: float

    def table(self) -> str:
        return (f"Вызовов модели:  ~{self.calls}\n"
                f"Токенов вход:    ~{self.input_tokens:,}\n"
                f"Токенов выход:   ~{self.output_tokens:,}\n"
                f"Веб-поисков:     до {self.web_searches}\n"
                f"Стоимость:       ${self.usd_low:.2f} – ${self.usd_high:.2f}")


def estimate(settings: Settings, sub_questions: int = 5, documents: int = 0) -> Estimate:
    """Оценка до запуска. Нижняя граница — 1 раунд, верхняя — max_rounds и все веб-поиски."""

    def run_cost(rounds: int) -> tuple[int, int, int, float, int]:
        plan: list[tuple[str, str, int]] = [("planner", "planner", 1)]
        plan += [("searcher", "searcher_web", sub_questions), ("searcher", "searcher_struct", sub_questions)]
        plan += [("searcher", "reader", documents)]
        plan += [("analyst", "analyst", rounds), ("critic", "critic", rounds)]
        plan += [("searcher", "searcher_web", rounds - 1), ("searcher", "searcher_struct", rounds - 1)]
        plan += [("hypothesis", "hypothesis", 1), ("writer", "writer", 1)]
        calls = inp = out = 0
        usd = 0.0
        for agent, kind, n in plan:
            i, o = TYPICAL_TOKENS[kind]
            calls += n
            inp += i * n
            out += o * n
            usd += call_cost(settings.agent(agent).model, i * n, o * n)
        searches = (sub_questions + rounds - 1) * settings.max_web_searches_per_question
        return calls, inp, out, usd, searches

    lo = run_cost(1)
    hi = run_cost(settings.max_rounds)
    return Estimate(calls=hi[0], input_tokens=hi[1], output_tokens=hi[2], web_searches=hi[4],
                    usd_low=lo[3] + lo[4] * WEB_SEARCH_USD * 0.5, usd_high=hi[3] * 1.5 + hi[4] * WEB_SEARCH_USD)
