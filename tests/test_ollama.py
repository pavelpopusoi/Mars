import json

import httpx
import pytest

from mars.config import Settings
from mars.llm import LLMError, OllamaLLM, make_llm
from mars.schemas import Synthesis
from mars.websearch import _real_url, extract_text


def _llm(handler) -> OllamaLLM:
    llm = make_llm(Settings(backend="ollama"))
    assert isinstance(llm, OllamaLLM)
    llm.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return llm


async def test_structured_retries_on_bad_json():
    answers = iter(["не json", json.dumps({"key_findings": ["A"], "consensus_points": [], "open_questions": [],
                                           "evidence_strength": "weak"})])
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"message": {"content": next(answers)}, "prompt_eval_count": 10,
                                         "eval_count": 5})

    llm = _llm(handler)
    out = await llm.structured("analyst", "sys", "prompt", Synthesis)
    assert out.key_findings == ["A"]
    assert seen[0]["model"] == "qwen3:8b" and "format" in seen[0]
    assert "не прошёл проверку" in seen[1]["messages"][1]["content"]
    assert llm.usage.calls == 2 and llm.usage.input_tokens == 20 and llm.usage.cost_usd == 0


async def test_missing_model_message():
    llm = _llm(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMError, match="ollama pull"):
        await llm.structured("analyst", "sys", "prompt", Synthesis)


def test_websearch_helpers():
    assert _real_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&rut=x") == "https://example.org/a"
    page = "<html><script>x()</script><p>" + "Полезный текст абзаца номер один. " * 3 + "</p></html>"
    assert "Полезный текст" in extract_text(page, 1000) and "x()" not in extract_text(page, 1000)
