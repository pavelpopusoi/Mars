"""Офлайн-тесты всего пайплайна на MockLLM — без сети и ключа."""
import json

from mars.config import Settings
from mars.knowledge_graph import KnowledgeGraph
from mars.llm import MockLLM
from mars.orchestrator import Orchestrator, rebuild_report
from mars.schemas import SearchFindings


async def test_full_pipeline(tmp_path):
    settings = Settings(backend="mock", runs_dir=str(tmp_path), max_rounds=2)
    result = await Orchestrator(settings, llm=MockLLM()).run("Влияет ли A на B?")

    d = result.run_dir
    for name in ["manifest.json", "plan.json", "findings.json", "synthesis.json", "critique.json",
                 "hypotheses.json", "draft.json", "meta.json", "graph.json", "graph.mmd", "graph.graphml",
                 "trace.jsonl", "report.md"]:
        assert (d / name).exists(), name

    md = result.report_markdown
    for section in ["## Резюме", "## Противоречия", "## Гипотезы", "## Граф знаний", "## Источники",
                    "## Воспроизводимость", "```mermaid"]:
        assert section in md

    meta = json.loads((d / "meta.json").read_text())
    assert meta["rounds"] == 2  # критик запросил доисследование в 1-м раунде
    trace = [json.loads(line) for line in (d / "trace.jsonl").read_text().splitlines()]
    agents = {t["agent"] for t in trace}
    assert agents == {"planner", "searcher", "analyst", "critic", "hypothesis", "writer"}


async def test_rebuild_is_deterministic(tmp_path):
    settings = Settings(backend="mock", runs_dir=str(tmp_path))
    result = await Orchestrator(settings, llm=MockLLM()).run("Q")
    before = result.report_path.read_text()
    rebuild_report(result.run_dir)
    assert result.report_path.read_text() == before


async def test_round_limit_respected(tmp_path):
    settings = Settings(backend="mock", runs_dir=str(tmp_path), max_rounds=1)
    result = await Orchestrator(settings, llm=MockLLM()).run("Q")
    meta = json.loads((result.run_dir / "meta.json").read_text())
    assert meta["rounds"] == 1


def test_knowledge_graph_merges_entities():
    f = SearchFindings.model_validate(MockLLM()._mock_SearchFindings("q1"))
    g = KnowledgeGraph.build([f], [], [])
    assert len(g.nodes) == 2  # «Фактор A» и «Показатель B» склеены
    assert len(g.edges) == 2
    assert g.to_mermaid().startswith("graph LR")
    assert "<graphml" in g.to_graphml()


async def test_user_documents_become_sources(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("Фактор A увеличивает B на 12%.", encoding="utf-8")
    settings = Settings(backend="mock", runs_dir=str(tmp_path / "runs"))
    result = await Orchestrator(settings, llm=MockLLM()).run("Q", documents=[doc])
    doc_findings = [f for f in result.findings if f.sub_question_id == "d1"]
    assert len(doc_findings) == 1
    assert doc_findings[0].sources[0].source_type == "user_document"
    assert "notes.md" in result.report_markdown
    assert "Reader" in " ".join(json.loads((result.run_dir / "meta.json").read_text())["steps"])


async def test_bad_document_rejected(tmp_path):
    import pytest

    from mars.llm import LLMError

    settings = Settings(backend="mock", runs_dir=str(tmp_path))
    with pytest.raises(LLMError):
        await Orchestrator(settings, llm=MockLLM()).run("Q", documents=[tmp_path / "missing.pdf"])


async def test_html_report_has_graph(tmp_path):
    settings = Settings(backend="mock", runs_dir=str(tmp_path))
    result = await Orchestrator(settings, llm=MockLLM()).run("Q")
    html = (result.run_dir / "report.html").read_text()
    assert '<svg id="graph"' in html and "Фактор A" in html and "<table>" in html
    assert "```mermaid" not in html


def test_cost_estimate_and_usage():
    from types import SimpleNamespace

    from mars.costs import call_cost, estimate
    from mars.llm import Usage

    e = estimate(Settings(), sub_questions=5, documents=2)
    assert 0 < e.usd_low < e.usd_high
    assert estimate(Settings(max_rounds=1)).usd_high < estimate(Settings(max_rounds=3)).usd_high
    assert call_cost("claude-fable-5-2", 1_000_000, 0) == 8.0

    u = Usage()
    u.add(SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=2000,
                          cache_creation_input_tokens=0, server_tool_use=None), "claude-fable-5-2")
    assert u.calls == 1 and u.cache_read_tokens == 2000 and u.cost_usd > 0


async def test_quality_check_catches_broken_citations(tmp_path):
    from mars.quality import evaluate, evaluate_run

    settings = Settings(backend="mock", runs_dir=str(tmp_path))
    result = await Orchestrator(settings, llm=MockLLM()).run("Q")
    q = evaluate_run(result.run_dir)
    assert q.score >= 80 and "## Проверка качества" in result.report_markdown
    assert (result.run_dir / "quality.json").exists()

    from mars.schemas import ReportDraft
    draft = ReportDraft.model_validate(json.loads((result.run_dir / "draft.json").read_text()))
    draft.executive_summary += " Выдуманная ссылка [q9-s9]."
    for f in result.findings:
        f.claims[0].source_ids = []
    bad = evaluate(result.plan, result.findings, result.critique, result.hypotheses, draft)
    failed = {c.name for c in bad.checks if not c.passed}
    assert {"Ссылки в тексте существуют", "У каждого утверждения есть источник"} <= failed
    assert bad.score < q.score
