"""Веб-интерфейс на mock-бэкенде."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mars.config import Settings  # noqa: E402
from mars.web.app import create_app  # noqa: E402


def test_web_run_end_to_end(tmp_path):
    app = create_app(Settings(backend="mock", runs_dir=str(tmp_path)))
    with TestClient(app) as client:
        assert "MARS" in client.get("/").text
        est = client.get("/api/estimate").json()
        assert est["usd_high"] > est["usd_low"] > 0

        r = client.post("/api/runs", data={"question": "Влияет ли A на B?", "mock": "true"},
                        files=[("files", ("notes.md", b"A raises B by 12%", "text/markdown"))])
        assert r.status_code == 200
        body = client.get(f"/api/runs/{r.json()['job_id']}/events").text
        assert '"type": "done"' in body and "Reader" in body

        hist = client.get("/api/history").json()
        assert len(hist) == 1
        run_id = hist[0]["run_id"]
        assert '<svg id="graph"' in client.get(f"/runs/{run_id}/report.html").text
        assert client.get(f"/runs/{run_id}/manifest.json").status_code == 404  # не в белом списке
        assert client.get("/runs/..%2F..%2Fetc/report.html").status_code == 404


def test_web_validation(tmp_path):
    app = create_app(Settings(backend="mock", runs_dir=str(tmp_path)))
    with TestClient(app) as client:
        assert client.post("/api/runs", data={"question": "hi"}).status_code == 422
        r = client.post("/api/runs", data={"question": "Нормальный вопрос?"},
                        files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
        assert r.status_code == 422
