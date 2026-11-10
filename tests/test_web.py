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


def _app_with_user(tmp_path):
    app = create_app(Settings(backend="mock", runs_dir=str(tmp_path)))
    app.state.store.set_user("pavel", "correct-horse")
    app.state.store.set_user("anna", "another-pass")
    return app


def test_login_required_and_runs_are_private(tmp_path):
    app = _app_with_user(tmp_path)
    with TestClient(app) as pavel, TestClient(app) as anna:
        assert pavel.get("/", follow_redirects=False).headers["location"] == "/login"
        assert pavel.get("/api/history").status_code == 401
        assert pavel.post("/api/runs", data={"question": "Влияет ли A на B?", "mock": "true"}).status_code == 401
        assert pavel.post("/login", data={"username": "pavel", "password": "wrong"}).status_code == 401

        r = pavel.post("/login", data={"username": "pavel", "password": "correct-horse"}, follow_redirects=False)
        assert r.status_code == 303 and "httponly" in r.headers["set-cookie"].lower()
        assert "samesite=strict" in r.headers["set-cookie"].lower()
        assert pavel.get("/api/me").json() == {"user": "pavel", "auth": True}

        job = pavel.post("/api/runs", data={"question": "Влияет ли A на B?", "mock": "true"}).json()["job_id"]
        assert '"type": "done"' in pavel.get(f"/api/runs/{job}/events").text
        run_id = pavel.get("/api/history").json()[0]["run_id"]
        assert pavel.get(f"/runs/{run_id}/report.html").status_code == 200

        anna.post("/login", data={"username": "anna", "password": "another-pass"})
        assert anna.get("/api/history").json() == []
        assert anna.get(f"/runs/{run_id}/report.html").status_code == 404
        assert anna.get(f"/api/runs/{job}/events").status_code == 404

        pavel.post("/logout")
        assert pavel.get("/api/history").status_code == 401


def test_login_throttled_and_cross_site_rejected(tmp_path):
    app = _app_with_user(tmp_path)
    with TestClient(app) as client:
        for _ in range(5):
            assert client.post("/login", data={"username": "pavel", "password": "nope"}).status_code == 401
        assert client.post("/login", data={"username": "pavel", "password": "correct-horse"}).status_code == 429
        r = client.post("/login", data={"username": "anna", "password": "another-pass"},
                        headers={"origin": "https://evil.example"})
        assert r.status_code == 403


def test_store_passwords_and_interrupted_jobs(tmp_path):
    from mars.web.store import Store

    store = Store(tmp_path / "t.db")
    assert not store.has_users()
    pytest.raises(ValueError, store.set_user, "pavel", "short")
    store.set_user("pavel", "long-enough")
    assert store.check_password("pavel", "long-enough") and not store.check_password("pavel", "x" * 12)
    assert not store.check_password("ghost", "long-enough")
    token = store.create_session("pavel")
    assert store.session_user(token) == "pavel" and store.session_user("forged") is None
    store.set_user("pavel", "new-password")  # смена пароля сбрасывает сессии
    assert store.session_user(token) is None

    store.add_job("j1", "pavel", "Q")
    Store(tmp_path / "t.db").fail_interrupted()
    job = store.get_job("j1")
    assert job["status"] == "error" and "перезапущен" in job["events"][-1]["text"]


def test_export_downloads(tmp_path):
    pytest.importorskip("fpdf")
    pytest.importorskip("docx")
    app = create_app(Settings(backend="mock", runs_dir=str(tmp_path)))
    with TestClient(app) as client:
        job = client.post("/api/runs", data={"question": "Влияет ли A на B?", "mock": "true"}).json()["job_id"]
        client.get(f"/api/runs/{job}/events")
        run_id = client.get("/api/history").json()[0]["run_id"]
        pdf = client.get(f"/runs/{run_id}/report.pdf")
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
        docx = client.get(f"/runs/{run_id}/report.docx")
        assert docx.status_code == 200 and docx.content[:2] == b"PK"
