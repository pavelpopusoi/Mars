"""Веб-интерфейс MARS: форма вопроса, прогресс шагов (SSE), отчёт с графом, экспорт.

Режимы:
* без пользователей в базе — локальный режим без входа (только для своего компьютера);
* есть пользователи (`mars user add имя`) — вход по паролю, у каждого свои запуски.
Запуски идут через очередь: одновременно выполняется не больше `max_concurrent_runs`.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse

from ..agents.reader import SUPPORTED
from ..config import Settings, load_settings
from ..costs import estimate
from ..export import to_docx, to_pdf
from ..llm import LLMError
from ..orchestrator import Orchestrator
from .store import SESSION_TTL, Store

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_ACTIVE_PER_USER = 3
LOGIN_ATTEMPTS, LOGIN_WINDOW = 5, 300
COOKIE = "mars_session"
LOCAL_USER = "local"
STATIC_FILES = {"report.html", "report.md", "graph.json", "graph.graphml", "trace.jsonl", "meta.json",
                "quality.json"}
EXPORTS = {"report.pdf": to_pdf, "report.docx": to_docx}


def create_app(settings: Settings | None = None) -> FastAPI:
    base = settings or load_settings()
    app = FastAPI(title="MARS", docs_url=None, redoc_url=None, openapi_url=None)
    store = Store(base.database)
    store.fail_interrupted()
    app.state.store = store
    runs_root = Path(base.runs_dir).resolve()
    slots = asyncio.Semaphore(max(1, base.max_concurrent_runs))
    wakeups: dict[str, asyncio.Event] = {}
    tasks: set[asyncio.Task] = set()
    failures: dict[str, list[float]] = {}

    # ------------------------------------------------------------ доступ
    def user_of(request: Request) -> str | None:
        if not store.has_users():
            return LOCAL_USER
        return store.session_user(request.cookies.get(COOKIE))

    def require_user(request: Request) -> str:
        user = user_of(request)
        if user is None:
            raise HTTPException(401, "Нужно войти")
        return user

    def check_origin(request: Request) -> None:
        """Защита от запросов с чужих сайтов: Origin, если он есть, должен совпадать с адресом сервера."""
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Запрос с другого сайта отклонён")

    def emit(job_id: str, kind: str, text: str) -> None:
        store.add_event(job_id, kind, text)
        if job_id in wakeups:
            wakeups[job_id].set()

    def run_file_path(run_id: str, name: str) -> Path:
        if not re.fullmatch(r"[\w\-]+", run_id):
            raise HTTPException(404)
        path = (runs_root / run_id / name).resolve()
        if runs_root not in path.parents:
            raise HTTPException(404)
        return path

    # ------------------------------------------------------------ вход
    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        if user_of(request):
            return RedirectResponse("/", 303)
        return LOGIN_HTML.replace("__ERROR__", "")

    @app.post("/login")
    async def login(request: Request, username: str = Form(...), password: str = Form(...)):
        check_origin(request)
        key = f"{username.lower()}|{request.client.host if request.client else ''}"
        now = time.time()
        recent = [t for t in failures.get(key, []) if now - t < LOGIN_WINDOW]
        if len(recent) >= LOGIN_ATTEMPTS:
            return HTMLResponse(LOGIN_HTML.replace(
                "__ERROR__", "Слишком много попыток. Подождите 5 минут и попробуйте снова."), 429)
        if not await asyncio.to_thread(store.check_password, username, password):
            failures[key] = [*recent, now]
            return HTMLResponse(LOGIN_HTML.replace("__ERROR__", "Неверное имя или пароль."), 401)
        failures.pop(key, None)
        response = RedirectResponse("/", 303)
        response.set_cookie(COOKIE, store.create_session(username), max_age=SESSION_TTL, httponly=True,
                            samesite="strict", secure=request.url.scheme == "https")
        return response

    @app.post("/logout")
    async def logout(request: Request):
        check_origin(request)
        store.drop_session(request.cookies.get(COOKIE))
        response = RedirectResponse("/login", 303)
        response.delete_cookie(COOKIE)
        return response

    # ------------------------------------------------------------ страницы и API
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        user = user_of(request)
        if user is None:
            return RedirectResponse("/login", 303)
        return INDEX_HTML

    @app.get("/api/me")
    async def me(request: Request) -> dict:
        user = require_user(request)
        return {"user": user, "auth": user != LOCAL_USER}

    @app.get("/api/estimate")
    async def api_estimate(request: Request, sub_questions: int = 5, documents: int = 0,
                           rounds: int | None = None) -> dict:
        require_user(request)
        s = base.model_copy(update={"max_rounds": max(1, min(rounds, 5))} if rounds else {})
        e = estimate(s, max(1, min(sub_questions, 10)), max(0, min(documents, s.max_documents)))
        return {"usd_low": round(e.usd_low, 2), "usd_high": round(e.usd_high, 2), "calls": e.calls,
                "model": s.ollama_model if s.backend == "ollama" else s.agent("analyst").model}

    @app.post("/api/runs")
    async def start_run(request: Request, question: str = Form(...), context: str = Form(""),
                        mock: bool = Form(False), rounds: int = Form(0),
                        files: list[UploadFile] = File(default=[])) -> dict:
        user = require_user(request)
        check_origin(request)
        question = question.strip()
        if len(question) < 5:
            raise HTTPException(422, "Вопрос слишком короткий")
        if len(files) > base.max_documents:
            raise HTTPException(422, f"Не больше {base.max_documents} файлов")
        if store.active_count(user) >= MAX_ACTIVE_PER_USER:
            raise HTTPException(429, f"У вас уже {MAX_ACTIVE_PER_USER} задачи в работе. Дождитесь завершения.")
        updates: dict = {}
        if mock:
            updates["backend"] = "mock"
        if rounds:
            updates["max_rounds"] = max(1, min(rounds, 5))
        settings = base.model_copy(update=updates)

        upload_dir = Path(tempfile.mkdtemp(prefix="mars-upload-"))
        docs: list[Path] = []
        try:
            for f in files:
                name = re.sub(r"[^\w.\-]+", "_", Path(f.filename or "file").name)
                if Path(name).suffix.lower() not in SUPPORTED:
                    raise HTTPException(422, f"Формат файла {name} не поддерживается")
                data = await f.read()
                if len(data) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Файл {name} больше 30 МБ")
                (upload_dir / name).write_bytes(data)
                docs.append(upload_dir / name)
        except HTTPException:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise

        job_id = uuid.uuid4().hex[:12]
        store.add_job(job_id, user, question)
        wakeups[job_id] = asyncio.Event()

        async def work() -> None:
            try:
                ahead = store.queued_ahead(job_id)
                if slots.locked():
                    emit(job_id, "step", f"В очереди, перед вами задач: {ahead}")
                async with slots:
                    store.set_status(job_id, "running")
                    orch = Orchestrator(settings, on_event=lambda msg: emit(job_id, "step", msg))
                    result = await orch.run(question, context, docs)
                store.set_status(job_id, "done", result.run_dir.name)
                emit(job_id, "done", result.run_dir.name)
            except LLMError as e:
                store.set_status(job_id, "error")
                emit(job_id, "error", str(e))
            except Exception as e:  # чтобы интерфейс не завис на «выполняется»
                store.set_status(job_id, "error")
                emit(job_id, "error", f"Внутренняя ошибка: {type(e).__name__}: {e}")
            finally:
                shutil.rmtree(upload_dir, ignore_errors=True)
                event = wakeups.pop(job_id, None)
                if event:
                    event.set()

        task = asyncio.create_task(work())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"job_id": job_id}

    @app.get("/api/runs/{job_id}/events")
    async def events(request: Request, job_id: str) -> StreamingResponse:
        user = require_user(request)
        job = store.get_job(job_id)
        if job is None or job["username"] != user:
            raise HTTPException(404)

        async def stream():
            sent = 0
            while True:
                current = store.get_job(job_id)
                for event in current["events"][sent:]:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                sent = len(current["events"])
                if current["status"] in ("done", "error"):
                    return
                wake = wakeups.get(job_id)
                if wake is None:
                    continue  # задача только что завершилась — дочитываем события
                wake.clear()
                try:
                    await asyncio.wait_for(wake.wait(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/history")
    async def history(request: Request) -> list[dict]:
        user = require_user(request)
        if user != LOCAL_USER:
            return [{"run_id": j["run_id"], "question": j["question"], "status": j["status"]}
                    for j in store.history(user) if j["run_id"]]
        out = []  # локальный режим: показываем и запуски из командной строки
        if runs_root.is_dir():
            for d in sorted((p for p in runs_root.iterdir() if p.is_dir()), reverse=True)[:30]:
                manifest, report = d / "manifest.json", d / "report.html"
                if manifest.is_file() and report.is_file():
                    m = json.loads(manifest.read_text(encoding="utf-8"))
                    out.append({"run_id": d.name, "question": m.get("question", ""), "status": "done"})
        return out

    @app.get("/runs/{run_id}/{name}")
    async def run_file(request: Request, run_id: str, name: str) -> FileResponse:
        user = require_user(request)
        if name not in STATIC_FILES and name not in EXPORTS:
            raise HTTPException(404)
        path = run_file_path(run_id, name)
        if user != LOCAL_USER and not store.owns_run(user, run_id):
            raise HTTPException(404)
        if name in EXPORTS and not path.is_file():
            if not (path.parent / "report.md").is_file():
                raise HTTPException(404)
            try:
                await asyncio.to_thread(EXPORTS[name], path.parent)
            except LLMError as e:
                raise HTTPException(501, str(e)) from e
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, filename=name if name in EXPORTS else None)

    return app


LOGIN_HTML = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MARS — вход</title><style>
:root{--bg:#f5f6f4;--surface:#fff;--fg:#17201f;--muted:#5b6763;--line:#d9ddd8;--accent:#0d6b62;--on:#fff;--err:#a8321f}
@media (prefers-color-scheme:dark){:root{--bg:#121716;--surface:#1a2120;--fg:#e6ebe9;--muted:#9aa7a3;--line:#2c3533;
--accent:#4fc2b3;--on:#0c1413;--err:#f08a75;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:16px;background:var(--bg);
color:var(--fg);font:16px/1.55 system-ui,sans-serif}
form{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:24px;width:100%;max-width:340px;
display:flex;flex-direction:column;gap:14px}h1{margin:0;font-size:1.5rem}
label{font-size:.85rem;color:var(--muted);display:flex;flex-direction:column;gap:4px}
input{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:9px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button{font:inherit;font-weight:600;background:var(--accent);color:var(--on);border:0;border-radius:6px;padding:10px;cursor:pointer}
.err{color:var(--err);font-size:.9rem;min-height:1.2em;margin:0}
</style></head><body><form method="post" action="/login"><h1>MARS</h1>
<label>Имя пользователя<input id="username" name="username" autocomplete="username" required autofocus></label>
<label>Пароль<input id="password" name="password" type="password" autocomplete="current-password" required></label>
<p class="err" role="alert">__ERROR__</p><button>Войти</button></form></body></html>"""


INDEX_HTML = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MARS</title><style>
:root{--bg:#f5f6f4;--surface:#fff;--fg:#17201f;--muted:#5b6763;--line:#d9ddd8;--accent:#0d6b62;--on:#fff;--err:#a8321f}
@media (prefers-color-scheme:dark){:root{--bg:#121716;--surface:#1a2120;--fg:#e6ebe9;--muted:#9aa7a3;--line:#2c3533;
--accent:#4fc2b3;--on:#0c1413;--err:#f08a75;color-scheme:dark}}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 16px 48px;display:grid;gap:24px;grid-template-columns:minmax(0,1fr)}
@media(min-width:900px){main{grid-template-columns:380px minmax(0,1fr)}header{grid-column:1/-1}}
h1{margin:0;font-size:1.7rem}header p{margin:4px 0 0;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px;display:flex;
flex-direction:column;gap:12px;min-width:0}
label{font-size:.85rem;color:var(--muted);display:flex;flex-direction:column;gap:4px}
textarea,input[type=number],input[type=file]{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);
border-radius:6px;padding:8px;width:100%}textarea{resize:vertical}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.row label{flex-direction:row;align-items:center;gap:6px}
button{font:inherit;font-weight:600;background:var(--accent);color:var(--on);border:0;border-radius:6px;padding:10px 16px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}#est{color:var(--muted);font-size:.88rem}
ol{margin:0;padding-left:1.3em;display:flex;flex-direction:column;gap:4px}ol:empty::after{content:"Шаги появятся здесь после запуска.";color:var(--muted);margin-left:-1.3em}
.err{color:var(--err)}h2{margin:0;font-size:1.05rem}
iframe{width:100%;height:78vh;border:1px solid var(--line);border-radius:10px;background:var(--surface)}
#hist{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px;font-size:.92rem}
#hist a{color:var(--accent);cursor:pointer;text-decoration:underline}#links{font-size:.9rem;display:flex;gap:14px;flex-wrap:wrap}
#links a{color:var(--accent)}
</style></head><body><main>
<header><div id="who" hidden style="float:right;display:flex;gap:10px;align-items:center;font-size:.9rem;color:var(--muted)"><span id="uname"></span><form method="post" action="/logout" style="margin:0"><button style="padding:6px 12px;font-weight:500">Выйти</button></form></div><h1>MARS</h1><p>Задайте исследовательский вопрос. Шесть агентов найдут источники, разберут противоречия и соберут отчёт.</p></header>
<div style="display:flex;flex-direction:column;gap:24px;min-width:0">
<form class="card" id="f">
<label>Вопрос<textarea id="q" name="question" rows="4" required minlength="5" placeholder="Сможет ли натрий-ионная технология вытеснить LFP в стационарных накопителях к 2030 году?"></textarea></label>
<label>Контекст (необязательно)<textarea id="ctx" name="context" rows="2"></textarea></label>
<label>Свои файлы: PDF, TXT, MD, CSV, JSON<input id="files" type="file" name="files" multiple accept=".pdf,.txt,.md,.csv,.json"></label>
<div class="row"><label>Раундов <input id="rounds" type="number" name="rounds" min="1" max="5" value="2" style="width:70px"></label>
<label><input id="mock" type="checkbox" name="mock" value="true"> Демо без API</label></div>
<div id="est"></div><button id="go">Начать исследование</button></form>
<div class="card"><h2>Ход работы</h2><ol id="steps"></ol><div id="links" hidden></div></div>
<div class="card"><h2>Прошлые запуски</h2><ul id="hist"></ul></div></div>
<div style="min-width:0"><iframe id="rep" title="Отчёт" hidden></iframe>
<div class="card" id="empty"><h2>Отчёт</h2><p style="margin:0;color:var(--muted)">Здесь откроется отчёт с источниками, таблицей противоречий, гипотезами и интерактивным графом знаний.</p></div></div>
</main><script>
const $=id=>document.getElementById(id);
function show(id){$('rep').src='/runs/'+id+'/report.html';$('rep').hidden=false;$('empty').hidden=true;
 $('links').hidden=false;$('links').innerHTML=['report.pdf','report.docx','report.md','graph.graphml','trace.jsonl'].map(n=>`<a href="/runs/${id}/${n}" target="_blank">${n}</a>`).join('')}
async function hist(){const r=await fetch('/api/history');const h=await r.json();$('hist').innerHTML='';
 if(!h.length)$('hist').innerHTML='<li style="color:var(--muted)">Пока нет запусков.</li>';
 h.forEach(x=>{const li=document.createElement('li'),a=document.createElement('a');a.textContent=x.question.slice(0,90);a.tabIndex=0;
 a.onclick=()=>show(x.run_id);a.onkeydown=e=>{if(e.key==='Enter')show(x.run_id)};li.append(a);$('hist').append(li)})}
async function est(){if($('mock').checked){$('est').textContent='Демо-режим: бесплатно, данные синтетические.';return}
 const r=await fetch(`/api/estimate?documents=${$('files').files.length}&rounds=${$('rounds').value||2}`);const e=await r.json();
 $('est').textContent=`Оценка: $${e.usd_low} – $${e.usd_high}, около ${e.calls} вызовов (${e.model}).`}
['files','rounds','mock'].forEach(i=>$(i).addEventListener('change',est));
$('f').addEventListener('submit',async ev=>{ev.preventDefault();$('go').disabled=true;$('steps').innerHTML='';
 const add=(t,c)=>{const li=document.createElement('li');li.textContent=t;if(c)li.className=c;$('steps').append(li)};
 const r=await fetch('/api/runs',{method:'POST',body:new FormData($('f'))});
 if(!r.ok){const e=await r.json().catch(()=>({}));add('Не удалось запустить: '+(e.detail||r.status),'err');$('go').disabled=false;return}
 const {job_id}=await r.json();const es=new EventSource('/api/runs/'+job_id+'/events');
 es.onmessage=m=>{const e=JSON.parse(m.data);if(e.type==='step')add(e.text);
  else{es.close();$('go').disabled=false;if(e.type==='done'){show(e.text);hist()}else add('Ошибка: '+e.text,'err')}};
 es.onerror=()=>{if(es.readyState===2){add('Соединение потеряно. Обновите страницу: готовый отчёт появится в прошлых запусках.','err');$('go').disabled=false}}});
fetch('/api/me').then(r=>r.json()).then(m=>{if(m.auth){$('uname').textContent=m.user;$('who').hidden=false}});
est();hist();
</script></body></html>"""
