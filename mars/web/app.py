"""Веб-интерфейс MARS: форма вопроса, прогресс шагов в реальном времени (SSE), отчёт с графом.

Запуск: `mars serve` (нужны зависимости: pip install -e ".[web]").
"""
from __future__ import annotations

import asyncio
import json
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from ..agents.reader import SUPPORTED
from ..config import Settings, load_settings
from ..costs import estimate
from ..llm import LLMError
from ..orchestrator import Orchestrator

MAX_UPLOAD_BYTES = 30 * 1024 * 1024


@dataclass
class Job:
    id: str
    question: str
    events: list[dict] = field(default_factory=list)
    status: str = "running"  # running | done | error
    run_id: str | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def emit(self, kind: str, text: str) -> None:
        self.events.append({"type": kind, "text": text})
        self.changed.set()


def create_app(settings: Settings | None = None) -> FastAPI:
    base = settings or load_settings()
    app = FastAPI(title="MARS")
    jobs: dict[str, Job] = {}
    tasks: set[asyncio.Task] = set()
    runs_root = Path(base.runs_dir).resolve()

    def run_path(run_id: str, name: str) -> Path:
        if not re.fullmatch(r"[\w\-]+", run_id) or name not in {"report.html", "report.md", "graph.json",
                                                              "graph.graphml", "trace.jsonl", "meta.json"}:
            raise HTTPException(404)
        path = (runs_root / run_id / name).resolve()
        if runs_root not in path.parents or not path.is_file():
            raise HTTPException(404)
        return path

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/api/estimate")
    async def api_estimate(sub_questions: int = 5, documents: int = 0, rounds: int | None = None) -> dict:
        s = base.model_copy(update={"max_rounds": rounds} if rounds else {})
        e = estimate(s, max(1, min(sub_questions, 10)), max(0, min(documents, s.max_documents)))
        return {"usd_low": round(e.usd_low, 2), "usd_high": round(e.usd_high, 2), "calls": e.calls,
                "model": s.agent("analyst").model}

    @app.post("/api/runs")
    async def start_run(question: str = Form(...), context: str = Form(""), mock: bool = Form(False),
                        rounds: int = Form(0), files: list[UploadFile] = File(default=[])) -> dict:
        question = question.strip()
        if len(question) < 5:
            raise HTTPException(422, "Вопрос слишком короткий")
        if len(files) > base.max_documents:
            raise HTTPException(422, f"Не больше {base.max_documents} файлов")
        updates: dict = {}
        if mock:
            updates["backend"] = "mock"
        if rounds:
            updates["max_rounds"] = max(1, min(rounds, 5))
        settings = base.model_copy(update=updates)

        upload_dir = Path(tempfile.mkdtemp(prefix="mars-upload-"))
        docs: list[Path] = []
        for f in files:
            name = re.sub(r"[^\w.\-]+", "_", Path(f.filename or "file").name)
            if Path(name).suffix.lower() not in SUPPORTED:
                raise HTTPException(422, f"Формат файла {name} не поддерживается")
            data = await f.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"Файл {name} больше 30 МБ")
            (upload_dir / name).write_bytes(data)
            docs.append(upload_dir / name)

        job = Job(id=uuid.uuid4().hex[:12], question=question)
        jobs[job.id] = job

        async def work() -> None:
            orch = Orchestrator(settings, on_event=lambda msg: job.emit("step", msg))
            try:
                result = await orch.run(question, context, docs)
                job.run_id = result.run_dir.name
                job.status = "done"
                job.emit("done", job.run_id)
            except LLMError as e:
                job.status = "error"
                job.emit("error", str(e))
            except Exception as e:  # чтобы интерфейс не завис на «выполняется»
                job.status = "error"
                job.emit("error", f"Внутренняя ошибка: {type(e).__name__}: {e}")

        task = asyncio.create_task(work())
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"job_id": job.id}

    @app.get("/api/runs/{job_id}/events")
    async def events(job_id: str) -> StreamingResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404)

        async def stream():
            sent = 0
            while True:
                while sent < len(job.events):
                    yield f"data: {json.dumps(job.events[sent], ensure_ascii=False)}\n\n"
                    sent += 1
                if job.status != "running":
                    return
                job.changed.clear()
                try:
                    await asyncio.wait_for(job.changed.wait(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/history")
    async def history() -> list[dict]:
        out = []
        if runs_root.is_dir():
            for d in sorted(runs_root.iterdir(), reverse=True)[:30]:
                manifest, report = d / "manifest.json", d / "report.html"
                if manifest.is_file() and report.is_file():
                    m = json.loads(manifest.read_text(encoding="utf-8"))
                    out.append({"run_id": d.name, "question": m.get("question", ""),
                                "started_at": m.get("started_at", "")})
        return out

    @app.get("/runs/{run_id}/{name}")
    async def run_file(run_id: str, name: str) -> FileResponse:
        return FileResponse(run_path(run_id, name))

    return app


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
<header><h1>MARS</h1><p>Задайте исследовательский вопрос. Шесть агентов найдут источники, разберут противоречия и соберут отчёт.</p></header>
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
 $('links').hidden=false;$('links').innerHTML=['report.md','graph.graphml','trace.jsonl'].map(n=>`<a href="/runs/${id}/${n}" target="_blank">${n}</a>`).join('')}
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
est();hist();
</script></body></html>"""
