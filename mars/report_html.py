"""Самодостаточный HTML-отчёт: текст из report.md + интерактивный граф знаний (без внешних библиотек)."""
from __future__ import annotations

import html
import json
import re

import markdown

from .knowledge_graph import KnowledgeGraph

_CSS = """
:root{--bg:#f5f6f4;--surface:#fff;--fg:#17201f;--muted:#5b6763;--line:#d9ddd8;--accent:#0d6b62;
--warn:#a8501f;--hyp:#6a4fb3}
@media (prefers-color-scheme:dark){:root{--bg:#121716;--surface:#1a2120;--fg:#e6ebe9;--muted:#9aa7a3;
--line:#2c3533;--accent:#4fc2b3;--warn:#e48a5a;--hyp:#a996e8;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:2rem;line-height:1.15}h2{margin-top:2.2em;padding-top:.6em;border-top:1px solid var(--line)}
a{color:var(--accent)}blockquote{margin:0;padding:10px 16px;background:var(--surface);
border-left:3px solid var(--accent);border-radius:4px}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.93rem}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}code{background:var(--surface);padding:1px 5px;border-radius:4px}
#graph{background:var(--surface);border:1px solid var(--line);border-radius:10px;width:100%;height:520px;
touch-action:none;cursor:grab}
#graph text{font:12px system-ui,sans-serif;fill:var(--fg);pointer-events:none}
#graph .edge{stroke:var(--muted);stroke-width:1.4;fill:none;opacity:.75}
#graph .edge.conflict{stroke:var(--warn);stroke-dasharray:5 4;stroke-width:2}
#graph .edge.hypothesis{stroke:var(--hyp);stroke-dasharray:2 4}
#graph .elabel{font-size:10.5px;fill:var(--muted)}
#graph .dim{opacity:.12}
.legend{display:flex;flex-wrap:wrap;gap:16px;color:var(--muted);font-size:.88rem;margin:8px 0}
.legend i{display:inline-block;width:22px;border-top:2px solid var(--muted);vertical-align:middle;margin-right:6px}
.legend i.c{border-top:2px dashed var(--warn)}.legend i.h{border-top:2px dotted var(--hyp)}
#info{min-height:1.6em;color:var(--muted);font-size:.9rem}
"""

_JS = r"""
(function(){
const G=__GRAPH__, svg=document.getElementById('graph'), info=document.getElementById('info');
const NS='http://www.w3.org/2000/svg', W=svg.clientWidth||900, H=520;
svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const nodes=G.nodes.map((n,i)=>({...n,x:W/2+Math.cos(i*2.4)*(120+i*9),y:H/2+Math.sin(i*2.4)*(90+i*7),vx:0,vy:0}));
const byId=Object.fromEntries(nodes.map(n=>[n.id,n]));
const edges=G.edges.filter(e=>byId[e.source]&&byId[e.target]).map(e=>({...e,s:byId[e.source],t:byId[e.target]}));
const pair={};edges.forEach(e=>{const k=[e.source,e.target].sort().join('|');e.k=(pair[k]=(pair[k]||0)+1)});
const el=(t,a,p)=>{const x=document.createElementNS(NS,t);for(const k in a)x.setAttribute(k,a[k]);(p||svg).appendChild(x);return x};
const defs=el('defs',{});const mk=el('marker',{id:'arr',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'},defs);
el('path',{d:'M0 0L10 5L0 10z',fill:css('--muted')},mk);
edges.forEach(e=>{e.p=el('path',{class:'edge '+(e.style||''),'marker-end':'url(#arr)'});e.l=el('text',{class:'elabel','text-anchor':'middle'});e.l.textContent=e.label});
nodes.forEach(n=>{n.r=n.kind==='hypothesis'?11:9+Math.min(8,n.degree||0);
 n.c=el('circle',{r:n.r,fill:n.kind==='hypothesis'?css('--hyp'):css('--accent'),stroke:css('--surface'),'stroke-width':2,style:'cursor:pointer'});
 n.tx=el('text',{'text-anchor':'middle'});n.tx.textContent=n.label.length>34?n.label.slice(0,33)+'…':n.label});
function draw(){edges.forEach(e=>{const dx=e.t.x-e.s.x,dy=e.t.y-e.s.y,d=Math.hypot(dx,dy)||1,b=(e.k-1)*26+(e.k>1?10:0);
 const mx=(e.s.x+e.t.x)/2-dy/d*b,my=(e.s.y+e.t.y)/2+dx/d*b;
 const ex=e.t.x-dx/d*(e.t.r+3),ey=e.t.y-dy/d*(e.t.r+3);
 e.p.setAttribute('d',`M${e.s.x} ${e.s.y}Q${mx} ${my} ${ex} ${ey}`);e.l.setAttribute('x',(e.s.x+2*mx+e.t.x)/4);e.l.setAttribute('y',(e.s.y+2*my+e.t.y)/4-3)});
 nodes.forEach(n=>{n.c.setAttribute('cx',n.x);n.c.setAttribute('cy',n.y);n.tx.setAttribute('x',n.x);n.tx.setAttribute('y',n.y+n.r+14)})}
let alpha=1,drag=null;
function step(){for(const a of nodes){for(const b of nodes){if(a===b)continue;const dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy+60,f=5200/d2;a.vx+=dx/Math.sqrt(d2)*f*alpha;a.vy+=dy/Math.sqrt(d2)*f*alpha}
 a.vx+=(W/2-a.x)*.012*alpha;a.vy+=(H/2-a.y)*.016*alpha}
 for(const e of edges){const dx=e.t.x-e.s.x,dy=e.t.y-e.s.y,d=Math.hypot(dx,dy)||1,f=(d-170)*.03*alpha;e.s.vx+=dx/d*f;e.s.vy+=dy/d*f;e.t.vx-=dx/d*f;e.t.vy-=dy/d*f}
 for(const n of nodes){if(n===drag)continue;n.vx*=.78;n.vy*=.78;n.x=Math.max(30,Math.min(W-30,n.x+n.vx));n.y=Math.max(24,Math.min(H-34,n.y+n.vy))}
alpha*=.985}
function tick(){step();draw();if(alpha>.02||drag)requestAnimationFrame(tick)}
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
if(reduce){for(let i=0;i<250;i++)step()}
function pt(ev){const r=svg.getBoundingClientRect();return{x:(ev.clientX-r.left)*W/r.width,y:(ev.clientY-r.top)*H/r.height}}
function focus(n){const near=new Set([n.id]);edges.forEach(e=>{if(e.s===n)near.add(e.t.id);if(e.t===n)near.add(e.s.id)});
 nodes.forEach(m=>{const d=!near.has(m.id);m.c.classList.toggle('dim',d);m.tx.classList.toggle('dim',d)});
 const mine=edges.filter(e=>e.s===n||e.t===n);edges.forEach(e=>{const d=!mine.includes(e);e.p.classList.toggle('dim',d);e.l.classList.toggle('dim',d)});
 info.textContent=n.label+' — связей: '+mine.length+(mine.length?'. '+mine.slice(0,4).map(e=>e.s.label+' → '+e.label+' → '+e.t.label+(e.claim_id?' ('+e.claim_id+')':'')).join('; '):'')}
function clear(){svg.querySelectorAll('.dim').forEach(x=>x.classList.remove('dim'));info.textContent='Наведите на узел или перетащите его.'}
nodes.forEach(n=>{n.c.addEventListener('pointerdown',ev=>{drag=n;svg.setPointerCapture(ev.pointerId);focus(n);if(alpha<.05){alpha=.3;requestAnimationFrame(tick)}});
 n.c.addEventListener('pointerenter',()=>{if(!drag)focus(n)});n.c.addEventListener('pointerleave',()=>{if(!drag)clear()})});
svg.addEventListener('pointermove',ev=>{if(!drag)return;const p=pt(ev);drag.x=p.x;drag.y=p.y;drag.vx=drag.vy=0;alpha=Math.max(alpha,.3)});
svg.addEventListener('pointerup',()=>{drag=null});
clear();if(reduce)draw();else requestAnimationFrame(tick);
})();
"""


def render_html(report_md: str, graph: KnowledgeGraph, title: str = "MARS report") -> str:
    placeholder = "MARSGRAPHPLACEHOLDER"
    md = re.sub(r"```mermaid\n.*?\n```", placeholder, report_md, flags=re.S)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
    graph_html = (
        '<svg id="graph" role="img" aria-label="Граф знаний"></svg>'
        '<div class="legend"><span><i></i>утверждение</span><span><i class="c"></i>противоречие</span>'
        '<span><i class="h"></i>опора гипотезы</span></div><div id="info"></div>'
    )
    body = body.replace(f"<p>{placeholder}</p>", graph_html).replace(placeholder, graph_html)
    data = json.dumps({"nodes": list(graph.nodes.values()), "edges": graph.edges}, ensure_ascii=False)
    data = data.replace("</", "<\\/")
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>{body}</main>"
        f"<script>{_JS.replace('__GRAPH__', data)}</script></body></html>"
    )
