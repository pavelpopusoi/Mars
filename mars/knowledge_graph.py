"""Граф знаний: сущности (узлы) и связи (рёбра) из утверждений + гипотезы и противоречия."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import pairwise

from .schemas import Contradiction, Hypothesis, SearchFindings


@dataclass
class KnowledgeGraph:
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)

    @staticmethod
    def _key(label: str) -> str:
        return re.sub(r"\s+", " ", label.strip().lower())

    def add_node(self, label: str, kind: str = "entity") -> str:
        key = self._key(label)
        node = self.nodes.setdefault(key, {"id": f"n{len(self.nodes) + 1}", "label": label.strip(), "kind": kind,
                                           "degree": 0})
        return node["id"]

    def add_edge(self, src: str, dst: str, label: str, **attrs) -> None:
        for e in self.edges:  # одинаковые связи из разных источников склеиваются, растёт вес
            if (e["source"], e["target"], e["label"], e.get("style")) == (src, dst, label, attrs.get("style")):
                e["weight"] = e.get("weight", 1) + 1
                if attrs.get("claim_id"):
                    e["claim_id"] = f'{e.get("claim_id", "")}, {attrs["claim_id"]}'.strip(", ")
                return
        self.edges.append({"source": src, "target": dst, "label": label, **attrs})
        for n in self.nodes.values():
            if n["id"] in (src, dst):
                n["degree"] += 1

    @classmethod
    def build(cls, findings: list[SearchFindings], contradictions: list[Contradiction],
              hypotheses: list[Hypothesis]) -> KnowledgeGraph:
        g = cls()
        claim_edges: dict[str, tuple[str, str]] = {}
        for f in findings:
            for c in f.claims:
                s, o = g.add_node(c.subject), g.add_node(c.object)
                g.add_edge(s, o, c.predicate, claim_id=c.id, confidence=c.confidence, sources=c.source_ids)
                claim_edges[c.id] = (s, o)
        for x in contradictions:
            ids = [cid for cid in x.claim_ids if cid in claim_edges]
            for a, b in pairwise(ids):
                g.add_edge(claim_edges[a][0], claim_edges[b][1], "противоречие", contradiction_id=x.id,
                           style="conflict")
        for h in hypotheses:
            hid = g.add_node(f"{h.id}: {h.statement[:60]}", kind="hypothesis")
            for cid in h.supporting_claim_ids:
                if cid in claim_edges:
                    g.add_edge(hid, claim_edges[cid][0], "опирается на", hypothesis_id=h.id, style="hypothesis")
        return g

    def to_json(self) -> str:
        return json.dumps({"nodes": list(self.nodes.values()), "edges": self.edges}, ensure_ascii=False, indent=2)

    def to_mermaid(self, max_edges: int = 60) -> str:
        def esc(t: str) -> str:
            return t.replace('"', "'").replace("\n", " ")[:70]

        lines = ["graph LR"]
        for n in self.nodes.values():
            shape = ('{{"%s"}}' if n["kind"] == "hypothesis" else '["%s"]') % esc(n["label"])
            lines.append(f"  {n['id']}{shape}")
        for e in self.edges[:max_edges]:
            arrow = "-.->" if e.get("style") in ("conflict", "hypothesis") else "-->"
            lines.append(f"  {e['source']} {arrow}|{esc(e['label'])}| {e['target']}")
        return "\n".join(lines)

    def to_graphml(self) -> str:
        from xml.sax.saxutils import escape

        out = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
               '<key id="label" for="all" attr.name="label" attr.type="string"/>',
               '<key id="kind" for="node" attr.name="kind" attr.type="string"/>',
               '<graph edgedefault="directed">']
        for n in self.nodes.values():
            out.append(f'<node id="{n["id"]}"><data key="label">{escape(n["label"])}</data>'
                       f'<data key="kind">{n["kind"]}</data></node>')
        for i, e in enumerate(self.edges):
            out.append(f'<edge id="e{i}" source="{e["source"]}" target="{e["target"]}">'
                       f'<data key="label">{escape(e["label"])}</data></edge>')
        out += ["</graph>", "</graphml>"]
        return "\n".join(out)
