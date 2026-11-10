"""Экспорт отчёта в Word (.docx) и PDF. Нужны необязательные зависимости: pip install -e ".[export]"."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMError


@dataclass
class Block:
    kind: str  # heading | para | bullets | numbered | table | quote
    text: str = ""
    level: int = 0
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_markdown(md: str, graph: dict | None = None) -> list[Block]:
    """Разбор того подмножества Markdown, которое выдаёт mars/report.py."""
    blocks: list[Block] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            lang = line[3:].strip()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            if lang == "mermaid":  # интерактивный граф заменяем таблицей связей
                blocks.append(_graph_table(graph))
            else:
                blocks.append(Block("para", "\n".join(lines[i + 1:j])))
            i = j + 1
        elif m := re.match(r"(#{1,4})\s+(.*)", line):
            blocks.append(Block("heading", m.group(2), level=len(m.group(1))))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.fullmatch(r"\|[\s\-:|]+\|?", lines[i].strip()):
                    rows.append(_cells(lines[i]))
                i += 1
            blocks.append(Block("table", rows=rows))
        elif line.startswith(">"):
            parts = []
            while i < len(lines) and lines[i].startswith(">"):
                parts.append(lines[i].lstrip("> ").strip())
                i += 1
            blocks.append(Block("quote", " ".join(p for p in parts if p)))
        elif re.match(r"[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"[-*]\s+", lines[i]):
                items.append(re.sub(r"^[-*]\s+", "", lines[i]))
                i += 1
            blocks.append(Block("bullets", items=items))
        elif re.match(r"\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i]))
                i += 1
            blocks.append(Block("numbered", items=items))
        elif line.strip():
            parts = []
            while i < len(lines) and lines[i].strip() and not re.match(r"(#{1,4}\s|\||>|```|[-*]\s|\d+\.\s)", lines[i]):
                parts.append(lines[i].strip())
                i += 1
            blocks.append(Block("para", " ".join(parts)))
        else:
            i += 1
    return blocks


def _graph_table(graph: dict | None) -> Block:
    if not graph or not graph.get("edges"):
        return Block("para", "Граф знаний доступен в report.html и graph.graphml.")
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    rows = [["От", "Связь", "К", "Утверждения"]]
    for e in graph["edges"][:60]:
        rows.append([labels.get(e["source"], "?"), e["label"], labels.get(e["target"], "?"),
                     e.get("claim_id") or e.get("contradiction_id") or e.get("hypothesis_id") or ""])
    return Block("table", rows=rows)


_INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))")


def inline_parts(text: str) -> list[tuple[str, str]]:
    """Разбить строку на куски: ('text'|'bold'|'code'|'link', содержимое)."""
    out: list[tuple[str, str]] = []
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            out.append(("bold", part[2:-2]))
        elif part.startswith("`"):
            out.append(("code", part[1:-1]))
        elif m := re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part):
            out.append(("link", f"{m.group(1)} ({m.group(2)})"))
        else:
            out.append(("text", part))
    return out


def plain(text: str) -> str:
    return "".join(t for _, t in inline_parts(text))


def _load(run_dir: str | Path) -> tuple[Path, list[Block]]:
    d = Path(run_dir)
    report = d / "report.md"
    if not report.is_file():
        raise LLMError(f"В папке {d} нет report.md")
    graph_file = d / "graph.json"
    graph = json.loads(graph_file.read_text(encoding="utf-8")) if graph_file.is_file() else None
    return d, parse_markdown(report.read_text(encoding="utf-8"), graph)


# ------------------------------------------------------------------ Word
def to_docx(run_dir: str | Path) -> Path:
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError as e:
        raise LLMError('Для экспорта в Word установите: pip install -e ".[export]"') from e
    d, blocks = _load(run_dir)
    doc = Document()
    doc.styles["Normal"].font.size = Pt(11)

    def fill(par, text: str) -> None:
        for kind, chunk in inline_parts(text):
            run = par.add_run(chunk)
            run.bold = kind == "bold"
            if kind == "code":
                run.font.name = "Consolas"

    for b in blocks:
        if b.kind == "heading":
            doc.add_heading(plain(b.text), level=min(b.level, 4) - 1)
        elif b.kind == "para":
            fill(doc.add_paragraph(), b.text)
        elif b.kind == "quote":
            fill(doc.add_paragraph(style="Intense Quote"), b.text)
        elif b.kind in ("bullets", "numbered"):
            style = "List Bullet" if b.kind == "bullets" else "List Number"
            for item in b.items:
                fill(doc.add_paragraph(style=style), item)
        elif b.kind == "table" and b.rows:
            cols = max(len(r) for r in b.rows)
            table = doc.add_table(rows=len(b.rows), cols=cols)
            table.style = "Light Grid Accent 1"
            for r, row in enumerate(b.rows):
                for c in range(cols):
                    cell = table.cell(r, c)
                    cell.text = ""
                    fill(cell.paragraphs[0], row[c] if c < len(row) else "")
                    if r == 0:
                        for run in cell.paragraphs[0].runs:
                            run.bold = True
            doc.add_paragraph()
    path = d / "report.docx"
    doc.save(str(path))
    return path


# ------------------------------------------------------------------ PDF
FONT_CANDIDATES = [  # нужен шрифт с кириллицей; (обычный, жирный)
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]


def find_font(font: str | None = None) -> tuple[str, str]:
    font = font or os.getenv("MARS_PDF_FONT")
    if font:
        if not Path(font).is_file():
            raise LLMError(f"Шрифт не найден: {font}")
        bold = re.sub(r"(\.ttf)$", r"-Bold\1", font, flags=re.I)
        return font, bold if Path(bold).is_file() else font
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).is_file():
            return regular, bold if Path(bold).is_file() else regular
    raise LLMError("Не найден шрифт с кириллицей для PDF. Укажите путь к .ttf: --font путь или MARS_PDF_FONT")


def to_pdf(run_dir: str | Path, font: str | None = None) -> Path:
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError as e:
        raise LLMError('Для экспорта в PDF установите: pip install -e ".[export]"') from e
    d, blocks = _load(run_dir)
    regular, bold = find_font(font)
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(18, 16, 18)
    pdf.add_font("mars", "", regular)
    pdf.add_font("mars", "B", bold)
    pdf.add_page()
    nl = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}
    sizes = {1: 20, 2: 15, 3: 12.5, 4: 11.5}

    def text(s: str, size: float = 10.5, style: str = "", indent: float = 0, gap: float = 2) -> None:
        pdf.set_font("mars", style, size)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(pdf.epw - indent, size * 0.5, s, align="L", **nl)
        pdf.ln(gap)

    for b in blocks:
        if b.kind == "heading":
            pdf.ln(3 if b.level > 1 else 0)
            text(plain(b.text), sizes.get(b.level, 11), "B", gap=1.5)
        elif b.kind == "para":
            text(plain(b.text))
        elif b.kind == "quote":
            pdf.set_text_color(70, 80, 78)
            text(plain(b.text), indent=5)
            pdf.set_text_color(0, 0, 0)
        elif b.kind in ("bullets", "numbered"):
            for n, item in enumerate(b.items, 1):
                mark = "•" if b.kind == "bullets" else f"{n}."
                text(f"{mark} {plain(item)}", indent=4, gap=0.6)
            pdf.ln(1.5)
        elif b.kind == "table" and b.rows:
            cols = max(len(r) for r in b.rows)
            pdf.set_font("mars", "", 8)
            with pdf.table(first_row_as_headings=True, line_height=4.2, padding=1.2, text_align="LEFT") as table:
                for row in b.rows:
                    out = table.row()
                    for c in range(cols):
                        out.cell(plain(row[c]) if c < len(row) else "")
            pdf.ln(3)
    path = d / "report.pdf"
    pdf.output(str(path))
    return path


def export(run_dir: str | Path, fmt: str = "all", font: str | None = None) -> list[Path]:
    out = []
    if fmt in ("docx", "all"):
        out.append(to_docx(run_dir))
    if fmt in ("pdf", "all"):
        out.append(to_pdf(run_dir, font))
    return out
