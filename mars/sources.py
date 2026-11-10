"""Научные базы: OpenAlex, arXiv, PubMed. Обычные HTTP-запросы без ключей; ошибки сети не роняют запуск."""
from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx

log = logging.getLogger("mars.sources")
ATOM = "{http://www.w3.org/2005/Atom}"
TIMEOUT = httpx.Timeout(20.0)
HEADERS = {"User-Agent": "mars-research/0.5"}


@dataclass
class Paper:
    title: str
    url: str
    database: str
    year: int | None = None
    authors: str = ""
    venue: str = ""
    abstract: str = ""
    citations: int | None = None

    def digest(self, max_abstract: int = 1500) -> str:
        meta = ", ".join(x for x in [self.authors, self.venue, str(self.year or ""),
                                     f"цитирований: {self.citations}" if self.citations is not None else ""] if x)
        return f"### {self.title}\n{self.url}\n[{self.database}] {meta}\n{self.abstract[:max_abstract]}".strip()


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _authors(names: list[str]) -> str:
    return ", ".join(names[:3]) + (" и др." if len(names) > 3 else "")


async def search_openalex(client: httpx.AsyncClient, query: str, limit: int, email: str = "") -> list[Paper]:
    params = {"search": query, "per-page": str(limit),
              "select": "id,doi,title,publication_year,abstract_inverted_index,primary_location,authorships,"
                        "cited_by_count"}
    if email:
        params["mailto"] = email
    r = await client.get("https://api.openalex.org/works", params=params)
    r.raise_for_status()
    out = []
    for w in r.json().get("results", []):
        index = w.get("abstract_inverted_index") or {}
        words = sorted((pos, word) for word, positions in index.items() for pos in positions)
        source = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        names = [(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []]
        out.append(Paper(title=_clean(w.get("title")), url=w.get("doi") or w.get("id") or "", database="OpenAlex",
                         year=w.get("publication_year"), authors=_authors([n for n in names if n]), venue=source,
                         abstract=" ".join(word for _, word in words), citations=w.get("cited_by_count")))
    return [p for p in out if p.title and p.url]


async def search_arxiv(client: httpx.AsyncClient, query: str, limit: int, email: str = "") -> list[Paper]:
    terms = " AND ".join(f"all:{t}" for t in re.findall(r"[\w\-]+", query)[:8])
    r = await client.get("https://export.arxiv.org/api/query",
                         params={"search_query": terms, "max_results": str(limit), "sortBy": "relevance"})
    r.raise_for_status()
    out = []
    for e in ET.fromstring(r.text).iter(f"{ATOM}entry"):
        published = e.findtext(f"{ATOM}published") or ""
        names = [_clean(a.findtext(f"{ATOM}name")) for a in e.iter(f"{ATOM}author")]
        out.append(Paper(title=_clean(e.findtext(f"{ATOM}title")), url=_clean(e.findtext(f"{ATOM}id")),
                         database="arXiv", year=int(published[:4]) if published[:4].isdigit() else None,
                         authors=_authors(names), venue="arXiv (препринт)",
                         abstract=_clean(e.findtext(f"{ATOM}summary"))))
    return [p for p in out if p.title and p.url]


async def search_pubmed(client: httpx.AsyncClient, query: str, limit: int, email: str = "") -> list[Paper]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    extra = {"tool": "mars-research", **({"email": email} if email else {})}
    r = await client.get(f"{base}/esearch.fcgi", params={"db": "pubmed", "term": query, "retmode": "json",
                                                         "retmax": str(limit), "sort": "relevance", **extra})
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    r = await client.get(f"{base}/efetch.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml",
                                                        **extra})
    r.raise_for_status()
    out = []
    for art in ET.fromstring(r.text).iter("PubmedArticle"):
        pmid = art.findtext("MedlineCitation/PMID") or ""
        a = art.find("MedlineCitation/Article")
        if a is None or not pmid:
            continue
        abstract = " ".join(_clean("".join(t.itertext())) for t in a.iter("AbstractText"))
        year = a.findtext("Journal/JournalIssue/PubDate/Year") or ""
        names = [_clean(f"{x.findtext('LastName') or ''} {x.findtext('Initials') or ''}") for x in a.iter("Author")]
        title_el = a.find("ArticleTitle")
        out.append(Paper(title=_clean("".join(title_el.itertext())) if title_el is not None else "",
                         url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", database="PubMed",
                         year=int(year) if year.isdigit() else None, authors=_authors([n for n in names if n]),
                         venue=_clean(a.findtext("Journal/Title")), abstract=abstract))
    return [p for p in out if p.title]


SEARCHERS = {"openalex": search_openalex, "arxiv": search_arxiv, "pubmed": search_pubmed}


def pick_queries(queries: list[str], n: int = 2) -> list[str]:
    """Научные базы индексируют в основном английские тексты — сначала берём запросы на латинице."""
    def latin_share(q: str) -> float:
        letters = [c for c in q if c.isalpha()]
        return sum(c.isascii() for c in letters) / len(letters) if letters else 0

    return sorted(queries, key=lambda q: -latin_share(q))[:n]


async def search_academic(queries: list[str], databases: list[str], per_source: int = 4, email: str = "",
                          client: httpx.AsyncClient | None = None) -> list[Paper]:
    """Параллельный поиск по выбранным базам. Недоступная база пропускается с предупреждением."""
    names = [d for d in databases if d in SEARCHERS]
    if not names or not queries:
        return []
    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS, follow_redirects=True)

    async def one(name: str, query: str) -> list[Paper]:
        try:
            return await SEARCHERS[name](client, query, per_source, email)
        except (httpx.HTTPError, ET.ParseError, ValueError, KeyError) as e:
            log.warning("База %s недоступна (%s): %s", name, query[:40], e)
            return []

    try:
        results = await asyncio.gather(*(one(n, q) for n in names for q in pick_queries(queries)))
    finally:
        if own:
            await client.aclose()
    seen: dict[str, Paper] = {}
    for paper in (p for batch in results for p in batch):
        seen.setdefault(re.sub(r"\W+", "", paper.title.lower())[:80], paper)  # одна статья из разных баз
    return list(seen.values())
