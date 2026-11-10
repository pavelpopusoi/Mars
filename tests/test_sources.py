"""Научные базы на поддельном HTTP-транспорте (без сети)."""
import httpx

from mars.sources import pick_queries, search_academic

OPENALEX = {"results": [{
    "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/abc", "title": "Sodium-ion batteries review",
    "publication_year": 2024, "cited_by_count": 42,
    "abstract_inverted_index": {"Sodium": [0], "cells": [1], "are": [2], "cheap": [3]},
    "primary_location": {"source": {"display_name": "Nature Energy"}},
    "authorships": [{"author": {"display_name": "A. One"}}, {"author": {"display_name": "B. Two"}}]}]}
ARXIV = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2401.00001v1</id><published>2024-01-02T00:00:00Z</published>
<title>Sodium  ion
 cathodes</title><summary> Layered oxides. </summary><author><name>C. Three</name></author></entry></feed>"""
ESEARCH = {"esearchresult": {"idlist": ["123"]}}
EFETCH = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
<Journal><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue><Title>Gut</Title></Journal>
<ArticleTitle>Microbiome and <i>mood</i></ArticleTitle><Abstract><AbstractText>RCT of probiotics.</AbstractText>
</Abstract><AuthorList><Author><LastName>Four</LastName><Initials>D</Initials></Author></AuthorList>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def handler(request: httpx.Request) -> httpx.Response:
    host, path = request.url.host, request.url.path
    if host == "api.openalex.org":
        return httpx.Response(200, json=OPENALEX)
    if host == "export.arxiv.org":
        assert request.url.params["search_query"].startswith("all:")
        return httpx.Response(200, text=ARXIV)
    if path.endswith("esearch.fcgi"):
        return httpx.Response(200, json=ESEARCH)
    if path.endswith("efetch.fcgi"):
        return httpx.Response(200, text=EFETCH)
    return httpx.Response(404)


async def test_all_databases_parsed_and_deduplicated():
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        papers = await search_academic(["sodium ion", "натрий-ионные"], ["openalex", "arxiv", "pubmed", "nope"],
                                       client=client)
    by_db = {p.database: p for p in papers}
    assert set(by_db) == {"OpenAlex", "arXiv", "PubMed"} and len(papers) == 3  # два запроса, дубли склеены
    assert by_db["OpenAlex"].abstract == "Sodium cells are cheap" and by_db["OpenAlex"].citations == 42
    assert by_db["arXiv"].title == "Sodium ion cathodes" and by_db["arXiv"].year == 2024
    assert by_db["PubMed"].url == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert by_db["PubMed"].title == "Microbiome and mood" and "Gut" in by_db["PubMed"].digest()


async def test_unavailable_database_is_skipped():
    def broken(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json=OPENALEX)
        raise httpx.ConnectError("blocked")

    async with httpx.AsyncClient(transport=httpx.MockTransport(broken)) as client:
        papers = await search_academic(["sodium"], ["openalex", "arxiv", "pubmed"], client=client)
    assert [p.database for p in papers] == ["OpenAlex"]


def test_latin_queries_first():
    queries = ["натрий-ионные аккумуляторы", "sodium-ion LFP cost", "Na-ion 2030 прогноз"]
    assert pick_queries(queries, 2)[0] == "sodium-ion LFP cost"
