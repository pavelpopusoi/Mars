"""Бесплатный веб-поиск без ключей: DuckDuckGo (HTML-версия) + Википедия, затем чтение страниц.

Используется с локальной моделью (backend: ollama), у которой нет серверного поиска, как у Claude.
Ошибки сети не роняют запуск: недоступный поисковик или страница просто пропускаются.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx

log = logging.getLogger("mars.websearch")
TIMEOUT = httpx.Timeout(20.0)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/130.0 Safari/537.36",
    "Accept-Language": "ru,en;q=0.8",
}
# Википедия требует от программ честный User-Agent, браузерный она отклоняет
WIKI_UA = "mars-research/0.6 (personal research tool; https://github.com/pavelpopusoi/Mars)"
SKIP_HOSTS = ("duckduckgo.com", "youtube.com", "facebook.com", "instagram.com", "tiktok.com", "vk.com/video")


@dataclass
class Page:
    url: str
    title: str
    snippet: str = ""
    text: str = ""


def _strip_tags(raw: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))).strip()


def _real_url(href: str) -> str:
    """DuckDuckGo оборачивает ссылки в //duckduckgo.com/l/?uddg=<url>."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


async def search_duckduckgo(client: httpx.AsyncClient, query: str, limit: int) -> list[Page]:
    r = await client.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "wt-wt"})
    r.raise_for_status()
    pages: list[Page] = []
    blocks = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=class="result__a"|$)',
                        r.text, flags=re.S)
    for href, title, rest in blocks:
        url = _real_url(html.unescape(href))
        if not url.startswith("http") or any(h in url for h in SKIP_HOSTS):
            continue
        snip = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', rest, flags=re.S)
        pages.append(Page(url=url, title=_strip_tags(title), snippet=_strip_tags(snip.group(1)) if snip else ""))
        if len(pages) >= limit:
            break
    return pages


async def search_wikipedia(client: httpx.AsyncClient, query: str, limit: int, lang: str = "ru") -> list[Page]:
    r = await client.get(f"https://{lang}.wikipedia.org/w/api.php",
                         params={"action": "query", "list": "search", "srsearch": query, "format": "json",
                                 "srlimit": str(limit)},
                         headers={"User-Agent": WIKI_UA})
    r.raise_for_status()
    out = []
    for item in r.json().get("query", {}).get("search", []):
        title = item.get("title", "")
        out.append(Page(url=f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}", title=title,
                        snippet=_strip_tags(item.get("snippet", ""))))
    return out


def extract_text(raw_html: str, max_chars: int) -> str:
    """Грубое извлечение основного текста: без скриптов, стилей, меню; берём абзацы."""
    body = re.sub(r"(?is)<(script|style|noscript|nav|header|footer|aside|form|svg)[^>]*>.*?</\1>", " ", raw_html)
    paras = [_strip_tags(p) for p in re.findall(r"(?is)<(?:p|li|h[1-4]|td)[^>]*>(.*?)</(?:p|li|h[1-4]|td)>", body)]
    text = "\n".join(p for p in paras if len(p) > 40)
    if len(text) < 300:  # страница без абзацев — берём весь текст
        text = _strip_tags(body)
    return text[:max_chars]


async def fetch_page(client: httpx.AsyncClient, page: Page, max_chars: int) -> Page:
    try:
        wiki = urlparse(page.url).netloc.endswith("wikipedia.org")
        r = await client.get(page.url, headers={"User-Agent": WIKI_UA} if wiki else None)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "html" in ctype or "text/plain" in ctype:
            page.text = extract_text(r.text, max_chars) if "html" in ctype else r.text[:max_chars]
    except (httpx.HTTPError, UnicodeDecodeError) as e:
        log.info("Страница недоступна %s: %s", page.url, e)
    return page


async def fetch_url(url: str, max_chars: int = 6000) -> Page:
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS, follow_redirects=True) as client:
        return await fetch_page(client, Page(url=url, title=url), max_chars)


async def web_search(queries: list[str], max_results: int = 5, fetch_top: int = 4, max_chars: int = 3000,
                     language: str = "ru", client: httpx.AsyncClient | None = None) -> list[Page]:
    """Ищет по всем запросам, убирает дубли, скачивает текст лучших страниц."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS, follow_redirects=True)

    async def one(fn, q: str) -> list[Page]:
        try:
            return await fn(client, q, max_results)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            log.warning("Поиск %s недоступен (%s): %s", fn.__name__, q[:40], e)
            return []

    try:
        tasks = [one(search_duckduckgo, q) for q in queries[:4]]
        tasks += [one(lambda c, q, n: search_wikipedia(c, q, 2, language), q) for q in queries[:2]]
        batches = await asyncio.gather(*tasks)
        seen: dict[str, Page] = {}
        # Чередуем результаты разных запросов, чтобы верх списка не занял один запрос
        for i in range(max(map(len, batches), default=0)):
            for b in batches:
                if i < len(b):
                    seen.setdefault(b[i].url.split("#")[0], b[i])
        pages = list(seen.values())
        await asyncio.gather(*(fetch_page(client, p, max_chars) for p in pages[:fetch_top]))
        return pages
    finally:
        if own:
            await client.aclose()


def digest(pages: list[Page]) -> str:
    parts = []
    for i, p in enumerate(pages, 1):
        body = p.text or p.snippet
        if body:
            parts.append(f"### [{i}] {p.title}\n{p.url}\n{body}")
    return "\n\n".join(parts)
