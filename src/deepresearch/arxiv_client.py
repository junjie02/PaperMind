import logging
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_ABS_BASE = "https://arxiv.org/abs"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"

BIBTEX_TEMPLATE = """@article{{{key},
  title = {{{title}}},
  author = {{{authors}}},
  journal = {{arXiv preprint arXiv:{arxiv_id}}},
  year = {{{year}}},
  month = {{{month}}},
  note = {{{abs_url}}},
}}"""


def _format_authors(authors: list[str]) -> str:
    return " and ".join(a.strip() for a in authors if a.strip())


def _xml_to_dict(xml_text: str) -> dict | None:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(xml_text)
        if root.tag == f"{{{ns['atom']}}}entry":
            entry = root
        else:
            entry = root.find("atom:entry", ns)
            if entry is None:
                return None

        title = entry.find("atom:title", ns)
        title_text = _clean_text(title.text if title is not None else "")

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None and name.text:
                authors.append(name.text.strip())

        summary = entry.find("atom:summary", ns)
        abstract = _clean_text(summary.text if summary is not None else "")

        published = entry.find("atom:published", ns)
        published_at = None
        year = "????"
        month = "??"
        if published is not None:
            try:
                dt = datetime.fromisoformat(published.text.replace("Z", "+00:00"))
                published_at = dt.strftime("%Y-%m-%d")
                year = dt.strftime("%Y")
                month = dt.strftime("%b").lower()
            except (ValueError, AttributeError):
                pass

        categories = []
        primary_class = None
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term", "")
            if term:
                categories.append(term)
        primary_category = entry.find("arxiv:primary_category", ns)
        if primary_category is not None:
            primary_class = primary_category.get("term")

        links = {}
        for link in entry.findall("atom:link", ns):
            rel = link.get("title", link.get("rel", ""))
            href = link.get("href", "")
            links[rel] = href

        return {
            "title": title_text,
            "authors": authors,
            "abstract": abstract,
            "published_at": published_at,
            "year": year,
            "month": month,
            "categories": categories,
            "primary_class": primary_class,
            "abs_url": links.get("abstract", ""),
            "pdf_url": links.get("pdf", ""),
        }
    except ET.ParseError as e:
        logger.error(f"Failed to parse arXiv XML: {e}")
        return None


def _clean_text(text: str | None) -> str:
    return " ".join((text or "").split())


class ArxivClient:
    def __init__(self, proxy: str = "", timeout: float = 10.0, retries: int = 2):
        self.retries = max(retries, 1)
        extra = {"proxy": proxy} if proxy else {}
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(timeout, 5.0)),
            follow_redirects=True,
            headers={"User-Agent": "DeepResearch/0.1 (mailto:research@example.com)"},
            **extra,
        )

    async def close(self):
        await self.client.aclose()

    async def _fetch_xml(self, arxiv_id: str) -> str | None:
        clean_id = _strip_version(arxiv_id)
        params = {"id_list": clean_id, "max_results": "1"}
        for attempt in range(self.retries):
            try:
                response = await self.client.get(ARXIV_API_BASE, params=params)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(
                    "arXiv API attempt %d/%d for %s: %s",
                    attempt + 1,
                    self.retries,
                    arxiv_id,
                    e,
                )
                if attempt < self.retries - 1:
                    await self._sleep(attempt)
        return None

    async def get_paper(self, arxiv_id: str) -> dict | None:
        xml_text = await self._fetch_xml(arxiv_id)
        if not xml_text:
            return None

        data = _xml_to_dict(xml_text)
        if not data:
            return None

        clean_id = _strip_version(arxiv_id)
        data["arxiv_id"] = clean_id

        if not data["abs_url"]:
            data["abs_url"] = f"{ARXIV_ABS_BASE}/{clean_id}"
        if not data["pdf_url"]:
            data["pdf_url"] = f"{ARXIV_PDF_BASE}/{clean_id}"

        data["bibtex"] = _make_bibtex(data)

        return data

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            response = await self.client.get(ARXIV_API_BASE, params=params)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        results = []
        try:
            root = ET.fromstring(response.text)
            for entry in root.findall("atom:entry", ns):
                paper = _xml_to_dict(ET.tostring(entry, encoding="unicode"))
                if paper:
                    # Extract arxiv ID from URL
                    for link in entry.findall("atom:link", ns):
                        href = link.get("href", "")
                        if "arxiv.org/abs/" in href:
                            paper["arxiv_id"] = href.split("/abs/")[-1].split("v")[0]
                            break
                    if not paper.get("arxiv_id"):
                        continue
                    paper["arxiv_id"] = _strip_version(paper["arxiv_id"])
                    if not paper["abs_url"]:
                        paper["abs_url"] = f"{ARXIV_ABS_BASE}/{paper['arxiv_id']}"
                    if not paper["pdf_url"]:
                        paper["pdf_url"] = f"{ARXIV_PDF_BASE}/{paper['arxiv_id']}"
                    paper["bibtex"] = _make_bibtex(paper)
                    results.append(paper)
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv search XML: {e}")

        return results

    async def _sleep(self, attempt: int) -> None:
        import asyncio

        await asyncio.sleep(2 ** attempt)


def _strip_version(arxiv_id: str) -> str:
    return arxiv_id.rsplit("v", 1)[0]


def _make_bibtex(data: dict) -> str:
    arxiv_id = data["arxiv_id"]
    return BIBTEX_TEMPLATE.format(
        key=arxiv_id.replace(".", ""),
        title=data.get("title", ""),
        authors=_format_authors(data.get("authors", [])),
        arxiv_id=arxiv_id,
        year=data.get("year", "????"),
        month=data.get("month", "??"),
        abs_url=data.get("abs_url", f"{ARXIV_ABS_BASE}/{arxiv_id}"),
    )
