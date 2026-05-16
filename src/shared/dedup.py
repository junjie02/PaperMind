import hashlib
import re


class DedupEngine:
    ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})")
    ARXIV_URL_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")

    def normalize_arxiv_id(self, raw: str) -> str | None:
        if not raw:
            return None
        if match := self.ARXIV_URL_PATTERN.search(raw):
            return match.group(1)
        if match := self.ARXIV_ID_PATTERN.search(raw):
            return match.group(1)
        return None

    def compute_paper_id(
        self,
        arxiv_id: str | None,
        source: str,
        source_url: str,
        title: str,
    ) -> str | None:
        if arxiv_id:
            normalized = self.normalize_arxiv_id(arxiv_id)
            if normalized:
                return f"arxiv:{normalized}"
        key = (source_url or title or "").strip().lower()
        if not key:
            return None
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        prefix = (source or "web").strip().lower() or "web"
        return f"{prefix}:{digest}"
