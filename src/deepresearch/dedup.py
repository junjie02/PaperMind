import re


class DedupEngine:
    ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})")
    ARXIV_URL_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")

    def normalize(self, raw: str) -> str | None:
        if not raw:
            return None
        if match := self.ARXIV_URL_PATTERN.search(raw):
            return match.group(1)
        if match := self.ARXIV_ID_PATTERN.search(raw):
            return match.group(1)
        return None
