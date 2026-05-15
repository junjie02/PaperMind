import re

_SLUG_LOWER = re.compile(r"[^a-z0-9一-鿿]+")
_SLUG_PRESERVE_CASE = re.compile(r"[^A-Za-z0-9._\-一-鿿]+")
_COLLAPSE_DASHES = re.compile(r"-+")


def slugify(text: str, max_len: int = 40, fallback: str = "survey",
            preserve_case: bool = False) -> str:
    """Normalize a string for safe use in directory names or file stems.

    - `preserve_case=False` (default): folds to lowercase, keeps [a-z0-9] and CJK.
      Suitable for run-dir slugs derived from research questions.
    - `preserve_case=True`: keeps original case + dot/underscore.
      Suitable for artifact file stems.
    """
    if preserve_case:
        cleaned = _SLUG_PRESERVE_CASE.sub("-", text.strip())
    else:
        cleaned = _SLUG_LOWER.sub("-", text.strip().lower())
    cleaned = _COLLAPSE_DASHES.sub("-", cleaned).strip("-._")
    if not cleaned:
        return fallback
    return cleaned[:max_len].rstrip("-._") or fallback
