"""PDF export: convert markdown survey to PDF using weasyprint."""

import logging
from pathlib import Path

import markdown
from weasyprint import HTML

logger = logging.getLogger(__name__)

_CSS = """\
@page {
    size: A4;
    margin: 2.5cm;
}
body {
    font-family: "Noto Serif CJK SC", "SimSun", "Times New Roman", serif;
    font-size: 12pt;
    line-height: 1.8;
    color: #333;
}
h1 {
    font-size: 20pt;
    text-align: center;
    margin-bottom: 1.5em;
}
h2 {
    font-size: 16pt;
    margin-top: 1.5em;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.3em;
}
h3 {
    font-size: 14pt;
    margin-top: 1.2em;
}
p {
    text-indent: 2em;
    margin: 0.5em 0;
}
code {
    font-family: "Courier New", monospace;
    font-size: 10pt;
    background: #f5f5f5;
    padding: 0.1em 0.3em;
}
"""


def md_to_pdf(md_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert a markdown file to PDF.

    Args:
        md_path: Path to the markdown file.
        output_path: Optional output PDF path. Defaults to same name with .pdf extension.

    Returns:
        Path to the generated PDF file.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    if output_path is None:
        output_path = md_path.with_suffix(".pdf")
    else:
        output_path = Path(output_path)

    md_text = md_path.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
    )

    full_html = f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{_CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html).write_pdf(str(output_path))
    logger.info("PDF exported: %s", output_path)
    return output_path
