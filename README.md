# DeepResearch

DeepResearch is a Claude Code based literature survey runner. Give it a
research topic and it launches Claude Code workers to gather high-quality
papers across multiple search directions, stores grouped metadata in SQLite,
and produces a local artifact for every paper.

Papers are **not** restricted to arXiv — workers may return conference papers
(OpenReview / ACL / NeurIPS / CVPR / ...), journal papers, or other authoritative
sources, as long as they're relevant and high quality.

Per-paper artifact handling:

- **arXiv papers** → the PDF is downloaded via the arXiv API into `pdfs/`.
- **Non-arXiv papers** → a Markdown file with the full metadata (title, authors,
  abstract, overview, BibTeX, source URL, ...) is written into `pdfs/`.

Each invocation is fully independent: a fresh `runs/{timestamp}-{slug}/`
directory is created containing that run's `papers.db` and `pdfs/` folder.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env
```

Edit `.env` if you need to set proxy or PDF download tuning. Claude Code must
be installed and authenticated because it is the worker backend. The
`DEEPSEEK_*` entries are kept as an optional hook for future LLM-based
post-processing — the default pipeline does not require them.

## Usage

```bash
# Default: Claude Code workers, 30 papers, auto-generated run dir
deepresearch "attention mechanisms in vision transformers"

# Smaller test run
deepresearch "diffusion models for video generation" -n 5

# Print structured metadata JSON
deepresearch "large language model reasoning" -n 10 --json

# Specify an output directory explicitly
deepresearch "graph neural networks survey" --out runs/gnn-survey

# Resume from an existing run directory (re-download missing artifacts, top up papers)
deepresearch "graph neural networks survey" --out runs/gnn-survey --resume

# Skip artifact writing (metadata only)
deepresearch "vision transformers" -n 5 --skip-pdf
```

### CLI Options

```text
-n, --num-papers    Target number of papers (default: 30)
-w, --workers       Parallel Claude Code workers (default: 1)
--out               Output directory (default: runs/{timestamp}-{slug}/)
--json              Print paper metadata as JSON
--resume            Resume from existing run dir (requires --out)
--skip-pdf          Skip artifact (PDF/MD) writing stage
--verbose, -v       Enable debug logging
```

The run is single-shot: workers are launched once with the user's research
question and the per-worker quota is derived from `-n` (e.g. `-n 10 -w 4` →
`[3, 3, 2, 2]`). Workers decide their own sub-directions and search keywords.
The final console output groups papers by the direction each worker actually
returned.

## Architecture

```text
User research question
  |
  v
Claude Code workers (each free to pick its own sub-directions / keywords)
  |
  +-- web search / fetch (arxiv, openreview, acl, ...)
  +-- structured metadata extraction
  |
  v
Dedup by paper_id + SQLite (runs/<id>/papers.db, grouped by direction)
  |
  v
ArtifactWriter
  ├─ arXiv → runs/<id>/pdfs/<arxiv_id>.pdf
  └─ other → runs/<id>/pdfs/<source>-<slug>.md
```

Failed PDF downloads are pushed back to the end of the queue and retried up to
`PDF_MAX_ATTEMPTS` times.

## Run Directory Layout

```
runs/
└── 20260515T134500-attention-in-vision-transformers/
    ├── papers.db
    └── pdfs/
        ├── 1706.03762.pdf
        ├── 2010.11929.pdf
        ├── openreview-low-rank-adaptation-of-large-language-models.md
        └── ...
```

## Configuration

```bash
SEARCH_TIMEOUT=900

# Optional: switch the Claude Code worker model (default = Claude Code's default)
# WORKER_MODEL=claude-haiku-4-5-20251001

# Optional PDF download settings
PDF_MAX_ATTEMPTS=3
PDF_RETRY_SLEEP=2.0
PDF_TIMEOUT=60

# Optional proxy
HTTP_PROXY=

# Optional OpenAI/DeepSeek-compatible LLM (not used by default; reserved for
# future post-processing like re-ranking / classification)
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-pro
```

## Output Fields

| Field | Purpose |
|---|---|
| `paper_id` | Dedup key. `arxiv:<id>` for arXiv papers, `<source>:<hash>` otherwise |
| `arxiv_id` | arXiv ID when applicable (e.g. `1512.03385`); null for non-arXiv papers |
| `source` | `arxiv` / `openreview` / `acl` / `neurips` / `cvpr` / `nature` / `web` / ... |
| `source_url` | Canonical landing page URL |
| `venue` | Publication venue, e.g. `ICLR 2024` / `arXiv preprint` |
| `search_direction` | The sub-direction this paper was found under |
| `title` | Paper title |
| `authors` | Author list |
| `abstract` | Paper abstract |
| `overview` | Short Chinese summary written by the worker |
| `bibtex` | BibTeX citation |
| `abs_url` / `pdf_url` | URLs (PDF URL optional for non-arXiv) |
| `artifact_rel_path` | Local path of the downloaded PDF or generated MD file |
| `published_at` | Published date, `YYYY-MM-DD` |
| `categories` / `primary_class` | Optional category tags |
| `relevance_score` | Worker-assessed relevance score (1-5) |
