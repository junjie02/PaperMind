# PaperMind

PaperMind is an automated literature survey tool. Give it a research topic and
it collects papers via Claude Code, downloads PDFs, clusters them by sub-topic,
and drafts a structured literature review outline — all in one command.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY (for outline generation)
```

Claude Code must be installed and authenticated (it's the search worker).

## One-Command Usage (PM)

```bash
# Full pipeline: search 15 papers + download PDFs + generate outline
PM "agent 安全研究" -n 15

# More papers, 2 parallel workers
PM "diffusion models for video generation" -n 30 -w 2

# Only collect papers, skip outline
PM "vision transformers" -n 10 --skip-outline

# Only metadata, skip PDF download too
PM "LLM reasoning" -n 20 --skip-pdf --skip-outline
```

### PM Options

```text
positional:
  question              Research topic

options:
  -n, --num-papers N    Target paper count (default: 30)
  -w, --workers N       Parallel Claude Code workers (default: 1)
  --out DIR             Output directory (default: runs/{timestamp}-{slug}/)
  --skip-pdf            Skip PDF/MD artifact download
  --skip-outline        Skip outline generation (papers only)
  -s, --sections N      Outline target sections (default: 8)
  --subsections N       Subsections per section (default: 2)
  --verbose, -v         Debug logging
```

## Split-Step Usage

You can also run each phase independently:

### Step 1: Collect papers

```bash
deepresearch "agent 安全研究" -n 15
# → runs/20260515T140000-agent-安全研究/papers.db + pdfs/
```

### Step 2: Generate outline (from an existing run)

```bash
outline --in runs/20260515T140000-agent-安全研究
# → runs/20260515T140000-agent-安全研究/outline.md
```

### deepresearch options

```text
-n, --num-papers N    Target paper count (default: 30)
-w, --workers N       Parallel Claude Code workers (default: 1)
--out DIR             Output directory
--json                Print metadata as JSON
--resume              Resume from existing run dir (requires --out)
--skip-pdf            Skip PDF/MD download
--verbose, -v         Debug logging
```

### outline options

```text
--in DIR              Input run directory (required, must contain papers.db)
--out FILE            Output file (default: {in_dir}/outline.md)
--force               Overwrite existing outline
--recluster           Force re-clustering even if DB already has groups
-s, --sections N      Target sections (default: 8)
--subsections N       Subsections per section (default: 2)
--topic TEXT          Override topic (default: derived from dir name)
--verbose, -v         Debug logging
```

## Architecture

```text
PM "research topic" -n 15
  │
  ├─ Phase 1: deepresearch (Claude Code)
  │    │
  │    ├── Claude Code workers → web search + metadata extraction
  │    ├── Dedup → SQLite (papers.db)
  │    └── ArtifactWriter
  │         ├─ arXiv papers → pdfs/{arxiv_id}.pdf
  │         └─ Other papers → pdfs/{source}-{slug}.md
  │
  └─ Phase 2: outline (LangGraph + DeepSeek)
       │
       ├── cluster_papers    (1 LLM call: group papers by sub-topic)
       ├── group_by_direction (pure Python: bucket + sort)
       ├── draft_per_group   (G LLM calls: one sub-outline per group)
       ├── merge_outlines    (1 LLM call: merge into final outline)
       └── render_references (pure Python: extract citations → bibtex)
       │
       └── → outline.md
```

## Run Directory Layout

```
runs/
└── 20260515T140000-agent-安全研究/
    ├── papers.db          # SQLite with all paper metadata
    ├── outline.md         # Generated literature review outline
    └── pdfs/
        ├── 2301.12345.pdf
        ├── 2405.67890.pdf
        ├── openreview-some-paper-title.md
        └── ...
```

## Configuration (.env)

```bash
# Claude Code worker model (leave empty for default Opus)
WORKER_MODEL=claude-haiku-4-5-20251001

# Search timeout per worker (seconds)
SEARCH_TIMEOUT=900

# PDF download
PDF_MAX_ATTEMPTS=3
PDF_RETRY_SLEEP=2.0
PDF_TIMEOUT=60

# HTTP proxy
HTTP_PROXY=

# LLM for outline generation (OpenAI-compatible)
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_MAX_TOKENS=4096
DEEPSEEK_JSON_MODE=1

# Outline temperature tuning (optional)
# OUTLINE_TEMP_CLUSTER=0.1
# OUTLINE_TEMP_DRAFT=0.6
# OUTLINE_TEMP_MERGE=0.3
# OUTLINE_CLUSTER_MAX=6
```

## Output Fields (papers.db)

| Field | Purpose |
|---|---|
| `paper_id` | Dedup key: `arxiv:<id>` or `<source>:<hash>` |
| `title` | Paper title |
| `authors` | Author list |
| `abstract` | Paper abstract |
| `overview` | Chinese summary by worker |
| `source` | `arxiv` / `openreview` / `acl` / `neurips` / ... |
| `source_url` | Canonical URL |
| `venue` | Publication venue |
| `arxiv_id` | arXiv ID (null for non-arXiv) |
| `search_direction` | Sub-topic cluster label |
| `bibtex` | BibTeX citation |
| `pdf_url` | PDF direct link |
| `artifact_rel_path` | Local PDF or MD path |
| `relevance_score` | 1-5 relevance rating |

## Future Roadmap

The LangGraph pipeline has seams for:
- **Reviewer node** — LLM-as-judge evaluates outline quality, loops back to re-draft if rejected
- **RAG content writer** — FAISS + embeddings retrieve relevant paper sections per outline chapter
- **Polish step** — final language and citation cleanup
