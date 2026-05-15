# PaperMind

PaperMind is an automated literature survey tool. Give it a research topic and it collects papers via Claude Code, downloads PDFs, clusters them by sub-topic, drafts a structured outline, and writes a full literature review — all in one command.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY
```

Claude Code must be installed and authenticated (used as the paper search worker).

## One-Command Usage (PM)

```bash
# Full pipeline: search + outline + write review
PM "agent 安全研究" -n 15

# More papers, 2 parallel search workers
PM "diffusion models for video generation" -n 30 -w 2

# Skip review writing
PM "vision transformers" -n 10 --skip-write

# Only collect papers
PM "LLM reasoning" -n 20 --skip-outline
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
  --skip-write          Skip review writing (Phase 3)
  -s, --sections N      Outline target sections (default: 8)
  --subsections N       Subsections per section (default: 2)
  --verbose, -v         Debug logging
```

## Split-Step Usage

Each phase can be run independently.

### Phase 1: Collect papers

```bash
deepresearch "agent 安全研究" -n 15
# → runs/20260515T140000-agent-安全研究/papers.db + pdfs/
```

### Phase 2: Generate outline

```bash
outline --in runs/20260515T140000-agent-安全研究
# → runs/.../outline.md
```

### Phase 3: Write review

```bash
# Build FAISS index from PDFs
write index --in runs/20260515T140000-agent-安全研究

# Write review (reuse existing index)
write write --in runs/20260515T140000-agent-安全研究 --skip-index
# → runs/.../review.md
```

The two `write` sub-commands can be run separately, which is useful for debugging or re-running just the writing step without re-embedding.

### deepresearch options

```text
-n, --num-papers N    Target paper count (default: 30)
-w, --workers N       Parallel Claude Code workers (default: 1)
--out DIR             Output directory
--skip-pdf            Skip PDF/MD download
--verbose, -v         Debug logging
```

### outline options

```text
--in DIR              Input run directory (required, must contain papers.db)
--out FILE            Output file (default: {in_dir}/outline.md)
--force               Overwrite existing outline
--recluster           Force re-clustering
-s, --sections N      Target sections (default: 8)
--subsections N       Subsections per section (default: 2)
--topic TEXT          Override topic
--verbose, -v         Debug logging
```

### write options

```text
# index sub-command
--in DIR              Run directory (required)

# write sub-command
--in DIR              Run directory (required)
--skip-index          Skip embedding, reuse existing faiss.index
--model MODEL         Override WRITER_MODEL for this run
--out FILE            Output path (default: {in_dir}/review.md)
--verbose, -v         Debug logging
```

## Architecture

```text
PM "research topic" -n 15
  │
  ├─ Phase 1: deepresearch (Claude Code workers)
  │    ├── Web search + metadata extraction
  │    ├── Dedup → SQLite (papers.db)
  │    └── ArtifactWriter
  │         ├─ arXiv → pdfs/{arxiv_id}.pdf
  │         └─ Other → pdfs/{source}-{slug}.md
  │
  ├─ Phase 2: outline (LangGraph + DeepSeek)
  │    ├── cluster_papers     (group by sub-topic)
  │    ├── draft_per_group    (sub-outline per cluster)
  │    ├── merge_outlines     (merge into final outline)
  │    └── → outline.md
  │
  └─ Phase 3: write (DeepSeek concurrent writers + FAISS RAG)
       ├── pdf_converter      (PyMuPDF: PDF → text)
       ├── chunker            (section-aware splitting)
       ├── indexer            (sentence-transformers → faiss.index)
       │
       └── Per ## section (concurrent):
            ├── Dual-path retrieval
            │    ├── Path A: direct vector search (title + outline text)
            │    └── Path B: LLM keyword expansion → vector search
            ├── DeepSeek write (draft with citations)
            ├── RAG verify     (re-retrieve on draft → fact-check)
            └── Loop until PASS or max_retries
            → review.md
```

## Run Directory Layout

```
runs/
└── 20260515T140000-agent-安全研究/
    ├── papers.db          # SQLite: all paper metadata
    ├── outline.md         # Literature review outline
    ├── review.md          # Full literature review (Phase 3 output)
    ├── faiss.index        # FAISS vector index
    ├── chunks.pkl         # Serialized paper chunks
    └── pdfs/
        ├── 2301.12345.pdf
        ├── openreview-some-paper.md
        └── ...
```

## Configuration (.env)

```bash
# DeepSeek API (used for outline + review writing)
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_MAX_TOKENS=4096

# Claude Code worker model (Phase 1 paper search)
WORKER_MODEL=claude-haiku-4-5-20251001

# HTTP proxy
HTTP_PROXY=http://127.0.0.1:7890

# Embedding model (CPU-friendly HuggingFace model)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Writer settings (all default to DEEPSEEK_* values if left empty)
WRITER_MODEL=              # leave empty to use DEEPSEEK_MODEL
WRITER_CONCURRENCY=4       # parallel section writers
WRITER_MAX_RETRIES=3       # write→verify→rewrite loops per section
WRITE_TIMEOUT=1800

# PDF download
PDF_MAX_ATTEMPTS=3
PDF_RETRY_SLEEP=2.0
PDF_TIMEOUT=60
SEARCH_TIMEOUT=900
```

## Output Fields (papers.db)

| Field | Purpose |
|---|---|
| `paper_id` | Dedup key: `arxiv:<id>` or `<source>:<hash>` |
| `title` | Paper title |
| `authors` | Author list (JSON) |
| `abstract` | Paper abstract |
| `overview` | Summary by search worker |
| `source` | `arxiv` / `openreview` / `acl` / ... |
| `source_url` | Canonical URL |
| `venue` | Publication venue |
| `arxiv_id` | arXiv ID (null for non-arXiv) |
| `bibtex` | BibTeX citation |
| `artifact_rel_path` | Local PDF or MD path |
| `relevance_score` | 1–5 relevance rating |
