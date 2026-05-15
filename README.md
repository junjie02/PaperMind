# DeepResearch

DeepResearch is a Claude Code based literature survey runner. Give it a
research topic and it launches Claude Code workers to search for related papers,
then stores structured metadata in SQLite.

The returned metadata includes `arxiv_id`, `title`, `authors`, `abstract`,
`abs_url`, `pdf_url`, categories, publication date, and BibTeX.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env
```

Edit `.env` and set:

```bash
DEEPSEEK_API_KEY=sk-your-real-key
SEARCH_BACKEND=claude
```

Claude Code must also be installed and authenticated because it is the default
worker backend.

## Usage

```bash
# Default: Claude Code workers, 30 papers
deepresearch "attention mechanisms in vision transformers"

# Smaller test run
deepresearch "diffusion models for video generation" -n 5

# Print structured metadata JSON
deepresearch "large language model reasoning" -n 10 --json

# Resume from an existing SQLite database
deepresearch "graph neural networks survey" --db gnn.db --resume

# Optional non-agent fallback for quick arXiv-only metadata lookup
deepresearch "vision transformers" --backend arxiv -n 10 --json
```

### CLI Options

```text
-n, --num-papers    Target number of papers (default: 30)
-w, --workers       Parallel Claude Code workers per round (default: 4)
--db                SQLite database path (default: papers.db)
--backend           claude or arxiv (default: claude)
--json              Print paper metadata as JSON
--resume            Resume from existing database
--verbose, -v       Enable debug logging
```

For the Claude backend, each round caps the number of active workers by the
remaining target. Worker paper quotas are split from `-n`; for example,
`-n 10 -w 4` assigns `[3, 3, 2, 2]` in the first round.

## Architecture

```text
User topic
  |
  v
SearchDiversifier (DeepSeek/OpenAI-compatible API)
  |
  v
Claude Code workers
  |
  +-- web search / fetch
  +-- arXiv metadata verification
  +-- JSON metadata output
  |
  v
Dedup + SQLite
```

`--backend arxiv` is a deterministic fallback that queries the arXiv API
directly. It is useful for smoke tests or when Claude Code is unavailable, but
it does not perform the richer agentic search.

## Configuration

```bash
DEEPSEEK_API_KEY=sk-your-real-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_MAX_TOKENS=4096
DEEPSEEK_JSON_MODE=1

SEARCH_BACKEND=claude
SEARCH_TIMEOUT=900

# Optional arXiv fallback settings
ARXIV_TIMEOUT=10
ARXIV_RETRIES=2

# Optional proxy
HTTP_PROXY=
```

## Output Fields

| Field | Purpose |
|---|---|
| `arxiv_id` | Normalized arXiv ID, for example `1512.03385` |
| `title` | Paper title |
| `authors` | JSON array of author names |
| `abstract` | Paper abstract |
| `overview` | Short Chinese summary |
| `bibtex` | BibTeX citation |
| `abs_url` | arXiv abstract URL |
| `pdf_url` | arXiv PDF URL |
| `published_at` | Published date, `YYYY-MM-DD` |
| `categories` | arXiv categories |
| `primary_class` | Primary arXiv category |
| `relevance_score` | Worker-assessed relevance score |
