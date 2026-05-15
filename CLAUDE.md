# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PaperMind is an automated literature survey tool. It uses Claude Code as a search worker to collect papers, then generates a structured literature review outline via a LangGraph pipeline backed by a DeepSeek/OpenAI-compatible LLM.

## Commands

```bash
# Install (editable, into a venv)
pip install -e .

# Run the full pipeline (search + outline)
PM "research topic" -n 15

# Run only paper collection
deepresearch "research topic" -n 15 -w 2

# Run only outline generation from an existing run
outline --in runs/<run-dir>
```

There are no tests, linters, or build steps configured beyond `pip install -e .`.

## Architecture

Three packages under `src/`, each with its own CLI entry point:

- **`papermind`** (`PM` command) — orchestrates the two phases sequentially. Calls deepresearch then outliner.
- **`deepresearch`** (`deepresearch` command) — paper collection phase. Spawns Claude Code as a subprocess (`claude --print --output-format stream-json --json-schema ...`) to do web searches and extract structured paper metadata. Results are deduplicated and stored in SQLite (`papers.db`). Optionally downloads PDFs (arXiv) or generates markdown summaries (other sources).
- **`outliner`** (`outline` command) — outline generation phase. A LangGraph `StateGraph` with nodes: `load_papers → cluster_papers → group_by_direction → draft_per_group → merge_outlines → review_outline (loop) → render_references`. Uses LangChain's `ChatOpenAI` pointed at a DeepSeek endpoint.

### Key design decisions

- `SearchClient` invokes `claude` CLI as a subprocess, not the Anthropic SDK. It streams JSON events and parses the final `type=result` event for structured output.
- The outliner graph uses `Command(goto=...)` for all transitions (no static edges). The `review_outline` node can loop back to itself up to `OUTLINE_MAX_REVISIONS` times.
- Config is a single `@dataclass` (`deepresearch/config.py`) populated from `.env` via `python-dotenv`. Both packages share it.
- Paper deduplication uses `arxiv:<id>` for arXiv papers and `<source>:<url-hash>` for others.

## Environment

- Python >= 3.12
- Requires `claude` CLI installed and authenticated (used as the search backend)
- `.env` file at project root (copy from `.env.example`). Key vars: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `WORKER_MODEL`, `HTTP_PROXY`.

## Output

Each run creates `runs/{timestamp}-{slug}/` containing:
- `papers.db` — SQLite with all paper metadata
- `pdfs/` — downloaded PDFs and markdown artifacts
- `outline.md` — final literature review outline
