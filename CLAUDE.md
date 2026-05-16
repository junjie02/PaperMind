# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PaperMind is an automated literature survey tool. A main Agent (LangGraph + MiniMax LLM) orchestrates multiple concurrent Sub-Agents to handle direction exploration, paper mining, writing, review, and polishing. Search and RAG capabilities are exposed as local MCP tools.

## Commands

```bash
# Install (editable, into a venv)
pip install -e .

# Run the full pipeline
PM "research topic" -n 30
PM "agent safety" -n 15 -v

# Resume from a specific phase (reuse existing run directory)
PM "research topic" --out runs/existing-run --resume build_index
PM "research topic" --out runs/existing-run --resume write_sections

# Skip reviewer phase
PM "research topic" -n 20 --skip-review

# Auto-generate PDF after completion
PM "research topic" -n 20 --pdf

# Convert existing markdown to PDF
PM --to-pdf runs/existing-run/survey.md
```

There are no tests, linters, or build steps configured beyond `pip install -e .`.

## Architecture

Seven packages under `src/`:

- **`papermind`** (`PM` command) — entry point, starts the LangGraph pipeline. Supports `--resume`, `--skip-review`, `--pdf`, `--to-pdf`.
- **`shared`** — `Config` dataclass, `PaperRecord`/`AgentTask`/`AgentResult` Pydantic models, SQLite `Database`, `DedupEngine`, `make_llm`/`make_review_llm` factories.
- **`mcp_servers`** — local MCP tool servers: `ddg_search.py` (DuckDuckGo + `fetch_pages_batch` for web scraping) and `rag_retrieval.py` (FAISS retriever).
- **`agents`** — Sub-Agent implementations, all single-prompt with conversation memory:
  - `ExplorerAgent` — background research on sub-directions (DDG search + page fetch, 3 iterations)
  - `ResearcherAgent` — targeted paper collection with PDF download (DDG search + selective fetch, up to 12 iterations, stops when target count reached)
  - `WriterAgent` — RAG-based iterative writing (multi-query parallel retrieval, adjacent chunk context, random exploration, available papers list injection)
  - `ReviewerAgent` — iterative fact-checking via RAG (verifies claims batch by batch)
  - `PolisherAgent` / `ConsistencyCheckerAgent` — per-section polish and cross-section consistency
- **`orchestrator`** — LangGraph `StateGraph`:
  - `MainAgent` — single system prompt, persistent conversation memory across all orchestration phases
  - `nodes.py` — Phase 1-3: explore → outline → research → coverage check
  - `nodes_writing.py` — Phase 4-7: build_index → write (with review loop) → polish → consistency → final_review → merge
- **`rag`** — FAISS indexing (`indexer.py`), chunking (`chunker.py`), retrieval with context expansion (`retriever.py`), paper DB helpers (`db.py`), PDF text extraction (`pdf_extractor.py`).
- **`evaluate`** — evaluation metrics script (`python -m evaluate <run_dir>`)

### Key design decisions

- Main Agent only orchestrates — no direct search or writing. All execution is delegated to Sub-Agents.
- Sub-Agents call MCP tool functions via direct Python import (not stdio protocol) to avoid process overhead.
- All Sub-Agents share the same MiniMax LLM config (`OPENAI_*` env vars). Final review uses separate `REVIEW_*` config.
- Main Agent decides Sub-Agent concurrency per phase (hard cap: 3). `asyncio.Semaphore` enforces this.
- Main Agent allocates per-question paper targets based on importance; user specifies total via `-n`.
- RAG retrieval returns top-2 matches + adjacent chunks (±1) + random exploration chunks (distance > 2).
- Writer receives full available papers list from DB to maximize citation coverage.
- Writer uses title-based citations `[Paper Title]`; `merge_final` resolves to numbered `[N]` via embedding similarity matching.
- References output in GB/T 7714 format.
- Final review node sends entire document to a separate (stronger) LLM for global polish before merge.
- JSON parse failures trigger automatic retry (up to 2 retries with error feedback to LLM).
- LLM HTTP calls have `max_retries=3` for transient failures (429/5xx).
- LangGraph graph uses `Command(goto=...)` for all transitions (no static edges).

## Environment

- Python >= 3.12
- `.env` file at project root (copy from `.env.example`). Key vars:
  - `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` — main LLM
  - `REVIEW_API_KEY`, `REVIEW_BASE_URL`, `REVIEW_MODEL` — final review LLM (optional, falls back to main)
  - `HTTP_PROXY` — for DDG search and PDF download

## Output

Each run creates `runs/{timestamp}-{slug}/` containing:
- `papers.db` — SQLite with all paper metadata
- `pdfs/` — downloaded PDF files + generated MD for papers without PDFs
- `faiss.index` + `chunks.pkl` — FAISS vector index
- `outline.md` — research outline with per-question paper targets
- `sections/` — individual section drafts (one md per sub_question)
- `draft.md` — combined body draft before polish
- `data/` — full agent I/O logs (JSON)
- `survey.md` — final literature review
- `survey.pdf` — PDF export (if `--pdf` used)
- `evaluation_report.json` — metrics (if evaluation run)
