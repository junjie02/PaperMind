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
```

There are no tests, linters, or build steps configured beyond `pip install -e .`.

## Architecture

Six packages under `src/`:

- **`papermind`** (`PM` command) — entry point, starts the LangGraph pipeline. Supports `--resume <phase>` to restart from any node.
- **`shared`** — `Config` dataclass, `PaperRecord`/`AgentTask`/`AgentResult` Pydantic models, SQLite `Database`, `DedupEngine`.
- **`mcp_servers`** — local MCP tool servers: `ddg_search.py` (DuckDuckGo via `ddgs` package, also provides `fetch_pages_batch` for web scraping) and `rag_retrieval.py` (FAISS retriever).
- **`agents`** — Sub-Agent implementations, all single-prompt with conversation memory:
  - `ExplorerAgent` — background research on sub-directions (DDG search + page fetch, 3 iterations)
  - `ResearcherAgent` — targeted paper collection with PDF download (DDG search + selective fetch, up to 12 iterations, stops when target count reached)
  - `WriterAgent` — RAG-based section writing with fact verification loop
  - `ReviewerAgent` — draft quality review
  - `PolisherAgent` / `ConsistencyCheckerAgent` — final polish and cross-section consistency
- **`orchestrator`** — LangGraph `StateGraph` with phases: explore → outline → research → coverage check → build_index → write → review → polish → merge.
  - `MainAgent` — single system prompt, persistent conversation memory across all orchestration phases
  - `nodes.py` — Phase 1-3 nodes
  - `nodes_writing.py` — Phase 4-6 nodes (includes PDF backfill before indexing)
- **`rag`** — FAISS indexing (`indexer.py`), chunking (`chunker.py`), retrieval (`retriever.py`), paper DB helpers (`db.py`), PDF text extraction (`pdf_extractor.py`). Falls back to abstract/overview if no PDF available.

### Key design decisions

- Main Agent only orchestrates — no direct search or writing. All execution is delegated to Sub-Agents.
- Sub-Agents call MCP tool functions via direct Python import (not stdio protocol) to avoid process overhead.
- All Sub-Agents share the same MiniMax LLM config (`OPENAI_*` env vars).
- Main Agent decides Sub-Agent concurrency per phase (hard cap: 3). `asyncio.Semaphore` enforces this.
- Main Agent allocates per-question paper targets based on importance; user specifies total via `-n`.
- `asyncio.wait_for` in `SubAgentBase.run()` enforces per-agent timeouts.
- Explorer and Researcher agents use single-prompt + conversation memory pattern (not multi-chain).
- Researcher downloads PDFs inline; `build_index` node backfills any missing artifacts before FAISS indexing.
- Paper deduplication uses `arxiv:<id>` for arXiv papers and `<source>:<url-hash>` for others.
- LangGraph graph uses `Command(goto=...)` for all transitions (no static edges).
- Introduction and Conclusion are written after all body sections complete (not sent to researcher).

## Environment

- Python >= 3.12
- `.env` file at project root (copy from `.env.example`). Key vars: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `HTTP_PROXY`.

## Output

Each run creates `runs/{timestamp}-{slug}/` containing:
- `papers.db` — SQLite with all paper metadata
- `pdfs/` — downloaded PDF files + generated MD for papers without PDFs
- `faiss.index` + `chunks.pkl` — FAISS vector index
- `outline.md` — research outline with per-question paper targets
- `data/` — full agent I/O logs (JSON)
- `survey.md` — final literature review
