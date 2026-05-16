import json
import logging
from datetime import datetime

import aiosqlite

from shared.models import PaperRecord

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    async def initialize(self):
        conn = await self._get_conn()
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id          TEXT PRIMARY KEY,
                title             TEXT NOT NULL,
                authors           TEXT NOT NULL,
                abstract          TEXT NOT NULL,
                overview          TEXT DEFAULT '',
                source            TEXT NOT NULL DEFAULT 'web',
                source_url        TEXT NOT NULL DEFAULT '',
                venue             TEXT DEFAULT NULL,
                arxiv_id          TEXT DEFAULT NULL,
                search_direction  TEXT NOT NULL DEFAULT '',
                published_at      TEXT DEFAULT NULL,
                categories        TEXT DEFAULT '[]',
                primary_class     TEXT DEFAULT NULL,
                bibtex            TEXT DEFAULT '',
                abs_url           TEXT NOT NULL DEFAULT '',
                pdf_url           TEXT DEFAULT '',
                artifact_rel_path TEXT DEFAULT NULL,
                search_round      INTEGER NOT NULL DEFAULT 0,
                worker_id         TEXT NOT NULL DEFAULT '',
                relevance_score   INTEGER DEFAULT 3,
                created_at        TEXT DEFAULT (datetime('now')),
                updated_at        TEXT DEFAULT (datetime('now'))
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_direction ON papers(search_direction)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id)")
        await conn.commit()

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def upsert(self, paper: PaperRecord) -> None:
        conn = await self._get_conn()
        now = datetime.utcnow().isoformat()
        await conn.execute(
            """
            INSERT INTO papers (
                paper_id, title, authors, abstract, overview,
                source, source_url, venue, arxiv_id, search_direction,
                published_at, categories, primary_class, bibtex,
                abs_url, pdf_url, artifact_rel_path,
                search_round, worker_id, relevance_score, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                title=excluded.title,
                authors=excluded.authors,
                abstract=excluded.abstract,
                overview=excluded.overview,
                source=excluded.source,
                source_url=excluded.source_url,
                venue=excluded.venue,
                arxiv_id=excluded.arxiv_id,
                search_direction=excluded.search_direction,
                published_at=excluded.published_at,
                categories=excluded.categories,
                primary_class=excluded.primary_class,
                bibtex=excluded.bibtex,
                abs_url=excluded.abs_url,
                pdf_url=excluded.pdf_url,
                artifact_rel_path=COALESCE(papers.artifact_rel_path, excluded.artifact_rel_path),
                relevance_score=excluded.relevance_score,
                updated_at=excluded.updated_at
            """,
            (
                paper.paper_id,
                paper.title,
                json.dumps(paper.authors, ensure_ascii=False),
                paper.abstract,
                paper.overview,
                paper.source,
                paper.source_url,
                paper.venue,
                paper.arxiv_id,
                paper.search_direction,
                paper.published_at,
                json.dumps(paper.categories, ensure_ascii=False),
                paper.primary_class,
                paper.bibtex,
                paper.abs_url,
                paper.pdf_url,
                paper.artifact_rel_path,
                paper.search_round,
                paper.worker_id,
                paper.relevance_score,
                now,
                now,
            ),
        )
        await conn.commit()

    async def update_artifact_path(self, paper_id: str, artifact_rel_path: str) -> None:
        conn = await self._get_conn()
        now = datetime.utcnow().isoformat()
        await conn.execute(
            "UPDATE papers SET artifact_rel_path=?, updated_at=? WHERE paper_id=?",
            (artifact_rel_path, now, paper_id),
        )
        await conn.commit()

    async def get_all_ids(self) -> set[str]:
        conn = await self._get_conn()
        cursor = await conn.execute("SELECT paper_id FROM papers")
        rows = await cursor.fetchall()
        return {r[0] for r in rows}

    async def get_papers(self) -> list[dict]:
        conn = await self._get_conn()
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            "SELECT * FROM papers ORDER BY search_direction, relevance_score DESC, created_at DESC"
        )
        rows = await cursor.fetchall()
        conn.row_factory = None
        results = []
        for row in rows:
            r = dict(row)
            r["authors"] = json.loads(r["authors"])
            r["categories"] = json.loads(r["categories"])
            results.append(r)
        return results
