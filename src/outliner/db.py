import json
import sqlite3
from pathlib import Path


def load_papers(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM papers "
            "ORDER BY search_direction, relevance_score DESC, created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["authors"] = json.loads(d["authors"]) if d["authors"] else []
        d["categories"] = json.loads(d["categories"]) if d["categories"] else []
        out.append(d)
    return out


def update_search_direction(db_path: Path, paper_id: str, direction: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE papers SET search_direction = ?, "
            "updated_at = datetime('now') WHERE paper_id = ?",
            (direction, paper_id),
        )
        conn.commit()
    finally:
        conn.close()


def count_papers(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    finally:
        conn.close()


def load_paper_titles(db_path: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT paper_id, title FROM papers").fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        conn.close()


def distinct_directions(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT DISTINCT search_direction FROM papers").fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows if r[0]}
