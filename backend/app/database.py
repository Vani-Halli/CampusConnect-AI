import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import BASE_DIR, settings

DB_PATH = settings.database_path
if not DB_PATH.is_absolute():
    DB_PATH = (BASE_DIR / DB_PATH).resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT,
            helpful INTEGER NOT NULL,
            category TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            category TEXT,
            mode TEXT,
            answer TEXT,
            retrieval_score REAL,
            source_count INTEGER,
            unanswered INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT NOT NULL,
            storage_name TEXT,
            status TEXT NOT NULL,
            page_count INTEGER,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            indexed INTEGER DEFAULT 0
        )
        """
    )
    document_columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
    if "storage_name" not in document_columns:
        conn.execute("ALTER TABLE documents ADD COLUMN storage_name TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_id TEXT NOT NULL UNIQUE,
            page_number INTEGER NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
        """
    )
    conn.commit()
    conn.close()


def record_feedback(question: str, answer: str, helpful: bool, category: str | None = None) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO feedback (question, answer, helpful, category) VALUES (?, ?, ?, ?)",
        (question, answer, int(helpful), category),
    )
    conn.commit()
    conn.close()


def record_analytics(question: str, category: str | None, mode: str, answer: str, retrieval_score: float, source_count: int, unanswered: int) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO analytics (question, category, mode, answer, retrieval_score, source_count, unanswered) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (question, category, mode, answer, retrieval_score, source_count, int(unanswered)),
    )
    conn.commit()
    conn.close()


def get_analytics_summary() -> dict[str, Any]:
    conn = get_connection()
    total_questions = conn.execute("SELECT COUNT(*) as count FROM analytics").fetchone()["count"]
    helpful_count = conn.execute("SELECT COUNT(*) as count FROM feedback WHERE helpful = 1").fetchone()["count"]
    not_helpful_count = conn.execute("SELECT COUNT(*) as count FROM feedback WHERE helpful = 0").fetchone()["count"]
    recent = conn.execute(
        "SELECT question, category, mode, created_at FROM analytics ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    categories = conn.execute(
        "SELECT category, COUNT(*) as count FROM analytics GROUP BY category ORDER BY count DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return {
        "total_questions": total_questions,
        "helpful_feedback": helpful_count,
        "not_helpful_feedback": not_helpful_count,
        "recent_queries": [dict(row) for row in recent],
        "frequently_requested_categories": [dict(row) for row in categories],
    }


def list_documents() -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_document(
    document_name: str,
    status: str,
    page_count: int | None = None,
    file_size: int | None = None,
    indexed: bool = True,
    storage_name: str | None = None,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO documents (document_name, storage_name, status, page_count, file_size, indexed) VALUES (?, ?, ?, ?, ?, ?)",
        (document_name, storage_name, status, page_count, file_size, int(indexed)),
    )
    conn.commit()
    doc_id = cursor.lastrowid
    conn.close()
    return int(doc_id)


def save_document_chunks(document_id: int, chunks: list[dict[str, Any]]) -> None:
    conn = get_connection()
    conn.executemany(
        "INSERT INTO document_chunks (document_id, chunk_id, page_number, content) VALUES (?, ?, ?, ?)",
        [(document_id, chunk["chunk_id"], chunk["page_number"], chunk["content"]) for chunk in chunks],
    )
    conn.commit()
    conn.close()


def list_document_chunks() -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT c.document_id, c.chunk_id, c.page_number, c.content, d.document_name "
        "FROM document_chunks AS c JOIN documents AS d ON d.id = c.document_id "
        "WHERE d.indexed = 1 ORDER BY d.id, c.page_number, c.id"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def remove_document(document_id: int) -> str | None:
    conn = get_connection()
    row = conn.execute("SELECT storage_name FROM documents WHERE id = ?", (document_id,)).fetchone()
    if row is None:
        conn.close()
        return None
    conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()
    conn.close()
    return row["storage_name"]


init_db()
