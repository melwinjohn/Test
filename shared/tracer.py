"""
Model call tracer with SQLite-backed replay.

Records every LLM call so experiments can be replayed without paying again.
Replay works by hashing (model, prompt) -> looking up the stored response.
"""
import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelTracer:
    """
    Persistent store for LLM calls.  Thread-safe via per-thread connections.

    Workflow
    --------
    1. Before each LLM call:  cached = tracer.get_cached(model, prompt)
    2. If cached:              use cached["response"] – free replay
    3. If not:                 call the API, then tracer.record(...)
    """

    def __init__(self, db_path: str = "traces/default.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        c = self._conn()
        c.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id      TEXT    NOT NULL,
                task_id       TEXT    DEFAULT '',
                worker_id     TEXT    DEFAULT '',
                model         TEXT    NOT NULL,
                prompt_hash   TEXT    NOT NULL,
                prompt        TEXT    NOT NULL,
                response      TEXT    NOT NULL,
                tokens_in     INTEGER DEFAULT 0,
                tokens_out    INTEGER DEFAULT 0,
                cost_usd      REAL    DEFAULT 0.0,
                duration_ms   INTEGER DEFAULT 0,
                metadata      TEXT    DEFAULT '{}',
                created_at    TEXT    NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_hash ON traces(model, prompt_hash)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_task ON traces(task_id)")
        c.commit()

    @staticmethod
    def _hash(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()[:24]

    # ------------------------------------------------------------------ #
    # Public write / read                                                  #
    # ------------------------------------------------------------------ #

    def record(
        self,
        *,
        model: str,
        prompt: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        task_id: str = "",
        worker_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        tid = trace_id or uuid.uuid4().hex
        c = self._conn()
        c.execute(
            """INSERT INTO traces
               (trace_id, task_id, worker_id, model, prompt_hash, prompt, response,
                tokens_in, tokens_out, cost_usd, duration_ms, metadata, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                tid,
                task_id,
                worker_id,
                model,
                self._hash(model, prompt),
                prompt,
                response,
                tokens_in,
                tokens_out,
                cost_usd,
                duration_ms,
                json.dumps(metadata or {}),
                datetime.utcnow().isoformat(),
            ),
        )
        c.commit()
        return tid

    def get_cached(self, model: str, prompt: str) -> Optional[Dict]:
        row = self._conn().execute(
            "SELECT * FROM traces WHERE model=? AND prompt_hash=? ORDER BY created_at DESC LIMIT 1",
            (model, self._hash(model, prompt)),
        ).fetchone()
        return dict(row) if row else None

    def get_task_traces(self, task_id: str) -> List[Dict]:
        rows = self._conn().execute(
            "SELECT * FROM traces WHERE task_id=? ORDER BY created_at",
            (task_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        row = self._conn().execute("""
            SELECT
                COUNT(*)              AS total_calls,
                SUM(tokens_in)        AS tokens_in,
                SUM(tokens_out)       AS tokens_out,
                SUM(cost_usd)         AS total_cost_usd,
                SUM(duration_ms)      AS total_ms,
                COUNT(DISTINCT task_id)  AS unique_tasks,
                COUNT(DISTINCT model)    AS unique_models
            FROM traces
        """).fetchone()
        return dict(row)

    def export_json(self, out_path: str) -> int:
        rows = self._conn().execute(
            "SELECT * FROM traces ORDER BY created_at"
        ).fetchall()
        data = [dict(r) for r in rows]
        Path(out_path).write_text(json.dumps(data, indent=2))
        return len(data)

    def task_summary(self) -> List[Dict]:
        rows = self._conn().execute("""
            SELECT
                task_id,
                COUNT(*) AS calls,
                SUM(tokens_in + tokens_out) AS total_tokens,
                SUM(cost_usd) AS cost_usd,
                SUM(duration_ms) AS duration_ms,
                MIN(created_at) AS started,
                MAX(created_at) AS ended
            FROM traces
            GROUP BY task_id
            ORDER BY started
        """).fetchall()
        return [dict(r) for r in rows]
