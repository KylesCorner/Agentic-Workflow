from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompactionState:
    summary: str = ""
    compacted_through_message_id: int = 0


class ConversationStore:
    """Persistent conversation history for local-code-agent sessions.

    Raw messages are retained permanently unless the session is explicitly
    reset. Compaction stores a rolling summary and a cutoff message ID; it does
    not delete the original transcript.
    """

    def __init__(self, repo: Path) -> None:
        self.repo = Path(repo).resolve()
        self.state_dir = self.repo / ".lca"
        self.db_path = self.state_dir / "conversations.db"

        self.state_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    created_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (session_id)
                        REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_session_messages_session
                ON session_messages(
                    session_id,
                    id
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_compaction (
                    session_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL
                        DEFAULT '',
                    compacted_through_message_id INTEGER
                        NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (session_id)
                        REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                )
                """
            )

    def ensure_session(
        self,
        session_id: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id
                )
                VALUES (?)
                ON CONFLICT(session_id)
                DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id,),
            )

    def append_message(
        self,
        session_id: str,
        message: dict[str, Any],
    ) -> int:
        """Append one normalized Ollama message and return its row ID."""

        role = str(
            message.get("role", "")
        ).strip()

        if not role:
            raise ValueError(
                "Conversation message has no role."
            )

        payload = json.dumps(
            message,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id
                )
                VALUES (?)
                ON CONFLICT(session_id)
                DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id,),
            )

            cursor = conn.execute(
                """
                INSERT INTO session_messages (
                    session_id,
                    role,
                    message_json
                )
                VALUES (?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    payload,
                ),
            )

            return int(cursor.lastrowid)

    def load_message_records(
        self,
        session_id: str,
        *,
        after_id: int = 0,
    ) -> list[tuple[int, dict[str, Any]]]:
        """Load persisted messages with their database IDs."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, message_json
                FROM session_messages
                WHERE session_id = ?
                  AND id > ?
                ORDER BY id ASC
                """,
                (
                    session_id,
                    after_id,
                ),
            ).fetchall()

        records: list[
            tuple[int, dict[str, Any]]
        ] = []

        for message_id, payload in rows:
            try:
                decoded = json.loads(payload)

            except json.JSONDecodeError:
                logger.warning(
                    "Skipping corrupt persisted "
                    "message id=%s session=%s",
                    message_id,
                    session_id,
                )
                continue

            if not isinstance(decoded, dict):
                logger.warning(
                    "Skipping non-object persisted "
                    "message id=%s session=%s",
                    message_id,
                    session_id,
                )
                continue

            # The current AgentRunner always supplies a fresh system prompt.
            if decoded.get("role") == "system":
                continue

            records.append(
                (
                    int(message_id),
                    decoded,
                )
            )

        return records

    def load_messages(
        self,
        session_id: str,
        *,
        after_id: int = 0,
    ) -> list[dict[str, Any]]:
        return [
            message
            for _, message
            in self.load_message_records(
                session_id,
                after_id=after_id,
            )
        ]

    def get_compaction_state(
        self,
        session_id: str,
    ) -> CompactionState:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    summary_text,
                    compacted_through_message_id
                FROM session_compaction
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return CompactionState()

        return CompactionState(
            summary=str(row[0] or ""),
            compacted_through_message_id=int(
                row[1] or 0
            ),
        )

    def save_compaction_state(
        self,
        session_id: str,
        *,
        summary: str,
        compacted_through_message_id: int,
    ) -> None:
        """Persist the rolling summary and transcript cutoff."""

        self.ensure_session(session_id)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_compaction (
                    session_id,
                    summary_text,
                    compacted_through_message_id,
                    updated_at
                )
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)

                ON CONFLICT(session_id)
                DO UPDATE SET
                    summary_text = excluded.summary_text,
                    compacted_through_message_id =
                        excluded.compacted_through_message_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    summary,
                    compacted_through_message_id,
                ),
            )

            conn.execute(
                """
                UPDATE sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (session_id,),
            )

    def clear_session(
        self,
        session_id: str,
    ) -> int:
        """Delete transcript and summary while preserving session identity."""

        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM session_messages
                WHERE session_id = ?
                """,
                (session_id,),
            )

            conn.execute(
                """
                DELETE FROM session_compaction
                WHERE session_id = ?
                """,
                (session_id,),
            )

            conn.execute(
                """
                INSERT INTO sessions (
                    session_id
                )
                VALUES (?)
                ON CONFLICT(session_id)
                DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id,),
            )

            return cursor.rowcount

    def count_messages(
        self,
        session_id: str,
    ) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM session_messages
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        return int(
            row[0] if row else 0
        )

    def list_sessions(
        self,
    ) -> list[dict[str, Any]]:
        """Return known sessions, newest activity first."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.id) AS message_count,
                    COALESCE(
                        c.compacted_through_message_id,
                        0
                    ) AS compacted_through
                FROM sessions AS s
                LEFT JOIN session_messages AS m
                    ON m.session_id = s.session_id
                LEFT JOIN session_compaction AS c
                    ON c.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                """
            ).fetchall()

        return [
            {
                "session_id": row[0],
                "created_at": row[1],
                "updated_at": row[2],
                "message_count": row[3],
                "compacted_through": row[4],
            }
            for row in rows
        ]
