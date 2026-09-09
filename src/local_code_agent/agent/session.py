from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AgentSession:
    session_id: str
    created_at: str


class SessionStore:
    """Persist the active agent session for a repository."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()
        self.state_dir = self.repo / ".lca"
        self.state_file = self.state_dir / "session.json"

    def load_or_create(self) -> AgentSession:
        """Load the current session, creating one if necessary."""

        if self.state_file.exists():
            try:
                data = json.loads(
                    self.state_file.read_text(encoding="utf-8")
                )

                session_id = self._normalize_id(
                    data["session_id"]
                )

                return AgentSession(
                    session_id=session_id,
                    created_at=data["created_at"],
                )

            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                # Corrupt/invalid session state: create a fresh one.
                pass

        return self.create()

    def create(self) -> AgentSession:
        """Create and activate a new persistent session."""

        session = AgentSession(
            session_id=str(uuid.uuid4()),
            created_at=self._now(),
        )

        self._write(session)

        return session

    def use(self, session_id: str) -> AgentSession:
        """Activate an explicitly supplied session ID."""

        session = AgentSession(
            session_id=self._normalize_id(session_id),
            created_at=self._now(),
        )

        self._write(session)

        return session

    def _write(self, session: AgentSession) -> None:
        self.state_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = self.state_file.with_suffix(".tmp")

        temporary.write_text(
            json.dumps(
                asdict(session),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(self.state_file)

    @staticmethod
    def _normalize_id(session_id: str) -> str:
        return str(uuid.UUID(session_id))

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()