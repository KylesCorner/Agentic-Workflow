import uuid
from pathlib import Path

import pytest

from local_code_agent.agent.session import SessionStore


def test_create_and_reload_session(repo: Path):
    store = SessionStore(repo)

    first = store.load_or_create()
    second = store.load_or_create()

    assert first == second

    uuid.UUID(first.session_id)


def test_explicit_session(repo: Path):
    store = SessionStore(repo)

    session_id = str(uuid.uuid4())

    session = store.use(session_id)

    assert session.session_id == session_id


def test_invalid_explicit_session(repo: Path):
    store = SessionStore(repo)

    with pytest.raises(ValueError):
        store.use("not-a-uuid")


def test_corrupt_session_file_creates_new_session(
    repo: Path,
):
    store = SessionStore(repo)

    original = store.load_or_create()

    store.state_file.write_text(
        "{broken json",
        encoding="utf-8",
    )

    replacement = store.load_or_create()

    assert (
        replacement.session_id
        != original.session_id
    )
