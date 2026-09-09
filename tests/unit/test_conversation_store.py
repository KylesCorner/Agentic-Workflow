from pathlib import Path

import pytest

from local_code_agent.agent.conversation_store import (
    ConversationStore,
)


def test_append_and_load_messages(repo: Path):
    store = ConversationStore(repo)

    session = "session-1"

    first_id = store.append_message(
        session,
        {
            "role": "user",
            "content": "hello",
        },
    )

    second_id = store.append_message(
        session,
        {
            "role": "assistant",
            "content": "hi",
        },
    )

    assert second_id > first_id

    assert store.load_messages(session) == [
        {
            "role": "user",
            "content": "hello",
        },
        {
            "role": "assistant",
            "content": "hi",
        },
    ]


def test_after_id_filter(repo: Path):
    store = ConversationStore(repo)

    session = "session-1"

    first_id = store.append_message(
        session,
        {
            "role": "user",
            "content": "first",
        },
    )

    store.append_message(
        session,
        {
            "role": "assistant",
            "content": "second",
        },
    )

    messages = store.load_messages(
        session,
        after_id=first_id,
    )

    assert len(messages) == 1
    assert messages[0]["content"] == "second"


def test_message_requires_role(repo: Path):
    store = ConversationStore(repo)

    with pytest.raises(
        ValueError,
        match="no role",
    ):
        store.append_message(
            "session",
            {"content": "oops"},
        )


def test_compaction_state(repo: Path):
    store = ConversationStore(repo)

    session = "session"

    store.save_compaction_state(
        session,
        summary="Important state",
        compacted_through_message_id=42,
    )

    state = store.get_compaction_state(
        session
    )

    assert state.summary == "Important state"
    assert (
        state.compacted_through_message_id
        == 42
    )


def test_clear_session(repo: Path):
    store = ConversationStore(repo)

    session = "session"

    store.append_message(
        session,
        {
            "role": "user",
            "content": "one",
        },
    )

    store.append_message(
        session,
        {
            "role": "assistant",
            "content": "two",
        },
    )

    store.save_compaction_state(
        session,
        summary="summary",
        compacted_through_message_id=1,
    )

    removed = store.clear_session(
        session
    )

    assert removed == 2
    assert store.count_messages(session) == 0

    state = store.get_compaction_state(
        session
    )

    assert state.summary == ""
    assert (
        state.compacted_through_message_id
        == 0
    )
