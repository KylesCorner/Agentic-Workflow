from pathlib import Path

from local_code_agent.rag.document_store import (
    SimpleDocumentStore,
)
from local_code_agent.rag.models import CodeChunk


def make_chunk(
    chunk_id: str,
    *,
    symbol: str = "example",
) -> CodeChunk:
    return CodeChunk(
        id=chunk_id,
        path="src/example.py",
        language="python",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=20,
        node_type="function_definition",
        symbol=symbol,
        parent_symbol=None,
        content=(
            f"def {symbol}():\n"
            "    return 42\n"
        ),
    )


def test_add_and_get_chunk(tmp_path: Path):
    store = SimpleDocumentStore(
        str(tmp_path / "rag.db")
    )

    chunk = make_chunk("chunk-1")

    store.add_chunk(chunk)

    result = store.get_chunk("chunk-1")

    assert result is not None
    assert result.id == "chunk-1"
    assert result.symbol == "example"
    assert store.size() == 1


def test_remove_chunk(tmp_path: Path):
    store = SimpleDocumentStore(
        str(tmp_path / "rag.db")
    )

    store.add_chunk(
        make_chunk("chunk-1")
    )

    assert store.remove_chunk(
        "chunk-1"
    )

    assert store.size() == 0


def test_search(tmp_path: Path):
    store = SimpleDocumentStore(
        str(tmp_path / "rag.db")
    )

    store.add_chunks(
        [
            make_chunk(
                "chunk-add",
                symbol="add_numbers",
            ),
            make_chunk(
                "chunk-delete",
                symbol="delete_user",
            ),
        ]
    )

    results = store.search(
        "add_numbers"
    )

    assert results
    assert (
        results[0].chunk.symbol
        == "add_numbers"
    )
