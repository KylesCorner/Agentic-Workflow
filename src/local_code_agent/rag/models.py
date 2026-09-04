from dataclasses import dataclass


@dataclass(slots=True)
class CodeChunk:
    id: str
    path: str
    language: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    node_type: str
    symbol: str | None
    content: str
    parent_symbol: str | None = None


@dataclass(slots=True)
class RetrievalResult:
    """A scored code chunk returned by retrieval."""

    chunk: CodeChunk
    score: float