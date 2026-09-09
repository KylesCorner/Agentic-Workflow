from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


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
    # Enhanced memory management fields
    tags: List[str] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass(slots=True)
class RetrievalResult:
    """A scored code chunk returned by retrieval."""

    chunk: CodeChunk
    score: float