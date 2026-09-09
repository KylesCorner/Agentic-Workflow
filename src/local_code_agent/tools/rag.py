"""Tree-sitter RAG tools exposed to the coding agent."""

from __future__ import annotations

from pathlib import Path

from local_code_agent.rag.retriever import RAGSystem


class RAGTools:
    """Agent tools for repository semantic search."""

    def __init__(
        self,
        repo: Path,
        *,
        session_id: str | None = None,
    ) -> None:
        self.repo = repo.resolve()
        self.session_id = session_id

        self.db_path = (
            self.repo / ".rag_index.db"
        )
        self.marker_path = (
            self.repo / ".rag_indexed"
        )

        self.rag_system = RAGSystem(
            db_path=str(self.db_path)
        )

    def _reset_index(self) -> None:
        """Delete the existing RAG index before a forced rebuild."""

        # SQLite may create these sidecar files depending on journal mode.
        paths = [
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
            self.marker_path,
        ]

        for path in paths:
            path.unlink(missing_ok=True)

        # Recreate the RAG system against a fresh database.
        self.rag_system = RAGSystem(
            db_path=str(self.db_path)
        )

    def rag_index_repository(
        self,
        force: bool = False,
    ) -> str:
        """Index repository code for semantic RAG search.

        Args:
            force: Delete the existing index and rebuild it from scratch.

        Returns:
            A summary of the indexing operation.

        This tool writes .rag_index.db and .rag_indexed in the repository.
        """

        if (
            not force
            and self.db_path.exists()
            and self.marker_path.exists()
            and self.rag_system.size() > 0
        ):
            return (
                "Repository is already indexed for RAG. "
                f"Current index contains {self.rag_system.size()} chunks. "
                "Use force=true to rebuild it."
            )

        if force:
            self._reset_index()

        self.rag_system.index_repository(
            self.repo
        )

        self.marker_path.touch()

        size = self.rag_system.size()

        return (
            "RAG indexing complete.\n"
            f"Repository: {self.repo}\n"
            f"Chunks: {size}\n"
            f"Database: {self.db_path}"
        )

    def rag_search(
        self,
        query: str,
        top_k: int = 5,
        hybrid_alpha: float = 0.5,
    ) -> str:
        """Semantically search indexed repository code.

        Use this for conceptual questions when the exact function,
        class, symbol, or text is not known.

        Args:
            query: Natural-language or code-related search query.
            top_k: Maximum number of chunks to return.
            hybrid_alpha: Retrieval weighting. 0.0 is lexical-only,
                1.0 is semantic-only, and 0.5 is balanced hybrid search.

        Returns:
            Ranked code chunks with paths, symbols, IDs, and line ranges.
        """

        if not query.strip():
            return "ERROR: RAG query cannot be empty."

        if top_k < 1:
            return "ERROR: top_k must be at least 1."

        # Protect the model context from giant retrievals.
        top_k = min(top_k, 20)

        if not 0.0 <= hybrid_alpha <= 1.0:
            return (
                "ERROR: hybrid_alpha must be between "
                "0.0 and 1.0."
            )

        if (
            not self.db_path.exists()
            or self.rag_system.size() == 0
        ):
            return (
                "RAG index is not available for this repository. "
                "Use rag_index_repository to create it, or use "
                "search_text/list_files for normal repository search."
            )

        results = self.rag_system.search(
            query,
            top_k=top_k,
            hybrid_alpha=hybrid_alpha,
            session_id=self.session_id,
        )

        if not results:
            return (
                "No relevant RAG chunks were found for this query."
            )

        output: list[str] = [
            f"Found {len(results)} relevant code chunks:",
            "",
        ]

        for index, result in enumerate(results, start=1):
            chunk = result.chunk

            output.extend(
                [
                    f"{index}. {chunk.symbol or '<anonymous>'}",
                    f"   ID: {chunk.id}",
                    (
                        f"   File: {chunk.path}:"
                        f"{chunk.start_line}-{chunk.end_line}"
                    ),
                    f"   Score: {result.score:.3f}",
                    f"   Type: {chunk.node_type}",
                    f"   Language: {chunk.language}",
                    (
                        "   Parent: "
                        f"{chunk.parent_symbol or '<none>'}"
                    ),
                    "   Content:",
                    chunk.content[:1000],
                    "",
                ]
            )

        return "\n".join(output)

    def rag_get_chunk(
        self,
        chunk_id: str,
    ) -> str:
        """Retrieve the complete source and metadata for one RAG chunk.

        Args:
            chunk_id: Chunk ID returned by rag_search.

        Returns:
            Full metadata and source text for the requested chunk.
        """

        chunk = self.rag_system.get_chunk(
            chunk_id
        )

        if chunk is None:
            return (
                f"RAG chunk '{chunk_id}' was not found."
            )

        return "\n".join(
            [
                "RAG Chunk",
                f"ID: {chunk.id}",
                f"File: {chunk.path}",
                (
                    f"Lines: "
                    f"{chunk.start_line}-{chunk.end_line}"
                ),
                f"Language: {chunk.language}",
                f"Type: {chunk.node_type}",
                f"Symbol: {chunk.symbol or '<anonymous>'}",
                (
                    "Parent: "
                    f"{chunk.parent_symbol or '<none>'}"
                ),
                "",
                "Content:",
                chunk.content,
            ]
        )