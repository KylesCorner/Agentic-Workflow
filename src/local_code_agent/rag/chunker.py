"""Tree-sitter semantic chunking implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tree_sitter import Node

from .languages import language_for_path, parser_for
from .models import CodeChunk


IDENTIFIER_TYPES = {"identifier", "field_identifier", "type_identifier", "namespace_identifier"}
CONTAINER_NODES = {"class_definition", "class_specifier", "struct_specifier", "namespace_definition"}


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _walk(node: Node):
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _find_identifier(node: Node, source: bytes) -> str | None:
    if node.type in IDENTIFIER_TYPES:
        return _text(node, source)
    for child in reversed(node.children):
        result = _find_identifier(child, source)
        if result:
            return result
    return None


def _symbol_name(node: Node, source: bytes) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return _text(name, source)
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        return _find_identifier(declarator, source)
    # Fallback to finding identifier in children
    for child in node.children:
        if child.type in IDENTIFIER_TYPES:
            return _text(child, source)
    return None


def _parent_symbol(node: Node, source: bytes) -> str | None:
    parent = node.parent
    while parent is not None:
        if parent.type in CONTAINER_NODES:
            name = _symbol_name(parent, source)
            if name:
                return name
        parent = parent.parent
    return None

class TreeSitterChunker:
    def chunk_file(
        self,
        path: Path,
        repo_root: Path,
    ) -> list[CodeChunk]:
        """Chunk a source file from disk."""

        source = path.read_bytes()
        relative = path.relative_to(repo_root)

        return self.chunk_content(
            source,
            path=relative,
        )

    def chunk_content(
        self,
        content: str | bytes,
        *,
        path: str | Path,
    ) -> list[CodeChunk]:
        """Chunk source code already loaded in memory.

        The path is used to determine the language and is also stored
        in the resulting CodeChunk metadata.
        """

        path = Path(path)

        config = language_for_path(path)
        if config is None:
            return []

        if isinstance(content, str):
            source = content.encode("utf-8")
        else:
            source = content

        tree = parser_for(config).parse(source)

        chunks: list[CodeChunk] = []

        for node in _walk(tree.root_node):
            if node.type not in config.semantic_nodes:
                continue

            chunk_text = _text(node, source)

            raw_id = (
                f"{path}:"
                f"{node.start_byte}:"
                f"{node.end_byte}:"
                f"{chunk_text}"
            ).encode()

            chunks.append(
                CodeChunk(
                    id=hashlib.sha256(raw_id).hexdigest(),
                    path=str(path),
                    language=config.name,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    node_type=node.type,
                    symbol=_symbol_name(node, source),
                    parent_symbol=_parent_symbol(node, source),
                    content=chunk_text,
                )
            )

        return chunks