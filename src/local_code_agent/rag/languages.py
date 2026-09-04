"""Tree-sitter language registry reserved for the later RAG milestone."""

from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_python


@dataclass(frozen=True)
class LanguageConfig:
    name: str
    extensions: frozenset[str]
    language: Language
    semantic_nodes: frozenset[str]


LANGUAGES = {
    "python": LanguageConfig(
        "python", frozenset({".py"}), Language(tree_sitter_python.language()),
        frozenset({"function_definition", "class_definition"}),
    ),
    "c": LanguageConfig(
        "c", frozenset({".c", ".h"}), Language(tree_sitter_c.language()),
        frozenset({"function_definition", "struct_specifier", "union_specifier", "enum_specifier", "type_definition"}),
    ),
    "cpp": LanguageConfig(
        "cpp", frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}),
        Language(tree_sitter_cpp.language()),
        frozenset({"function_definition", "class_specifier", "struct_specifier", "enum_specifier", "namespace_definition", "template_declaration"}),
    ),
}


def language_for_path(path: Path) -> LanguageConfig | None:
    suffix = path.suffix.lower()
    for config in LANGUAGES.values():
        if suffix in config.extensions:
            return config
    return None


def parser_for(config: LanguageConfig) -> Parser:
    return Parser(config.language)
