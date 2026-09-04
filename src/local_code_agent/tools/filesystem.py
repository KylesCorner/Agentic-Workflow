from __future__ import annotations

import fnmatch
import os
from pathlib import Path


class FileSystemTools:
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()

    def _resolve(self, path: str) -> Path:
        candidate = (self.repo / path).resolve()
        try:
            candidate.relative_to(self.repo)
        except ValueError as exc:
            raise ValueError(f"Path escapes repository root: {path}") from exc
        return candidate

    def read_file(self, path: str, start_line: int = 1, end_line: int = 400) -> str:
        """Read a UTF-8 text file from the repository.

        Args:
            path: Repository-relative file path.
            start_line: First 1-based line to return.
            end_line: Last 1-based line to return.
        """
        file_path = self._resolve(path)
        if not file_path.is_file():
            return f"ERROR: file not found: {path}"
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), max(start, int(end_line)))
        body = "\n".join(f"{i:>6}: {lines[i - 1]}" for i in range(start, end + 1))
        return f"FILE: {path}\nLINES: {start}-{end} of {len(lines)}\n\n{body}"

    def list_files(self, path: str = ".", max_entries: int = 200) -> str:
        """List repository files/directories below a path.

        Args:
            path: Repository-relative directory.
            max_entries: Maximum number of entries to return.
        """
        root = self._resolve(path)
        if not root.exists():
            return f"ERROR: path not found: {path}"
        if root.is_file():
            return str(root.relative_to(self.repo))

        ignored_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".code-agent"}
        entries: list[str] = []
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in ignored_dirs)
            current_path = Path(current)
            for d in dirs:
                entries.append(str((current_path / d).relative_to(self.repo)) + "/")
                if len(entries) >= max_entries:
                    return "\n".join(entries) + "\n... truncated ..."
            for name in sorted(files):
                entries.append(str((current_path / name).relative_to(self.repo)))
                if len(entries) >= max_entries:
                    return "\n".join(entries) + "\n... truncated ..."
        return "\n".join(entries) if entries else "(empty)"

    def search_text(self, query: str, path: str = ".", glob_pattern: str = "*") -> str:
        """Search text files in the repository for a literal string.

        Args:
            query: Literal text to search for.
            path: Repository-relative directory or file.
            glob_pattern: Filename glob, for example '*.cpp' or '*.py'.
        """
        root = self._resolve(path)
        candidates = [root] if root.is_file() else root.rglob("*")
        ignored_parts = {".git", ".venv", "venv", "__pycache__", "node_modules", ".code-agent"}
        matches: list[str] = []

        for file_path in candidates:
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(self.repo)
            if any(part in ignored_parts for part in rel.parts):
                continue
            if not fnmatch.fnmatch(file_path.name, glob_pattern):
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    matches.append(f"{rel}:{number}: {line.strip()}")
                    if len(matches) >= 200:
                        return "\n".join(matches) + "\n... truncated ..."
        return "\n".join(matches) if matches else "No matches."

    def write_file(self, path: str, content: str) -> str:
        """Create or replace a UTF-8 text file in the repository.

        Args:
            path: Repository-relative file path.
            content: Complete new file content.
        """
        file_path = self._resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"WROTE: {path} ({len(content.encode('utf-8'))} bytes)"

    def replace_in_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        """Replace exact text in a repository file.

        Args:
            path: Repository-relative file path.
            old_text: Exact text to replace.
            new_text: Replacement text.
            replace_all: Replace every occurrence instead of requiring exactly one.
        """
        file_path = self._resolve(path)
        if not file_path.is_file():
            return f"ERROR: file not found: {path}"
        text = file_path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count == 0:
            return "ERROR: old_text was not found; no changes made."
        if not replace_all and count != 1:
            return f"ERROR: old_text occurs {count} times; make the match more specific or set replace_all=true."
        updated = text.replace(old_text, new_text, -1 if replace_all else 1)
        file_path.write_text(updated, encoding="utf-8")
        return f"UPDATED: {path}; replacements={count if replace_all else 1}"
