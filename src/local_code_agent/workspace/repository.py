from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Workspace:
    """Resolved working directory for an agent session."""

    requested_path: Path
    root: Path
    is_git_repository: bool


def discover_workspace(path: Path) -> Workspace:
    """Resolve a requested path and, when possible, normalize it to the Git root.

    Non-Git directories are still valid workspaces; Git/GitHub tools will simply
    return errors when used there.
    """
    requested = path.expanduser().resolve()
    if not requested.exists():
        raise ValueError(f"Workspace path does not exist: {requested}")
    if not requested.is_dir():
        raise ValueError(f"Workspace path is not a directory: {requested}")

    try:
        result = subprocess.run(
            ["git", "-C", str(requested), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return Workspace(requested, requested, False)

    if result.returncode != 0:
        return Workspace(requested, requested, False)

    root_text = result.stdout.strip()
    if not root_text:
        return Workspace(requested, requested, False)

    return Workspace(requested, Path(root_text).resolve(), True)
