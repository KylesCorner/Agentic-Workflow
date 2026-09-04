from __future__ import annotations

import subprocess
from pathlib import Path


class GitTools:
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return f"ERROR (git exit {result.returncode}):\n{output}"
        return output or "OK"

    def git_status(self) -> str:
        """Show branch and working tree status."""
        return self._git("status", "--short", "--branch")

    def git_diff(self, staged: bool = False) -> str:
        """Show current Git diff.

        Args:
            staged: Show staged changes instead of unstaged changes.
        """
        args = ["diff"]
        if staged:
            args.append("--cached")
        return self._git(*args)

    def git_log(self, count: int = 10) -> str:
        """Show recent commits.

        Args:
            count: Maximum commits to show.
        """
        count = max(1, min(int(count), 50))
        return self._git("log", f"-{count}", "--oneline", "--decorate")

    def git_add(self, path: str) -> str:
        """Stage one repository path.

        Args:
            path: Repository-relative path to stage.
        """
        return self._git("add", "--", path)

    def git_commit(self, message: str) -> str:
        """Commit staged changes. Only use when the user explicitly requests a commit.

        Args:
            message: Git commit message.
        """
        return self._git("commit", "-m", message)

    def git_create_branch(self, name: str) -> str:
        """Create and switch to a new branch. Only use when explicitly requested.

        Args:
            name: New branch name.
        """
        return self._git("switch", "-c", name)
