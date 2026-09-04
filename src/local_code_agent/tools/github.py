from __future__ import annotations

import json
import subprocess
from pathlib import Path


class GitHubTools:
    def __init__(self, repo: Path) -> None:
        self.repo = repo.resolve()

    def _gh(self, *args: str) -> str:
        try:
            result = subprocess.run(
                ["gh", *args],
                cwd=self.repo,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            return "ERROR: GitHub CLI 'gh' is not installed or is not on PATH."
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return f"ERROR (gh exit {result.returncode}):\n{output}"
        return output or "OK"

    def github_repo_view(self) -> str:
        """Show GitHub metadata for the current repository."""
        return self._gh("repo", "view", "--json", "nameWithOwner,url,description,defaultBranchRef")

    def github_issue_view(self, number: int) -> str:
        """Read a GitHub issue from the current repository.

        Args:
            number: Issue number.
        """
        return self._gh(
            "issue", "view", str(int(number)),
            "--json", "number,title,body,state,labels,comments,url",
        )

    def github_pr_view(self, number: int) -> str:
        """Read a GitHub pull request from the current repository.

        Args:
            number: Pull request number.
        """
        return self._gh(
            "pr", "view", str(int(number)),
            "--json", "number,title,body,state,baseRefName,headRefName,files,comments,url",
        )

    def github_create_pr(self, title: str, body: str, base: str = "") -> str:
        """Create a GitHub pull request. Only use when the user explicitly requests it.

        Args:
            title: Pull request title.
            body: Pull request body.
            base: Optional base branch. Empty uses GitHub's default.
        """
        args = ["pr", "create", "--title", title, "--body", body]
        if base:
            args.extend(["--base", base])
        return self._gh(*args)

    def github_push(self, remote: str = "origin", branch: str = "") -> str:
        """Push the current branch to a remote. Only use when the user explicitly requests it.

        Args:
            remote: Git remote name.
            branch: Optional branch name; empty pushes the current branch.
        """
        cmd = ["git", "push", remote]
        if branch:
            cmd.append(branch)
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            return "ERROR: git is not installed or is not on PATH."
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return f"ERROR (git push exit {result.returncode}):\n{output}"
        return output or "OK"
