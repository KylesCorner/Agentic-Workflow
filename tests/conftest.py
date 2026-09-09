from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Empty temporary repository/workspace."""
    return tmp_path


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Temporary local Git repository with test identity configured."""

    if shutil.which("git") is None:
        pytest.skip("git is not installed")

    subprocess.run(
        ["git", "init", "-q"],
        cwd=tmp_path,
        check=True,
    )

    subprocess.run(
        ["git", "config", "user.name", "Pytest"],
        cwd=tmp_path,
        check=True,
    )

    subprocess.run(
        ["git", "config", "user.email", "pytest@example.com"],
        cwd=tmp_path,
        check=True,
    )

    return tmp_path
