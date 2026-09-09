from pathlib import Path

import pytest

from local_code_agent.workspace.repository import (
    discover_workspace,
)


@pytest.mark.integration
def test_nested_path_resolves_to_git_root(
    git_repo: Path,
):
    nested = git_repo / "src/package"
    nested.mkdir(parents=True)

    workspace = discover_workspace(
        nested
    )

    assert workspace.is_git_repository
    assert workspace.root == (
        git_repo.resolve()
    )
    assert workspace.requested_path == (
        nested.resolve()
    )


@pytest.mark.integration
def test_non_git_workspace(tmp_path: Path):
    directory = tmp_path / "plain"
    directory.mkdir()

    workspace = discover_workspace(
        directory
    )

    assert not workspace.is_git_repository
    assert workspace.root == (
        directory.resolve()
    )


def test_missing_workspace_raises(
    tmp_path: Path,
):
    with pytest.raises(ValueError):
        discover_workspace(
            tmp_path / "missing"
        )
