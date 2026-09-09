from pathlib import Path

import pytest

from local_code_agent.tools.git import GitTools


@pytest.mark.integration
def test_git_add_commit_log(
    git_repo: Path,
):
    tools = GitTools(git_repo)

    (
        git_repo / "hello.txt"
    ).write_text(
        "hello\n",
        encoding="utf-8",
    )

    assert tools.git_add(
        "hello.txt"
    ) == "OK"

    commit = tools.git_commit(
        "initial commit"
    )

    assert "ERROR" not in commit

    log = tools.git_log(1)

    assert "initial commit" in log


@pytest.mark.integration
def test_git_diff(
    git_repo: Path,
):
    tools = GitTools(git_repo)

    file = git_repo / "hello.txt"

    file.write_text(
        "one\n",
        encoding="utf-8",
    )

    tools.git_add("hello.txt")
    tools.git_commit("initial")

    file.write_text(
        "one\ntwo\n",
        encoding="utf-8",
    )

    diff = tools.git_diff()

    assert "+two" in diff
