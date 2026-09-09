from pathlib import Path

import pytest

from local_code_agent.tools.filesystem import FileSystemTools


def test_write_and_read_file(repo: Path):
    fs = FileSystemTools(repo)

    result = fs.write_file(
        "src/example.py",
        "VALUE = 42\n",
    )

    assert result.startswith("WROTE:")

    contents = fs.read_file(
        "src/example.py"
    )

    assert "VALUE = 42" in contents


def test_nested_directories_are_created(repo: Path):
    fs = FileSystemTools(repo)

    fs.write_file(
        "a/b/c/test.txt",
        "hello",
    )

    assert (
        repo / "a/b/c/test.txt"
    ).read_text() == "hello"


def test_path_escape_is_rejected(repo: Path):
    fs = FileSystemTools(repo)

    with pytest.raises(
        ValueError,
        match="escapes repository root",
    ):
        fs.read_file("../secret.txt")


def test_search_text(repo: Path):
    fs = FileSystemTools(repo)

    fs.write_file(
        "src/a.py",
        "needle = 123\n",
    )

    fs.write_file(
        "src/b.py",
        "something_else = 456\n",
    )

    result = fs.search_text(
        "needle",
        path="src",
        glob_pattern="*.py",
    )

    assert "src/a.py:1" in result
    assert "needle = 123" in result
    assert "src/b.py" not in result


def test_replace_in_file(repo: Path):
    fs = FileSystemTools(repo)

    fs.write_file(
        "config.txt",
        "MODE=old\n",
    )

    result = fs.replace_in_file(
        "config.txt",
        "MODE=old",
        "MODE=new",
    )

    assert result.startswith("UPDATED:")
    assert (
        repo / "config.txt"
    ).read_text() == "MODE=new\n"


def test_replace_requires_unique_match(repo: Path):
    fs = FileSystemTools(repo)

    fs.write_file(
        "values.txt",
        "foo\nfoo\n",
    )

    result = fs.replace_in_file(
        "values.txt",
        "foo",
        "bar",
    )

    assert "occurs 2 times" in result
