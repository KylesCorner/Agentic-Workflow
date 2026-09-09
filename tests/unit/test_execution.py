from pathlib import Path

from local_code_agent.tools.execution import ExecutionTools


def test_auto_detects_pytest(
    repo: Path,
    monkeypatch,
):
    (repo / "pyproject.toml").write_text(
        "[project]\nname='test'\n"
    )

    tools = ExecutionTools(repo)

    calls = []

    def fake_run(command, timeout=300):
        calls.append((command, timeout))
        return "OK"

    monkeypatch.setattr(
        tools,
        "_run",
        fake_run,
    )

    assert tools.run_tests() == "OK"

    assert calls == [
        (
            ["python", "-m", "pytest"],
            300,
        )
    ]


def test_auto_detects_python_build(
    repo: Path,
    monkeypatch,
):
    (repo / "pyproject.toml").write_text(
        "[project]\nname='test'\n"
    )

    tools = ExecutionTools(repo)

    calls = []

    monkeypatch.setattr(
        tools,
        "_run",
        lambda command, timeout=300: (
            calls.append(command) or "OK"
        ),
    )

    assert tools.run_build() == "OK"

    assert calls == [
        [
            "python",
            "-m",
            "compileall",
            "-q",
            ".",
        ]
    ]


def test_invalid_test_kind(repo: Path):
    tools = ExecutionTools(repo)

    result = tools.run_tests("garbage")

    assert result.startswith("ERROR:")
