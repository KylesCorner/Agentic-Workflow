from local_code_agent.tools.registry import (
    ExplicitAction,
    Permission,
    ToolRegistry,
)


def test_executes_read_tool():
    def echo(text: str) -> str:
        return text

    registry = ToolRegistry()
    registry.add(echo)

    assert registry.execute(
        "echo",
        {"text": "hello"},
    ) == "hello"


def test_unknown_tool():
    registry = ToolRegistry()

    assert registry.execute("missing", {}) == (
        "ERROR: unknown tool: missing"
    )


def test_tool_exception_is_returned_to_agent():
    def explode() -> str:
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.add(explode)

    result = registry.execute("explode", {})

    assert "ERROR running explode" in result
    assert "RuntimeError" in result
    assert "boom" in result


def test_commit_requires_explicit_user_request():
    def commit(message: str) -> str:
        return f"committed: {message}"

    registry = ToolRegistry(auto_approve=True)

    registry.add(
        commit,
        Permission.WRITE,
        explicit_action=ExplicitAction.COMMIT,
    )

    registry.begin_turn(
        "Please fix the tests."
    )

    denied = registry.execute(
        "commit",
        {"message": "tests"},
    )

    assert "DENIED BY POLICY" in denied

    registry.begin_turn(
        "Please commit these changes."
    )

    allowed = registry.execute(
        "commit",
        {"message": "tests"},
    )

    assert allowed == "committed: tests"


def test_large_tool_result_is_truncated():
    def huge() -> str:
        return "x" * 5000

    registry = ToolRegistry(
        max_result_chars=1000,
    )

    registry.add(huge)

    result = registry.execute("huge", {})

    assert "TOOL OUTPUT TRUNCATED" in result
    assert len(result) < 5000
