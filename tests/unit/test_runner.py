from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from copy import deepcopy

from local_code_agent.agent import runner as runner_module
from local_code_agent.tools.registry import ToolRegistry

@dataclass
class FakeFunction:
    name: str
    arguments: dict[str, Any]

    def model_dump(self, **_):
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class FakeToolCall:
    function: FakeFunction

    def model_dump(self, **_):
        return {
            "type": "function",
            "function": self.function.model_dump(),
        }


@dataclass
class FakeMessage:
    content: str = ""
    role: str = "assistant"
    tool_calls: list[FakeToolCall] = field(
        default_factory=list
    )

    def model_dump(self, **_):
        result = {
            "role": self.role,
            "content": self.content,
        }

        if self.tool_calls:
            result["tool_calls"] = [
                call.model_dump()
                for call in self.tool_calls
            ]

        return result


@dataclass
class FakeResponse:
    message: FakeMessage


class FakeClient:
    def __init__(
        self,
        responses: list[FakeResponse],
    ):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        # Snapshot the call at the moment it happened.
        self.calls.append(deepcopy(kwargs))

        if not self.responses:
            raise AssertionError(
                "FakeClient ran out of responses"
            )

        return self.responses.pop(0)

class DummyRAGSystem:
    def size(self):
        return 0


class DummyRAGTools:
    def __init__(
        self,
        repo: Path,
        *,
        session_id=None,
    ):
        self.repo = Path(repo)
        self.session_id = session_id
        self.db_path = (
            self.repo / "rag_index.db"
        )
        self.rag_system = DummyRAGSystem()


def make_runner(
    tmp_path,
    monkeypatch,
    client,
    registry,
    *,
    max_iterations=20,
):
    monkeypatch.setattr(
        runner_module,
        "RAGTools",
        DummyRAGTools,
    )

    return runner_module.AgentRunner(
        client=client,
        model="fake-model",
        registry=registry,
        repo=tmp_path,
        max_iterations=max_iterations,
    )


def test_normal_response(
    tmp_path,
    monkeypatch,
):
    client = FakeClient(
        [
            FakeResponse(
                FakeMessage(
                    content="Hello!"
                )
            )
        ]
    )

    runner = make_runner(
        tmp_path,
        monkeypatch,
        client,
        ToolRegistry(),
    )

    assert runner.ask("Hi") == "Hello!"
    assert len(client.calls) == 1


def test_native_tool_call(
    tmp_path,
    monkeypatch,
):
    def echo(text: str) -> str:
        return f"ECHO:{text}"

    registry = ToolRegistry()
    registry.add(echo)

    client = FakeClient(
        [
            FakeResponse(
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            FakeFunction(
                                name="echo",
                                arguments={
                                    "text": "hello"
                                },
                            )
                        )
                    ]
                )
            ),
            FakeResponse(
                FakeMessage(
                    content="Finished."
                )
            ),
        ]
    )

    runner = make_runner(
        tmp_path,
        monkeypatch,
        client,
        registry,
    )

    result = runner.ask(
        "Use echo."
    )

    assert result == "Finished."
    assert len(client.calls) == 2

    second_messages = (
        client.calls[1]["messages"]
    )

    assert (
        second_messages[-1]["role"]
        == "tool"
    )

    assert (
        second_messages[-1]["content"]
        == "ECHO:hello"
    )


def test_qwen_text_tool_recovery(
    tmp_path,
    monkeypatch,
):
    def echo(text: str) -> str:
        return f"ECHO:{text}"

    registry = ToolRegistry()
    registry.add(echo)

    malformed = """
    <function=echo>
    <parameter=text>"hello"
    </tool_call>
    """

    client = FakeClient(
        [
            FakeResponse(
                FakeMessage(
                    content=malformed
                )
            ),
            FakeResponse(
                FakeMessage(
                    content="Recovered."
                )
            ),
        ]
    )

    runner = make_runner(
        tmp_path,
        monkeypatch,
        client,
        registry,
    )

    assert (
        runner.ask("Use echo.")
        == "Recovered."
    )

    assert len(client.calls) == 2

    tool_message = (
        client.calls[1]["messages"][-1]
    )

    assert tool_message["role"] == "tool"
    assert tool_message["content"] == (
        "ECHO:hello"
    )


def test_iteration_limit_forces_final_response(
    tmp_path,
    monkeypatch,
):
    def echo(text: str) -> str:
        return text

    registry = ToolRegistry()
    registry.add(echo)

    client = FakeClient(
        [
            FakeResponse(
                FakeMessage(
                    tool_calls=[
                        FakeToolCall(
                            FakeFunction(
                                "echo",
                                {"text": "hello"},
                            )
                        )
                    ]
                )
            ),
            FakeResponse(
                FakeMessage(
                    content=(
                        "Tool budget exhausted."
                    )
                )
            ),
        ]
    )

    runner = make_runner(
        tmp_path,
        monkeypatch,
        client,
        registry,
        max_iterations=1,
    )

    result = runner.ask("Do something.")

    assert result == (
        "Tool budget exhausted."
    )

    assert len(client.calls) == 2

    assert (
        "tools"
        not in client.calls[1]
    )
