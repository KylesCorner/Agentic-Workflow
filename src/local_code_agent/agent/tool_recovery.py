from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RecoveredToolCall:
    name: str
    arguments: dict[str, Any]


_TOOL_CALL_RE = re.compile(
    r"(?:<tool_call>\s*)?"
    r"<function=([A-Za-z_][A-Za-z0-9_]*)>"
    r"(.*?)"
    r"</tool_call>",
    re.DOTALL,
)

_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*"
    r"(.*?)"
    r"(?=\s*<parameter=|\s*</tool_call>|$)",
    re.DOTALL,
)


def _parse_value(value: str) -> Any:
    value = value.strip()

    # Handle JSON-like values when the model emits them.
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass

    # Otherwise keep it as a normal string.
    return value


def recover_tool_calls(content: str) -> list[RecoveredToolCall]:
    """Recover malformed Qwen-style tool calls leaked into message content.

    Qwen3-Coder occasionally emits:

        <function=read_file>
        <parameter=path> foo.py
        </tool_call>

    without the opening <tool_call> tag. Ollama may then return the call as
    normal assistant content rather than message.tool_calls.
    """

    recovered: list[RecoveredToolCall] = []

    for match in _TOOL_CALL_RE.finditer(content):
        name = match.group(1)
        body = match.group(2)

        arguments: dict[str, Any] = {}

        for param in _PARAMETER_RE.finditer(body):
            key = param.group(1)
            value = _parse_value(param.group(2))
            arguments[key] = value

        recovered.append(
            RecoveredToolCall(
                name=name,
                arguments=arguments,
            )
        )

    return recovered