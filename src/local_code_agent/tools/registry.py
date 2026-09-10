from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from rich.console import Console
from rich.prompt import Confirm


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


class ExplicitAction(str, Enum):
    """Actions that are never allowed merely because --yes was supplied."""

    COMMIT = "commit"
    BRANCH = "branch"
    PUSH = "push"
    PULL_REQUEST = "pull_request"


_EXPLICIT_ACTION_PATTERNS: dict[ExplicitAction, tuple[re.Pattern[str], ...]] = {
    ExplicitAction.COMMIT: (
        re.compile(r"^\s*commit\s*[.!]?\s*$", re.I),
        re.compile(r"\b(?:please|can you|could you|go ahead and|then)\s+commit\b", re.I),
        re.compile(r"\b(?:make|create)\s+(?:a\s+)?commit\b", re.I),
        re.compile(r"\bcommit\s+(?:these|the|my|our|your|this|those)\s+(?:changes|files?)\b", re.I),
    ),
    ExplicitAction.BRANCH: (
        re.compile(r"^\s*(?:new\s+)?branch\s*[.!]?\s*$", re.I),
        re.compile(r"\b(?:create|make)\s+(?:a\s+)?(?:new\s+)?branch\b", re.I),
        re.compile(r"\b(?:switch|checkout)\s+(?:to\s+)?(?:a\s+)?(?:new\s+)?branch\b", re.I),
        re.compile(r"\bbranch\s+off\b", re.I),
    ),
    ExplicitAction.PUSH: (
        re.compile(r"^\s*push\s*[.!]?\s*$", re.I),
        re.compile(r"\b(?:please|can you|could you|go ahead and|then)\s+push\b", re.I),
        re.compile(r"\bpush\s+(?:these|the|my|our|your|this|those)\s+(?:changes|commits?|branch)\b", re.I),
        re.compile(r"\bpush\s+(?:it|to\s+(?:origin|upstream))\b", re.I),
    ),
    ExplicitAction.PULL_REQUEST: (
        re.compile(r"^\s*(?:pr|pull request)\s*[.!]?\s*$", re.I),
        re.compile(r"\b(?:create|open|make|submit)\s+(?:a\s+)?(?:pr|pull request)\b", re.I),
        re.compile(r"\b(?:please|can you|could you|go ahead and|then)\s+(?:create|open)\s+(?:a\s+)?(?:pr|pull request)\b", re.I),
    ),
}


@dataclass(slots=True)
class ToolSpec:
    function: Callable[..., Any]
    permission: Permission
    explicit_action: ExplicitAction | None = None


class ToolRegistry:
    def __init__(
        self,
        *,
        auto_approve: bool = False,
        console: Console | None = None,
        max_result_chars: int = 40_000,
    ) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.auto_approve = auto_approve
        self.console = console or Console()
        self.max_result_chars = max(1_000, int(max_result_chars))
        self._current_user_request = ""

    def add(
        self,
        function: Callable[..., Any],
        permission: Permission = Permission.READ,
        *,
        explicit_action: ExplicitAction | None = None,
    ) -> None:
        self._tools[function.__name__] = ToolSpec(
            function=function,
            permission=permission,
            explicit_action=explicit_action,
        )

    def begin_turn(self, user_request: str) -> None:
        """Record the exact user turn used for explicit-action policy checks."""
        self._current_user_request = user_request

    @property
    def schemas(self) -> list[Callable[..., Any]]:
        return [spec.function for spec in self._tools.values()]

    def _explicitly_requested(self, action: ExplicitAction) -> bool:
        request = self._current_user_request.strip()
        return any(pattern.search(request) for pattern in _EXPLICIT_ACTION_PATTERNS[action])

    @staticmethod
    def _preview_arguments(arguments: dict[str, Any], limit: int = 1_200) -> str:
        parts: list[str] = []
        remaining = limit
        for key, value in arguments.items():
            rendered = repr(value)
            if len(rendered) > 300:
                rendered = rendered[:280] + f"... <{len(rendered) - 280} chars omitted>"
            piece = f"{key}={rendered}"
            if len(piece) > remaining:
                parts.append("...")
                break
            parts.append(piece)
            remaining -= len(piece) + 2
        return ", ".join(parts)

    def _truncate_result(self, result: str) -> str:
        if len(result) <= self.max_result_chars:
            return result

        head_chars = int(self.max_result_chars * 0.75)
        tail_chars = self.max_result_chars - head_chars
        omitted = len(result) - self.max_result_chars
        return (
            result[:head_chars]
            + f"\n\n... TOOL OUTPUT TRUNCATED ({omitted} chars omitted) ...\n\n"
            + result[-tail_chars:]
        )

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        spec = self._tools.get(name)

        if spec is None:
            return f"ERROR: unknown tool: {name}"

        # Always sanitize model-generated arguments before
        # permission checks, previews, or tool execution.
        arguments = self.prepare_arguments(
            name,
            arguments,
        )

        if (
            spec.explicit_action is not None
            and not self._explicitly_requested(
                spec.explicit_action
            )
        ):
            return (
                f"DENIED BY POLICY: {name} requires the user to explicitly request "
                f"the '{spec.explicit_action.value}' action in the current message. "
                "The --yes flag does not bypass this policy."
            )

        if (
            spec.permission is not Permission.READ
            and not self.auto_approve
        ):
            preview = self._preview_arguments(
                arguments
            )

            self.console.print(
                f"\n[bold yellow]"
                f"{spec.permission.value.upper()} tool:[/] "
                f"{name}({preview})"
            )

            if not Confirm.ask(
                "Approve this tool call?",
                default=False,
            ):
                return f"DENIED BY USER: {name}"

        try:
            result = str(
                spec.function(**arguments)
            )

        except Exception as exc:
            result = (
                f"ERROR running {name}: "
                f"{type(exc).__name__}: {exc}"
            )

        return self._truncate_result(result)
    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def prepare_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize model-generated tool arguments."""

        arguments = dict(arguments)

        if name != "web_search":
            return arguments

        query = str(
            arguments.get("query", "")
        ).strip()

        recency_terms = (
            "latest",
            "current",
            "recent",
            "newest",
            "today",
        )

        query_lower = query.lower()
        request_lower = (
            self._current_user_request.lower()
        )

        recency_requested = any(
            term in query_lower
            or term in request_lower
            for term in recency_terms
        )

        # Detect a model-added trailing year.
        year_match = re.search(
            r"\s+((?:19|20)\d{2})\s*$",
            query,
        )

        if recency_requested and year_match:
            year = year_match.group(1)

            # Preserve the year if the USER explicitly
            # requested that year.
            user_requested_year = re.search(
                rf"\b{re.escape(year)}\b",
                self._current_user_request,
            )

            if not user_requested_year:
                query = query[
                    :year_match.start()
                ].rstrip()

        arguments["query"] = " ".join(
            query.split()
        )

        return arguments