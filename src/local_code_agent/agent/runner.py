from __future__ import annotations

from pathlib import Path

from ollama import Client
from rich.console import Console

from local_code_agent.agent.prompts import system_prompt
from local_code_agent.agent.tool_recovery import recover_tool_calls
from local_code_agent.tools.registry import ToolRegistry


class AgentRunner:
    def __init__(
        self,
        *,
        client: Client,
        model: str,
        registry: ToolRegistry,
        repo: Path,
        max_iterations: int = 20,
        console: Console | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.registry = registry
        self.max_iterations = max_iterations
        self.console = console or Console()

        self.messages: list = [
            {
                "role": "system",
                "content": system_prompt(repo),
            },
        ]

    def reset_context(self) -> None:
        """Reset the conversation context to default (keep only system prompt)."""
        self.messages = [
            {
                "role": "system",
                "content": self.messages[0]["content"],
            },
        ]

    def ask(self, user_message: str) -> str:
        self.registry.begin_turn(user_message)

        self.messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        for _ in range(self.max_iterations):
            response = self.client.chat(
                model=self.model,
                messages=self.messages,
                tools=self.registry.schemas,
                options={
                    "temperature": 0.2,
                },
            )

            message = response.message

            #
            # 1. Normal/native Ollama tool calling.
            #
            if message.tool_calls:
                self.messages.append(message)

                for call in message.tool_calls:
                    name = call.function.name
                    arguments = dict(call.function.arguments or {})

                    self.console.print(
                        f"[dim]tool → {name}[/]"
                    )

                    result = self.registry.execute(
                        name,
                        arguments,
                    )

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "content": str(result),
                        }
                    )

                # Give the tool results back to the model.
                continue

            #
            # 2. Qwen3-Coder fallback.
            #
            # Qwen may occasionally emit a textual tool call instead of
            # populating message.tool_calls.
            #
            content = message.content or ""
            recovered = recover_tool_calls(content)

            if recovered:
                self.console.print(
                    "[yellow]warning:[/] "
                    "recovered malformed Qwen tool call"
                )

                #
                # Only allow calls to tools that are actually registered.
                #
                valid_calls = []

                for call in recovered:
                    if not self.registry.has_tool(call.name):
                        self.console.print(
                            "[yellow]warning:[/] "
                            f"ignoring unknown recovered tool: "
                            f"{call.name}"
                        )
                        continue

                    valid_calls.append(call)

                #
                # If the text looked like a tool call but none of the
                # requested tools are valid, stop rather than fabricating
                # an executable action.
                #
                if not valid_calls:
                    self.messages.append(message)
                    return content

                #
                # Convert the malformed textual call into the structured
                # assistant tool-call message Ollama normally produces.
                #
                synthetic_calls = []

                for index, call in enumerate(valid_calls):
                    synthetic_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "index": index,
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                    )

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": synthetic_calls,
                    }
                )

                #
                # Execute each recovered call exactly once.
                #
                for call in valid_calls:
                    self.console.print(
                        f"[dim]tool → {call.name}[/]"
                    )

                    result = self.registry.execute(
                        call.name,
                        call.arguments,
                    )

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_name": call.name,
                            "content": str(result),
                        }
                    )

                # Give the recovered tool results back to the model.
                continue

            #
            # 3. Normal assistant response.
            #
            self.messages.append(message)
            return content

        return (
            f"Stopped after {self.max_iterations} "
            "tool iterations."
        )