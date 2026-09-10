from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import typer
from ollama import Client
from rich.console import Console
from rich.markdown import Markdown

from local_code_agent.agent.runner import AgentRunner
from local_code_agent.agent.session import SessionStore
from local_code_agent.config import settings
from local_code_agent.tools.docker import DockerTools
from local_code_agent.tools.execution import ExecutionTools
from local_code_agent.tools.filesystem import FileSystemTools
from local_code_agent.tools.git import GitTools
from local_code_agent.tools.github import GitHubTools
from local_code_agent.tools.rag import RAGTools
from local_code_agent.tools.registry import ExplicitAction, Permission, ToolRegistry
from local_code_agent.tools.tavily import TavilyTools
from local_code_agent.tools.unix import UnixTools
from local_code_agent.workspace import discover_workspace

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def build_registry(
    repo: Path,
    *,
    web: bool,
    auto_approve: bool,
    session_id: str | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(
        auto_approve=auto_approve,
        console=console,
        max_result_chars=settings.code_agent_max_tool_output_chars,
    )

    fs = FileSystemTools(repo)
    git = GitTools(repo)
    github = GitHubTools(repo)
    execution = ExecutionTools(repo)
    rag = RAGTools(
        repo,
        session_id=session_id,
    )
    unix = UnixTools(repo)
    docker = DockerTools(repo)

    registry.add(
        rag.rag_index_repository,
        Permission.WRITE,
    )

    registry.add(
        rag.rag_search,
        Permission.READ,
    )

    registry.add(
        rag.rag_get_chunk,
        Permission.READ,
    )

    registry.add(fs.read_file)
    registry.add(fs.list_files)
    registry.add(fs.search_text)
    registry.add(fs.write_file, Permission.WRITE)
    registry.add(fs.replace_in_file, Permission.WRITE)

    registry.add(git.git_status)
    registry.add(git.git_diff)
    registry.add(git.git_log)
    registry.add(git.git_add, Permission.WRITE)

    registry.add(
        git.git_commit,
        Permission.WRITE,
        explicit_action=ExplicitAction.COMMIT,
    )

    registry.add(
        git.git_create_branch,
        Permission.WRITE,
        explicit_action=ExplicitAction.BRANCH,
    )

    registry.add(github.github_repo_view)
    registry.add(github.github_issue_view)
    registry.add(github.github_pr_view)

    registry.add(
        github.github_create_pr,
        Permission.EXECUTE,
        explicit_action=ExplicitAction.PULL_REQUEST,
    )

    registry.add(
        github.github_push,
        Permission.EXECUTE,
        explicit_action=ExplicitAction.PUSH,
    )

    registry.add(
        execution.run_tests,
        Permission.EXECUTE,
    )
    registry.add(
        execution.run_build,
        Permission.EXECUTE,
    )

    # Unix tools.
    # registry.add(unix.grep, Permission.READ)
    # registry.add(unix.find, Permission.READ)
    # registry.add(unix.sed, Permission.WRITE)

    # Docker tools.
    registry.add(
        docker.docker_run,
        Permission.EXECUTE,
    )
    registry.add(
        docker.docker_ps,
        Permission.READ,
    )
    registry.add(
        docker.docker_images,
        Permission.READ,
    )
    registry.add(
        docker.docker_build,
        Permission.WRITE,
    )

    if web:
        tavily = TavilyTools(
            settings.tavily_api_key
        )
        registry.add(tavily.web_search)

    return registry


@app.command()
def chat(
    repo: Path = typer.Option(
        Path.cwd(),
        "--repo",
        "-r",
        help="Repository/workspace path.",
    ),
    model: str = typer.Option(
        settings.ollama_model,
        "--model",
        "-m",
    ),
    web: bool = typer.Option(
        False,
        "--web/--no-web",
        help="Enable Tavily web search.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help=(
            "Auto-approve ordinary "
            "write/execute tools."
        ),
    ),
    session_id: str | None = typer.Option(
        None,
        "--session",
        help=(
            "Resume a specific persistent "
            "session UUID."
        ),
    ),
    new_session: bool = typer.Option(
        False,
        "--new-session",
        help=(
            "Start a new persistent "
            "agent session."
        ),
    ),
) -> None:
    """Start an interactive local coding-agent session."""

    #
    # 1. Resolve workspace.
    #
    try:
        workspace = discover_workspace(repo)
    except ValueError as exc:
        console.print(
            f"[bold red]Workspace error:[/] {exc}"
        )
        raise typer.Exit(code=2) from exc

    if not workspace.is_git_repository:
        console.print(
            "[yellow]Warning:[/] "
            f"{workspace.root} is not "
            "a Git repository."
        )

    elif (
        workspace.root
        != workspace.requested_path
    ):
        console.print(
            "[dim]Resolved Git root: "
            f"{workspace.root}[/]"
        )

    #
    # 2. Resolve persistent session BEFORE
    #    building tools or AgentRunner.
    #
    if session_id and new_session:
        console.print(
            "[bold red]Session error:[/] "
            "--session and --new-session "
            "cannot be used together."
        )
        raise typer.Exit(code=2)

    session_store = SessionStore(
        workspace.root
    )

    try:
        if new_session:
            session = session_store.create()

        elif session_id:
            session = session_store.use(
                session_id
            )

        else:
            session = (
                session_store.load_or_create()
            )

    except ValueError as exc:
        console.print(
            f"[bold red]Session error:[/] {exc}"
        )
        raise typer.Exit(code=2) from exc

    #
    # 3. Build tool registry with the resolved
    #    session ID.
    #
    try:
        registry = build_registry(
            workspace.root,
            web=web,
            auto_approve=yes,
            session_id=session.session_id,
        )

    except RuntimeError as exc:
        console.print(
            f"[bold red]Tool setup error:[/] {exc}"
        )
        raise typer.Exit(code=2) from exc

    #
    # 4. Create runner using the same session ID.
    #
    client = Client(
        host=settings.ollama_host
    )

    runner = AgentRunner(
        client=client,
        model=model,
        registry=registry,
        repo=workspace.root,
        max_iterations=(
            settings
            .code_agent_max_tool_iterations
        ),
        console=console,
        session_id=session.session_id,
    )

    console.print(
        f"[bold]Local Code Agent[/]  "
        f"model={model}"
    )
    console.print(
        f"repo={workspace.root}"
    )
    console.print(
        f"session={session.session_id}"
    )
    console.print(
        f"ollama={settings.ollama_host}  "
        f"tavily={'on' if web else 'off'}"
    )

    if yes:
        console.print(
            "[yellow]--yes:[/] ordinary "
            "writes/builds/tests auto-approved; "
            "commit/branch/push/PR still require "
            "an explicit user request."
        )

    console.print(
        "Type [bold]/exit[/] to quit."
    )
    console.print(
        "Type [bold]/reset[/] to reset "
        "the conversation context."
    )
    console.print(
        "Type [bold]/session[/] to show "
        "the current persistent session ID."
    )
    console.print()

    while True:
        try:
            prompt = console.input(
                "[bold cyan]you>[/] "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not prompt:
            continue

        if prompt in {
            "/exit",
            "/quit",
        }:
            break

        if prompt == "/reset":
            runner.reset_context()
            console.print(
                "[yellow]Context reset.[/]"
            )
            continue

        if prompt == "/session":
            console.print(
                f"[dim]{runner.session_id}[/]"
            )
            continue

        try:
            answer = runner.ask(prompt)

        except Exception as exc:
            console.print(
                "[bold red]Agent error:[/] "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if not isinstance(answer, str):
            console.print(
                "[bold red]Agent error:[/] "
                "runner returned a non-string "
                "response"
            )
            continue

        console.print(
            Markdown(answer)
        )
        console.print()


def _probe_tool_call(
    client: Client,
    model: str,
) -> tuple[bool, str]:
    def tool_probe(value: str) -> str:
        """Return a value unchanged. Args: value: The string to return."""
        return value

    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Call the tool_probe tool "
                        "exactly once with "
                        "value='ok'. "
                        "Do not answer this request "
                        "without using the tool."
                    ),
                }
            ],
            tools=[tool_probe],
        )

    except Exception as exc:
        return (
            False,
            "request failed: "
            f"{type(exc).__name__}: {exc}",
        )

    calls = (
        response.message.tool_calls
        or []
    )

    if not calls:
        return (
            False,
            "model returned no tool call",
        )

    call = calls[0]

    if (
        call.function.name
        != "tool_probe"
    ):
        return (
            False,
            "model called unexpected tool: "
            f"{call.function.name}",
        )

    value = dict(
        call.function.arguments or {}
    ).get("value")

    if value != "ok":
        return (
            False,
            "tool arguments were unexpected: "
            f"{call.function.arguments}",
        )

    return (
        True,
        "tool call schema and arguments "
        "look good",
    )


@app.command()
def doctor(
    model: str = typer.Option(
        settings.ollama_model,
        "--model",
        "-m",
    ),
    probe_tools: bool = typer.Option(
        False,
        "--probe-tools",
        help=(
            "Ask the configured model to "
            "perform one harmless tool call."
        ),
    ),
) -> None:
    """Check dependencies, Ollama connectivity, and optional tool-calling support."""

    console.print(
        f"Ollama host: "
        f"{settings.ollama_host}"
    )
    console.print(
        f"Model: {model}"
    )

    git_path = shutil.which("git")
    gh_path = shutil.which("gh")

    console.print(
        f"git: "
        f"{git_path or '[red]NOT FOUND[/]'}"
    )
    console.print(
        f"gh:  "
        f"{gh_path or '[yellow]NOT FOUND[/]'}"
    )

    if gh_path:
        auth = subprocess.run(
            [
                gh_path,
                "auth",
                "status",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if auth.returncode == 0:
            console.print(
                "GitHub auth: [green]OK[/]"
            )
        else:
            console.print(
                "GitHub auth: "
                "[yellow]not authenticated[/]"
            )

    tavily_installed = (
        importlib.util.find_spec("tavily")
        is not None
    )

    console.print(
        "Tavily SDK: "
        + (
            "installed"
            if tavily_installed
            else (
                "not installed "
                "(install "
                "local-code-agent[web])"
            )
        )
    )

    if settings.tavily_api_key:
        console.print(
            "Tavily API key: configured"
        )
    else:
        console.print(
            "Tavily API key: "
            "not configured "
            "(keyless search is available "
            "but rate-limited)"
        )

    try:
        client = Client(
            host=settings.ollama_host
        )

        models = client.list()

        names = [
            (
                getattr(
                    item,
                    "model",
                    None,
                )
                or getattr(
                    item,
                    "name",
                    "<unknown>",
                )
            )
            for item in models.models
        ]

        console.print(
            "Ollama connection: "
            f"[green]OK[/] "
            f"({len(names)} models)"
        )

        for name in names[:20]:
            marker = (
                " [green]<- selected[/]"
                if name == model
                else ""
            )

            console.print(
                f"  - {name}{marker}"
            )

        if model not in names:
            console.print(
                "[yellow]Warning:[/] "
                f"selected model '{model}' "
                "was not in Ollama's "
                "model list."
            )

    except Exception as exc:
        console.print(
            "Ollama connection: "
            "[red]FAILED[/] "
            f"{type(exc).__name__}: {exc}"
        )
        raise typer.Exit(
            code=1
        ) from exc

    if probe_tools:
        ok, detail = _probe_tool_call(
            client,
            model,
        )

        if ok:
            console.print(
                "Tool-call probe: "
                f"[green]OK[/] — {detail}"
            )
        else:
            console.print(
                "Tool-call probe: "
                f"[red]FAILED[/] — {detail}"
            )
            raise typer.Exit(code=1)