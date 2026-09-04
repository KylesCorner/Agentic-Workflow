# Local Code Agent

A local coding-agent CLI for C, C++, and Python. It talks to an Ollama server over localhost, LAN, or Tailscale and can inspect/edit repositories, use Git/GitHub, run constrained tests/builds, and optionally search the web with Tavily.

The Tree-sitter RAG subsystem remains in `local_code_agent.rag` but is intentionally **not connected yet**.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional Tavily support only when you want web search:

```bash
pip install -e '.[web]'
```

Install the future Tree-sitter RAG dependencies separately:

```bash
pip install -e '.[rag]'
```

For development tooling:

```bash
pip install -e '.[dev]'
```

## Configure

Copy `.env.example` to `.env`, or export the variables in your shell:

```dotenv
OLLAMA_HOST=http://100.x.x.x:11434
OLLAMA_MODEL=qwen3-coder:latest
TAVILY_API_KEY=tvly-...

CODE_AGENT_MAX_TOOL_ITERATIONS=20
CODE_AGENT_MAX_TOOL_OUTPUT_CHARS=40000
```

Tavily currently supports keyless `search()` calls, so `TAVILY_API_KEY` is optional for basic use, although keyless mode is rate-limited.

## Verify the local stack

```bash
lca doctor
```

The most useful first test is the harmless tool-call probe:

```bash
lca doctor --probe-tools
```

That verifies that the selected Ollama model actually emits a correctly formed tool call. A model can be reachable and still be a poor choice for agentic tool use.

## Start the agent

Run from anywhere inside a Git repository:

```bash
lca chat
```

The CLI resolves nested paths back to the repository root automatically.

Optional web search:

```bash
lca chat --web
```

Choose a model:

```bash
lca chat --model qwen3-coder:30b
```

Point at another repository:

```bash
lca chat --repo ~/repos/my-project
```

Auto-approve ordinary write/build/test tools:

```bash
lca chat --yes
```

`--yes` deliberately does **not** grant blanket permission to commit, create/switch branches, push, or open pull requests. Those operations are blocked unless the current user message explicitly requests that action.

Examples:

```text
Fix the parser bug and run the tests.
```

The agent may edit files and run approved tests, but it may not commit or push.

```text
Commit these changes with a concise message.
```

A commit is now eligible to run.

```text
Push this branch and open a pull request against main.
```

Push and PR creation are now eligible to run.

## Current tools

Read-only:

- `read_file`
- `list_files`
- `search_text`
- `git_status`
- `git_diff`
- `git_log`
- `github_repo_view`
- `github_issue_view`
- `github_pr_view`
- `web_search` when `--web` is enabled

Approval required unless `--yes`:

- `write_file`
- `replace_in_file`
- `git_add`
- `run_tests`
- `run_build`

Explicit request required **in addition to** normal approval policy:

- `git_commit`
- `git_create_branch`
- `github_push`
- `github_create_pr`

The agent intentionally has no arbitrary shell tool.

## GitHub

GitHub integration uses the `gh` CLI:

```bash
gh auth login
```

`lca doctor` also reports whether `gh` is installed and authenticated.

## Repository instructions

If the target repository contains `AGENTS.md`, it is included in the system prompt automatically.

Example:

```markdown
# Repository Instructions

- C++17
- Build with: cmake --build build
- Never edit third_party/
- Run ctest before proposing a commit
```

## Suggested MVP test sequence

Before adding RAG, test the agent on real repositories in increasing order of risk:

```text
1. "Explain how this repository is structured. Do not edit anything."
2. "Find where FooManager is implemented and explain the call path."
3. "Fix this small bug. Show me the diff, but do not commit."
4. "Run the tests for the change."
5. "Commit these changes with a concise message."
6. "Read issue 12 and tell me what code would need to change."
7. "Implement issue 12, test it, and show the diff. Do not push."
8. "Push this branch and open a pull request."
9. Start with `--web` and ask for current library/API documentation; verify that the final response preserves Tavily URLs.
```

This sequence isolates model/tool-call failures from repository retrieval problems. RAG should come only after these behaviors are dependable.

## RAG later

The optional `rag/` package already contains the Tree-sitter language registry and semantic chunk model for C, C++, and Python. The next RAG milestone should add embeddings, local vector storage, indexing/update policy, and a read-only `rag_search()` tool without changing the core agent loop.
