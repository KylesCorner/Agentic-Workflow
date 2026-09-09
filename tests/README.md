# Agentic Workflow pytest suite

This package contains a replacement pytest suite for Agentic-Workflow.

## Install test dependencies

Because the current RAG package eagerly imports Tree-sitter components, install
both the dev and rag extras:

```bash
uv sync --extra dev --extra rag
```

## Run

```bash
pytest
```

Run only fast/unit tests:

```bash
pytest tests/unit
```

Run integration tests:

```bash
pytest -m integration
```

Skip integration tests:

```bash
pytest -m "not integration"
```

The default suite does not require a live Ollama server, Tavily, GitHub network
access, or Docker.
