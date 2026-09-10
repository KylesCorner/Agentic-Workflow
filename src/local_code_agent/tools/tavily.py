from __future__ import annotations

import re
from typing import Any


class TavilyTools:
    """Optional Tavily integration loaded only for --web sessions."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise RuntimeError(
                "Tavily support is not installed. Install it with: "
                "pip install 'local-code-agent[web]'"
            ) from exc

        self.client: Any = (
            TavilyClient(api_key=api_key)
            if api_key
            else TavilyClient()
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Remove model-invented trailing years from recency queries."""

        recency_terms = (
            "latest",
            "current",
            "recent",
            "newest",
            "today",
        )

        if any(
            term in query.lower()
            for term in recency_terms
        ):
            query = re.sub(
                r"\s+(?:19|20)\d{2}\s*$",
                "",
                query,
            )

        return " ".join(query.split())

    def web_search(
        self,
        query: str,
        max_results: int = 5,
    ) -> str:
        """Search the public web for current technical information."""

        query = self._normalize_query(query)


        max_results = max(1, min(int(max_results), 10))

        response = self.client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
        )

        parts: list[str] = []

        for idx, result in enumerate(
            response.get("results", []),
            start=1,
        ):
            parts.append(
                f"[{idx}] {result.get('title', '')}\n"
                f"URL: {result.get('url', '')}\n"
                f"{result.get('content', '')}"
            )

        return (
            "\n\n".join(parts)
            if parts
            else "No web results."
        )