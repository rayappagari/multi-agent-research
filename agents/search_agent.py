"""Search Agent — generates sub-queries and collects raw search results."""

import logging
import json
import anthropic

from tools.web_search_tool import WebSearchTool
from tools.citation_tracker import CitationTracker
from schemas.tool_errors import ToolException

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Decomposes a research query into focused sub-queries,
    executes each via WebSearchTool, and returns deduplicated
    raw result dicts (title, url, snippet).
    """

    DECOMPOSE_PROMPT = """You are a research assistant decomposing a broad query into focused sub-queries.

Given the research question below, produce {n} distinct sub-queries that together cover the topic well.
Each sub-query must be short (≤10 words) and target a different angle (e.g. background, recent developments,
criticism, data, examples).

Research question: {query}

Respond ONLY with a JSON array of strings. No markdown, no explanation."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        search_tool: WebSearchTool,
        citation_tracker: CitationTracker,
        model: str = "claude-sonnet-4-20250514",
        sub_queries: int = 4,
        results_per_query: int = 6,
    ):
        self.client = client
        self.search_tool = search_tool
        self.citation_tracker = citation_tracker
        self.model = model
        self.sub_queries = sub_queries
        self.results_per_query = results_per_query

    # ------------------------------------------------------------------

    def run(self, query: str) -> list[dict]:
        """
        Main entry point.

        Returns a deduplicated list of result dicts:
          [{title, url, snippet}, ...]
        """
        logger.info("SearchAgent: decomposing query '%s'", query)
        sub_qs = self._decompose(query)
        logger.info("SearchAgent: sub-queries → %s", sub_qs)

        seen_urls: set[str] = set()
        all_results: list[dict] = []

        for sq in sub_qs:
            try:
                results = self.search_tool.search(sq, max_results=self.results_per_query)
            except ToolException as exc:
                logger.warning("SearchAgent: sub-query '%s' failed — %s", sq, exc)
                continue

            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)

        logger.info("SearchAgent: collected %d unique results", len(all_results))
        return all_results

    # ------------------------------------------------------------------

    def _decompose(self, query: str) -> list[str]:
        """Ask Claude to produce focused sub-queries."""
        prompt = self.DECOMPOSE_PROMPT.format(n=self.sub_queries, query=query)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            sub_qs: list[str] = json.loads(raw)
            # Always include the original query
            if query not in sub_qs:
                sub_qs.insert(0, query)
            return sub_qs[: self.sub_queries]
        except Exception as exc:
            logger.error("SearchAgent: decompose failed (%s) — falling back to original query", exc)
            return [query]
