"""Web search tool — wraps the Anthropic built-in web_search capability."""

import json
import logging
import anthropic

from schemas.tool_errors import ErrorCode, ToolError, ToolException

logger = logging.getLogger(__name__)

# Anthropic tool definition passed to the API
WEB_SEARCH_TOOL_DEFINITION = {
    "type": "web_search_20250305",
    "name": "web_search",
}


class WebSearchTool:
    """
    Uses Claude's native web_search tool to retrieve a list of search results
    for a given query.

    Returns a list of dicts: [{title, url, snippet}, ...]
    """

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def search(self, query: str, max_results: int = 8) -> list[dict]:
        """
        Execute a web search and return structured results.

        Args:
            query: Natural-language search query.
            max_results: Maximum number of results to return.

        Returns:
            List of result dicts with keys: title, url, snippet.
        """
        if not query or not query.strip():
            raise ToolException(
                ToolError(
                    code=ErrorCode.SEARCH_INVALID_QUERY,
                    message="Search query must not be empty.",
                    tool_name="web_search",
                    recoverable=False,
                )
            )

        prompt = (
            f"Search the web for: {query}\n\n"
            f"Return the top {max_results} results as a JSON array. "
            "Each element must have exactly these keys: title, url, snippet. "
            "Respond ONLY with the JSON array — no markdown, no preamble."
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                tools=[WEB_SEARCH_TOOL_DEFINITION],
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError as exc:
            raise ToolException(
                ToolError(
                    code=ErrorCode.SEARCH_RATE_LIMITED,
                    message=str(exc),
                    tool_name="web_search",
                    recoverable=True,
                )
            ) from exc
        except Exception as exc:
            raise ToolException(
                ToolError(
                    code=ErrorCode.UNEXPECTED_ERROR,
                    message=str(exc),
                    tool_name="web_search",
                    recoverable=True,
                )
            ) from exc

        # Extract text content from the response
        raw_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw_text += block.text

        if not raw_text.strip():
            logger.warning("WebSearchTool: empty response for query '%s'", query)
            return []

        # Strip optional markdown fences
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.strip()

        try:
            results: list[dict] = json.loads(clean)
            return results[:max_results]
        except json.JSONDecodeError:
            logger.error("WebSearchTool: failed to parse JSON response: %s", raw_text[:200])
            return []
