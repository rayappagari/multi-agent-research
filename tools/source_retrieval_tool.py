"""Source retrieval tool — fetches and cleans web page content."""

import logging
import re
import urllib.request
import urllib.error
from urllib.parse import urlparse

from schemas.source import Source, SourceType
from schemas.tool_errors import ErrorCode, ToolError, ToolException

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds

# Simple heuristics for source type classification
_TYPE_HINTS: list[tuple[str, SourceType]] = [
    (r"arxiv\.org|scholar\.google|pubmed|doi\.org|researchgate", SourceType.ACADEMIC),
    (r"\.pdf$", SourceType.PDF),
    (r"reuters|bbc|cnn|nytimes|apnews|theguardian", SourceType.NEWS),
    (r"medium\.com|substack|wordpress|blogger", SourceType.BLOG),
]


def _classify_url(url: str) -> SourceType:
    for pattern, stype in _TYPE_HINTS:
        if re.search(pattern, url, re.IGNORECASE):
            return stype
    return SourceType.WEB


def _strip_html(html: str) -> str:
    """Very lightweight HTML→text conversion (no external deps)."""
    # Remove script/style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
        html = html.replace(entity, char)
    # Collapse whitespace
    html = re.sub(r"\s+", " ", html).strip()
    return html


class SourceRetrievalTool:
    """
    Downloads a URL and returns a populated Source object.
    Falls back to snippet text when the URL is unreachable.
    """

    def __init__(self, timeout: int = _TIMEOUT):
        self.timeout = timeout

    def retrieve(self, url: str, title: str = "", snippet: str = "") -> Source:
        """
        Fetch content from *url* and return a Source.

        Args:
            url: Target URL.
            title: Known title (from search results); used as fallback.
            snippet: Known snippet (from search results); used as fallback.

        Returns:
            Source populated with whatever content was retrievable.
        """
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ToolException(
                ToolError(
                    code=ErrorCode.RETRIEVAL_NOT_FOUND,
                    message=f"Invalid URL: {url}",
                    tool_name="source_retrieval",
                    recoverable=False,
                )
            )

        source_type = _classify_url(url)
        content = snippet  # default fallback

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; ResearchBot/1.0; +https://example.com/bot)"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_bytes = resp.read(500_000)  # cap at ~500 KB
                charset = resp.headers.get_content_charset("utf-8") or "utf-8"
                raw_html = raw_bytes.decode(charset, errors="replace")
                content = _strip_html(raw_html)

                # Try to extract <title> if no title was supplied
                if not title:
                    m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
                    if m:
                        title = _strip_html(m.group(1))

        except urllib.error.HTTPError as exc:
            code = (
                ErrorCode.RETRIEVAL_FORBIDDEN
                if exc.code in (401, 403)
                else ErrorCode.RETRIEVAL_NOT_FOUND
            )
            logger.warning("SourceRetrievalTool HTTP %s for %s — using snippet", exc.code, url)
            if not content:
                raise ToolException(
                    ToolError(code=code, message=str(exc), tool_name="source_retrieval", recoverable=True)
                ) from exc

        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning("SourceRetrievalTool timeout/error for %s — using snippet", url)
            if not content:
                raise ToolException(
                    ToolError(
                        code=ErrorCode.RETRIEVAL_TIMEOUT,
                        message=str(exc),
                        tool_name="source_retrieval",
                        recoverable=True,
                    )
                ) from exc

        return Source(
            url=url,
            title=title or url,
            content=content,
            source_type=source_type,
        )
