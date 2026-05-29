"""Citation tracker — assigns stable IDs and prevents duplicates."""

import hashlib
import logging
from typing import Optional

from schemas.source import Source
from schemas.tool_errors import ErrorCode, ToolError, ToolException

logger = logging.getLogger(__name__)


def _url_hash(url: str) -> str:
    """Short deterministic ID derived from the URL."""
    return hashlib.md5(url.encode()).hexdigest()[:8].upper()


class CitationTracker:
    """
    Single-process citation registry.

    Assigns each unique URL a short citation ID like [SRC-A1B2C3D4]
    and raises on true duplicates to avoid redundant retrieval.
    """

    def __init__(self):
        self._registry: dict[str, Source] = {}  # citation_id → Source
        self._url_to_id: dict[str, str] = {}     # canonical_url → citation_id
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, source: Source) -> Source:
        """
        Register *source*, assign a citation_id, and return it.
        Raises ToolException(CITATION_DUPLICATE) if the URL was already registered.
        """
        canonical = self._canonicalize(source.url)

        if canonical in self._url_to_id:
            existing_id = self._url_to_id[canonical]
            raise ToolException(
                ToolError(
                    code=ErrorCode.CITATION_DUPLICATE,
                    message=f"URL already registered as [{existing_id}]: {source.url}",
                    tool_name="citation_tracker",
                    recoverable=True,
                    context={"citation_id": existing_id, "url": source.url},
                )
            )

        self._counter += 1
        cid = f"SRC-{_url_hash(canonical)}"
        source.citation_id = cid
        self._url_to_id[canonical] = cid
        self._registry[cid] = source
        logger.debug("CitationTracker: registered [%s] %s", cid, source.title[:60])
        return source

    def get(self, citation_id: str) -> Optional[Source]:
        return self._registry.get(citation_id)

    def get_by_url(self, url: str) -> Optional[Source]:
        cid = self._url_to_id.get(self._canonicalize(url))
        return self._registry.get(cid) if cid else None

    def all_sources(self) -> list[Source]:
        return list(self._registry.values())

    def all_ids(self) -> list[str]:
        return list(self._registry.keys())

    def is_registered(self, url: str) -> bool:
        return self._canonicalize(url) in self._url_to_id

    def count(self) -> int:
        return len(self._registry)

    def to_bibliography(self) -> str:
        """Return a plain-text numbered bibliography."""
        lines = []
        for i, src in enumerate(self._registry.values(), 1):
            author = f" {src.author}." if src.author else ""
            date = f" ({src.published_date})" if src.published_date else ""
            lines.append(f"[{i}] [{src.citation_id}]{author} {src.title}.{date} {src.url}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _canonicalize(url: str) -> str:
        """Strip trailing slashes and lowercase the scheme+host."""
        url = url.strip().rstrip("/")
        # lowercase scheme and host only
        if "://" in url:
            scheme, rest = url.split("://", 1)
            host, *path_parts = rest.split("/", 1)
            url = f"{scheme.lower()}://{host.lower()}" + (f"/{path_parts[0]}" if path_parts else "")
        return url
