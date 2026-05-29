"""License policy tool — heuristically determines if a source may be cited."""

import re
import logging

from schemas.source import Source, LicenseType
from schemas.tool_errors import ErrorCode, ToolError, ToolException

logger = logging.getLogger(__name__)

# Domains known to allow open citation / fair-use research
_OPEN_DOMAINS: list[str] = [
    "wikipedia.org",
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "plos.org",
    "doaj.org",
    "openalex.org",
    "semanticscholar.org",
    "ncbi.nlm.nih.gov",
]

# Domains that are paywalled / proprietary
_RESTRICTED_DOMAINS: list[str] = [
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "thetimes.co.uk",
    "newyorker.com",
]

# Content patterns indicating Creative Commons
_CC_PATTERNS = [
    r"creative\s+commons",
    r"cc[-\s]by",
    r"cc0",
    r"open\s+access",
    r"licensed\s+under\s+cc",
]

# Patterns indicating restricted use
_RESTRICTED_PATTERNS = [
    r"all rights reserved",
    r"©\s*\d{4}",
    r"do not reproduce",
    r"no part.*may be reproduced",
    r"subscription required",
]


def _domain_of(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1).lower() if m else ""


class LicensePolicyTool:
    """
    Determines the likely license type of a source and whether it is
    acceptable to use in research output.
    """

    def check(self, source: Source) -> Source:
        """
        Annotate *source* with a LicenseType and return it.
        Raises ToolException with LICENSE_RESTRICTED if the source
        must not be used.
        """
        domain = _domain_of(source.url)
        content_lower = source.content.lower()

        # 1. Domain allow-list (open)
        if any(d in domain for d in _OPEN_DOMAINS):
            source.license_type = LicenseType.OPEN
            return source

        # 2. Domain block-list (proprietary / paywalled)
        if any(d in domain for d in _RESTRICTED_DOMAINS):
            source.license_type = LicenseType.PROPRIETARY
            raise ToolException(
                ToolError(
                    code=ErrorCode.LICENSE_RESTRICTED,
                    message=f"Source '{source.url}' is from a restricted domain ({domain}).",
                    tool_name="license_policy",
                    recoverable=True,
                    context={"url": source.url, "domain": domain},
                )
            )

        # 3. Content-based CC detection
        if any(re.search(p, content_lower) for p in _CC_PATTERNS):
            source.license_type = LicenseType.CC
            return source

        # 4. Content-based restriction detection
        if any(re.search(p, content_lower) for p in _RESTRICTED_PATTERNS):
            logger.warning(
                "LicensePolicyTool: source %s appears proprietary — marking fair-use only",
                source.url,
            )
            source.license_type = LicenseType.FAIR_USE
            return source

        # 5. Default: treat as fair-use (publicly accessible web content)
        source.license_type = LicenseType.FAIR_USE
        return source

    def is_citeable(self, source: Source) -> bool:
        """Return True if the source may be included in research output."""
        return source.license_type in (
            LicenseType.OPEN,
            LicenseType.CC,
            LicenseType.FAIR_USE,
            LicenseType.UNKNOWN,
        )
