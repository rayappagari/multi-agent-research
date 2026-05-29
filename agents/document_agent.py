"""Document Agent — retrieves, filters, and scores source documents."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import anthropic

from tools.source_retrieval_tool import SourceRetrievalTool
from tools.license_policy_tool import LicensePolicyTool
from tools.citation_tracker import CitationTracker
from schemas.source import Source
from schemas.tool_errors import ErrorCode, ToolException

logger = logging.getLogger(__name__)


class DocumentAgent:
    """
    For each raw search result:
      1. Retrieve full content + license check (parallel across all URLs)
      2. Batch-score all candidates for relevance in a single Claude call
      3. Filter by threshold, cap at max_sources, register with CitationTracker
    """

    BATCH_RELEVANCE_PROMPT = """Rate how relevant each document is to the research query.

Query: {query}

{items}

Respond ONLY with a JSON array of {n} floats between 0.0 (irrelevant) and 1.0 (highly relevant),
one per document in the same order. No explanation."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        retrieval_tool: SourceRetrievalTool,
        license_tool: LicensePolicyTool,
        citation_tracker: CitationTracker,
        model: str = "claude-sonnet-4-20250514",
        relevance_threshold: float = 0.4,
        max_sources: int = 12,
        max_workers: int = 8,
    ):
        self.client = client
        self.retrieval_tool = retrieval_tool
        self.license_tool = license_tool
        self.citation_tracker = citation_tracker
        self.model = model
        self.relevance_threshold = relevance_threshold
        self.max_sources = max_sources
        self.max_workers = max_workers

    # ------------------------------------------------------------------

    def run(self, query: str, raw_results: list[dict]) -> list[Source]:
        """
        Process raw search results into curated Source objects.

        Args:
            query: Original research query (used for relevance scoring).
            raw_results: List of {title, url, snippet} dicts from SearchAgent.

        Returns:
            List of accepted, citation-registered Source objects.
        """
        # Phase 1: retrieve + license check in parallel
        candidates: list[Source] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._retrieve_and_check, r): r
                for r in raw_results
            }
            for future in as_completed(futures):
                src = future.result()
                if src is not None:
                    candidates.append(src)

        logger.info("DocumentAgent: %d candidates after retrieval + license check", len(candidates))

        if not candidates:
            return []

        # Phase 2: batch relevance scoring (single API call)
        scores = self._score_batch(query, candidates)
        for src, score in zip(candidates, scores):
            src.relevance_score = score

        # Phase 3: sort by relevance, filter, cap, register citations
        accepted: list[Source] = []
        for src in sorted(candidates, key=lambda s: s.relevance_score, reverse=True):
            if len(accepted) >= self.max_sources:
                break
            if src.relevance_score < self.relevance_threshold:
                continue
            try:
                src = self.citation_tracker.register(src)
            except ToolException as exc:
                if exc.error.code == ErrorCode.CITATION_DUPLICATE:
                    logger.debug("DocumentAgent: duplicate citation for %s", src.url)
                    continue
                raise
            accepted.append(src)
            logger.info(
                "DocumentAgent: accepted [%s] relevance=%.2f  %s",
                src.citation_id,
                src.relevance_score,
                src.title[:60],
            )

        logger.info("DocumentAgent: accepted %d / %d sources", len(accepted), len(raw_results))
        return accepted

    # ------------------------------------------------------------------

    def _retrieve_and_check(self, result: dict) -> Optional[Source]:
        """Retrieve one URL and run license policy. Returns None if rejected."""
        url = result.get("url", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")

        if not url:
            return None

        if self.citation_tracker.is_registered(url):
            logger.debug("DocumentAgent: skip duplicate %s", url)
            return None

        try:
            source = self.retrieval_tool.retrieve(url, title=title, snippet=snippet)
        except ToolException as exc:
            logger.warning("DocumentAgent: retrieval failed for %s — %s", url, exc)
            source = Source(url=url, title=title or url, content=snippet)
        except Exception as exc:
            logger.warning("DocumentAgent: unexpected retrieval error for %s — %s", url, exc)
            if not snippet:
                return None
            source = Source(url=url, title=title or url, content=snippet)

        try:
            source = self.license_tool.check(source)
        except ToolException as exc:
            if exc.error.code == ErrorCode.LICENSE_RESTRICTED:
                logger.info("DocumentAgent: skipping restricted source %s", url)
                return None

        if not self.license_tool.is_citeable(source):
            return None

        return source

    def _score_batch(self, query: str, sources: list[Source]) -> list[float]:
        """Score all sources for relevance in a single Claude call."""
        items = "\n\n".join(
            f"[{i + 1}] Title: {src.title}\nExcerpt: {src.content[:500]}"
            for i, src in enumerate(sources)
        )
        prompt = self.BATCH_RELEVANCE_PROMPT.format(
            query=query, items=items, n=len(sources)
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max(64, len(sources) * 8),
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            scores: list = json.loads(raw)
            if len(scores) != len(sources):
                logger.warning(
                    "DocumentAgent: batch scoring returned %d scores for %d sources — padding",
                    len(scores), len(sources),
                )
                scores = (scores + [0.5] * len(sources))[: len(sources)]
            return [max(0.0, min(1.0, float(s))) for s in scores]
        except Exception as exc:
            logger.warning("DocumentAgent: batch scoring failed (%s) — defaulting to 0.5", exc)
            return [0.5] * len(sources)
