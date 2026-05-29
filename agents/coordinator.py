"""Coordinator — orchestrates the full multi-agent research pipeline."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from agents.search_agent import SearchAgent
from agents.document_agent import DocumentAgent
from agents.synthesis_agent import SynthesisAgent
from agents.report_agent import ReportAgent
from tools.web_search_tool import WebSearchTool
from tools.source_retrieval_tool import SourceRetrievalTool
from tools.license_policy_tool import LicensePolicyTool
from tools.citation_tracker import CitationTracker
from schemas.report import Report, ReportStatus

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Tunable parameters for the research pipeline."""

    model: str = "claude-sonnet-4-6"
    sub_queries: int = 4
    results_per_query: int = 6
    max_sources: int = 12
    relevance_threshold: float = 0.4
    retrieval_timeout: int = 10
    max_source_chars_in_prompt: int = 800


@dataclass
class PipelineResult:
    report: Optional[Report] = None
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    stages_completed: list[str] = field(default_factory=list)


class Coordinator:
    """
    Wires together all agents and tools and drives the research pipeline:

      query
        → SearchAgent  (decompose + web search)
        → DocumentAgent (retrieve + license + relevance filter + citation)
        → SynthesisAgent (thematic analysis)
        → ReportAgent (assemble + persist)
        → Report
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._build_pipeline()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(self, query: str) -> PipelineResult:
        """
        Run the full research pipeline for *query*.

        Returns a PipelineResult containing the finished Report (or error info).
        """
        start = time.time()
        result = PipelineResult()

        logger.info("=" * 60)
        logger.info("Coordinator: starting research for: %s", query)
        logger.info("=" * 60)

        # ── Stage 1: Search ────────────────────────────────────────────
        try:
            raw_results = self.search_agent.run(query)
            result.stages_completed.append("search")
            logger.info("Stage 1 DONE: %d raw results", len(raw_results))
        except Exception as exc:
            msg = f"Search stage failed: {exc}"
            logger.exception(msg)
            result.errors.append(msg)
            result.elapsed_seconds = time.time() - start
            return result

        if not raw_results:
            result.errors.append("No search results found for query.")
            result.elapsed_seconds = time.time() - start
            return result

        # ── Stage 2: Document retrieval & filtering ───────────────────
        try:
            sources = self.document_agent.run(query, raw_results)
            result.stages_completed.append("document")
            logger.info("Stage 2 DONE: %d accepted sources", len(sources))
        except Exception as exc:
            msg = f"Document stage failed: {exc}"
            logger.exception(msg)
            result.errors.append(msg)
            result.elapsed_seconds = time.time() - start
            return result

        if not sources:
            result.errors.append("No usable sources found after filtering.")
            result.elapsed_seconds = time.time() - start
            return result

        # ── Stage 3: Synthesis ─────────────────────────────────────────
        try:
            synthesis = self.synthesis_agent.run(query, sources)
            result.stages_completed.append("synthesis")
            logger.info("Stage 3 DONE: %d sections", len(synthesis.get("sections", [])))
        except Exception as exc:
            msg = f"Synthesis stage failed: {exc}"
            logger.exception(msg)
            result.errors.append(msg)
            result.elapsed_seconds = time.time() - start
            return result

        # ── Stage 4: Report assembly ───────────────────────────────────
        try:
            report = self.report_agent.run(query, synthesis, sources)
            result.report = report
            result.stages_completed.append("report")
            logger.info(
                "Stage 4 DONE: report '%s' (%d words)", report.title, report.word_count
            )
        except Exception as exc:
            msg = f"Report stage failed: {exc}"
            logger.exception(msg)
            result.errors.append(msg)

        result.elapsed_seconds = round(time.time() - start, 2)
        logger.info(
            "Coordinator: pipeline finished in %.1fs — stages: %s",
            result.elapsed_seconds,
            result.stages_completed,
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> None:
        cfg = self.config

        self.client = anthropic.Anthropic()
        self.citation_tracker = CitationTracker()

        self.search_agent = SearchAgent(
            client=self.client,
            search_tool=WebSearchTool(self.client, model=cfg.model),
            citation_tracker=self.citation_tracker,
            model=cfg.model,
            sub_queries=cfg.sub_queries,
            results_per_query=cfg.results_per_query,
        )

        self.document_agent = DocumentAgent(
            client=self.client,
            retrieval_tool=SourceRetrievalTool(timeout=cfg.retrieval_timeout),
            license_tool=LicensePolicyTool(),
            citation_tracker=self.citation_tracker,
            model=cfg.model,
            relevance_threshold=cfg.relevance_threshold,
            max_sources=cfg.max_sources,
        )

        self.synthesis_agent = SynthesisAgent(
            client=self.client,
            model=cfg.model,
            max_source_chars=cfg.max_source_chars_in_prompt,
        )

        self.report_agent = ReportAgent()
