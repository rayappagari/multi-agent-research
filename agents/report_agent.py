"""Report Agent — assembles the final Report object and persists it."""

import json
import logging
from pathlib import Path

from schemas.report import Report, ReportSection, ReportStatus
from schemas.source import Source

logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).parent.parent / "memory" / "run_state.json"


class ReportAgent:
    """
    Assembles a Report from the synthesis skeleton + accepted sources,
    writes it to disk, and returns the finished Report.
    """

    def __init__(self, state_file: Path = _STATE_FILE):
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        synthesis: dict,
        sources: list[Source],
    ) -> Report:
        """
        Build and persist the final Report.

        Args:
            query: Original research query.
            synthesis: Dict from SynthesisAgent (title, summary, sections).
            sources: Accepted Source objects from DocumentAgent.

        Returns:
            Finalised Report object.
        """
        report = Report(
            query=query,
            title=synthesis.get("title", f"Research Report: {query}"),
            summary=synthesis.get("summary", ""),
            sources=sources,
        )
        report.status = ReportStatus.IN_PROGRESS

        for sec_data in synthesis.get("sections", []):
            section = ReportSection(
                heading=sec_data.get("heading", ""),
                body=sec_data.get("body", ""),
                citation_ids=sec_data.get("citation_ids", []),
            )
            report.sections.append(section)

        report.finalize()
        self._persist(report)

        logger.info(
            "ReportAgent: report complete — %d words, %d sources, %d sections",
            report.word_count,
            report.source_count,
            len(report.sections),
        )
        return report

    # ------------------------------------------------------------------

    def _persist(self, report: Report) -> None:
        """Append or update the run_state.json file."""
        existing: list[dict] = []
        if self.state_file.exists():
            try:
                existing = json.loads(self.state_file.read_text())
                if not isinstance(existing, list):
                    existing = [existing]
            except json.JSONDecodeError:
                existing = []

        entry = self._slim_dict(report)
        updated = False
        for i, e in enumerate(existing):
            if e.get("query") == report.query:
                existing[i] = entry
                updated = True
                break
        if not updated:
            existing.append(entry)

        self.state_file.write_text(json.dumps(existing, indent=2))
        logger.debug("ReportAgent: persisted report to %s", self.state_file)

    @staticmethod
    def _slim_dict(report: Report) -> dict:
        """Return a report dict with source content stripped (metadata only)."""
        d = report.to_dict()
        _KEEP = {"url", "title", "citation_id", "relevance_score", "source_type", "author", "published_date"}
        d["sources"] = [{k: v for k, v in src.items() if k in _KEEP} for src in d["sources"]]
        return d

    def load_history(self) -> list[dict]:
        """Return all previously persisted reports."""
        if not self.state_file.exists():
            return []
        try:
            return json.loads(self.state_file.read_text())
        except json.JSONDecodeError:
            return []
