"""Data models for research reports."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

from schemas.source import Source


class ReportStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ReportSection:
    heading: str
    body: str
    citation_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "body": self.body,
            "citation_ids": self.citation_ids,
        }


@dataclass
class Report:
    query: str
    title: str
    summary: str
    sections: list[ReportSection] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    status: ReportStatus = ReportStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    word_count: int = 0
    source_count: int = 0

    def finalize(self):
        self.completed_at = datetime.utcnow().isoformat()
        self.status = ReportStatus.COMPLETE
        all_text = self.summary + " ".join(s.body for s in self.sections)
        self.word_count = len(all_text.split())
        self.source_count = len(self.sources)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
            "sources": [s.to_dict() for s in self.sources],
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "word_count": self.word_count,
            "source_count": self.source_count,
        }

    def to_markdown(self) -> str:
        """Render the report as a Markdown document."""
        lines = [f"# {self.title}", "", f"> {self.summary}", ""]

        for section in self.sections:
            lines.append(f"## {section.heading}")
            lines.append(section.body)
            if section.citation_ids:
                refs = ", ".join(f"[{cid}]" for cid in section.citation_ids)
                lines.append(f"*Sources: {refs}*")
            lines.append("")

        lines.append("## References")
        for src in self.sources:
            cid = src.citation_id or "?"
            author = f" — {src.author}" if src.author else ""
            lines.append(f"- [{cid}] [{src.title}]({src.url}){author}")

        return "\n".join(lines)
