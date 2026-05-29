"""Data models for research sources."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class SourceType(str, Enum):
    WEB = "web"
    PDF = "pdf"
    ACADEMIC = "academic"
    NEWS = "news"
    BLOG = "blog"
    UNKNOWN = "unknown"


class LicenseType(str, Enum):
    OPEN = "open"
    CC = "creative_commons"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"
    FAIR_USE = "fair_use"


@dataclass
class Source:
    url: str
    title: str
    content: str
    source_type: SourceType = SourceType.UNKNOWN
    license_type: LicenseType = LicenseType.UNKNOWN
    author: Optional[str] = None
    published_date: Optional[str] = None
    retrieved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    citation_id: Optional[str] = None
    relevance_score: float = 0.0
    word_count: int = 0
    snippet: str = ""

    def __post_init__(self):
        self.word_count = len(self.content.split())
        self.snippet = self.content[:300] + "..." if len(self.content) > 300 else self.content

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "source_type": self.source_type.value,
            "license_type": self.license_type.value,
            "author": self.author,
            "published_date": self.published_date,
            "retrieved_at": self.retrieved_at,
            "citation_id": self.citation_id,
            "relevance_score": self.relevance_score,
            "word_count": self.word_count,
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        data = data.copy()
        data["source_type"] = SourceType(data.get("source_type", "unknown"))
        data["license_type"] = LicenseType(data.get("license_type", "unknown"))
        return cls(**data)
