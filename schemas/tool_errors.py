"""Structured error types for agent tools."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCode(str, Enum):
    # Search errors
    SEARCH_RATE_LIMITED = "SEARCH_RATE_LIMITED"
    SEARCH_NO_RESULTS = "SEARCH_NO_RESULTS"
    SEARCH_INVALID_QUERY = "SEARCH_INVALID_QUERY"

    # Retrieval errors
    RETRIEVAL_TIMEOUT = "RETRIEVAL_TIMEOUT"
    RETRIEVAL_FORBIDDEN = "RETRIEVAL_FORBIDDEN"
    RETRIEVAL_NOT_FOUND = "RETRIEVAL_NOT_FOUND"
    RETRIEVAL_PARSE_FAILED = "RETRIEVAL_PARSE_FAILED"

    # License errors
    LICENSE_RESTRICTED = "LICENSE_RESTRICTED"
    LICENSE_CHECK_FAILED = "LICENSE_CHECK_FAILED"

    # Citation errors
    CITATION_DUPLICATE = "CITATION_DUPLICATE"
    CITATION_INVALID = "CITATION_INVALID"

    # General
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"


@dataclass
class ToolError:
    code: ErrorCode
    message: str
    tool_name: str
    recoverable: bool = True
    context: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "tool_name": self.tool_name,
            "recoverable": self.recoverable,
            "context": self.context or {},
        }

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.tool_name}: {self.message}"


class ToolException(Exception):
    """Raised when a tool encounters a non-recoverable error."""

    def __init__(self, error: ToolError):
        self.error = error
        super().__init__(str(error))
