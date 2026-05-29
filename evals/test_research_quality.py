"""Evaluation suite for multi-agent research pipeline quality."""

import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.source import Source, SourceType, LicenseType
from schemas.report import Report, ReportSection, ReportStatus
from schemas.tool_errors import ErrorCode, ToolError, ToolException
from tools.citation_tracker import CitationTracker
from tools.license_policy_tool import LicensePolicyTool
from tools.source_retrieval_tool import SourceRetrievalTool, _classify_url


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_source(url="https://example.com/article", title="Test Article", content="Some content about AI."):
    return Source(url=url, title=title, content=content)


# ─────────────────────────────────────────────────────────────────────────────
# Schema tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceSchema(unittest.TestCase):

    def test_word_count_computed(self):
        src = Source(url="https://a.com", title="T", content="one two three four five")
        self.assertEqual(src.word_count, 5)

    def test_snippet_truncated(self):
        long_content = "word " * 200
        src = Source(url="https://a.com", title="T", content=long_content)
        self.assertTrue(src.snippet.endswith("..."))
        self.assertLessEqual(len(src.snippet), 310)

    def test_roundtrip_dict(self):
        src = _make_source()
        restored = Source.from_dict(src.to_dict())
        self.assertEqual(src.url, restored.url)
        self.assertEqual(src.title, restored.title)

    def test_source_type_enum(self):
        src = Source(url="https://a.com", title="T", content="c", source_type=SourceType.ACADEMIC)
        d = src.to_dict()
        self.assertEqual(d["source_type"], "academic")


class TestReportSchema(unittest.TestCase):

    def test_finalize_sets_counts(self):
        report = Report(
            query="test",
            title="Title",
            summary="Summary here.",
            sections=[ReportSection(heading="H", body="Body text here.")],
            sources=[_make_source()],
        )
        report.finalize()
        self.assertEqual(report.status, ReportStatus.COMPLETE)
        self.assertGreater(report.word_count, 0)
        self.assertEqual(report.source_count, 1)
        self.assertIsNotNone(report.completed_at)

    def test_to_markdown(self):
        report = Report(query="q", title="My Report", summary="Overview.")
        src = _make_source()
        src.citation_id = "SRC-12345678"
        report.sources = [src]
        report.sections = [ReportSection(heading="Intro", body="Text.", citation_ids=["SRC-12345678"])]
        md = report.to_markdown()
        self.assertIn("# My Report", md)
        self.assertIn("## Intro", md)
        self.assertIn("## References", md)
        self.assertIn("SRC-12345678", md)


# ─────────────────────────────────────────────────────────────────────────────
# Tool tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCitationTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = CitationTracker()

    def test_register_assigns_id(self):
        src = _make_source()
        registered = self.tracker.register(src)
        self.assertIsNotNone(registered.citation_id)
        self.assertTrue(registered.citation_id.startswith("SRC-"))

    def test_duplicate_raises(self):
        src1 = _make_source()
        src2 = _make_source()  # same URL
        self.tracker.register(src1)
        with self.assertRaises(ToolException) as ctx:
            self.tracker.register(src2)
        self.assertEqual(ctx.exception.error.code, ErrorCode.CITATION_DUPLICATE)

    def test_is_registered(self):
        src = _make_source()
        self.assertFalse(self.tracker.is_registered(src.url))
        self.tracker.register(src)
        self.assertTrue(self.tracker.is_registered(src.url))

    def test_url_canonicalization(self):
        src1 = _make_source(url="https://Example.COM/article/")
        src2 = _make_source(url="https://example.com/article")
        self.tracker.register(src1)
        with self.assertRaises(ToolException):
            self.tracker.register(src2)  # same canonical URL

    def test_all_sources(self):
        for i in range(3):
            self.tracker.register(_make_source(url=f"https://example.com/article-{i}"))
        self.assertEqual(self.tracker.count(), 3)

    def test_bibliography(self):
        src = _make_source()
        self.tracker.register(src)
        bib = self.tracker.to_bibliography()
        self.assertIn("Test Article", bib)
        self.assertIn("https://example.com/article", bib)


class TestLicensePolicyTool(unittest.TestCase):

    def setUp(self):
        self.tool = LicensePolicyTool()

    def test_open_domain(self):
        src = _make_source(url="https://arxiv.org/abs/2301.00001")
        result = self.tool.check(src)
        self.assertEqual(result.license_type, LicenseType.OPEN)
        self.assertTrue(self.tool.is_citeable(result))

    def test_restricted_domain_raises(self):
        src = _make_source(url="https://www.wsj.com/articles/some-article")
        with self.assertRaises(ToolException) as ctx:
            self.tool.check(src)
        self.assertEqual(ctx.exception.error.code, ErrorCode.LICENSE_RESTRICTED)

    def test_cc_content_detected(self):
        src = _make_source(
            url="https://someblog.com/post",
            content="This work is licensed under CC-BY 4.0.",
        )
        result = self.tool.check(src)
        self.assertEqual(result.license_type, LicenseType.CC)

    def test_default_fair_use(self):
        src = _make_source(url="https://somesite.com/page", content="Just some random content.")
        result = self.tool.check(src)
        self.assertEqual(result.license_type, LicenseType.FAIR_USE)
        self.assertTrue(self.tool.is_citeable(result))


class TestSourceRetrievalTool(unittest.TestCase):

    def test_url_classification(self):
        self.assertEqual(_classify_url("https://arxiv.org/abs/123"), SourceType.ACADEMIC)
        self.assertEqual(_classify_url("https://medium.com/post"), SourceType.BLOG)
        self.assertEqual(_classify_url("https://bbc.com/news/article"), SourceType.NEWS)
        self.assertEqual(_classify_url("https://randomsite.com/page"), SourceType.WEB)

    def test_invalid_url_raises(self):
        tool = SourceRetrievalTool()
        with self.assertRaises(ToolException) as ctx:
            tool.retrieve("not-a-url")
        self.assertEqual(ctx.exception.error.code, ErrorCode.RETRIEVAL_NOT_FOUND)

    def test_fallback_to_snippet(self):
        """When network fails, should fall back to snippet without raising if snippet present."""
        tool = SourceRetrievalTool(timeout=1)
        # Using a guaranteed-unreachable URL with a good snippet
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            src = tool.retrieve(
                "https://unreachable.example.com/page",
                title="Fallback Title",
                snippet="This is fallback content from the search snippet.",
            )
        self.assertEqual(src.content, "This is fallback content from the search snippet.")
        self.assertEqual(src.title, "Fallback Title")


# ─────────────────────────────────────────────────────────────────────────────
# Integration-style quality checks (no live API calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineQualityMetrics(unittest.TestCase):
    """
    Validates quality constraints on a mock pipeline output.
    These run without an API key.
    """

    def _build_mock_report(self, n_sources: int = 5, n_sections: int = 3) -> Report:
        tracker = CitationTracker()
        sources = []
        for i in range(n_sources):
            src = Source(
                url=f"https://example.com/source-{i}",
                title=f"Source {i}: AI Research",
                content=f"Detailed content about artificial intelligence topic {i}. " * 20,
            )
            src.relevance_score = 0.7 + (i * 0.05)
            tracker.register(src)
            sources.append(src)

        sections = [
            ReportSection(
                heading=f"Section {j}",
                body=f"Analysis of topic {j}. " * 50,
                citation_ids=[sources[j % n_sources].citation_id],
            )
            for j in range(n_sections)
        ]

        report = Report(
            query="What is the current state of AI research?",
            title="State of AI Research 2025",
            summary="AI research has advanced dramatically across multiple domains.",
            sections=sections,
            sources=sources,
        )
        report.finalize()
        return report

    def test_minimum_sources(self):
        report = self._build_mock_report(n_sources=5)
        self.assertGreaterEqual(report.source_count, 3, "Report must have at least 3 sources")

    def test_minimum_sections(self):
        report = self._build_mock_report(n_sections=3)
        self.assertGreaterEqual(len(report.sections), 2, "Report must have at least 2 sections")

    def test_minimum_word_count(self):
        report = self._build_mock_report()
        self.assertGreater(report.word_count, 50, "Report body must have >50 words")

    def test_all_sections_have_citations(self):
        report = self._build_mock_report()
        for section in report.sections:
            self.assertTrue(
                len(section.citation_ids) > 0,
                f"Section '{section.heading}' has no citations",
            )

    def test_citations_reference_known_sources(self):
        report = self._build_mock_report()
        known_ids = {s.citation_id for s in report.sources}
        for section in report.sections:
            for cid in section.citation_ids:
                self.assertIn(cid, known_ids, f"Unknown citation id: {cid}")

    def test_report_status_complete(self):
        report = self._build_mock_report()
        self.assertEqual(report.status, ReportStatus.COMPLETE)

    def test_markdown_output_valid(self):
        report = self._build_mock_report()
        md = report.to_markdown()
        self.assertIn("# ", md)          # has h1 title
        self.assertIn("## References", md)
        self.assertIn("SRC-", md)        # has citation IDs

    def test_no_duplicate_source_urls(self):
        report = self._build_mock_report()
        urls = [s.url for s in report.sources]
        self.assertEqual(len(urls), len(set(urls)), "Duplicate source URLs found")

    def test_serialization_roundtrip(self):
        report = self._build_mock_report()
        d = report.to_dict()
        self.assertEqual(d["query"], report.query)
        self.assertEqual(len(d["sources"]), len(report.sources))
        self.assertEqual(len(d["sections"]), len(report.sections))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Running multi_agent_research evaluation suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
