"""Synthesis Agent — distils accepted sources into structured insights."""

import json
import logging
import anthropic

from schemas.source import Source

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """
    Takes a list of accepted Source objects and produces a structured
    synthesis: a title, an executive summary, and a list of thematic
    sections with inline citation references.
    """

    SYNTHESIS_PROMPT = """You are a senior research analyst. Your task is to synthesise the provided sources
into a coherent research report structure.

Research query: {query}

--- SOURCES ---
{sources_block}
--- END SOURCES ---

Produce ONLY a JSON object with this exact schema:
{{
  "title": "<concise report title>",
  "summary": "<2-3 sentence executive summary>",
  "sections": [
    {{
      "heading": "<section heading>",
      "body": "<3-5 paragraph analysis>",
      "citation_ids": ["SRC-XXXXXXXX", ...]
    }}
  ]
}}

Requirements:
- 3-5 sections covering different thematic angles
- Every claim must reference at least one citation_id from the sources list
- citation_ids must be exact IDs from the provided sources
- Respond ONLY with the JSON object — no markdown fences, no preamble"""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-20250514",
        max_source_chars: int = 800,
    ):
        self.client = client
        self.model = model
        self.max_source_chars = max_source_chars  # per source in the prompt

    # ------------------------------------------------------------------

    def run(self, query: str, sources: list[Source]) -> dict:
        """
        Synthesise sources into a report skeleton.

        Returns:
            dict with keys: title, summary, sections
            (sections is a list of {heading, body, citation_ids})
        """
        if not sources:
            return {
                "title": f"Research Report: {query}",
                "summary": "No sources were available for synthesis.",
                "sections": [],
            }

        sources_block = self._format_sources_block(sources)
        prompt = self.SYNTHESIS_PROMPT.format(query=query, sources_block=sources_block)

        logger.info("SynthesisAgent: synthesising %d sources for '%s'", len(sources), query)

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("SynthesisAgent: failed to parse JSON — returning stub")
            result = {
                "title": f"Research Report: {query}",
                "summary": raw[:500],
                "sections": [],
            }

        # Validate citation IDs — remove any that aren't in our source list
        valid_ids = {s.citation_id for s in sources if s.citation_id}
        for section in result.get("sections", []):
            section["citation_ids"] = [
                cid for cid in section.get("citation_ids", []) if cid in valid_ids
            ]

        logger.info(
            "SynthesisAgent: produced %d sections", len(result.get("sections", []))
        )
        return result

    # ------------------------------------------------------------------

    def _format_sources_block(self, sources: list[Source]) -> str:
        """Render sources as a compact block for the prompt."""
        parts = []
        for src in sources:
            content_excerpt = src.content[: self.max_source_chars]
            parts.append(
                f"[{src.citation_id}]\n"
                f"Title: {src.title}\n"
                f"URL: {src.url}\n"
                f"Type: {src.source_type.value}\n"
                f"Excerpt: {content_excerpt}"
            )
        return "\n\n---\n\n".join(parts)
