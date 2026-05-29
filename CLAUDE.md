# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires Python 3.13+ and the `anthropic` package. Set `ANTHROPIC_API_KEY` in `.env` before running anything — the entry point checks for it on startup.

No `requirements.txt` exists; install the Anthropic SDK manually:
```
pip install anthropic
```

## Commands

```bash
# Run a research query
python main.py "your research question here"

# Save output to Markdown or JSON
python main.py "your question" --out report.md
python main.py "your question" --json report.json

# View history of past runs
python main.py --history

# Run the test/eval suite
python main.py --eval
# or directly:
python evals/test_research_quality.py

# Tune pipeline parameters (all optional, these are the defaults)
python main.py "query" --sub-queries 4 --max-sources 12 --relevance-threshold 0.4 --model claude-sonnet-4-6

# Enable debug logging
python main.py "query" --verbose
```

## Architecture

The system is a **4-stage sequential pipeline** driven by `Coordinator` (`agents/coordinator.py`). All agents share a single `anthropic.Anthropic` client and a single `CitationTracker` instance that is constructed once in `Coordinator._build_pipeline()`.

### Pipeline stages

1. **SearchAgent** (`agents/search_agent.py`) — Calls Claude to decompose the query into `N` sub-queries, then executes each via `WebSearchTool`. Returns a deduplicated list of `{title, url, snippet}` dicts.

2. **DocumentAgent** (`agents/document_agent.py`) — For each raw result: fetches full HTML content (`SourceRetrievalTool`), checks license policy (`LicensePolicyTool`), scores relevance 0–1 via Claude, and registers accepted sources with `CitationTracker`. Sources below `relevance_threshold` or from restricted domains (wsj.com, ft.com, bloomberg.com, etc.) are dropped silently.

3. **SynthesisAgent** (`agents/synthesis_agent.py`) — Formats accepted sources into a compact block (capped at `max_source_chars` per source) and sends a single Claude call asking for a structured JSON response: `{title, summary, sections[]}`. Citation IDs are validated post-parse to strip any hallucinated IDs.

4. **ReportAgent** (`agents/report_agent.py`) — Assembles the final `Report` dataclass, calls `report.finalize()` to compute word/source counts, and appends/upserts the result to `memory/run_state.json` (same-query runs overwrite).

### Tools

- **`WebSearchTool`** (`tools/web_search_tool.py`) — Wraps Anthropic's native `web_search_20250305` tool type. Prompts Claude with a structured JSON-response request and parses the returned array.
- **`SourceRetrievalTool`** (`tools/source_retrieval_tool.py`) — Plain `urllib` HTTP fetch capped at 500 KB, lightweight regex HTML stripper, and URL-based `SourceType` classification. Falls back to snippet text on network failure rather than raising when a snippet is available.
- **`LicensePolicyTool`** (`tools/license_policy_tool.py`) — Heuristic only: domain allow-list → block-list → CC content patterns → `FAIR_USE` default. Raises `ToolException(LICENSE_RESTRICTED)` for blocked domains; all other citable types pass through.
- **`CitationTracker`** (`tools/citation_tracker.py`) — Assigns deterministic `SRC-XXXXXXXX` IDs (MD5 hash of canonical URL, first 8 hex chars). Canonicalizes URLs by lowercasing scheme+host and stripping trailing slashes. Raises `CITATION_DUPLICATE` on re-registration.

### Schemas

- **`Source`** (`schemas/source.py`) — Core data unit flowing through the pipeline. `__post_init__` computes `word_count` and `snippet` (300-char truncation) automatically.
- **`Report` / `ReportSection`** (`schemas/report.py`) — Final output. `report.finalize()` sets `status=COMPLETE`, timestamps, and computes aggregate counts. `to_markdown()` renders a full document with a `## References` section.
- **`ToolError` / `ToolException`** (`schemas/tool_errors.py`) — All tool failures raise `ToolException` wrapping a typed `ToolError` with an `ErrorCode` enum. Agents catch specific `ErrorCode` values (e.g. `LICENSE_RESTRICTED`, `CITATION_DUPLICATE`) to decide whether to skip or abort.

### State persistence

Run history is stored in `memory/run_state.json` as a JSON array of serialized `Report` dicts. `ReportAgent` upserts by query string. `main.py --history` reads this file directly.

### PipelineConfig

All tunables live in `PipelineConfig` (`agents/coordinator.py`). Key fields:
- `sub_queries` — number of search sub-queries generated per research question
- `max_sources` — hard cap on accepted sources fed to synthesis
- `relevance_threshold` — float 0–1; sources below this are dropped
- `max_source_chars_in_prompt` — per-source character cap in the synthesis prompt (controls token cost)
