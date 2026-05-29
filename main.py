"""
multi_agent_research — main entry point.

Usage:
    python main.py "your research question here"
    python main.py "your research question here" --out report.md
    python main.py --history
    python main.py --eval
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from agents.coordinator import Coordinator, PipelineConfig
from schemas.report import ReportStatus


# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "httpcore", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-agent research pipeline powered by Claude.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", nargs="?", help="Research question to investigate.")
    p.add_argument("--out", metavar="FILE", help="Save Markdown report to FILE.")
    p.add_argument("--json", metavar="FILE", dest="json_out", help="Save JSON report to FILE.")
    p.add_argument("--history", action="store_true", help="Print past run summaries.")
    p.add_argument("--eval", action="store_true", help="Run the evaluation test suite.")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")

    # Pipeline tuning
    p.add_argument("--sub-queries", type=int, default=4, metavar="N")
    p.add_argument("--max-sources", type=int, default=12, metavar="N")
    p.add_argument("--relevance-threshold", type=float, default=0.4, metavar="F")
    p.add_argument("--model", default="claude-sonnet-4-6", metavar="MODEL")
    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.\n"
            "       export ANTHROPIC_API_KEY=sk-ant-...",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_history() -> None:
    state_file = Path(__file__).parent / "memory" / "run_state.json"
    if not state_file.exists() or state_file.read_text().strip() in ("", "[]"):
        print("No past runs found.")
        return
    runs: list[dict] = json.loads(state_file.read_text())
    print(f"\n{'─'*60}")
    print(f"  Past research runs ({len(runs)} total)")
    print(f"{'─'*60}")
    for i, run in enumerate(runs, 1):
        status = run.get("status", "?")
        title = run.get("title", run.get("query", "?"))
        words = run.get("word_count", 0)
        sources = run.get("source_count", 0)
        completed = run.get("completed_at", "?")[:19]
        print(f"  [{i:2}] {status:10}  {words:5} words  {sources:2} sources  {completed}  {title}")
    print()


def _run_evals() -> None:
    eval_path = Path(__file__).parent / "evals" / "test_research_quality.py"
    result = subprocess.run(
        [sys.executable, str(eval_path)],
        cwd=str(Path(__file__).parent),
    )
    sys.exit(result.returncode)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _configure_logging(args.verbose)
    log = logging.getLogger("main")

    # ── Special modes ─────────────────────────────────────────────────
    if args.eval:
        _run_evals()
        return  # unreachable

    if args.history:
        _print_history()
        return

    if not args.query:
        parser.print_help()
        sys.exit(0)

    _check_api_key()

    # ── Build config ──────────────────────────────────────────────────
    config = PipelineConfig(
        model=args.model,
        sub_queries=args.sub_queries,
        max_sources=args.max_sources,
        relevance_threshold=args.relevance_threshold,
    )

    # ── Run pipeline ──────────────────────────────────────────────────
    print(f"\n🔍  Researching: {args.query}\n")
    coordinator = Coordinator(config=config)
    result = coordinator.research(args.query)

    # ── Handle errors ─────────────────────────────────────────────────
    if result.errors:
        print("\n⚠️  Pipeline errors:")
        for err in result.errors:
            print(f"   • {err}")

    if not result.report or result.report.status != ReportStatus.COMPLETE:
        print("\n❌  Research did not complete successfully.")
        sys.exit(1)

    report = result.report

    # ── Console output ────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  {report.title}")
    print(f"{'═'*60}")
    print(f"\n{report.summary}\n")

    for section in report.sections:
        print(f"{'─'*60}")
        print(f"  {section.heading}")
        print(f"{'─'*60}")
        # Print first 500 chars of body to keep console manageable
        body_preview = section.body[:500]
        if len(section.body) > 500:
            body_preview += "…"
        print(body_preview)
        if section.citation_ids:
            print(f"\n  Sources: {', '.join(section.citation_ids)}")
        print()

    print(f"{'═'*60}")
    print(f"  📚 {report.source_count} sources  •  {report.word_count} words  •  {result.elapsed_seconds}s")
    print(f"{'═'*60}\n")

    # ── Save Markdown ─────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"✅  Markdown report saved to {out_path}")

    # ── Save JSON ─────────────────────────────────────────────────────
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"✅  JSON report saved to {json_path}")

    print()


if __name__ == "__main__":
    main()
