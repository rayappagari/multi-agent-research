"""
FastAPI server for the multi_agent_research UI.

Endpoints:
  GET  /              → serves index.html
  POST /research      → starts a research run, returns run_id
  GET  /stream/{id}   → SSE stream of pipeline events
  GET  /report/{id}   → returns finished report JSON
  GET  /history       → list of past runs
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import AsyncIterator

import anthropic
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# ── Project root on path ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from agents.coordinator import Coordinator, PipelineConfig
from agents.report_agent import ReportAgent
from schemas.report import ReportStatus

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("server")

# ── In-memory run store ───────────────────────────────────────────────
# { run_id: { "status": str, "events": [...], "report": dict | None } }
_runs: dict[str, dict] = {}

app = FastAPI(title="Ask Kian API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).parent / "ui"

# ── Password config ───────────────────────────────────────────────────
# Set APP_PASSWORD in Railway environment variables
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
_sessions: set[str] = set()  # active session tokens

def _make_token() -> str:
    return secrets.token_hex(32)

def _check_session(request: Request) -> bool:
    token = request.cookies.get("session")
    return token in _sessions



# ── Request / Response models ─────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    max_sources: int = 12
    sub_queries: int = 4
    relevance_threshold: float = 0.4


class ResearchResponse(BaseModel):
    run_id: str
    query: str

class DebateResponse(BaseModel):
    run_id_for: str
    run_id_against: str
    query: str


# ── SSE event helpers ─────────────────────────────────────────────────

def _emit(run_id: str, event_type: str, data: dict):
    """Append an event to the run's event list."""
    _runs[run_id]["events"].append({"type": event_type, "data": data})


# ── Streaming pipeline runner ─────────────────────────────────────────

async def _run_pipeline(run_id: str, query: str, config: PipelineConfig):
    """Run the research pipeline in a thread and emit SSE events."""
    run = _runs[run_id]

    _emit(run_id, "stage", {
        "stage": "start",
        "message": f'Starting research for "{query}"',
        "icon": "🔍"
    })

    loop = asyncio.get_event_loop()

    # ── We monkey-patch the agents to emit progress events ────────────
    coordinator = Coordinator(config=config)

    # Patch search agent
    original_search = coordinator.search_agent.run
    def patched_search(q):
        _emit(run_id, "stage", {
            "stage": "search",
            "message": "Decomposing query into sub-searches...",
            "icon": "🔎"
        })
        result = original_search(q)
        _emit(run_id, "progress", {
            "stage": "search",
            "message": f"Found {len(result)} unique results across sub-queries",
            "count": len(result)
        })
        return result
    coordinator.search_agent.run = patched_search

    # Patch document agent
    original_doc = coordinator.document_agent.run
    def patched_doc(q, raw):
        _emit(run_id, "stage", {
            "stage": "document",
            "message": f"Retrieving and filtering {len(raw)} sources...",
            "icon": "📄"
        })
        result = original_doc(q, raw)
        _emit(run_id, "progress", {
            "stage": "document",
            "message": f"Accepted {len(result)} quality sources after filtering",
            "count": len(result),
            "sources": [{"title": s.title, "url": s.url,
                         "citation_id": s.citation_id,
                         "relevance": round(s.relevance_score, 2),
                         "type": s.source_type.value} for s in result]
        })
        return result
    coordinator.document_agent.run = patched_doc

    # Patch synthesis agent
    original_synth = coordinator.synthesis_agent.run
    def patched_synth(q, sources):
        _emit(run_id, "stage", {
            "stage": "synthesis",
            "message": f"Synthesising {len(sources)} sources into report...",
            "icon": "🧠"
        })
        result = original_synth(q, sources)
        sections = result.get("sections", [])
        for i, section in enumerate(sections):
            _emit(run_id, "section", {
                "index": i,
                "total": len(sections),
                "heading": section["heading"],
                "body": section["body"],
                "citation_ids": section.get("citation_ids", [])
            })
        _emit(run_id, "progress", {
            "stage": "synthesis",
            "message": f"Generated {len(sections)} thematic sections",
            "sections": [s["heading"] for s in sections]
        })
        return result
    coordinator.synthesis_agent.run = patched_synth

    # Patch report agent
    original_report = coordinator.report_agent.run
    def patched_report(q, synthesis, sources):
        _emit(run_id, "stage", {
            "stage": "report",
            "message": "Assembling final report...",
            "icon": "📝"
        })
        result = original_report(q, synthesis, sources)
        return result
    coordinator.report_agent.run = patched_report

    # ── Run pipeline in thread (it's synchronous) ─────────────────────
    try:
        pipeline_result = await loop.run_in_executor(
            None, lambda: coordinator.research(query)
        )

        if pipeline_result.report and pipeline_result.report.status == ReportStatus.COMPLETE:
            report = pipeline_result.report
            run["report"] = report.to_dict()
            run["status"] = "complete"
            _emit(run_id, "complete", {
                "message": "Research complete!",
                "word_count": report.word_count,
                "source_count": report.source_count,
                "elapsed": pipeline_result.elapsed_seconds,
                "title": report.title
            })
        else:
            errors = pipeline_result.errors or ["Pipeline did not complete"]
            run["status"] = "error"
            _emit(run_id, "error", {"message": " | ".join(errors)})

    except Exception as exc:
        logger.exception("Pipeline error for run %s", run_id)
        run["status"] = "error"
        _emit(run_id, "error", {"message": str(exc)})


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not _check_session(request):
        login_path = STATIC_DIR / "login.html"
        if login_path.exists():
            return HTMLResponse(content=login_path.read_text(encoding="utf-8"))
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found</h1>", status_code=404)


@app.post("/login")
async def login(request: Request, response: Response):
    body = await request.json()
    password = body.get("password", "")
    if password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Incorrect password")
    token = _make_token()
    _sessions.add(token)
    response.set_cookie(
        key="session", value=token,
        httponly=True, samesite="lax",
        max_age=60 * 60 * 24 * 7  # 7 days
    )
    return {"ok": True}


@app.post("/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    _sessions.discard(token)
    response.delete_cookie("session")
    return {"ok": True}


@app.post("/debate", response_model=DebateResponse)
async def start_debate(req: ResearchRequest, request: Request):
    if not _check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set on the server")

    run_id_for     = str(uuid.uuid4())[:8]
    run_id_against = str(uuid.uuid4())[:8]

    q_for     = f"{req.query} — Build the strongest possible case FOR this. Focus on supporting evidence, benefits, and arguments in favour."
    q_against = f"{req.query} — Build the strongest possible case AGAINST this. Focus on opposing evidence, risks, and arguments against."

    config = PipelineConfig(
        sub_queries=req.sub_queries,
        max_sources=req.max_sources,
        relevance_threshold=req.relevance_threshold,
    )

    _runs[run_id_for]     = {"status": "running", "events": [], "report": None, "query": q_for}
    _runs[run_id_against] = {"status": "running", "events": [], "report": None, "query": q_against}

    asyncio.create_task(_run_pipeline(run_id_for,     q_for,     config))
    asyncio.create_task(_run_pipeline(run_id_against, q_against, config))

    return DebateResponse(run_id_for=run_id_for, run_id_against=run_id_against, query=req.query)


@app.get("/bg")
async def get_background(request: Request, query: str = Query(default="")):
    if not _check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not key or not query.strip():
        return JSONResponse({"url": None, "credit": None})

    def _fetch():
        try:
            params = urllib.parse.urlencode({
                "query": query[:80],
                "orientation": "landscape",
                "client_id": key,
            })
            req = urllib.request.Request(
                f"https://api.unsplash.com/photos/random?{params}",
                headers={"Accept-Version": "v1"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read())
                return {
                    "url": data["urls"]["regular"],
                    "credit": data["user"]["name"],
                    "credit_link": data["user"]["links"]["html"],
                }
        except Exception:
            return {"url": None, "credit": None}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _fetch)
    return JSONResponse(result)


async def _serve_cached(run_id: str, report: dict):
    """Replay a cached report over SSE without touching the pipeline."""
    await asyncio.sleep(0.15)  # let the client connect first
    _emit(run_id, "cached", {
        "message": "Loaded from recent reports",
        "title": report.get("title", ""),
    })
    sections = report.get("sections", [])
    for i, section in enumerate(sections):
        _emit(run_id, "section", {
            "index": i, "total": len(sections),
            "heading": section["heading"],
            "body": section["body"],
            "citation_ids": section.get("citation_ids", []),
        })
        await asyncio.sleep(0.06)
    _emit(run_id, "complete", {
        "message": "Research complete!",
        "word_count": report.get("word_count", 0),
        "source_count": report.get("source_count", 0),
        "elapsed": 0,
        "title": report.get("title", ""),
        "from_cache": True,
    })
    _runs[run_id]["status"] = "complete"


@app.post("/research", response_model=ResearchResponse)
async def start_research(req: ResearchRequest, request: Request):
    if not _check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not req.query.strip():
        raise HTTPException(400, "Query must not be empty")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set on the server")

    run_id = str(uuid.uuid4())[:8]

    # Return cached report if the same query was run before
    try:
        history = ReportAgent().load_history()
        cached = next(
            (r for r in history
             if r.get("query", "").strip().lower() == req.query.strip().lower()),
            None
        )
    except Exception:
        cached = None

    if cached:
        _runs[run_id] = {"status": "running", "events": [], "report": cached, "query": req.query}
        asyncio.create_task(_serve_cached(run_id, cached))
        return ResearchResponse(run_id=run_id, query=req.query)

    _runs[run_id] = {"status": "running", "events": [], "report": None, "query": req.query}
    config = PipelineConfig(
        sub_queries=req.sub_queries,
        max_sources=req.max_sources,
        relevance_threshold=req.relevance_threshold,
    )
    asyncio.create_task(_run_pipeline(run_id, req.query, config))

    return ResearchResponse(run_id=run_id, query=req.query)


@app.get("/stream/{run_id}")
async def stream_events(run_id: str, request: Request):
    if not _check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if run_id not in _runs:
        raise HTTPException(404, f"Run {run_id} not found")

    async def generator() -> AsyncIterator[dict]:
        sent = 0
        while True:
            run = _runs[run_id]
            events = run["events"]

            while sent < len(events):
                evt = events[sent]
                yield {"event": evt["type"], "data": json.dumps(evt["data"])}
                sent += 1

            if run["status"] in ("complete", "error"):
                break

            await asyncio.sleep(0.3)

    return EventSourceResponse(generator())


@app.get("/report/{run_id}")
async def get_report(run_id: str, request: Request):
    if not _check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if run_id not in _runs:
        raise HTTPException(404, f"Run {run_id} not found")
    run = _runs[run_id]
    if run["status"] != "complete":
        raise HTTPException(400, f"Run {run_id} is not complete yet (status: {run['status']})")
    return run["report"]


@app.get("/headlines")
async def get_headlines(request: Request):
    if not _check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    def _fetch():
        try:
            news_key = os.environ.get("NEWS_API_KEY")
            if news_key:
                params = urllib.parse.urlencode({
                    "language": "en", "pageSize": 10, "apiKey": news_key,
                })
                req = urllib.request.Request(
                    f"https://newsapi.org/v2/top-headlines?{params}",
                    headers={"User-Agent": "AskKian/1.0"}
                )
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read())
                    return [
                        {"title": a["title"], "url": a["url"], "source": a["source"]["name"]}
                        for a in data.get("articles", [])[:10]
                        if a.get("title") and a.get("url")
                    ]
            else:
                req = urllib.request.Request(
                    "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=10",
                    headers={"User-Agent": "AskKian/1.0"}
                )
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read())
                    return [
                        {
                            "title": h.get("title", ""),
                            "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                            "source": "Hacker News",
                        }
                        for h in data.get("hits", [])[:10]
                        if h.get("title")
                    ]
        except Exception as e:
            logger.error("Headlines fetch failed: %s", e)
            return []

    loop = asyncio.get_event_loop()
    headlines = await loop.run_in_executor(None, _fetch)
    return JSONResponse({"headlines": headlines})


@app.get("/cache-status")
async def cache_status(request: Request):
    if not _check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        history = ReportAgent().load_history()
        return JSONResponse({
            "count": len(history),
            "entries": [
                {
                    "query": r.get("query", ""),
                    "title": r.get("title", ""),
                    "word_count": r.get("word_count", 0),
                    "source_count": r.get("source_count", 0),
                    "completed_at": r.get("completed_at", ""),
                }
                for r in history
            ]
        })
    except Exception as e:
        return JSONResponse({"count": 0, "entries": [], "error": str(e)})


@app.get("/history")
async def get_history(request: Request):
    if not _check_session(request):
        raise HTTPException(status_code=401, detail="Not authenticated")
    agent = ReportAgent()
    return agent.load_history()


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔬 Research Assistant server starting...")
    print("   Open http://localhost:8000 in your browser\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
