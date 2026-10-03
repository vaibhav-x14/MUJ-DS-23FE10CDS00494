import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omnirag.pipeline import OmniRAGEngine
from omnirag.core.schemas import OmniRAGResult
from prompts.prompt_manager import prompt_catalog
from config.settings import settings

app = FastAPI(
    title="OmniRAG API",
    description="Multi-Hop Agentic RAG System with Self-Reflection & Graph Augmentation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = OmniRAGEngine()
engine.index()

STATIC_DIR = Path(__file__).resolve().parent / "static"


class QueryRequest(BaseModel):
    query: str


@app.get("/api/health")
def health_check():
    db_stats = engine.db.get_stats()
    return {
        "status": "healthy",
        "provider": settings.llm.provider,
        "model": settings.llm.model_name,
        "is_indexed": engine._is_indexed,
        "chunks_indexed": len(engine.chunks),
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY") or settings.llm.api_key),
        "database": db_stats,
    }


@app.get("/api/database")
def get_database_stats():
    return engine.db.get_stats()


@app.post("/api/query", response_model=OmniRAGResult)
def execute_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = engine.query(req.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prompts")
def get_prompts():
    return prompt_catalog.list_templates()


@app.get("/api/config")
def get_configuration():
    return {
        "system": settings.system.model_dump(),
        "llm": {
            "provider": settings.llm.provider,
            "model_name": settings.llm.model_name,
            "temperature": settings.llm.temperature,
            "max_output_tokens": settings.llm.max_output_tokens,
        },
        "retrieval": settings.retrieval.model_dump(),
        "knowledge_graph": settings.knowledge_graph.model_dump(),
        "agent": settings.agent.model_dump(),
    }


# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>OmniRAG Web Dashboard Loading...</h1>")
