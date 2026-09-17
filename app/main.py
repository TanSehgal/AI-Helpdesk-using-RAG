"""FastAPI backend for the AI Campus Helpdesk.

Endpoints:
  POST /ask     - answer a student's question using the multi-agent RAG pipeline
  GET  /health  - report the health of the LLM backend and vector store
"""
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agents.router import route_and_answer
from app.llm.ollama_client import check_ollama_health
from app.models.schemas import AskRequest, AskResponse, HealthResponse, SourceDocument
from app.rag.vector_store import get_collection
from app.utils.config import LLM_PROVIDER, OLLAMA_MODEL, TOP_K
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Campus Helpdesk API",
    description="RAG-based multi-agent chatbot for campus services (research prototype).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ollama_ok = check_ollama_health() if LLM_PROVIDER == "ollama" else True

    vector_store_ready = False
    try:
        collection = get_collection()
        vector_store_ready = collection.count() > 0
    except Exception as exc:
        logger.error("Vector store health check failed: %s", exc)

    status = "ok" if (ollama_ok and vector_store_ready) else "degraded"
    return HealthResponse(
        status=status,
        ollama_available=ollama_ok,
        vector_store_ready=vector_store_ready,
        llm_provider=LLM_PROVIDER,
        llm_model=OLLAMA_MODEL,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    start = time.perf_counter()
    try:
        result = route_and_answer(question, top_k=TOP_K)
    except Exception as exc:
        logger.exception("Unexpected error while answering question")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
    elapsed = time.perf_counter() - start

    sources = sorted({c.source for c in result.retrieved})
    retrieved_documents = [
        SourceDocument(
            document_name=c.source,
            department=c.department,
            text_snippet=(c.text[:300] + "...") if len(c.text) > 300 else c.text,
        )
        for c in result.retrieved
    ]

    return AskResponse(
        answer=result.answer,
        department=result.department,
        agent=result.agent_name,
        sources=sources,
        retrieved_documents=retrieved_documents,
        response_time=round(elapsed, 3),
        grounded=result.grounded,
    )
