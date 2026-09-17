"""Pydantic request/response models for the FastAPI backend."""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="The student's question")


class SourceDocument(BaseModel):
    document_name: str
    department: str
    text_snippet: str


class AskResponse(BaseModel):
    answer: str
    department: str
    agent: str
    sources: list[str]
    retrieved_documents: list[SourceDocument]
    response_time: float
    grounded: bool


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    vector_store_ready: bool
    llm_provider: str
    llm_model: str
