"""Retrieves the most relevant chunks for a query, optionally filtered by department."""
from dataclasses import dataclass

from app.rag.vector_store import get_collection
from app.utils.config import TOP_K
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    department: str
    document_type: str
    distance: float


def retrieve(query: str, department: str | None = None, top_k: int = TOP_K) -> list[RetrievedChunk]:
    """Runs a similarity search against the vector store.

    If `department` is given, only chunks belonging to that department are
    considered. This is what lets each department agent search only its own
    knowledge base.
    """
    collection = get_collection()
    where = {"department": department} if department else None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )
    except Exception as exc:
        logger.error("Vector store query failed: %s", exc)
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []
    for text, meta, distance in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                text=text,
                source=meta.get("source", "unknown"),
                department=meta.get("department", "unknown"),
                document_type=meta.get("document_type", "unknown"),
                distance=distance,
            )
        )
    return chunks
