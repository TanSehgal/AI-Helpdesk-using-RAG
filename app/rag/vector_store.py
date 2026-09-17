"""ChromaDB vector store wrapper.

Design choice (kept simple on purpose): all departments share ONE ChromaDB
collection, and each stored chunk carries a `department` metadata field.
Department-specific retrieval is done with a metadata filter rather than by
creating four separate collections. This is easier to build, inspect, and
explain than managing multiple collections, while still giving each
department a clearly separated knowledge base.
"""
import chromadb

from app.rag.chunker import Chunk
from app.rag.embeddings import get_embedding_function
from app.utils.config import CHROMA_PERSIST_DIRECTORY
from app.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "campus_helpdesk"


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)


def get_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client: chromadb.ClientAPI | None = None):
    """Deletes the existing collection (if any) so the index can be rebuilt from scratch."""
    client = client or get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet
    return get_collection(client)


def add_chunks(collection, chunks: list[Chunk], batch_size: int = 100) -> None:
    """Embeds and stores chunks in the collection, in batches for progress visibility."""
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        ids = [f"{c.source}-{c.chunk_index}-{start + i}" for i, c in enumerate(batch)]
        documents = [c.text for c in batch]
        metadatas = [
            {
                "department": c.department,
                "source": c.source,
                "document_name": c.source,
                "document_type": c.doc_type,
                "chunk_index": c.chunk_index,
            }
            for c in batch
        ]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Indexed chunks %d-%d of %d", start + 1, min(start + batch_size, total), total)
