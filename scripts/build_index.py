#!/usr/bin/env python3
"""Builds (or rebuilds) the ChromaDB vector index from the knowledge_base directory.

Usage:
    python scripts/build_index.py

This script:
1. Loads all .txt/.pdf files from knowledge_base/
2. Splits them into overlapping chunks
3. Generates embeddings for each chunk
4. Stores everything in the persistent ChromaDB collection
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunker import split_documents
from app.rag.document_loader import load_documents
from app.rag.vector_store import add_chunks, reset_collection
from app.utils.config import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_BASE_DIR
from app.utils.logger import get_logger

logger = get_logger("build_index")


def main() -> None:
    start = time.perf_counter()
    logger.info("Building vector index from: %s", KNOWLEDGE_BASE_DIR)

    documents = load_documents(KNOWLEDGE_BASE_DIR)
    if not documents:
        logger.error("No documents found in knowledge base directory. Aborting.")
        sys.exit(1)
    logger.info("Loaded %d document(s).", len(documents))

    chunks = split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logger.info("Split into %d chunk(s) (chunk_size=%d, overlap=%d).", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    logger.info("Resetting collection and generating embeddings (this may take a moment)...")
    collection = reset_collection()
    add_chunks(collection, chunks)

    elapsed = time.perf_counter() - start
    logger.info("Done. Indexed %d chunks from %d documents in %.1fs.", len(chunks), len(documents), elapsed)
    logger.info("Collection now contains %d items.", collection.count())


if __name__ == "__main__":
    main()
