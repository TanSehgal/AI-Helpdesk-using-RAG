"""Splits documents into overlapping text chunks for embedding.

Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
paragraph/sentence boundaries first and only falls back to hard character
cuts when necessary. This keeps chunks readable, which matters for a project
that needs to be easy to debug and explain.
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.document_loader import RawDocument


@dataclass
class Chunk:
    text: str
    source: str
    department: str
    doc_type: str
    chunk_index: int


def split_documents(
    documents: list[RawDocument], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for doc in documents:
        pieces = splitter.split_text(doc.text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    department=doc.department,
                    doc_type=doc.doc_type,
                    chunk_index=i,
                )
            )
    return chunks
