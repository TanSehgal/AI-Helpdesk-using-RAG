from app.rag.chunker import split_documents
from app.rag.document_loader import RawDocument


def test_split_documents_produces_chunks_within_size_limit():
    long_text = "This is a sentence about hostel rules. " * 100
    doc = RawDocument(text=long_text, source="test.txt", department="Hostel", doc_type="txt")

    chunks = split_documents([doc], chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 250  # small buffer for separator boundaries
        assert chunk.source == "test.txt"
        assert chunk.department == "Hostel"


def test_split_documents_empty_list_returns_empty():
    assert split_documents([], chunk_size=500, chunk_overlap=50) == []
