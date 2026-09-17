"""Tests for the retriever. Requires the vector index to have been built first
(python scripts/build_index.py), since these tests query the real ChromaDB collection."""
import pytest

from app.rag.retriever import retrieve
from app.rag.vector_store import get_collection


def _index_exists() -> bool:
    try:
        return get_collection().count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _index_exists(), reason="Vector index not built yet")


def test_retrieve_hostel_question_returns_hostel_chunks():
    results = retrieve("What documents are required for hostel admission?", department="Hostel", top_k=3)
    assert len(results) > 0
    assert all(r.department == "Hostel" for r in results)


def test_retrieve_respects_top_k():
    results = retrieve("library book borrowing rules", department="Student Services", top_k=2)
    assert len(results) <= 2


def test_retrieve_without_department_filter_searches_everything():
    results = retrieve("hostel fees", department=None, top_k=5)
    assert len(results) > 0
