from app.rag.document_loader import load_documents
from app.utils.config import KNOWLEDGE_BASE_DIR


def test_load_documents_returns_all_kb_files():
    docs = load_documents(KNOWLEDGE_BASE_DIR)
    assert len(docs) == 10
    filenames = {d.source for d in docs}
    assert "hostel.txt" in filenames
    assert "academics.txt" in filenames


def test_documents_have_department_assigned():
    docs = load_documents(KNOWLEDGE_BASE_DIR)
    for doc in docs:
        assert doc.department in {"Academic", "Hostel", "Administration", "Student Services"}
        assert doc.text.strip() != ""


def test_hostel_document_department_is_correct():
    docs = load_documents(KNOWLEDGE_BASE_DIR)
    hostel_doc = next(d for d in docs if d.source == "hostel.txt")
    assert hostel_doc.department == "Hostel"
