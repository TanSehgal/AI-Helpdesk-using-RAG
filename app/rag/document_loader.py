"""Loads TXT and PDF documents from the knowledge base directory.

Each department has its own set of files. The department is inferred from a
filename -> department mapping so that metadata can be attached during ingestion.
"""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Maps a knowledge-base filename (without extension) to the department it belongs to.
FILE_DEPARTMENT_MAP = {
    "academics": "Academic",
    "examinations": "Academic",
    "hostel": "Hostel",
    "fees": "Administration",
    "scholarships": "Administration",
    "certificates": "Administration",
    "library": "Student Services",
    "student_services": "Student Services",
    "transportation": "Student Services",
    "it_support": "Student Services",
}

DEFAULT_DEPARTMENT = "Student Services"


@dataclass
class RawDocument:
    text: str
    source: str  # filename
    department: str
    doc_type: str  # "txt" or "pdf"


def _infer_department(filename_stem: str) -> str:
    return FILE_DEPARTMENT_MAP.get(filename_stem.lower(), DEFAULT_DEPARTMENT)


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(knowledge_base_dir: str) -> list[RawDocument]:
    """Loads every .txt and .pdf file from the knowledge base directory."""
    kb_path = Path(knowledge_base_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_path}")

    documents: list[RawDocument] = []
    for path in sorted(kb_path.iterdir()):
        if path.suffix.lower() == ".txt":
            text = _load_txt(path)
            doc_type = "txt"
        elif path.suffix.lower() == ".pdf":
            text = _load_pdf(path)
            doc_type = "pdf"
        else:
            continue

        if not text.strip():
            logger.warning("Skipping empty document: %s", path.name)
            continue

        department = _infer_department(path.stem)
        documents.append(
            RawDocument(text=text, source=path.name, department=department, doc_type=doc_type)
        )
        logger.info("Loaded %s (%s, %d chars)", path.name, department, len(text))

    return documents
