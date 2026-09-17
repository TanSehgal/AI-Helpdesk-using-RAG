"""Router Agent: decides which department agent should handle a question.

Primary strategy: ask the LLM to classify the question (semantic routing).
Fallback strategy: if the LLM call fails, or returns something that is not a
recognized department, fall back to simple keyword matching so the system
never crashes just because routing was ambiguous.
"""
from app.agents.academic_agent import academic_agent
from app.agents.administration_agent import administration_agent
from app.agents.hostel_agent import hostel_agent
from app.agents.services_agent import services_agent
from app.llm.ollama_client import LLMUnavailableError, generate
from app.llm.prompts import ROUTER_PROMPT_TEMPLATE
from app.utils.config import DEPARTMENTS
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEPARTMENT_AGENTS = {
    "Academic": academic_agent,
    "Hostel": hostel_agent,
    "Administration": administration_agent,
    "Student Services": services_agent,
}

# Keyword fallback: used only when the LLM router is unavailable or gives an
# unrecognized answer. Order matters slightly less here since we score all
# departments and pick the best match.
KEYWORD_MAP = {
    "Academic": [
        "course", "subject", "attendance", "exam", "syllabus", "grade", "cgpa",
        "semester", "credit", "backlog", "result", "class", "professor", "faculty",
    ],
    "Hostel": [
        "hostel", "room", "mess", "warden", "roommate", "accommodation", "dormitory",
    ],
    "Administration": [
        "fee", "scholarship", "certificate", "document", "bonafide", "transcript",
        "refund", "tuition", "admission fee", "transfer certificate",
    ],
    "Student Services": [
        "library", "book", "id card", "transport", "bus", "wifi", "wi-fi", "it support",
        "password", "email", "internet", "computer lab",
    ],
}


def _keyword_route(question: str) -> str:
    q = question.lower()
    scores = {dept: 0 for dept in DEPARTMENTS}
    for dept, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in q:
                scores[dept] += 1

    best_dept = max(scores, key=lambda d: scores[d])
    if scores[best_dept] == 0:
        # No keyword matched anything; default to Student Services as the
        # general catch-all department.
        return "Student Services"
    return best_dept


def _llm_route(question: str) -> str | None:
    prompt = ROUTER_PROMPT_TEMPLATE.format(question=question)
    try:
        raw = generate(prompt, temperature=0.0)
    except LLMUnavailableError as exc:
        logger.warning("LLM router unavailable, falling back to keywords: %s", exc)
        return None

    cleaned = raw.strip().strip(".").strip()
    for dept in DEPARTMENTS:
        if dept.lower() in cleaned.lower():
            return dept
    return None


def route(question: str) -> str:
    """Returns the department name best suited to answer the question."""
    department = _llm_route(question)
    if department is None:
        department = _keyword_route(question)
        logger.info("Routed via keyword fallback -> %s", department)
    else:
        logger.info("Routed via LLM semantic router -> %s", department)
    return department


def get_agent_for_department(department: str):
    return DEPARTMENT_AGENTS.get(department, DEPARTMENT_AGENTS["Student Services"])


def route_and_answer(question: str, top_k: int | None = None):
    """End-to-end: route the question, then delegate to the chosen department agent."""
    department = route(question)
    agent = get_agent_for_department(department)
    if top_k is not None:
        return agent.answer(question, top_k=top_k)
    return agent.answer(question)
