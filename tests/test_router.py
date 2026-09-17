"""Tests for the Router Agent's keyword fallback (does not require Ollama to be running)."""
from app.agents.router import _keyword_route, get_agent_for_department


def test_keyword_route_hostel():
    assert _keyword_route("What documents are needed for hostel admission?") == "Hostel"


def test_keyword_route_academic():
    assert _keyword_route("What is the minimum attendance required for exams?") == "Academic"


def test_keyword_route_administration():
    assert _keyword_route("How do I apply for a scholarship?") == "Administration"


def test_keyword_route_student_services():
    assert _keyword_route("How many books can I borrow from the library?") == "Student Services"


def test_keyword_route_defaults_to_student_services_when_no_match():
    assert _keyword_route("asdkjaslkdj random gibberish") == "Student Services"


def test_get_agent_for_department_returns_correct_agent():
    agent = get_agent_for_department("Hostel")
    assert agent.department == "Hostel"
    assert agent.agent_name == "Hostel Agent"


def test_get_agent_for_unknown_department_defaults_to_student_services():
    agent = get_agent_for_department("Nonexistent")
    assert agent.department == "Student Services"
