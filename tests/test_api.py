"""API-level tests using FastAPI's TestClient. Uses monkeypatching so these
tests don't require Ollama to be running."""
from fastapi.testclient import TestClient

from app.agents.base_agent import AgentResult
from app.main import app

client = TestClient(app)


def test_empty_question_returns_400():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400


def test_missing_question_field_returns_422():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_health_endpoint_returns_expected_fields():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama_available" in data
    assert "vector_store_ready" in data


def test_ask_returns_expected_shape(monkeypatch):
    def fake_route_and_answer(question, top_k=None):
        return AgentResult(
            answer="Test answer.",
            department="Hostel",
            agent_name="Hostel Agent",
            retrieved=[],
            grounded=True,
        )

    monkeypatch.setattr("app.main.route_and_answer", fake_route_and_answer)

    response = client.post("/ask", json={"question": "What documents do I need for hostel admission?"})
    assert response.status_code == 200
    data = response.json()
    assert data["department"] == "Hostel"
    assert data["agent"] == "Hostel Agent"
    assert data["answer"] == "Test answer."
    assert "response_time" in data
