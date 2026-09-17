"""Student Services Agent: library, ID cards, transportation, IT support, general services."""
from app.agents.base_agent import DepartmentAgent

services_agent = DepartmentAgent(
    department="Student Services", agent_name="Student Services Agent"
)
