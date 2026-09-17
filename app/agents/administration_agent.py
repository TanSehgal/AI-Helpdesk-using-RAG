"""Administration Agent: fees, scholarships, certificates, documents, procedures."""
from app.agents.base_agent import DepartmentAgent

administration_agent = DepartmentAgent(
    department="Administration", agent_name="Administration Agent"
)
