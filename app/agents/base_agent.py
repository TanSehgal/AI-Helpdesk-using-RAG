"""Shared logic for all department agents.

Each department agent (Academic, Hostel, Administration, Student Services) is a
thin wrapper around this shared RAG-answer routine, configured with its own
department name and display name. Keeping the logic in one place avoids
duplicating the same retrieve-then-generate flow four times.
"""
from dataclasses import dataclass, field

from app.llm.ollama_client import LLMUnavailableError, generate
from app.llm.prompts import NO_CONTEXT_FALLBACK, RAG_ANSWER_PROMPT_TEMPLATE
from app.rag.retriever import RetrievedChunk, retrieve
from app.utils.config import RELEVANCE_DISTANCE_THRESHOLD, TOP_K
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Substring that the strict RAG prompt asks the model to output when the
# context does not contain the answer. Used to flag "ungrounded" answers.
NOT_FOUND_MARKER = "could not find"


@dataclass
class AgentResult:
    answer: str
    department: str
    agent_name: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    grounded: bool = True


class DepartmentAgent:
    def __init__(self, department: str, agent_name: str):
        self.department = department
        self.agent_name = agent_name

    def answer(self, question: str, top_k: int = TOP_K) -> AgentResult:
        retrieved = retrieve(question, department=self.department, top_k=top_k)

        # Skip the (slow) LLM call entirely when nothing retrieved is actually
        # relevant to the question - this both avoids wasted generation time
        # and gives a clean, honest "not found" instead of forcing the LLM to
        # work from irrelevant context.
        relevant = [c for c in retrieved if c.distance <= RELEVANCE_DISTANCE_THRESHOLD]

        if not relevant:
            return AgentResult(
                answer=NO_CONTEXT_FALLBACK,
                department=self.department,
                agent_name=self.agent_name,
                retrieved=[],
                grounded=False,
            )
        retrieved = relevant

        context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in retrieved)
        prompt = RAG_ANSWER_PROMPT_TEMPLATE.format(
            agent_name=self.agent_name,
            department=self.department,
            context=context,
            question=question,
        )

        try:
            answer = generate(prompt)
        except LLMUnavailableError as exc:
            logger.error("LLM generation failed: %s", exc)
            return AgentResult(
                answer=f"Sorry, the AI model is currently unavailable ({exc}). "
                "Please try again shortly.",
                department=self.department,
                agent_name=self.agent_name,
                retrieved=retrieved,
                grounded=False,
            )

        grounded = NOT_FOUND_MARKER not in answer.lower()
        return AgentResult(
            answer=answer,
            department=self.department,
            agent_name=self.agent_name,
            retrieved=retrieved,
            grounded=grounded,
        )
