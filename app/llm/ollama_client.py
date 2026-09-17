"""LLM abstraction layer.

Provides a single `generate(prompt)` function used by the rest of the app.
Internally it can talk to either:
  - a local Ollama server (default, LLM_PROVIDER=ollama)
  - a cloud, OpenAI-compatible chat completions API (LLM_PROVIDER=cloud)

This keeps the rest of the codebase (agents, router) independent of which
backend actually serves the model, which is what makes cloud deployment
possible without rewriting agent logic.
"""
import requests

from app.utils.config import (
    CLOUD_LLM_API_KEY,
    CLOUD_LLM_BASE_URL,
    CLOUD_LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMUnavailableError(Exception):
    """Raised when the configured LLM backend cannot be reached or fails."""


OLLAMA_TIMEOUT_SECONDS = 240  # CPU-only inference with small local models can be slow


def _generate_ollama(prompt: str, temperature: float = 0.2) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # num_predict caps output length: keeps helpdesk answers concise and
        # bounds worst-case latency on slow CPU-only inference.
        "options": {"temperature": temperature, "num_predict": 400},
    }
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise LLMUnavailableError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is Ollama running? Try: ollama serve"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMUnavailableError("Ollama request timed out.") from exc
    except requests.exceptions.HTTPError as exc:
        raise LLMUnavailableError(
            f"Ollama returned an error. Is the model '{OLLAMA_MODEL}' pulled? "
            f"Try: ollama pull {OLLAMA_MODEL}"
        ) from exc

    data = response.json()
    return data.get("response", "").strip()


def _generate_cloud(prompt: str, temperature: float = 0.2) -> str:
    if not CLOUD_LLM_API_KEY:
        raise LLMUnavailableError(
            "LLM_PROVIDER=cloud but CLOUD_LLM_API_KEY is not set. "
            "Set it in your .env file."
        )
    url = f"{CLOUD_LLM_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {CLOUD_LLM_API_KEY}"}
    payload = {
        "model": CLOUD_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise LLMUnavailableError(f"Cloud LLM request failed: {exc}") from exc

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate(prompt: str, temperature: float = 0.2) -> str:
    """Generates a completion from the configured LLM provider."""
    if LLM_PROVIDER == "cloud":
        return _generate_cloud(prompt, temperature)
    return _generate_ollama(prompt, temperature)


def check_ollama_health() -> bool:
    """Returns True if the Ollama server responds, used by the /health endpoint."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
