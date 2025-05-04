"""AI-powered helper utilities (e.g., GPT-based recommendation generator)."""
from __future__ import annotations

import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

# Load variables from a .env file in project root (if present)
load_dotenv()

# Module-level logger
logger = logging.getLogger(__name__)


def _openai_client():
    """Return an OpenAI client instance or *None* if package / key missing."""
    try:
        import openai  # type: ignore
    except ModuleNotFoundError:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key)


def generate_recommendations(metrics: Dict[str, float]) -> List[str]:
    """Return a static set of generic recommendations.

    The previous implementation attempted to call the OpenAI API.  To simplify
    the code-base and avoid external dependencies, we now always return the
    fallback list directly.
    """

    return [
        "Run additional hyper-parameter tuning to explore learning-rate and epsilon schedules.",
        "Back-test the agent across different market regimes (bull, bear, sideways).",
        "Experiment with additional technical indicators or alternative reward functions to enhance learning.",
    ] 