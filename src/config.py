"""Environment configuration, model endpoints, and cost/telemetry helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_DIR = REPO_ROOT / "starter"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def load_env() -> None:
    """Load environment variables from starter/.env without overriding explicit env vars."""
    env_path = STARTER_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


@dataclass(frozen=True)
class AppConfig:
    database_url: str
    openrouter_api_key: str
    gemini_api_key: str
    embedding_model: str
    embedding_dimensions: int
    agent_model: str
    rerank_model: str
    vision_model: str


def get_config() -> AppConfig:
    """Return validated runtime configuration from environment and starter/.env."""
    load_env()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing. Configure starter/.env first.")

    return AppConfig(
        database_url=database_url,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "").strip() or "openai/text-embedding-3-small",
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
        agent_model=os.getenv("AGENT_MODEL", "").strip() or "gemini-3.5-flash-lite",
        rerank_model=os.getenv("RERANK_MODEL", "").strip() or "gemini-3.1-flash-lite-preview",
        vision_model=os.getenv("VISION_MODEL", "").strip() or "gemini-3.5-flash-lite",
    )


# Reference pricing (USD per 1M tokens) for telemetry reporting when provider API omits USD cost.
# We always record raw token counts alongside these reference USD estimates.
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (0.50, 3.00),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "openai/text-embedding-3-small": (0.02, 0.0),
}


def estimate_llm_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from token counts using reference rates (0.0 if unknown)."""
    in_rate, out_rate = MODEL_PRICING_PER_1M.get(model, (0.0, 0.0))
    return round((prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0, 8)

