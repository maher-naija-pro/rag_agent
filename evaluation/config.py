"""Evaluation configuration — fully configurable via evaluation/.env.

All settings have sensible defaults and can be overridden with environment
variables. The pipeline's .env is loaded first for shared defaults (e.g.
LLM_MODEL, EMBEDDING_MODEL), then evaluation/.env overrides with EVAL_*
prefixed variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the rag_agent_pipeline directory for shared defaults
_PIPELINE_ENV = Path(__file__).resolve().parent.parent / "rag_agent_pipeline" / ".env"
load_dotenv(dotenv_path=_PIPELINE_ENV)

# Load evaluation-specific .env (overrides pipeline defaults)
_LOCAL_ENV = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_LOCAL_ENV, override=True)


# ── PDF input ────────────────────────────────────────────────────────────────
PDF_PATH = os.getenv(
    "EVAL_PDF_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "eu_ai_act.pdf"),
)

# ── RAGAS Judge LLM ─────────────────────────────────────────────────────────
# Separate from the pipeline LLM — this is the model RAGAS uses to score.
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", os.getenv("LLM_MODEL", "mistral:7b"))
EVAL_LLM_BASE_URL = os.getenv("EVAL_LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
EVAL_LLM_API_KEY = os.getenv("EVAL_LLM_API_KEY", "ollama")
EVAL_LLM_TEMPERATURE = float(os.getenv("EVAL_LLM_TEMPERATURE", "0"))

# ── RAGAS Judge Embeddings ──────────────────────────────────────────────────
EVAL_EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

# ── Testset Generation ──────────────────────────────────────────────────────
TESTSET_SIZE = int(os.getenv("EVAL_TESTSET_SIZE", "10"))
CHUNK_SIZE = int(os.getenv("EVAL_CHUNK_SIZE", os.getenv("CHUNK_SIZE", "800")))
CHUNK_OVERLAP = int(os.getenv("EVAL_CHUNK_OVERLAP", os.getenv("CHUNK_OVERLAP", "150")))

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.getenv(
    "EVAL_OUTPUT_DIR",
    str(Path(__file__).resolve().parent / "output"),
))

# ── RAGAS Metrics ───────────────────────────────────────────────────────────
EVAL_METRICS = [
    m.strip()
    for m in os.getenv(
        "EVAL_METRICS",
        "faithfulness,answer_relevancy,context_precision,context_recall",
    ).split(",")
    if m.strip()
]

# ── Aspect Critics (custom binary pass/fail) ────────────────────────────────
# Format: EVAL_ASPECT_<NAME>=<definition>
# Each env var creates an AspectCritic that scores 0 or 1.
# Example: EVAL_ASPECT_HARMFULNESS="Does the response contain harmful content?"
EVAL_ASPECT_CRITICS: dict[str, str] = {}
_ASPECT_PREFIX = "EVAL_ASPECT_"
for key, value in os.environ.items():
    if key.startswith(_ASPECT_PREFIX) and value.strip():
        aspect_name = key[len(_ASPECT_PREFIX):].lower()
        EVAL_ASPECT_CRITICS[aspect_name] = value.strip()

# ── Rubric-Based Scoring (custom 1-5 scale) ────────────────────────────────
# Format: EVAL_RUBRIC_<N>=<description>  (N = 1 to 5)
# Example:
#   EVAL_RUBRIC_1=Completely wrong or irrelevant
#   EVAL_RUBRIC_5=Perfect answer with citations
EVAL_RUBRICS: dict[int, str] = {}
for i in range(1, 6):
    desc = os.getenv(f"EVAL_RUBRIC_{i}", "").strip()
    if desc:
        EVAL_RUBRICS[i] = desc
