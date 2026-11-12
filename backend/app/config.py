"""
config.py - every number that controls the scoring lives in ONE place.

Each value can be overridden with an environment variable (see ../../.env.example),
so you can try another model or other weights without editing any code.
"""

import os


def _float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to `default` if missing or invalid."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to `default` if missing or invalid."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1) Signal 1 - semantic matching with a BERT encoder (SBERT or BGE)
# ---------------------------------------------------------------------------
# Default: SBERT "all-MiniLM-L6-v2" - small (~90 MB), fast on CPU, 384-number vectors.
# To try BGE instead:  EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  (or a job-specific
# BGE fine-tune), rebuild, and compare both models with nDCG.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# MiniLM reads at most 256 tokens (~190 words) and silently cuts the rest.
# So long text is split into chunks of at most this many words before embedding.
CHUNK_WORDS = _int("CHUNK_WORDS", 150)

# ---------------------------------------------------------------------------
# 2) Fusion weights - like college marks: theory 50 / practical 30 / viva 20
# ---------------------------------------------------------------------------
W_SEMANTIC = _float("W_SEMANTIC", 0.5)
W_LEXICAL = _float("W_LEXICAL", 0.3)
W_LLM = _float("W_LLM", 0.2)

# Inside the lexical score, must-have skills count more than nice-to-have ones.
LEX_MUST_WEIGHT = 0.7
LEX_NICE_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# 3) Signal 2 - LLM evaluation (OPTIONAL)
# ---------------------------------------------------------------------------
# Any OpenAI-compatible chat API works: xAI Grok, OpenAI, Groq, OpenRouter,
# or a local Ollama server. Leave LLM_API_KEY empty to switch this signal off;
# its 20% weight is then shared between semantic and lexical.
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.x.ai/v1").strip().rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_TIMEOUT_SECONDS = _float("LLM_TIMEOUT_SECONDS", 60)
LLM_MAX_WORKERS = _int("LLM_MAX_WORKERS", 4)  # parallel LLM calls

# Rubric the LLM judge scores (each 0-10). The weights add up to 1.0.
LLM_RUBRIC = {"skills": 0.4, "experience": 0.3, "projects": 0.2, "education": 0.1}

# ---------------------------------------------------------------------------
# 4) Integrity checks (resume gaming)
# ---------------------------------------------------------------------------
TINY_FONT_PT = _float("TINY_FONT_PT", 5.0)         # smaller than this = unreadable text
NEAR_WHITE_MIN = _int("NEAR_WHITE_MIN", 240)       # all RGB channels >= this = "white"
JD_COPY_MIN_WORDS = _int("JD_COPY_MIN_WORDS", 10)  # only check sentences this long or longer
JD_COPY_NGRAM = 8                                  # compare 8-word sequences ("shingles")
JD_COPY_OVERLAP = 0.8                              # >= 80% shared shingles = copied sentence
MIN_WORDS_FOR_TEXT_LAYER = 30                      # fewer words => probably a scanned image

# Score penalty per flag:  final = final * (1 - penalty), total capped at MAX_PENALTY.
PENALTIES = {"hidden_text": 0.30, "tiny_font": 0.15, "jd_copy": 0.20}
MAX_PENALTY = 0.5

# ---------------------------------------------------------------------------
# 5) Upload limits
# ---------------------------------------------------------------------------
MAX_FILES = _int("MAX_FILES", 50)
MAX_FILE_MB = _float("MAX_FILE_MB", 5)
MAX_PAGES = _int("MAX_PAGES", 5)  # only the first MAX_PAGES pages of a resume are read
