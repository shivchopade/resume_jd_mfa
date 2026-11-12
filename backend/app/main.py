"""
main.py - the REST API (FastAPI).

Endpoints
---------
GET  /api/health          is the service up? which model, weights and LLM status?
GET  /api/sample          the bundled sample JD + skill lists + resume file names
POST /api/screen          upload a JD + resume PDFs (+ optional labels.csv) -> ranking
POST /api/screen/sample   run the whole pipeline on the bundled sample data

Interactive API docs (Swagger): http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config, evaluation, llm_eval, pipeline, semantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("resume-screener")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model once at startup, so the first request is not slow."""
    try:
        semantic.get_model()
        log.info("Embedding model ready: %s", config.EMBEDDING_MODEL)
    except Exception as exc:  # e.g. no internet and the model is not in the image
        log.error("Could not load the embedding model (%s). Screening will fail until it loads.", exc)
    log.info("LLM signal: %s", "ON" if llm_eval.is_enabled() else "OFF (no LLM_API_KEY / LLM_MODEL)")
    yield


app = FastAPI(
    title="Hybrid Resume Screener",
    description="SBERT/BGE semantic matching + LLM rubric + lexical matching, "
                "weighted fusion and nDCG evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)

# In Docker the frontend reaches us through nginx (same origin), so CORS is not needed.
# Allowing all origins just makes local experiments easier (e.g. opening index.html directly).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    """Service status, shown in the header of the frontend."""
    return {
        "status": "ok",
        "embedding_model": config.EMBEDDING_MODEL,
        "model_loaded": semantic.is_loaded(),
        "llm_enabled": llm_eval.is_enabled(),
        "llm_model": config.LLM_MODEL or None,
        "weights": {"semantic": config.W_SEMANTIC, "lexical": config.W_LEXICAL, "llm": config.W_LLM},
    }


@app.get("/api/sample")
def sample():
    """The sample JD and skill lists, so the frontend can fill its form with them."""
    data = pipeline.load_sample_data()
    return {
        "jd_text": data["jd_text"],
        "must_have": data["must_have"],
        "nice_to_have": data["nice_to_have"],
        "resumes": [name for name, _ in data["files"]],
        "labels": data["labels"],
    }


@app.post("/api/screen")
def screen(
    jd_text: str = Form(..., description="Full job description text"),
    must_have: str = Form("", description="Comma-separated must-have skills"),
    nice_to_have: str = Form("", description="Comma-separated nice-to-have skills"),
    k: int = Form(10, ge=1, le=100, description="k for nDCG@k"),
    resumes: list[UploadFile] = File(..., description="Resume PDFs"),
    labels: UploadFile | None = File(None, description="Optional labels.csv (filename,relevance)"),
):
    """Score and rank uploaded resumes against one job description."""
    if len(resumes) > config.MAX_FILES:
        raise HTTPException(400, f"Upload at most {config.MAX_FILES} resumes at a time.")

    max_bytes = int(config.MAX_FILE_MB * 1024 * 1024)
    files = []
    for upload in resumes:
        name = os.path.basename(upload.filename or "resume.pdf")
        data = upload.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"{name} is larger than {config.MAX_FILE_MB:g} MB.")
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, f"{name} is not a PDF file.")
        files.append((name, data))

    label_map = None
    if labels is not None and labels.filename:
        try:
            label_map = evaluation.parse_labels_csv(labels.file.read())
        except Exception as exc:
            raise HTTPException(400, f"Could not read the labels CSV: {exc}")

    return _run(jd_text, files, must_have, nice_to_have, label_map, k)


@app.post("/api/screen/sample")
def screen_sample(k: int = 10):
    """Run the full pipeline on the bundled sample data (1 JD, 6 resumes, HR labels)."""
    data = pipeline.load_sample_data()
    return _run(data["jd_text"], data["files"], data["must_have"], data["nice_to_have"],
                data["labels"], k)


def _run(jd_text, files, must_have, nice_to_have, labels, k):
    """Call the pipeline and turn errors into clear HTTP responses."""
    try:
        return pipeline.screen(jd_text, files, must_have, nice_to_have, labels, k)
    except ValueError as exc:  # bad input, e.g. empty JD
        raise HTTPException(400, str(exc))
    except (OSError, ImportError, RuntimeError) as exc:  # model could not be loaded
        log.exception("Screening failed")
        raise HTTPException(503, f"Embedding model not available: {exc}")
