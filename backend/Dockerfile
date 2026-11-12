# ---------------------------------------------------------------------------
# Backend image: FastAPI + PyMuPDF + SBERT/BGE embeddings (CPU only)
# Build + run from the project root with:  docker compose up --build
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# No .pyc files, unbuffered logs, no pip cache; Hugging Face models live inside the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# A normal (non-root) user to run the app
RUN useradd --create-home appuser

# 1) CPU-only PyTorch (~200 MB) instead of the default CUDA build (~2 GB).
#    Installed first, so this slow layer stays cached when you change other things.
RUN pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu

# 2) The rest of the Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Download the embedding model INTO the image, so the container works offline later.
#    To use BGE instead:  EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 docker compose up --build
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN mkdir -p "$HF_HOME" && chown appuser "$HF_HOME"
USER appuser
RUN python -c "import os; from sentence_transformers import SentenceTransformer; \
SentenceTransformer(os.environ['EMBEDDING_MODEL'], device='cpu')"

# 4) Application code last, so code changes don't re-run the slow steps above
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
