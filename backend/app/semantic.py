"""
semantic.py - signal 1 of 3: meaning-based matching with a BERT encoder (SBERT / BGE).

How the score is computed
-------------------------
1. The encoder turns a piece of text into a vector (384 numbers for MiniLM)
   that represents its MEANING. Texts with similar meaning point in a similar
   direction, even when the words differ ("built REST APIs with Flask" vs
   "Python backend development").

2. Cosine similarity between two vectors A and B:
        cos(A, B) = (A . B) / (|A| * |B|)          range -1..1, higher = closer
   Every vector is L2-normalised (|A| = |B| = 1), so cosine = plain dot product.

3. Resumes are longer than the model's 256-token limit, so the JD and every
   resume are split into chunks. For each JD chunk we take its BEST matching
   resume chunk, then average over the JD chunks:
        S_sem = mean over JD chunks j of ( max over resume chunks r of cos(j, r) )
   (For a short JD this is simply "the best-matching resume chunk".)

4. Manhattan distance is also computed, ONLY as a comparison metric for the report:
        D = sum_i |A_i - B_i|          similarity = 1 / (1 + D)
   Sentence-transformer models are trained for cosine, so cosine is what we fuse.
"""

from __future__ import annotations

import logging

import numpy as np

from . import config

log = logging.getLogger(__name__)

_model = None  # loaded once, then reused for every request


def get_model():
    """Load the embedding model the first time it is needed (takes a few seconds)."""
    global _model
    if _model is None:
        # Heavy import (PyTorch), so it only happens when the model is really needed.
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s ...", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _model


def set_model(model) -> None:
    """Use a different encoder object. The tests use this to avoid downloading a model."""
    global _model
    _model = model


def is_loaded() -> bool:
    return _model is not None


def embed(texts: list[str]) -> np.ndarray:
    """Encode texts into L2-normalised vectors. Shape: (number of texts, vector size)."""
    vectors = get_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def score_resumes(jd_chunks: list[str], resumes_chunks: list[list[str]]) -> list[dict]:
    """
    Semantic scores of many resumes against ONE job description.

    All resume chunks are encoded in a single batched call (much faster than
    encoding resume by resume). Returns one dict per resume:
        {"cosine": S_sem, "manhattan_similarity": 1 / (1 + D)}
    """
    jd_vectors = embed(jd_chunks)
    all_chunks = [chunk for chunks in resumes_chunks for chunk in chunks]
    all_vectors = embed(all_chunks) if all_chunks else np.zeros((0, jd_vectors.shape[1]))
    jd_document = _mean_vector(jd_vectors)

    results, start = [], 0
    for chunks in resumes_chunks:
        vectors = all_vectors[start:start + len(chunks)]
        start += len(chunks)
        if len(vectors) == 0:
            results.append({"cosine": 0.0, "manhattan_similarity": 0.0})
            continue

        # cosine for every (JD chunk, resume chunk) pair -> matrix of shape (jd, resume)
        similarity_matrix = jd_vectors @ vectors.T
        cosine = float(similarity_matrix.max(axis=1).mean())

        # Manhattan (L1) distance between whole-document vectors, for comparison only
        manhattan = float(np.abs(jd_document - _mean_vector(vectors)).sum())

        results.append({
            "cosine": round(cosine, 4),
            "manhattan_similarity": round(1.0 / (1.0 + manhattan), 4),
        })
    return results


def _mean_vector(vectors: np.ndarray) -> np.ndarray:
    """Average chunk vectors into one document vector, re-normalised to length 1."""
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean
