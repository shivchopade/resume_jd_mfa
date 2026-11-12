"""
Shared test setup.

The real SBERT model is ~90 MB and needs PyTorch. To keep the tests fast and
offline, every test uses FakeEncoder instead: a tiny "bag of words" vector.
It is NOT a semantic model, it only lets the rest of the pipeline run.

Tests also ignore your .env: they always use the default weights with the LLM off.
"""

import hashlib
import re

import numpy as np
import pytest

from app import config, semantic


class FakeEncoder:
    """Stand-in for SentenceTransformer: hashed word counts, L2-normalised."""

    dim = 512

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kwargs):
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9+#]+", text.lower()):
                vectors[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Same starting point for every test: fake encoder, default weights, LLM off."""
    monkeypatch.setattr(config, "W_SEMANTIC", 0.5)
    monkeypatch.setattr(config, "W_LEXICAL", 0.3)
    monkeypatch.setattr(config, "W_LLM", 0.2)
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    semantic.set_model(FakeEncoder())
    yield
