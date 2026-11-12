"""
evaluation.py - box "Evaluation: nDCG". Is our ranking any good?

nDCG compares OUR ranking with an answer key made by HR. HR rates candidates
for a JD from 0 to 3 (3 = perfect fit, 2 = good, 1 = weak, 0 = not relevant).

    gain(rel) = 2^rel - 1                      3 -> 7,  2 -> 3,  1 -> 1,  0 -> 0
    DCG@k     = sum over positions i = 1..k of  gain(rel_i) / log2(i + 1)
                (position 1 counts fully, position 2 is divided by 1.58,
                 position 3 by 2 ... because HR mostly looks at the top)
    IDCG@k    = DCG@k of the perfect ordering (HR labels sorted high -> low)
    nDCG@k    = DCG@k / IDCG@k                 1.0 = exactly HR's order

    e.g. we rank labels [3, 1, 2]:  DCG = 7 + 1/1.58 + 3/2 = 9.13
         perfect order  [3, 2, 1]:  IDCG = 7 + 3/1.58 + 1/2 = 9.39
         nDCG = 9.13 / 9.39 = 0.97

Weight tuning: try every weight combination in steps of 0.1 that adds up to 1
(66 combinations), compute nDCG@k for each and keep the best one.
Use 30-50 labelled resumes per JD for a trustworthy result.
"""

from __future__ import annotations

import csv
import io
import math
from itertools import product

from . import fusion


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain of the first k positions."""
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_relevances: list[float], k: int) -> float:
    """HR labels in OUR ranking order -> nDCG@k between 0 and 1."""
    ideal = dcg_at_k(sorted(ranked_relevances, reverse=True), k)
    return dcg_at_k(ranked_relevances, k) / ideal if ideal > 0 else 0.0


def parse_labels_csv(data: bytes) -> dict[str, float]:
    """
    Read a labels file into {filename: relevance}. Expected format:

        filename,relevance
        ananya_sharma.pdf,3
        rahul_verma.pdf,2

    ('rating' or 'label' are also accepted instead of 'relevance'.)
    """
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    labels: dict[str, float] = {}
    for row in reader:
        row = {
            (key or "").strip().lower(): (value.strip() if isinstance(value, str) else "")
            for key, value in row.items()
        }
        name = row.get("filename") or row.get("file") or row.get("resume")
        value = row.get("relevance") or row.get("rating") or row.get("label")
        if name and value:
            labels[name] = max(0.0, min(3.0, float(value)))
    if not labels:
        raise ValueError("no rows found - expected a header 'filename,relevance'")
    return labels


def weight_grid(step: float = 0.1, include_llm: bool = True):
    """Every (semantic, lexical, llm) weight combination in steps of `step` that adds up to 1."""
    steps = round(1 / step)
    for sem, lex in product(range(steps + 1), repeat=2):
        llm = steps - sem - lex
        if llm < 0 or (not include_llm and llm != 0):
            continue
        yield {"semantic": sem / steps, "lexical": lex / steps, "llm": llm / steps}


def evaluate(filenames, raw, penalties, labels, current_weights, k: int = 10) -> dict:
    """nDCG@k of the current weights, plus a grid search for the weights with the best nDCG."""
    relevance = [labels.get(name, 0.0) for name in filenames]
    unlabelled = [name for name in filenames if name not in labels]

    def ndcg_for(weights: dict) -> float:
        _, finals = fusion.fuse(raw, weights, penalties)
        order = fusion.rank(finals)
        return ndcg_at_k([relevance[i] for i in order], k)

    current = ndcg_for(current_weights)
    trials = [(ndcg_for(w), w) for w in weight_grid(include_llm="llm" in raw)]
    best_score = max(score for score, _ in trials)

    # Small label sets often have several combinations tied at the top. Among the
    # ties, keep the one closest to the current weights, so results stay stable.
    tied = [w for score, w in trials if abs(score - best_score) < 1e-9]
    best = min(tied, key=lambda w: sum(abs(w[s] - current_weights.get(s, 0.0)) for s in w))

    return {
        "k": k,
        "ndcg_current": round(current, 4),
        "best_weights": best,
        "ndcg_best": round(best_score, 4),
        "combinations_tried": len(trials),
        "combinations_tied_at_best": len(tied),
        "labelled": len(filenames) - len(unlabelled),
        "unlabelled": unlabelled,  # counted as relevance 0
    }
