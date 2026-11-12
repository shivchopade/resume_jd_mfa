"""
fusion.py - box "Weights assign": combine the three signals into ONE score and rank.

Step 1 - min-max normalisation, per JD, for each signal separately:
        S_hat = (S - min S) / (max S - min S)
    The best candidate for this JD gets 1, the weakest gets 0.
    Why: raw cosine scores bunch up between ~0.3 and ~0.8 while the other signals
    use 0-1, so without this step the signals are not comparable.
    If all candidates have the same value (or there is only one), everyone gets 0.5.

    e.g. cosine scores for one JD: lowest 0.38, highest 0.88, Rahul 0.78
         S_hat = (0.78 - 0.38) / (0.88 - 0.38) = 0.80

Step 2 - weighted sum (like theory 50 + practical 30 + viva 20 in college marks):
        Final = 0.5 * S_hat_sem + 0.3 * S_hat_lex + 0.2 * S_hat_llm
    e.g. 0.5 * 0.80 + 0.3 * 0.60 + 0.2 * 0.70 = 0.72
    If the LLM is off, its 20% is shared in proportion: 0.625 semantic / 0.375 lexical.

Step 3 - integrity penalty:
        Final = Final * (1 - penalty)   hidden text 0.30, tiny font 0.15, JD copy 0.20 (max 0.5)

Step 4 - sort by Final, highest first  ->  the ranking for this JD.
"""

from __future__ import annotations

from . import config


def min_max(values: list[float]) -> list[float]:
    """Rescale values so the lowest becomes 0 and the highest becomes 1."""
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def effective_weights(llm_available: bool, weights: dict | None = None) -> dict:
    """Weights that add up to 1. The LLM weight is dropped when the LLM signal is unavailable."""
    w = dict(weights or {
        "semantic": config.W_SEMANTIC,
        "lexical": config.W_LEXICAL,
        "llm": config.W_LLM,
    })
    if not llm_available:
        w["llm"] = 0.0
    total = sum(w.values())
    if total <= 0:
        raise ValueError("At least one weight must be greater than 0")
    return {name: value / total for name, value in w.items()}


def penalty_for(flags: list[str]) -> float: 
    """Total integrity penalty for a list of flags (capped at MAX_PENALTY)."""
    return min(config.MAX_PENALTY, sum(config.PENALTIES.get(flag, 0.0) for flag in set(flags)))


def fuse(raw: dict[str, list[float]], weights: dict, penalties: list[float]):
    """
    raw       = {"semantic": [...], "lexical": [...], "llm": [...]}  one value per candidate
    weights   = {"semantic": 0.5, "lexical": 0.3, "llm": 0.2}
    penalties = [0.0, 0.3, ...]                                       one value per candidate

    Returns (normalised scores per signal, final score per candidate).
    """
    normalised = {signal: min_max(values) for signal, values in raw.items()}
    finals = []
    for i, penalty in enumerate(penalties):
        score = sum(weights.get(signal, 0.0) * normalised[signal][i] for signal in normalised)
        finals.append(round(score * (1.0 - penalty), 4))
    return normalised, finals


def rank(finals: list[float]) -> list[int]:
    """Candidate indices sorted by final score, best first."""
    return sorted(range(len(finals)), key=lambda i: finals[i], reverse=True)
