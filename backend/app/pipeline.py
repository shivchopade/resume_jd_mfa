"""
pipeline.py - runs the whole architecture for ONE job description and MANY resumes.

    resume PDFs --> preprocessing.py  (PyMuPDF parsing + integrity checks)
                         |
          +--------------+---------------+
          v              v               v
     semantic.py    llm_eval.py     lexical.py      <- 3 signals, computed independently
     (SBERT/BGE)    (optional)      (keywords)
          +--------------+---------------+
                         v
                    fusion.py        normalise + weights 50/30/20 + penalty -> ranking
                         v
                  evaluation.py      nDCG@k against HR labels (only if labels are given)

Used by both the REST API (main.py) and the terminal demo (scripts/demo_cli.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config, evaluation, fusion, lexical, llm_eval, preprocessing, semantic

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def screen(
    jd_text: str,
    files: list[tuple[str, bytes]],
    must_have_raw: str = "",
    nice_to_have_raw: str = "",
    labels: dict[str, float] | None = None,
    k: int = 10,
) -> dict:
    """Score and rank resumes for one JD. `files` is a list of (filename, pdf_bytes)."""
    started = time.perf_counter()
    jd_text = (jd_text or "").strip()
    if not jd_text:
        raise ValueError("The job description is empty.")
    if not files:
        raise ValueError("Upload at least one resume PDF.")

    # --- 1. What is the recruiter looking for? ------------------------------
    must = lexical.parse_skill_list(must_have_raw)
    nice = lexical.parse_skill_list(nice_to_have_raw)
    auto_extracted = False
    if not must and not nice:  # no lists given -> take known skills from the JD itself
        must = lexical.extract_skills(jd_text)
        auto_extracted = True

    # --- 2. Preprocessing + integrity checks (and duplicate uploads) --------
    parsed, seen = [], {}
    for name, data in files:
        resume = preprocessing.parse_resume_pdf(name, data, jd_text)
        if resume.sha256 in seen:
            resume.flags.append("duplicate_file")
            resume.evidence["duplicate_file"] = [f"same file as {seen[resume.sha256]}"]
        else:
            seen[resume.sha256] = name
        parsed.append(resume)

    scorable = [r for r in parsed if r.readable and r.chunks]
    unreadable = [r for r in parsed if not (r.readable and r.chunks)]

    jd_chunks = preprocessing.chunk_words(jd_text)
    llm_info = {"status": "off", "model": config.LLM_MODEL or None, "error": None}
    weights = fusion.effective_weights(llm_available=False)
    candidates, evaluation_result = [], None

    if scorable:
        # --- 3. The three signals ------------------------------------------
        sem = semantic.score_resumes(jd_chunks, [r.chunks for r in scorable])
        lex = [lexical.score(r.text, must, nice) for r in scorable]

        llm = None
        if llm_eval.is_enabled():
            llm = llm_eval.evaluate_many(jd_text, [r.text for r in scorable])
            failed = [x for x in llm if "error" in x]
            if failed:
                # Keep the comparison fair: if any call failed, nobody gets an LLM score.
                llm_info.update(status="failed", error=failed[0]["error"])
                llm = None
            else:
                llm_info["status"] = "on"

        raw = {"semantic": [s["cosine"] for s in sem], "lexical": [x["score"] for x in lex]}
        if llm is not None:
            raw["llm"] = [x["score"] for x in llm]

        # --- 4. Normalise + weighted fusion + penalty + ranking -------------
        weights = fusion.effective_weights(llm_available="llm" in raw)
        penalties = [fusion.penalty_for(r.flags) for r in scorable]
        normalised, finals = fusion.fuse(raw, weights, penalties)

        for position, i in enumerate(fusion.rank(finals), start=1):
            r = scorable[i]
            candidates.append({
                "rank": position,
                "filename": r.filename,
                "final_score": finals[i],
                "scores": {
                    "semantic": {
                        "raw": sem[i]["cosine"],
                        "normalised": round(normalised["semantic"][i], 4),
                        "manhattan_similarity": sem[i]["manhattan_similarity"],
                    },
                    "lexical": {
                        "raw": lex[i]["score"],
                        "normalised": round(normalised["lexical"][i], 4),
                        **{key: lex[i][key] for key in
                           ("must_matched", "must_missing", "nice_matched", "nice_missing")},
                    },
                    "llm": None if llm is None else {
                        "raw": llm[i]["score"],
                        "normalised": round(normalised["llm"][i], 4),
                        "rubric": llm[i]["rubric"],
                        "reasoning": llm[i]["reasoning"],
                    },
                },
                "penalty": penalties[i],
                "flags": r.flags,
                "evidence": r.evidence,
                "sections_found": [name for name in r.sections if name != "header"],
                "hr_label": labels.get(r.filename) if labels else None,
            })

        # --- 5. Optional evaluation against HR labels -----------------------
        if labels:
            evaluation_result = evaluation.evaluate(
                [r.filename for r in scorable], raw, penalties, labels, weights, k
            )

    # Unreadable files (corrupt, password-protected, scanned) go to the bottom.
    for r in unreadable:
        candidates.append({
            "rank": None, "filename": r.filename, "final_score": 0.0, "scores": None,
            "penalty": 0.0, "flags": r.flags, "evidence": r.evidence, "sections_found": [],
            "hr_label": labels.get(r.filename) if labels else None,
        })

    return {
        "job": {
            "must_have": must,
            "nice_to_have": nice,
            "skills_auto_extracted": auto_extracted,
            "jd_chunks": len(jd_chunks),
        },
        "embedding_model": config.EMBEDDING_MODEL,
        "weights": {name: round(value, 3) for name, value in weights.items()},
        "llm": llm_info,
        "candidates": candidates,
        "evaluation": evaluation_result,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def load_sample_data() -> dict:
    """The bundled demo: one JD, six resume PDFs and HR labels (backend/sample_data/)."""
    skills = json.loads((SAMPLE_DIR / "skills.json").read_text(encoding="utf-8"))
    return {
        "jd_text": (SAMPLE_DIR / "job_description.txt").read_text(encoding="utf-8"),
        "must_have": skills["must_have"],
        "nice_to_have": skills["nice_to_have"],
        "files": [(p.name, p.read_bytes()) for p in sorted((SAMPLE_DIR / "resumes").glob("*.pdf"))],
        "labels": evaluation.parse_labels_csv((SAMPLE_DIR / "labels.csv").read_bytes()),
    }
