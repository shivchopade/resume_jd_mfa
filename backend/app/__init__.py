"""
Hybrid resume screener - backend package.

Pipeline (one module per box in the architecture diagram):

    preprocessing.py  PDF parsing (PyMuPDF) + integrity checks + section split
    semantic.py       Signal 1: SBERT / BGE embeddings + cosine similarity
    llm_eval.py       Signal 2: LLM rubric evaluation (optional)
    lexical.py        Signal 3: exact keyword matching (must-have / nice-to-have)
    fusion.py         Min-max normalisation + weighted sum (50 / 30 / 20) + ranking
    evaluation.py     nDCG@k against HR labels + weight tuning (grid search)
    pipeline.py       Runs all of the above for one JD and many resumes
    main.py           REST API (FastAPI)
    config.py         Every tunable number in one place
"""
