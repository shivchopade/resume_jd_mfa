"""
demo_cli.py - run the full pipeline from the terminal (no browser needed).

With Docker (from the project root):
    docker compose run --rm backend python scripts/demo_cli.py
    docker compose run --rm backend python scripts/demo_cli.py --resumes /data/pdfs --jd /data/jd.txt

Without Docker (from the backend/ folder, after `pip install -r requirements.txt`):
    python scripts/demo_cli.py

With no arguments it uses the bundled sample data (1 JD, 6 resumes, HR labels).
"""

import argparse
import sys
from pathlib import Path

# Make the "app" package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import evaluation, pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid resume screener - terminal demo")
    parser.add_argument("--jd", help="path to a .txt job description (default: sample JD)")
    parser.add_argument("--resumes", help="folder with resume PDFs (default: sample resumes)")
    parser.add_argument("--labels", help="labels.csv with HR ratings 0-3, for nDCG")
    parser.add_argument("--must", help='must-have skills, e.g. "Python, SQL, Docker"')
    parser.add_argument("--nice", help='nice-to-have skills, e.g. "AWS, Git"')
    parser.add_argument("--k", type=int, default=10, help="k for nDCG@k (default 10)")
    args = parser.parse_args()

    sample = pipeline.load_sample_data()
    custom = bool(args.jd or args.resumes)

    jd_text = Path(args.jd).read_text(encoding="utf-8") if args.jd else sample["jd_text"]
    if args.resumes:
        files = [(p.name, p.read_bytes()) for p in sorted(Path(args.resumes).glob("*.pdf"))]
    else:
        files = sample["files"]
    must = args.must if args.must is not None else ("" if custom else sample["must_have"])
    nice = args.nice if args.nice is not None else ("" if custom else sample["nice_to_have"])
    if args.labels:
        labels = evaluation.parse_labels_csv(Path(args.labels).read_bytes())
    else:
        labels = None if custom else sample["labels"]

    print(f"Scoring {len(files)} resume(s) ... (the first run loads the embedding model)")
    result = pipeline.screen(jd_text, files, must, nice, labels, args.k)

    w = result["weights"]
    print(f"\nEmbedding model : {result['embedding_model']}")
    print(f"LLM signal      : {result['llm']['status']}"
          + (f" ({result['llm']['error']})" if result["llm"]["error"] else ""))
    print(f"Weights         : semantic {w['semantic']:.3f} | lexical {w['lexical']:.3f} | llm {w['llm']:.3f}")
    print(f"Must-have       : {', '.join(result['job']['must_have']) or '-'}")
    print(f"Nice-to-have    : {', '.join(result['job']['nice_to_have']) or '-'}\n")

    header = f"{'#':>2}  {'candidate':<22} {'final':>6} {'cosine':>7} {'lexical':>8} {'llm':>6}  flags"
    print(header)
    print("-" * len(header))
    for c in result["candidates"]:
        if c["scores"] is None:
            print(f"{'-':>2}  {c['filename']:<22} {'-':>6} {'-':>7} {'-':>8} {'-':>6}  {', '.join(c['flags'])}")
            continue
        s = c["scores"]
        llm = f"{s['llm']['raw']:.2f}" if s["llm"] else "-"
        print(f"{c['rank']:>2}  {c['filename']:<22} {c['final_score']:>6.3f} "
              f"{s['semantic']['raw']:>7.3f} {s['lexical']['raw']:>8.3f} {llm:>6}  "
              f"{', '.join(c['flags']) or '-'}")

    ev = result["evaluation"]
    if ev:
        bw = ev["best_weights"]
        print(f"\nnDCG@{ev['k']} with current weights : {ev['ndcg_current']:.4f}")
        print(f"Best of {ev['combinations_tried']} weight combinations: "
              f"semantic {bw['semantic']:.1f} | lexical {bw['lexical']:.1f} | llm {bw['llm']:.1f}"
              f"  ->  nDCG@{ev['k']} {ev['ndcg_best']:.4f}")
    print(f"\nDone in {result['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
