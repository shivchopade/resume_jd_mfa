# Hybrid Resume Screener

Ranks resumes for a job description by combining three independent signals into one weighted score:

- **Semantic** — SBERT / BGE embeddings + cosine similarity (understands meaning)
- **LLM evaluation** — rubric-based score with written reasoning (optional, supporting signal)
- **Lexical** — exact must-have / nice-to-have keyword matching

Before scoring, every PDF goes through **integrity checks** (hidden white text, tiny fonts, sentences copied from the JD). Ranking quality is measured with **nDCG@k** against HR labels, which is also used to tune the weights.

## Architecture

```mermaid
flowchart LR
    R[Recruiter] -->|uploads| JD[Job description]
    C[Candidates] -->|apply| PDF[Resume PDFs]
    JD --> P
    PDF --> P[Preprocessing + integrity checks<br/>PyMuPDF parsing]
    P --> S[Semantic<br/>SBERT / BGE + cosine]
    P --> L[LLM evaluation<br/>rubric score]
    P --> K[Lexical matching<br/>exact keywords]
    S --> F[Normalise + weighted fusion<br/>0.5 / 0.3 / 0.2]
    L --> F
    K --> F
    F --> RK[Ranked candidates per JD]
    RK -.-> E[nDCG@k vs HR labels<br/>tunes the weights]
```

| Box in the diagram | Code |
|---|---|
| Preprocessing + integrity checks | `backend/app/preprocessing.py` |
| SBERT / BGE + cosine (+ Manhattan for comparison) | `backend/app/semantic.py` |
| LLM based evaluation | `backend/app/llm_eval.py` |
| Lexical matching | `backend/app/lexical.py` |
| Weights assign | `backend/app/fusion.py` |
| Evaluation: nDCG | `backend/app/evaluation.py` |
| Whole flow | `backend/app/pipeline.py` |

## Quick start (Docker)

You only need [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine with Compose v2) and about 3 GB of free disk.

```bash
git clone https://github.com/shivchopade/hybrid-resume-screener.git
cd hybrid-resume-screener
docker compose up --build
```

Then open **http://localhost:8080** and press **Try sample data**.

The first build downloads PyTorch (CPU version) and the embedding model, so it takes a few minutes. After that, the app starts in seconds and works offline.

| URL | What |
|---|---|
| http://localhost:8080 | Web app |
| http://localhost:8000/docs | Interactive API docs (Swagger) |

Stop with `Ctrl + C`, or `docker compose down`.

### Terminal demo (no browser)

```bash
docker compose run --rm backend python scripts/demo_cli.py
```

With your own files (put them in a `my_data` folder):

```bash
docker compose run --rm -v "$PWD/my_data:/data" backend \
  python scripts/demo_cli.py --jd /data/jd.txt --resumes /data/resumes \
  --must "Python, SQL" --nice "AWS, Git" --labels /data/labels.csv
```

### Tests

```bash
docker compose run --rm backend pytest -q
```

The tests use a tiny stand-in encoder, so they run offline in about a second.

## How the score works

### 1. Preprocessing and integrity checks

- PyMuPDF reads every text span with its font size, colour and position.
- **Hidden text** (white or transparent), **tiny text** (< 5 pt) and **sentences copied from the JD** are flagged and **removed before scoring**, so the trick doesn't work. Each flag also adds a penalty.
- White text on a dark header or sidebar is visible, so it is *not* flagged.
- Duplicate uploads and corrupt, password-protected or scanned PDFs are flagged too.
- The text is split into sections (skills, experience, projects …) and chunks of at most 150 words, because MiniLM reads only 256 tokens and silently cuts the rest.

### 2. Three signals

| Signal | Formula |
|---|---|
| Semantic | `cos(A, B) = A·B / (‖A‖·‖B‖)`, then `S_sem = mean over JD chunks of (max over resume chunks of cos)` |
| LLM | `S_llm = (0.4·skills + 0.3·experience + 0.2·projects + 0.1·education) / 10` |
| Lexical | `S_lex = 0.7 · (must-have found / must-have total) + 0.3 · (nice-to-have found / nice-to-have total)` |

Manhattan distance between the document vectors is also reported (`similarity = 1 / (1 + D)`), for comparison only. SBERT and BGE are trained for cosine, so cosine is what gets fused.

### 3. Normalise, then fuse with weights

```
Ŝ     = (S − min S) / (max S − min S)        per JD, for each signal separately
Final = 0.5·Ŝ_sem + 0.3·Ŝ_lex + 0.2·Ŝ_llm
Final = Final × (1 − penalty)                 hidden 0.30, tiny 0.15, JD copy 0.20 (max 0.50)
```

It works like college marks: theory 50%, practical 30%, viva 20%. Normalising first matters, because raw cosine scores bunch up between about 0.3 and 0.8 while the other signals use 0 to 1. If the LLM is off, its 20% is shared in proportion (0.625 semantic, 0.375 lexical).

**Worked example:** for one JD the lowest cosine is 0.38 and the highest is 0.88. Rahul's cosine is 0.78, so `Ŝ_sem = (0.78 − 0.38) / 0.50 = 0.80`. With `Ŝ_lex = 0.60` and `Ŝ_llm = 0.70`:
`Final = 0.5·0.80 + 0.3·0.60 + 0.2·0.70 = 0.72`. Sorting all candidates by `Final` gives the ranking.

### 4. Evaluation with nDCG@k

HR rates candidates from 0 to 3 (3 = perfect fit). nDCG compares our order with HR's:

```
gain(rel) = 2^rel − 1                  3 → 7, 2 → 3, 1 → 1, 0 → 0
DCG@k     = Σ gain(rel_i) / log2(i + 1)
nDCG@k    = DCG@k / IDCG@k             IDCG = DCG of HR's perfect order; 1.0 = same order
```

**Example:** our order has HR labels `[3, 1, 2]` → DCG = 7 + 1/1.58 + 3/2 = 9.13. The perfect order `[3, 2, 1]` gives IDCG = 9.39, so nDCG = 0.97.

**Weight tuning:** the backend tries all 66 weight combinations (steps of 0.1 that add up to 1) and reports the one with the highest nDCG. Use 30–50 labelled resumes per JD before trusting the tuned weights.

Labels file (`filename,relevance`):

```csv
filename,relevance
ananya_sharma.pdf,3
rahul_verma.pdf,2
```

## Configuration

Copy `.env.example` to `.env` and change what you need (all settings are optional):

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | SBERT or BGE model (rebuild after changing) |
| `W_SEMANTIC`, `W_LEXICAL`, `W_LLM` | `0.5`, `0.3`, `0.2` | Fusion weights |
| `LLM_API_KEY` | empty | Empty = LLM signal off |
| `LLM_MODEL` | empty | Any chat model your provider offers |
| `LLM_BASE_URL` | `https://api.x.ai/v1` | Any OpenAI-compatible API (xAI, OpenAI, Groq, Ollama …) |

Switch to BGE and compare with nDCG:

```bash
echo "EMBEDDING_MODEL=BAAI/bge-small-en-v1.5" >> .env
docker compose up --build
```

## Sample data

`backend/sample_data/` has one JD, six fictional resumes and HR labels. Each resume tests something:

| Resume | HR label | What it tests |
|---|---|---|
| `ananya_sharma.pdf` | 3 | Strong fit; white text on a dark header band must **not** be flagged |
| `rahul_verma.pdf` | 2 | Good fit, but no Docker or AWS |
| `sneha_patil.pdf` | 1 | Data analyst: Python + SQL, no backend work |
| `priya_joshi.pdf` | 1 | Copies two sentences from the JD → `jd_copy` |
| `amit_kulkarni.pdf` | 0 | Frontend developer |
| `rohan_deshmukh.pdf` | 0 | Hides keywords in white and 1 pt text → `hidden_text`, `tiny_font` |

Regenerate them with `python backend/scripts/make_sample_pdfs.py`.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status, model, weights, LLM on/off |
| `GET` | `/api/sample` | Sample JD and skill lists |
| `POST` | `/api/screen` | Multipart form: `jd_text`, `must_have`, `nice_to_have`, `k`, `resumes` (PDFs), optional `labels` (CSV) |
| `POST` | `/api/screen/sample?k=10` | Run everything on the sample data |

```bash
curl -X POST http://localhost:8000/api/screen \
  -F "jd_text=<backend/sample_data/job_description.txt" \
  -F "must_have=Python, SQL, Docker" -F "nice_to_have=AWS, Git" \
  -F "resumes=@backend/sample_data/resumes/rahul_verma.pdf" \
  -F "resumes=@backend/sample_data/resumes/ananya_sharma.pdf"
```

## Project structure

```
hybrid-resume-screener/
├── docker-compose.yml        # runs backend + frontend together
├── .env.example              # optional settings (copy to .env)
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py           # REST API (FastAPI)
│   │   ├── pipeline.py       # runs the whole architecture
│   │   ├── preprocessing.py  # PyMuPDF parsing + integrity checks
│   │   ├── semantic.py       # SBERT / BGE + cosine (+ Manhattan)
│   │   ├── llm_eval.py       # LLM rubric evaluation (optional)
│   │   ├── lexical.py        # keyword matching
│   │   ├── fusion.py         # normalisation + weights + ranking
│   │   ├── evaluation.py     # nDCG@k + weight tuning
│   │   └── config.py         # all tunable numbers
│   ├── scripts/
│   │   ├── demo_cli.py       # terminal demo
│   │   └── make_sample_pdfs.py
│   ├── sample_data/          # JD, 6 resumes, labels.csv
│   └── tests/
└── frontend/
    ├── Dockerfile            # nginx
    ├── nginx.conf            # serves the UI, forwards /api to the backend
    └── public/               # index.html, styles.css, app.js
```

## Run without Docker (optional)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In a second terminal, serve the frontend and point it at the backend:

```bash
python -m http.server 5500 -d frontend/public
```

Open http://localhost:5500/?api=http://localhost:8000/api

## Limitations

- Min-max normalisation is relative to the uploaded batch. With only one or two resumes, the normalised values are extreme (0 / 1 or 0.5), so also look at the raw scores.
- The hidden-text check covers white and transparent text and dark vector shapes. Text hidden behind an image is not detected.
- If any LLM call fails, the LLM signal is dropped for the whole batch, so every candidate is compared on the same signals.
- The sample HR labels are illustrative. Real weight tuning needs real labels from recruiters.
#   r e s u m e _ j d _ m f a  
 