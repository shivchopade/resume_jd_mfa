# Hybrid Resume Screener — Quick Guide

## Summary

This project **ranks resumes (PDFs) against a job description** and explains why each candidate got their score.

It combines three signals into one final score:

| Signal | What it does | Default weight |
|---|---|---|
| **Semantic** | Compares the *meaning* of the resume and the job description using SBERT/BGE embeddings and cosine similarity | 0.5 |
| **Lexical** | Checks for exact must-have and nice-to-have keywords (e.g. Python, SQL, Docker) | 0.3 |
| **LLM** (optional) | An LLM scores the resume against a rubric and gives written reasoning | 0.2 |

If no LLM key is set, the LLM signal is switched off and its weight is shared between the other two.

**Cheat detection:** before scoring, each PDF is checked for hidden white text, tiny fonts (under 5 pt) and sentences copied from the job description. Flagged text is removed and the candidate's score is reduced. Duplicate, corrupt, password-protected and scanned PDFs are flagged too.

**Quality measurement:** if you give HR ratings (0–3) for the resumes, the app computes **nDCG@k** to show how closely its ranking matches HR's. It also tries all 66 weight combinations to find the best one.

**Tech stack:** Python, FastAPI, PyMuPDF, sentence-transformers, plain HTML/JS frontend served by nginx, Docker Compose.

---

## How to run

### Option 1: Docker (recommended)

Requirements: Docker Desktop, **running**, and about 3 GB of free disk.

```bash
docker compose up --build
```

Then open:

- Web app: http://localhost:8080
- API docs (Swagger): http://localhost:8000/docs

The first build takes a few minutes because it downloads PyTorch and the embedding model. Later starts take seconds. Stop with `Ctrl + C` or `docker compose down`.

### Option 2: Without Docker (Windows, PowerShell)

Requirements: Python 3.12 or newer.

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Start the API
uvicorn app.main:app --reload
```

In a second terminal, from the project root, serve the frontend:

```powershell
python -m http.server 5500 -d frontend/public
```

Open http://localhost:5500/?api=http://localhost:8000/api

(On Linux/macOS, activate with `source .venv/bin/activate`.)

### Option 3: Terminal demo only (no browser)

```bash
cd backend
python scripts/demo_cli.py
```

This scores the 6 bundled sample resumes and prints a ranking table. It takes about 50 seconds the first time because it downloads the embedding model.

---

## How to use

### In the web app

1. Open http://localhost:8080.
2. Press **Try sample data** to see an example, or provide your own inputs:
   - the **job description** text,
   - **must-have** skills (e.g. `Python, SQL, Docker`),
   - **nice-to-have** skills (e.g. `AWS, Git`),
   - **resume PDFs**,
   - optionally a **labels CSV** to get nDCG.
3. Read the ranked list. Each candidate shows the final score, the three sub-scores and any integrity flags.

### With your own files in the terminal

```bash
python scripts/demo_cli.py --jd my_data/jd.txt --resumes my_data/resumes \
  --must "Python, SQL" --nice "AWS, Git" --labels my_data/labels.csv
```

Labels file format (`relevance` is 0–3, where 3 is a perfect fit):

```csv
filename,relevance
ananya_sharma.pdf,3
rahul_verma.pdf,2
```

### With the API

```bash
curl -X POST http://localhost:8000/api/screen \
  -F "jd_text=<backend/sample_data/job_description.txt" \
  -F "must_have=Python, SQL, Docker" -F "nice_to_have=AWS, Git" \
  -F "resumes=@backend/sample_data/resumes/rahul_verma.pdf" \
  -F "resumes=@backend/sample_data/resumes/ananya_sharma.pdf"
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, model, weights, LLM on/off |
| GET | `/api/sample` | Sample JD and skill lists |
| POST | `/api/screen` | Screen your own resumes |
| POST | `/api/screen/sample` | Run on the bundled sample data |

### Run the tests

```bash
cd backend
pytest -q
```

---

## Configuration (optional)

Copy `.env.example` to `.env` and change what you need.

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (e.g. `BAAI/bge-small-en-v1.5`) |
| `W_SEMANTIC`, `W_LEXICAL`, `W_LLM` | 0.5, 0.3, 0.2 | Fusion weights |
| `LLM_API_KEY` | empty | Empty means the LLM signal is off |
| `LLM_MODEL` | empty | Any chat model your provider offers |
| `LLM_BASE_URL` | `https://api.x.ai/v1` | Any OpenAI-compatible API |

---

## Example output (sample data, LLM off)

```
 #  candidate            final  cosine  lexical  flags
 1  ananya_sharma.pdf    1.000   0.770    1.000  -
 2  rahul_verma.pdf      0.816   0.736    0.617  -
 3  sneha_patil.pdf      0.484   0.497    0.467  -
 4  priya_joshi.pdf      0.324   0.505    0.233  jd_copy
 5  amit_kulkarni.pdf    0.295   0.437    0.150  -
 6  rohan_deshmukh.pdf   0.000   0.231    0.000  hidden_text, tiny_font

nDCG@10 = 1.0000 (ranking matches the HR labels)
```

---

## Project layout

```
backend/app/        API and pipeline (preprocessing, semantic, lexical, llm_eval, fusion, evaluation)
backend/scripts/    demo_cli.py, make_sample_pdfs.py
backend/sample_data/ sample JD, 6 resumes, labels.csv
backend/tests/      pytest tests
frontend/public/    index.html, styles.css, app.js
docker-compose.yml  runs backend + frontend together
```

## Limitations

- Scores are normalised within the uploaded batch, so with only 1–2 resumes the normalised values are extreme. Check the raw scores too.
- Text hidden behind an image is not detected.
- If any LLM call fails, the LLM signal is dropped for the whole batch so everyone is compared fairly.
- The sample HR labels are illustrative. Tune weights with 30–50 real labelled resumes per job.
