"""
API tests: the endpoints the frontend calls. Uses FastAPI's TestClient (no server needed).
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"


def test_health():
    with TestClient(app) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_enabled"] is False  # no LLM_API_KEY in tests


def test_screen_sample():
    with TestClient(app) as client:
        response = client.post("/api/screen/sample?k=5")
    assert response.status_code == 200
    body = response.json()
    assert len(body["candidates"]) == 6
    assert body["evaluation"]["k"] == 5


def test_screen_upload_with_labels():
    pdfs = sorted((SAMPLE / "resumes").glob("*.pdf"))[:3]
    files = [("resumes", (p.name, p.read_bytes(), "application/pdf")) for p in pdfs]
    files.append(("labels", ("labels.csv", (SAMPLE / "labels.csv").read_bytes(), "text/csv")))
    data = {
        "jd_text": (SAMPLE / "job_description.txt").read_text(),
        "must_have": "Python, SQL, Docker",
        "nice_to_have": "AWS, Git",
    }
    with TestClient(app) as client:
        response = client.post("/api/screen", data=data, files=files)
    assert response.status_code == 200, response.text
    assert [c["rank"] for c in response.json()["candidates"]] == [1, 2, 3]


def test_rejects_non_pdf_and_empty_jd():
    with TestClient(app) as client:
        not_pdf = client.post("/api/screen", data={"jd_text": "Python developer"},
                              files=[("resumes", ("cv.txt", b"hello", "text/plain"))])
        empty_jd = client.post("/api/screen", data={"jd_text": "  "},
                               files=[("resumes", ("a.pdf", b"%PDF-1.4", "application/pdf"))])
    assert not_pdf.status_code == 400
    assert empty_jd.status_code == 400


def test_duplicate_upload_is_flagged():
    pdf = (SAMPLE / "resumes" / "rahul_verma.pdf").read_bytes()
    files = [("resumes", ("a.pdf", pdf, "application/pdf")),
             ("resumes", ("b.pdf", pdf, "application/pdf"))]
    with TestClient(app) as client:
        body = client.post("/api/screen", data={"jd_text": "Python developer with SQL"},
                           files=files).json()
    flagged = [c for c in body["candidates"] if "duplicate_file" in c["flags"]]
    assert len(flagged) == 1
