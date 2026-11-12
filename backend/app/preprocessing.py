"""
preprocessing.py - box 1 of the architecture: "Preprocessing and integrity checks".

PDF bytes  ->  visible text + sections + chunks + integrity flags

How it works
------------
1. PyMuPDF opens the PDF and gives us every text "span" (a run of text with
   one font) together with its font size, colour, opacity and position.

2. Integrity checks - catching resume gaming:
     hidden_text   white or fully transparent text that a human cannot see,
                   but a keyword matcher would happily count
     tiny_font     text smaller than TINY_FONT_PT (e.g. 1 pt keyword stuffing)
     jd_copy       sentences pasted word-for-word from the job description
     duplicate_file / corrupt_pdf / encrypted_pdf / no_text_layer
   Hidden text, tiny text and JD-copied sentences are REMOVED before scoring, so
   the trick simply doesn't work (and the flag adds a penalty on top).
   White text on a dark box (e.g. a dark header band) IS visible, so it is kept.

3. The visible text is grouped into sections (skills, experience, projects ...)
   and cut into chunks small enough for the embedding model (CHUNK_WORDS words).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import pymupdf  # PyMuPDF (its old import name was "fitz")

from . import config

# Lines that usually start a resume section  ->  canonical section name.
SECTION_HEADINGS = {
    "summary": "summary", "profile": "summary", "objective": "summary", "about me": "summary",
    "professional summary": "summary", "career objective": "summary",
    "skills": "skills", "technical skills": "skills", "key skills": "skills",
    "core skills": "skills", "tech stack": "skills", "skills and tools": "skills",
    "experience": "experience", "work experience": "experience",
    "professional experience": "experience", "employment": "experience",
    "internship": "experience", "internships": "experience",
    "projects": "projects", "academic projects": "projects", "personal projects": "projects",
    "education": "education", "academics": "education", "qualifications": "education",
    "certifications": "certifications", "certificates": "certifications", "courses": "certifications",
    "achievements": "achievements", "awards": "achievements", "hackathons": "achievements",
}


@dataclass
class ParsedResume:
    """Everything the three scorers need to know about one resume."""

    filename: str
    sha256: str                                                  # file fingerprint (duplicate check)
    text: str = ""                                               # VISIBLE text only
    sections: dict[str, str] = field(default_factory=dict)       # {"skills": "...", ...}
    chunks: list[str] = field(default_factory=list)              # what the encoder reads
    flags: list[str] = field(default_factory=list)               # integrity problems found
    evidence: dict[str, list[str]] = field(default_factory=dict)  # examples behind each flag
    readable: bool = True                                        # False => cannot be scored
    pages: int = 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def parse_resume_pdf(filename: str, data: bytes, jd_text: str = "") -> ParsedResume:
    """Open one resume PDF and return its visible text, sections, chunks and integrity flags."""
    resume = ParsedResume(filename=filename, sha256=hashlib.sha256(data).hexdigest())

    # --- 1. Open the file safely -------------------------------------------
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        resume.flags.append("corrupt_pdf")
        resume.readable = False
        return resume
    if doc.needs_pass:  # password-protected PDF
        resume.flags.append("encrypted_pdf")
        resume.readable = False
        return resume

    resume.pages = doc.page_count
    if doc.page_count > config.MAX_PAGES:
        resume.flags.append("too_many_pages")  # informative only; we read the first MAX_PAGES

    # --- 2. Walk every text span and keep only what a human can see --------
    visible_lines: list[str] = []  # for section splitting
    blocks: list[str] = []         # paragraph-like units, for JD-copy detection
    hidden: list[str] = []
    tiny: list[str] = []

    for page in doc.pages(0, min(doc.page_count, config.MAX_PAGES)):
        dark_boxes = _dark_filled_boxes(page)
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # 0 = text block, 1 = image block
                continue
            block_lines = []
            for line in block["lines"]:
                kept = []
                for span in line["spans"]:
                    text = span["text"]
                    if not text.strip():
                        continue
                    transparent = span.get("alpha", 255) == 0
                    white_on_white = _is_near_white(span["color"]) and not _on_dark_box(
                        span["bbox"], dark_boxes
                    )
                    if transparent or white_on_white:
                        hidden.append(text.strip())
                        continue
                    if span["size"] < config.TINY_FONT_PT:
                        tiny.append(text.strip())
                        continue
                    kept.append(text)
                line_text = "".join(kept).strip()
                if line_text:
                    block_lines.append(line_text)
            if block_lines:
                visible_lines.extend(block_lines)
                blocks.append(" ".join(block_lines))
    doc.close()

    if hidden:
        resume.flags.append("hidden_text")
        resume.evidence["hidden_text"] = _preview(hidden)
    if tiny:
        resume.flags.append("tiny_font")
        resume.evidence["tiny_font"] = _preview(tiny)

    # --- 3. Text-layer check (scanned resumes have no real text) ----------
    resume.text = "\n".join(visible_lines)
    if len(resume.text.split()) < config.MIN_WORDS_FOR_TEXT_LAYER:
        resume.flags.append("no_text_layer")
        if not resume.text.strip():
            resume.readable = False
            return resume

    # --- 4. Sections ------------------------------------------------------
    resume.sections = split_sections(visible_lines)

    # --- 5. Sentences copied from the JD ----------------------------------
    copied = find_jd_copies(blocks, jd_text)
    if copied:
        resume.flags.append("jd_copy")
        resume.evidence["jd_copy"] = _preview(copied, max_len=120)
        # A sentence pasted from the JD says nothing about the candidate, so it is
        # removed before scoring - exactly like hidden text.
        resume.sections = {
            name: cleaned
            for name, body in resume.sections.items()
            if (cleaned := _remove_sentences(body, copied))
        }

    # --- 6. Final text (used by lexical + LLM) and chunks (used by the encoder)
    resume.text = "\n\n".join(
        body if name == "header" else f"{name.upper()}\n{body}"
        for name, body in resume.sections.items()
    )
    resume.chunks = make_chunks(resume.sections)
    return resume


# ---------------------------------------------------------------------------
# Integrity-check helpers
# ---------------------------------------------------------------------------
def _is_near_white(color: int) -> bool:
    """PyMuPDF stores a span colour as one integer 0xRRGGBB. True if R, G and B are all ~255."""
    red, green, blue = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return min(red, green, blue) >= config.NEAR_WHITE_MIN


def _dark_filled_boxes(page) -> list:
    """
    Rectangles on the page filled with a dark colour (a dark header band or sidebar).
    White text drawn on top of these IS visible, so it must not be flagged as hidden.
    (Limitation: dark background *images* are not checked, only vector shapes.)
    """
    boxes = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if not fill or len(fill) != 3:
            continue
        red, green, blue = fill  # floats between 0 and 1
        brightness = 0.299 * red + 0.587 * green + 0.114 * blue
        if brightness < 0.5:
            boxes.append(pymupdf.Rect(drawing["rect"]))
    return boxes


def _on_dark_box(bbox, dark_boxes) -> bool:
    """True if the centre of a text span lies inside one of the dark rectangles."""
    x0, y0, x1, y1 = bbox
    centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
    return any(centre in box for box in dark_boxes)


def _words(text: str) -> list[str]:
    """Lower-case word tokens. Keeps '+' and '#' so 'c++' and 'c#' survive."""
    return re.findall(r"[a-z0-9+#]+", text.lower())


# Sentence ends (. ! ? ;) or any common bullet glyph: • · ▪ ● ◦ ‣ ⁃ ■ ► ➢ ✓ and the
# private-use bullet that Word's Symbol font produces in exported PDFs.
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?;])\s+|\s*[•·▪●◦‣⁃■►➢✓]\s*"
)


def _shingles(words: list[str], size: int) -> set[tuple[str, ...]]:
    """All overlapping word sequences of length `size` ('shingles')."""
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def find_jd_copies(blocks: list[str], jd_text: str) -> list[str]:
    """
    Return resume sentences copied (almost) word-for-word from the job description.

    Method: cut the JD into overlapping 8-word sequences. If >= 80% of a resume
    sentence's 8-word sequences also appear in the JD, the sentence was copied.
    Short sentences (< JD_COPY_MIN_WORDS words) are ignored, because sharing a
    short phrase like "Python and SQL" is normal.
    """
    if not jd_text.strip():
        return []
    size = config.JD_COPY_NGRAM
    jd_shingles = _shingles(_words(jd_text), size)
    copied = []
    for block in blocks:
        # split a paragraph into sentences and bullet points
        for sentence in _SENTENCE_SPLIT.split(block):
            words = _words(sentence)
            if len(words) < config.JD_COPY_MIN_WORDS:
                continue
            sentence_shingles = _shingles(words, size)
            overlap = len(sentence_shingles & jd_shingles) / len(sentence_shingles)
            if overlap >= config.JD_COPY_OVERLAP:
                copied.append(sentence.strip())
    return copied


def _remove_sentences(text: str, sentences: list[str]) -> str:
    """Delete the given sentences from a text (whitespace-insensitive). Returns '' if nothing is left."""
    flat = " ".join(text.split())
    for sentence in sentences:
        flat = flat.replace(" ".join(sentence.split()), " ")
    flat = " ".join(flat.split())
    # nothing meaningful left (only bullet glyphs / punctuation)?
    return flat if re.search(r"[A-Za-z0-9]", flat) else ""


def _preview(items: list[str], limit: int = 3, max_len: int = 80) -> list[str]:
    """A few unique, shortened examples to show the recruiter why a flag was raised."""
    unique = list(dict.fromkeys(item for item in items if item))
    return [item if len(item) <= max_len else item[: max_len - 1] + "…" for item in unique[:limit]]


# ---------------------------------------------------------------------------
# Sections and chunks
# ---------------------------------------------------------------------------
def _heading_of(line: str) -> str | None:
    """If a line looks like a section heading ('SKILLS', 'Work Experience:'), return the section."""
    key = " ".join(re.sub(r"[^a-z ]", " ", line.lower()).split())
    if 0 < len(key.split()) <= 3:
        return SECTION_HEADINGS.get(key)
    return None


def split_sections(lines: list[str]) -> dict[str, str]:
    """Group lines under the latest heading. Text before the first heading goes to 'header'."""
    sections: dict[str, list[str]] = {}
    current = "header"
    for line in lines:
        heading = _heading_of(line)
        if heading:
            current = heading
            continue
        sections.setdefault(current, []).append(line)
    return {name: " ".join(body) for name, body in sections.items() if body}


def chunk_words(text: str, size: int | None = None) -> list[str]:
    """Split text into pieces of at most `size` words (the encoder's reading limit)."""
    size = size or config.CHUNK_WORDS
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def make_chunks(sections: dict[str, str]) -> list[str]:
    """One or more chunks per section, prefixed with the section name for extra context."""
    chunks = []
    for name, body in sections.items():
        prefix = "" if name == "header" else f"{name.capitalize()}: "
        chunks.extend(prefix + piece for piece in chunk_words(body))
    return chunks
